"""Train and load shadow (reference) models for LiRA and RMIA.

A shadow model is a model trained on a random half of the data pool, so that
for any query point some shadows trained on it (IN) and some did not (OUT).
RMIA needs only the OUT side; LiRA needs both. Both attacks consume the exact
same artifacts, so one module produces them.

Recipe fidelity is the whole point. LiRA calibrates the target's output against
the distribution of outputs *under the target's training recipe*: shadows must
therefore be built the same way the target was (same init, same adaptation,
same defenses). We get that by construction here -- shadows are trained through
the very same :class:`~trustml.training.trainer.Trainer` as the targets, driven
by a config that mirrors the target experiment. (The previous version trained
plain from-scratch models with no pretrain, no last-block adaptation and no
DP/TRADES, which mis-calibrated the reference distribution.)

Layout written to ``<out_dir>/``::

    manifest.json      # pool/subset/seed parameters (identical from every shard)
    masks.npy          # bool [num_models, pool_size]; True = index was trained on
    shadow_<k>.pt      # standard trustml checkpoints (model state + config)

Masks are a deterministic function of (pool_size, subset_fraction, base seed,
num_models), so every shard can regenerate the full matrix without coordination
and ``load_shadow_models`` simply uses whichever ``shadow_<k>.pt`` exist. That
makes it safe to split the (expensive) training across parallel jobs.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from trustml.core.model_loading import load_checkpoint_model
from trustml.core.seed import seed_everything
from trustml.data.transforms import CIFAR10_MEAN, CIFAR10_STD, cifar_transform
from trustml.defenses.registry import build_defenses
from trustml.evaluation.clean import evaluate_clean
from trustml.logging.base import Logger
from trustml.models.registry import build_model
from trustml.training.trainer import Trainer

MANIFEST_NAME = "manifest.json"
MASKS_NAME = "masks.npy"
CHECKPOINT_GLOB = "shadow_*.pt"
_CHECKPOINT_RE = re.compile(r"shadow_(\d+)\.pt$")

# Loaded shadow sets, keyed by (dir, device, max_models). Scoring a whole sweep
# asks for the same set once per target; see load_shadow_models.
_SHADOW_CACHE: dict[tuple, tuple] = {}


def resolve_device(cfg: DictConfig) -> torch.device:
    requested = str(cfg.get("training", {}).get("device", "auto")).lower()
    if requested in {"auto", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            raise RuntimeError("training.device=cuda but torch.cuda.is_available() is False.")
        return torch.device("cpu")
    return torch.device(requested)


def subset_indices(pool_size: int, fraction: float, seed: int) -> np.ndarray:
    """A deterministic random subset of pool indices for one shadow model."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}")
    n = int(round(pool_size * fraction))
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(pool_size, generator=generator)[:n]
    return np.sort(perm.numpy())


def build_masks(pool_size: int, num_models: int, fraction: float, base_seed: int) -> np.ndarray:
    """Full [num_models, pool_size] membership matrix; True = model k trained on i.

    Deterministic in its arguments alone, so parallel shards agree without
    exchanging anything.
    """
    masks = np.zeros((int(num_models), int(pool_size)), dtype=bool)
    for k in range(int(num_models)):
        masks[k, subset_indices(pool_size, fraction, int(base_seed) + k)] = True
    return masks


def out_mask_for_indices(masks: np.ndarray, pool_indices) -> np.ndarray:
    """Shadow OUT-mask for query pool indices: [num_shadows, num_queries].

    True where a shadow did NOT train on that pool index.
    """
    pool_indices = np.asarray(list(pool_indices), dtype=np.int64)
    return ~masks[:, pool_indices]


def in_mask_for_indices(masks: np.ndarray, pool_indices) -> np.ndarray:
    """Shadow IN-mask for query pool indices (the complement of the OUT-mask)."""
    pool_indices = np.asarray(list(pool_indices), dtype=np.int64)
    return masks[:, pool_indices]


class _ShadowLogger(Logger):
    """Minimal logger for shadow training.

    ``output_dir`` stays None on purpose: it makes ``Trainer.save_checkpoint``
    skip the artifact registry, so 64 shadow models never pollute the
    experiment alias registry under ``outputs/_artifacts``.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self.output_dir = None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, float] = {}

    def log_metrics(self, metrics, step=None) -> None:
        self.metrics.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})


def _cifar10_pool(cfg: DictConfig):
    data_cfg = cfg.data
    root = Path(str(data_cfg.get("root", "./data"))).expanduser()
    mean = tuple(data_cfg.get("mean", CIFAR10_MEAN))
    std = tuple(data_cfg.get("std", CIFAR10_STD))
    download = bool(data_cfg.get("download", True))
    augmentation = bool(data_cfg.get("augmentation", True))
    train_tfm = cifar_transform(train=True, mean=mean, std=std, augmentation=augmentation)
    eval_tfm = cifar_transform(train=False, mean=mean, std=std, augmentation=False)
    pool = torchvision.datasets.CIFAR10(root=str(root), train=True, transform=train_tfm, download=download)
    test = torchvision.datasets.CIFAR10(root=str(root), train=False, transform=eval_tfm, download=download)
    return pool, test


def _shadow_loaders(cfg: DictConfig, pool, test, subset: np.ndarray) -> dict:
    batch_size = int(cfg.data.get("batch_size", 128))
    num_workers = int(cfg.data.get("num_workers", 0))
    # persistent_workers stays False: several loaders with lingering worker
    # pools make the interpreter exit non-zero on headless VMs (a finished
    # DataSphere job then reports ERROR).
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "persistent_workers": False}
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **kwargs)
    return {
        "train": DataLoader(Subset(pool, subset.tolist()), batch_size=batch_size, shuffle=True, **kwargs),
        "val": test_loader,
        "test": test_loader,
    }


def _train_one_shadow(cfg: DictConfig, k: int, subset: np.ndarray, pool, test,
                      out_dir: Path, seed: int, progress: bool) -> dict:
    seed_everything(seed)
    work_dir = out_dir / f"_work_{k}"
    logger = _ShadowLogger(work_dir / "checkpoints")

    model = build_model(cfg.model, root_cfg=cfg)  # keeps weights_path -> the shared pretrain
    defenses = build_defenses(cfg.get("defenses", []))
    data = _shadow_loaders(cfg, pool, test, subset)

    trainer = Trainer(cfg=cfg, model=model, data=data, defenses=defenses, logger=logger)
    trainer.fit()

    # Mirror how targets are scored: prefer the best-val checkpoint. Trainer has
    # already written it in the trustml checkpoint format (state + embedded cfg),
    # so the shadow artifact is just that file under its final name.
    produced = logger.checkpoint_dir / "best.pt"
    if not produced.exists():
        produced = logger.checkpoint_dir / "latest.pt"
    if not produced.exists():
        raise RuntimeError(f"shadow {k}: Trainer produced no checkpoint in {logger.checkpoint_dir}")
    destination = out_dir / f"shadow_{k}.pt"
    shutil.move(str(produced), str(destination))

    metrics = evaluate_clean(trainer.model, data["test"], trainer.device,
                             max_batches=cfg.training.get("max_eval_batches", None))
    shutil.rmtree(work_dir, ignore_errors=True)
    record = {"index": k, "checkpoint": destination.name, "seed": seed,
              "num_train": int(subset.size), "test_acc": float(metrics.get("clean_acc", 0.0))}
    if progress:
        print(f"[shadow] {k}: n_train={subset.size} test_acc={record['test_acc']:.4f}", flush=True)
    return record


def train_shadow_models(cfg: DictConfig, out_dir: str | Path, device: torch.device | None = None,
                        model_range: tuple[int, int] | None = None, progress: bool = True) -> dict:
    """Train the configured shadow models (optionally only a shard) and persist them.

    ``model_range`` is a half-open [start, end) slice of model indices, so the
    expensive sweep can be split across parallel jobs. Masks and manifest are
    written identically by every shard.
    """
    if str(cfg.data.get("name", "cifar10")).lower() != "cifar10":
        raise ValueError("Shadow-model training is only implemented for cifar10.")
    device = device or resolve_device(cfg)
    shadow_cfg = cfg.get("shadow", cfg.get("references", {})) or {}
    num_models = int(shadow_cfg.get("num_models", 64))
    fraction = float(shadow_cfg.get("subset_fraction", 0.5))
    base_seed = int(shadow_cfg.get("seed", 1000))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool, test = _cifar10_pool(cfg)
    pool_size = len(pool)
    masks = build_masks(pool_size, num_models, fraction, base_seed)
    np.save(out_dir / MASKS_NAME, masks)

    manifest = {
        "dataset": "cifar10",
        "pool_size": pool_size,
        "subset_fraction": fraction,
        "shadow_seed": base_seed,
        "num_models": num_models,
        "masks_file": MASKS_NAME,
        "model_arch": str(cfg.model.get("name", "resnet18")),
        "recipe": {
            "weights_path": str(cfg.model.get("weights_path", None)),
            "adaptation": str(cfg.get("adaptation", {}).get("method", "full_finetune")),
            "epochs": int(cfg.training.get("epochs", 1)),
            "defenses": [str(d.get("name")) for d in (cfg.get("defenses", []) or []) if d.get("enabled", True)],
        },
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    start, end = model_range if model_range is not None else (0, num_models)
    start, end = max(0, int(start)), min(int(end), num_models)
    records = []
    for k in range(start, end):
        if (out_dir / f"shadow_{k}.pt").exists():
            if progress:
                print(f"[shadow] {k}: already present, skipping", flush=True)
            continue
        subset = np.flatnonzero(masks[k])
        records.append(_train_one_shadow(cfg, k, subset, pool, test, out_dir, base_seed + k, progress))
    manifest["models"] = records
    return manifest


def load_shadow_models(shadow_dir: str | Path, device: torch.device,
                       root_cfg: DictConfig | None = None, max_models: int | None = None):
    """Load whatever shadow checkpoints exist, with their membership masks.

    Returns ``(models, masks, manifest)`` where ``masks[r]`` is the pool mask of
    the r-th *loaded* model (rows are selected to match the checkpoints found),
    so partially-complete shards are usable without bookkeeping.

    Checkpoints are rebuilt with :func:`load_checkpoint_model`, which replays the
    DP BatchNorm->GroupNorm surgery from the embedded config -- a plain
    ``build_model`` + ``load_state_dict`` would fail on DP-trained shadows.

    Results are cached per (directory, device, max_models) for the lifetime of
    the process: scoring a sweep re-requests the same shadow set once per target,
    and re-reading 16-64 checkpoints of ~75MB each time dominates the runtime.
    """
    if root_cfg is not None:
        raise ValueError(
            "Shadow checkpoints must be rebuilt from their embedded configs; "
            "target/root config overrides are not permitted."
        )
    cache_key = (str(Path(shadow_dir).resolve()), str(device), max_models)
    cached = _SHADOW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    shadow_dir = Path(shadow_dir)
    manifest = json.loads((shadow_dir / MANIFEST_NAME).read_text())
    masks = np.load(shadow_dir / manifest.get("masks_file", MASKS_NAME))

    found = []
    for path in sorted(shadow_dir.glob(CHECKPOINT_GLOB)):
        match = _CHECKPOINT_RE.search(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    found.sort()
    if max_models is not None:
        found = found[: int(max_models)]
    if not found:
        raise FileNotFoundError(f"No {CHECKPOINT_GLOB} checkpoints in {shadow_dir}")

    models = []
    rows = []
    for index, path in found:
        if index >= masks.shape[0]:
            raise ValueError(f"{path.name} has index {index} beyond the mask matrix ({masks.shape[0]} rows)")
        model, _ = load_checkpoint_model(str(path))
        model.to(device).eval()
        models.append(model)
        rows.append(index)
    result = (models, masks[rows], manifest)
    _SHADOW_CACHE[cache_key] = result
    return result


def clear_shadow_cache() -> None:
    """Drop cached shadow models (frees memory between shadow sets)."""
    _SHADOW_CACHE.clear()
