"""Post-lock evaluation of the ordered CIFAR-10 official test split.

This module is deliberately separate from development evaluation.  It accepts
only one of the 24 predeclared manifest rows and refuses to instantiate the
official test split without an explicit acknowledgement flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler

from trustml.attacks.autoattack import AutoAttackAttack
from trustml.attacks.pgd import PGDAttack, _set_grad_sample_hooks
from trustml.core.config import load_config
from trustml.core.model_loading import load_checkpoint_model
from trustml.data.transforms import CIFAR10_MEAN, CIFAR10_STD, cifar_transform


EXPECTED_EXAMPLES = 10_000
ATTACK_SEED = 42
OFFICIAL_BATCH_SIZE = 128
PGD_EPSILON = 8.0 / 255.0
PGD_STEP_SIZE = 2.0 / 255.0
PGD_STEPS = 20
PGD_RESTARTS = 5
AA_RADII = {"aa8": 8.0 / 255.0, "aa4": 4.0 / 255.0}
AA_COMPONENTS = ("apgd-ce", "apgd-t", "fab-t", "square")
OFFICIAL_SAMPLE_ID_SHA256 = "a74675bfed3834ac1c3140b2d7fb7d93d90a76608cc08515a5019a1cb2075ff4"
FROZEN_BINDINGS = {
    ("H-Natural", 42): "hce_c030_lr0050_s42",
    ("H-Natural", 43): "hce_locked_s43",
    ("H-Natural", 44): "hce_locked_s44",
    ("H-Joint", 42): "ha_b1_c030_lr0050_s42",
    ("H-Joint", 43): "ha_locked_s43",
    ("H-Joint", 44): "ha_locked_s44",
    ("H-C", 42): "hc_q1of3_c100_lr0025_s42",
    ("H-C", 43): "hc_locked_s43",
    ("H-C", 44): "hc_locked_s44",
    ("H-JointMatched", 42): "ha_b1_c100_lr0025_s42",
    ("H-JointMatched", 43): "ha_jointmatched_s43",
    ("H-JointMatched", 44): "ha_jointmatched_s44",
    ("S-Natural", 42): "sce_g1000_lr050_s42",
    ("S-Natural", 43): "sce_g1000_lr050_s43",
    ("S-Natural", 44): "sce_g1000_lr050_s44",
    ("S-Joint", 42): "sa_lr100_s42",
    ("S-Joint", 43): "sa_g0010_lr100_s43",
    ("S-Joint", 44): "sa_g0010_lr100_s44",
    ("S-C", 42): "sc_g0100_q025_lr100_s42",
    ("S-C", 43): "sc_locked_s43",
    ("S-C", 44): "sc_locked_s44",
    ("S-JointMatched", 42): "sa_g0100_lr100_s42",
    ("S-JointMatched", 43): "sa_g0100_lr100_s43",
    ("S-JointMatched", 44): "sa_g0100_lr100_s44",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def resolve_local_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path while rejecting absolute and escaping paths."""

    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"manifest path must be package-relative: {relative!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeError(f"manifest path escapes the package: {relative!r}")
    return resolved


def read_binding(manifest_path: Path, evaluation_id: str) -> tuple[Path, dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed_bindings = {
        (row["paper_arm"], int(row["seed"])): row["run_id"] for row in rows
    }
    if len(rows) != len(FROZEN_BINDINGS) or observed_bindings != FROZEN_BINDINGS:
        raise RuntimeError("official-test manifest differs from the frozen 24-row design")
    if any(row.get("selection_locked_before_official_test") != "true" for row in rows):
        raise RuntimeError("official-test manifest contains a row without a post-lock declaration")
    matches = [row for row in rows if row["evaluation_id"] == evaluation_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"official-test manifest must contain exactly one row for {evaluation_id!r}"
        )
    root = manifest_path.resolve().parent.parent
    row = matches[0]
    if row.get("selection_locked_before_official_test") != "true":
        raise RuntimeError("official-test access requires a predeclared post-lock row")
    return root, row


def _official_test_loader(cfg, *, acknowledged: bool) -> DataLoader:
    if not acknowledged:
        raise RuntimeError(
            "official-test access is blocked; pass --acknowledge-official-test only "
            "after model selection and evaluation settings are frozen"
        )

    root = Path(str(cfg.get("root", "./data"))).expanduser()
    mean = tuple(float(value) for value in cfg.get("mean", CIFAR10_MEAN))
    std = tuple(float(value) for value in cfg.get("std", CIFAR10_STD))
    transform = cifar_transform(train=False, mean=mean, std=std, augmentation=False)
    dataset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=False,
        transform=transform,
        download=False,
    )
    if len(dataset) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"official CIFAR-10 test split has {len(dataset)} examples, "
            f"expected {EXPECTED_EXAMPLES}"
        )
    if bool(getattr(dataset, "train", True)):
        raise RuntimeError("dataset identity check failed: expected CIFAR-10 train=False")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )
    if not isinstance(loader.sampler, SequentialSampler):
        raise RuntimeError("official test loader must use the sequential sampler")
    return loader


def _targets_from_dataset(loader: DataLoader) -> np.ndarray:
    targets = getattr(loader.dataset, "targets", None)
    if targets is None:
        raise RuntimeError("CIFAR-10 dataset does not expose ordered targets")
    result = np.asarray(targets, dtype=np.uint8)
    if result.shape != (EXPECTED_EXAMPLES,):
        raise RuntimeError("official test target vector has an unexpected shape")
    if np.any(result > 9):
        raise RuntimeError("official test target vector contains an invalid class")
    return result


def _validate_model(model: torch.nn.Module) -> None:
    if model.training:
        raise RuntimeError("model.eval() invariant failed")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("FP32 parameter invariant failed")
    probability_layers = (torch.nn.Softmax, torch.nn.LogSoftmax)
    if any(isinstance(module, probability_layers) for module in model.modules()):
        raise RuntimeError("official evaluation requires model logits, not probabilities")


@torch.no_grad()
def _clean_correctness(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    correct: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        if logits.ndim != 2 or logits.shape[1] != 10:
            raise RuntimeError("model output is not a ten-class logits matrix")
        if not torch.isfinite(logits).all():
            raise RuntimeError("clean evaluation produced non-finite logits")
        correct.append(logits.argmax(dim=1).cpu().eq(labels))
        targets.append(labels.detach().cpu().long())
    mask = torch.cat(correct).numpy().astype(np.bool_)
    target = torch.cat(targets).numpy().astype(np.uint8)
    if mask.shape != (EXPECTED_EXAMPLES,) or target.shape != (EXPECTED_EXAMPLES,):
        raise RuntimeError("clean evaluation did not cover all ordered official examples")
    return mask, target


def reduce_restart_correctness(restart_masks: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the per-example worst case over independent attack restarts."""

    if len(restart_masks) != PGD_RESTARTS:
        raise RuntimeError(f"PGD requires exactly {PGD_RESTARTS} restart masks")
    if not restart_masks:
        raise RuntimeError("PGD restart mask collection is empty")
    shape = restart_masks[0].shape
    if any(mask.dtype != torch.bool or mask.shape != shape for mask in restart_masks):
        raise RuntimeError("PGD restart masks must be aligned boolean vectors")
    return torch.stack(list(restart_masks), dim=0).all(dim=0)


def _pgd_correctness(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg,
) -> tuple[np.ndarray, np.ndarray, float]:
    attack = PGDAttack(
        OmegaConf.create(
            {
                "epsilon": PGD_EPSILON,
                "step_size": PGD_STEP_SIZE,
                "steps": PGD_STEPS,
                "restarts": PGD_RESTARTS,
                "max_batches": None,
            }
        ),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    std = torch.tensor(cfg.data.std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    correct: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    max_pixel_delta = 0.0
    _set_grad_sample_hooks(model, False)
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            restart_masks: list[torch.Tensor] = []
            for _ in range(PGD_RESTARTS):
                adversarial = attack.generate(model, images, labels)
                if not torch.isfinite(adversarial).all():
                    raise RuntimeError("PGD produced non-finite adversarial inputs")
                max_pixel_delta = max(
                    max_pixel_delta,
                    float(((adversarial - images).abs() * std).max().item()),
                )
                with torch.no_grad():
                    logits = model(adversarial)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("PGD evaluation produced non-finite logits")
                restart_masks.append(logits.argmax(dim=1).eq(labels).cpu())
            correct.append(reduce_restart_correctness(restart_masks))
            targets.append(labels.detach().cpu().long())
    finally:
        _set_grad_sample_hooks(model, True)

    mask = torch.cat(correct).numpy().astype(np.bool_)
    target = torch.cat(targets).numpy().astype(np.uint8)
    if mask.shape != (EXPECTED_EXAMPLES,) or target.shape != (EXPECTED_EXAMPLES,):
        raise RuntimeError("PGD evaluation did not cover all ordered official examples")
    if max_pixel_delta > PGD_EPSILON + 1e-6:
        raise RuntimeError("PGD exceeded the Linf perturbation budget")
    return mask, target, max_pixel_delta


def validate_autoattack_trace(trace: Mapping[str, Any], expected_examples: int) -> None:
    if trace.get("official_component_order") != list(AA_COMPONENTS):
        raise RuntimeError("unexpected AutoAttack standard component order")
    clean = trace.get("clean_correct")
    final = trace.get("final_robust_correct")
    attack_time_final = trace.get("attack_time_final_robust_correct")
    masks = trace.get("correct_after_component", {})
    if not isinstance(clean, list) or len(clean) != expected_examples:
        raise RuntimeError("AutoAttack clean mask is incomplete")
    if not isinstance(final, list) or len(final) != expected_examples:
        raise RuntimeError("AutoAttack final mask is incomplete")
    if not isinstance(attack_time_final, list) or len(attack_time_final) != expected_examples:
        raise RuntimeError("AutoAttack attack-time final mask is incomplete")
    if trace.get("final_robust_correct_source") != "posthoc_fp32_logits":
        raise RuntimeError("official AutoAttack headline is not based on post-hoc FP32 logits")

    previous = np.asarray(clean, dtype=np.bool_)
    for component in AA_COMPONENTS:
        values = masks.get(component)
        if not isinstance(values, list) or len(values) != expected_examples:
            raise RuntimeError(f"AutoAttack mask is incomplete for {component}")
        current = np.asarray(values, dtype=np.bool_)
        if np.any(current & ~previous):
            raise RuntimeError(f"AutoAttack survivor mask increased after {component}")
        reached = int(trace["component_reached_examples"][component])
        if reached != int(previous.sum()):
            raise RuntimeError(f"AutoAttack reach count is inconsistent for {component}")
        previous = current
    if not np.array_equal(previous, np.asarray(attack_time_final, dtype=np.bool_)):
        raise RuntimeError("AutoAttack attack-time final and post-Square masks differ")
    if not np.array_equal(
        np.asarray(trace.get("posthoc_final_robust_correct"), dtype=np.bool_),
        np.asarray(final, dtype=np.bool_),
    ):
        raise RuntimeError("AutoAttack headline differs from its post-hoc FP32 mask")
    if np.any(np.asarray(final, dtype=np.bool_) & ~np.asarray(clean, dtype=np.bool_)):
        raise RuntimeError("AutoAttack post-hoc robust mask is not a subset of clean correctness")


def _run_autoattack(
    *,
    label: str,
    epsilon: float,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg,
    output_json: Path,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    torch.manual_seed(ATTACK_SEED)
    torch.cuda.manual_seed_all(ATTACK_SEED)
    attack = AutoAttackAttack(
        OmegaConf.create(
            {
                "epsilon": epsilon,
                "norm": "Linf",
                "version": "standard",
                "batch_size": batch_size,
                "seed": ATTACK_SEED,
                "expected_examples": EXPECTED_EXAMPLES,
                "require_full_test": True,
                "verbose": False,
                "log_path": str(output_json.with_suffix(f".{label}.autoattack.log")),
                "capture_component_trace": True,
                "headline_from_posthoc": True,
            }
        ),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    started = time.perf_counter()
    metrics = attack.run(model, loader, device=device, progress=False)
    elapsed = time.perf_counter() - started
    trace = attack.last_component_trace
    if trace is None:
        raise RuntimeError(f"{label}: AutoAttack component trace was not captured")
    validate_autoattack_trace(trace, EXPECTED_EXAMPLES)
    diagnostics = attack.last_diagnostics or {}
    output_diagnostics = attack.last_output_diagnostics or {}
    if not diagnostics.get("validation_passed", False):
        raise RuntimeError(f"{label}: pixel and normalization diagnostics failed")
    if int(diagnostics.get("n_examples", -1)) != EXPECTED_EXAMPLES:
        raise RuntimeError(f"{label}: pixel diagnostics have an unexpected sample count")
    max_perturbation = float(metrics["attack/autoattack_max_perturbation"])
    if not np.isfinite(max_perturbation) or max_perturbation > epsilon + 1e-6:
        raise RuntimeError(f"{label}: AutoAttack exceeded the Linf perturbation budget")
    return metrics, trace, {
        "pixel_wrapper": diagnostics,
        "attack_output": output_diagnostics,
    }, elapsed


def _trace_arrays(prefix: str, trace: Mapping[str, Any]) -> dict[str, np.ndarray]:
    masks = trace["correct_after_component"]
    return {
        f"{prefix}_clean_correct": np.asarray(trace["clean_correct"], dtype=np.bool_),
        f"{prefix}_after_apgd_ce": np.asarray(masks["apgd-ce"], dtype=np.bool_),
        f"{prefix}_after_apgd_dlr": np.asarray(masks["apgd-t"], dtype=np.bool_),
        f"{prefix}_after_fab": np.asarray(masks["fab-t"], dtype=np.bool_),
        f"{prefix}_after_square": np.asarray(masks["square"], dtype=np.bool_),
        f"{prefix}_attack_time_final_correct": np.asarray(
            trace["attack_time_final_robust_correct"], dtype=np.bool_
        ),
        f"{prefix}_final_correct": np.asarray(trace["final_robust_correct"], dtype=np.bool_),
    }


def _write_compact_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        if value.shape != (EXPECTED_EXAMPLES,):
            raise RuntimeError(f"compact output array {name!r} has an unexpected shape")
    with tempfile.NamedTemporaryFile(
        prefix=path.stem + ".",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {name: array_sha256(value) for name, value in sorted(arrays.items())}


def _autoattack_version() -> str:
    try:
        return importlib.metadata.version("autoattack")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _gpu_metadata(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "model": torch.cuda.get_device_name(index),
        "device_index": int(index),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def evaluate(
    *,
    root: Path,
    row: Mapping[str, str],
    acknowledged: bool,
    batch_size: int = OFFICIAL_BATCH_SIZE,
    data_root: Path | None = None,
) -> dict[str, Any]:
    if not acknowledged:
        raise RuntimeError(
            "official-test access requires --acknowledge-official-test after the lock"
        )
    if row.get("selection_locked_before_official_test") != "true":
        raise RuntimeError("manifest row is not approved for post-lock official evaluation")
    if batch_size != OFFICIAL_BATCH_SIZE:
        raise RuntimeError(f"official-test batch size is locked to {OFFICIAL_BATCH_SIZE}")
    if not torch.cuda.is_available():
        raise RuntimeError("the complete official attack suite requires a CUDA GPU")

    config_path = resolve_local_path(root, row["config_path"])
    checkpoint_path = resolve_local_path(root, row["checkpoint_path"])
    output_json = resolve_local_path(root, row["output_json"])
    output_npz = resolve_local_path(root, row["output_npz"])
    if output_json.exists() or output_npz.exists():
        raise RuntimeError("refusing to overwrite an existing official-test output")
    if not config_path.is_file():
        raise RuntimeError(f"configuration is missing: {row['config_path']}")
    if sha256(config_path) != row["config_sha256"]:
        raise RuntimeError("official-test configuration SHA-256 mismatch")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"training checkpoint is missing: {row['checkpoint_path']}")

    cfg = load_config(config_path)
    if not bool(cfg.protocol.get("test_locked", False)):
        raise RuntimeError("training configuration does not declare the official test locked")
    if str(cfg.experiment.name) != row["run_id"]:
        raise RuntimeError("configuration run identity differs from the official-test manifest")
    if int(cfg.experiment.seed) != int(row["seed"]):
        raise RuntimeError("configuration seed differs from the official-test manifest")
    cfg.data.download = False
    cfg.data.augmentation = False
    cfg.data.batch_size = int(batch_size)
    cfg.data.num_workers = min(2, int(cfg.data.get("num_workers", 2)))
    if data_root is not None:
        cfg.data.root = str(data_root)

    torch.manual_seed(ATTACK_SEED)
    torch.cuda.manual_seed_all(ATTACK_SEED)
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(True)
    device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    loader = _official_test_loader(cfg.data, acknowledged=acknowledged)
    dataset_targets = _targets_from_dataset(loader)
    sample_ids = np.arange(EXPECTED_EXAMPLES, dtype=np.int32)
    if canonical_hash(sample_ids.tolist()) != OFFICIAL_SAMPLE_ID_SHA256:
        raise RuntimeError("official sample-order identity check failed")

    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != row["checkpoint_sha256"]:
        raise RuntimeError("official-test checkpoint SHA-256 differs from the frozen manifest")
    model, saved_cfg = load_checkpoint_model(str(checkpoint_path), cfg_override=cfg)
    if str(saved_cfg.experiment.name) != row["run_id"]:
        raise RuntimeError("checkpoint run identity differs from the official-test manifest")
    if int(saved_cfg.experiment.seed) != int(row["seed"]):
        raise RuntimeError("checkpoint seed differs from the official-test manifest")
    model = model.float().to(device).eval()
    _validate_model(model)

    total_started = time.perf_counter()
    clean_started = time.perf_counter()
    clean_correct, clean_targets = _clean_correctness(model, loader, device)
    clean_seconds = time.perf_counter() - clean_started
    if not np.array_equal(clean_targets, dataset_targets):
        raise RuntimeError("clean loader changed official sample identity or order")

    pgd_started = time.perf_counter()
    pgd_correct, pgd_targets, pgd_max_delta = _pgd_correctness(model, loader, device, cfg)
    pgd_seconds = time.perf_counter() - pgd_started
    if not np.array_equal(pgd_targets, dataset_targets):
        raise RuntimeError("PGD loader changed official sample identity or order")

    aa8_metrics, aa8_trace, aa8_diagnostics, aa8_seconds = _run_autoattack(
        label="aa8",
        epsilon=AA_RADII["aa8"],
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        output_json=output_json,
        batch_size=batch_size,
    )
    aa4_metrics, aa4_trace, aa4_diagnostics, aa4_seconds = _run_autoattack(
        label="aa4",
        epsilon=AA_RADII["aa4"],
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        output_json=output_json,
        batch_size=batch_size,
    )
    _validate_model(model)
    aa8_clean = np.asarray(aa8_trace["clean_correct"], dtype=np.bool_)
    aa4_clean = np.asarray(aa4_trace["clean_correct"], dtype=np.bool_)
    if not np.array_equal(clean_correct, aa8_clean) or not np.array_equal(
        clean_correct, aa4_clean
    ):
        raise RuntimeError(
            "clean correctness differs across direct, AA8, and AA4 normalization paths"
        )

    arrays: dict[str, np.ndarray] = {
        "sample_id": sample_ids,
        "target": dataset_targets,
        "clean_correct": clean_correct,
        "pgd20x5_correct": pgd_correct,
        **_trace_arrays("aa8", aa8_trace),
        **_trace_arrays("aa4", aa4_trace),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    array_hashes = _write_compact_arrays(output_npz, arrays)
    compact_hash = sha256(output_npz)

    observed_counts = {
        "clean_correct": int(clean_correct.sum()),
        "pgd20x5_correct": int(pgd_correct.sum()),
        "aa8_correct": int(arrays["aa8_final_correct"].sum()),
        "aa4_correct": int(arrays["aa4_final_correct"].sum()),
    }
    reference_counts = {
        "clean_correct": int(row["reference_clean_correct"]),
        "pgd20x5_correct": int(row["reference_pgd20x5_correct"]),
        "aa8_correct": int(row["reference_aa8_correct"]),
        "aa4_correct": int(row["reference_aa4_correct"]),
    }
    reference_delta = {
        name: observed_counts[name] - reference_counts[name] for name in observed_counts
    }
    diagnostic_warnings = []
    for radius, diagnostic in (
        ("aa8", aa8_diagnostics["attack_output"]),
        ("aa4", aa4_diagnostics["attack_output"]),
    ):
        correctness = int(
            diagnostic.get("attack_time_vs_posthoc_correctness_disagreement_count", 0)
        )
        predictions = int(
            diagnostic.get("attack_time_vs_posthoc_prediction_disagreement_count", 0)
        )
        if correctness or predictions:
            diagnostic_warnings.append(
                {
                    "radius": radius,
                    "attack_time_vs_posthoc_correctness_disagreements": correctness,
                    "attack_time_vs_posthoc_prediction_disagreements": predictions,
                    "attack_time_correct": int(diagnostic["attack_time_adv_correct"]),
                    "posthoc_correct": int(diagnostic["posthoc_adv_correct"]),
                    "headline_source": diagnostic["headline_source"],
                }
            )
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "post_lock_official_test",
        "intended_use": "confirmation_only_not_model_selection",
        "evaluation_id": row["evaluation_id"],
        "rq": row["rq"],
        "paper_arm": row["paper_arm"],
        "training_seed": int(row["seed"]),
        "run_id": row["run_id"],
        "sample_count": EXPECTED_EXAMPLES,
        "sample_order": "official CIFAR-10 test indices 0 through 9999",
        "sample_id_sha256": OFFICIAL_SAMPLE_ID_SHA256,
        "target_sha256": array_hashes["target"],
        "config_path": row["config_path"],
        "config_sha256": row["config_sha256"],
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_sha256": checkpoint_hash,
        "compact_output_path": row["output_npz"],
        "compact_output_sha256": compact_hash,
        "per_array_sha256": array_hashes,
        "protocol": {
            "precision": "FP32",
            "model_mode": "eval",
            "model_outputs": "logits",
            "attack_seed": ATTACK_SEED,
            "normalization": (
                "one dataset normalization for clean and PGD; AutoAttack receives recovered "
                "pixel tensors and applies that same normalization exactly once in its wrapper"
            ),
            "pgd": {
                "norm": "Linf",
                "epsilon": PGD_EPSILON,
                "step_size": PGD_STEP_SIZE,
                "steps": PGD_STEPS,
                "restarts": PGD_RESTARTS,
                "restart_reduction": "per-example logical AND",
            },
            "autoattack": {
                "version": "standard",
                "package_version": _autoattack_version(),
                "norm": "Linf",
                "radii": AA_RADII,
                "component_order": list(AA_COMPONENTS),
            },
        },
        "observed_counts": observed_counts,
        "observed_accuracy": {
            name.removesuffix("_correct"): value / EXPECTED_EXAMPLES
            for name, value in observed_counts.items()
        },
        "paper_reference_counts": reference_counts,
        "difference_from_paper_reference_counts": reference_delta,
        "matches_paper_reference_counts": all(value == 0 for value in reference_delta.values()),
        "diagnostic_warnings": diagnostic_warnings,
        "autoattack_metrics": {"aa8": aa8_metrics, "aa4": aa4_metrics},
        "invariants": {
            "post_lock_acknowledged": True,
            "not_authorized_for_tuning": True,
            "ordered_test_ids_0_to_9999": True,
            "sample_order_sha256_valid": True,
            "model_eval": True,
            "fp32": True,
            "logits_not_probabilities": True,
            "single_normalization_path": True,
            "clean_correctness_equal_across_evaluators": True,
            "finite_logits_and_adversarial_inputs": True,
            "pgd_per_example_worst_case": True,
            "pgd_linf_bound_valid": True,
            "autoattack_linf_bounds_valid": True,
            "autoattack_stage_monotonicity_valid": True,
            "autoattack_headline_from_posthoc_fp32_logits": True,
            "compact_output_complete": True,
        },
        "diagnostics": {
            "pgd_max_pixel_linf": pgd_max_delta,
            "aa8": aa8_diagnostics,
            "aa4": aa4_diagnostics,
        },
        "execution": {
            "command": [sys.executable, "-m", "trustml.official_test", *sys.argv[1:]],
            "python": platform.python_version(),
            "gpu": _gpu_metadata(device),
            "batch_size": int(batch_size),
            "duration_seconds": {
                "clean": clean_seconds,
                "pgd20x5": pgd_seconds,
                "aa8": aa8_seconds,
                "aa4": aa4_seconds,
                "total": time.perf_counter() - total_started,
            },
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "diagnostics"}, indent=2))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/official_test.csv"),
    )
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(OFFICIAL_BATCH_SIZE,),
        default=OFFICIAL_BATCH_SIZE,
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--acknowledge-official-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.acknowledge_official_test:
        parser.error(
            "--acknowledge-official-test is required and confirms that model selection "
            "and evaluation settings were frozen before official-test access"
        )
    root, row = read_binding(args.manifest, args.evaluation_id)
    evaluate(
        root=root,
        row=row,
        acknowledged=args.acknowledge_official_test,
        batch_size=args.batch_size,
        data_root=args.data_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
