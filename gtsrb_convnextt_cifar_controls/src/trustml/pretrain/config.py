from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf


EPSILON = 8.0 / 255.0
STEP_SIZE = 2.0 / 255.0
ALLOWED_RUNTIME_KEYS = {
    "name",
    "dataset_root",
    "output_root",
    "num_workers",
    "wandb_mode",
    "artifact_persistence",
    "platform_job_id_env",
    "git_commit_env",
    "smoke",
}


def load_pretrain_config(config_path: str | Path, runtime_path: str | Path) -> DictConfig:
    config_path = Path(config_path).resolve()
    runtime_path = Path(runtime_path).resolve()
    raw = OmegaConf.load(config_path)
    base_ref = raw.pop("base_config", None)
    if base_ref:
        base_path = (config_path.parent / str(base_ref)).resolve()
        base = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base, raw)
    else:
        cfg = raw

    runtime_file = OmegaConf.load(runtime_path)
    if set(runtime_file.keys()) != {"runtime"}:
        raise ValueError("runtime config must contain only the top-level 'runtime' mapping")
    unknown = set(runtime_file.runtime.keys()) - ALLOWED_RUNTIME_KEYS
    if unknown:
        raise ValueError(f"runtime config contains non-infrastructure keys: {sorted(unknown)}")
    cfg = OmegaConf.merge(cfg, runtime_file)
    cfg.provenance.config_path = _project_relative(config_path)
    cfg.provenance.runtime_config_path = _project_relative(runtime_path)
    OmegaConf.resolve(cfg)
    validate_pretrain_config(cfg)
    cfg.provenance.config_hash = scientific_config_hash(cfg)
    cfg.logging.wandb.name = wandb_run_name(cfg)
    OmegaConf.resolve(cfg)
    return cfg


def validate_pretrain_config(cfg: DictConfig) -> None:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    final_v1 = str(cfg.get("protocol", {}).get("name", "legacy_sweep")) == "final_v1"
    require(int(cfg.experiment.seed) in ({42, 43, 44} if final_v1 else {42}), "invalid experiment.seed")
    require(int(cfg.experiment.split_seed) == (20260818 if final_v1 else 20260817), "unexpected split_seed")
    require(str(cfg.data.name).lower() == "cifar100", "data.name must be cifar100")
    require(int(cfg.data.num_classes) == 100, "data.num_classes must be 100")
    require(int(cfg.data.image_size) == 32, "data.image_size must be 32")
    require(int(cfg.data.train_per_class) == (490 if final_v1 else 450), "unexpected train_per_class")
    require(int(cfg.data.val_per_class) == (10 if final_v1 else 50), "unexpected val_per_class")
    require(bool(cfg.data.normalize_in_transform) is False, "data transforms must emit [0,1] pixels")

    require(str(cfg.model.architecture) == "cifar_resnet18", "model must be cifar_resnet18")
    require(str(cfg.model.normalization) == "group_norm", "model normalization must be group_norm")
    require(int(cfg.model.group_norm_groups) == 32, "group_norm_groups must be 32")
    require(str(cfg.model.trainable_scope) == "full_model", "the full model must be trainable")

    require(int(cfg.training.epochs) == (200 if final_v1 else 100), "unexpected training.epochs")
    require(int(cfg.training.batch_size) == (256 if final_v1 else 128), "unexpected training.batch_size")
    require(str(cfg.training.precision) == "fp32", "precision must be fp32")
    require(bool(cfg.training.amp) is False, "AMP must be disabled")
    require(bool(cfg.training.tf32) is False, "TF32 must be disabled")
    require(bool(cfg.training.shuffle) is True, "shuffle must be enabled")
    require(bool(cfg.training.drop_last) is False, "drop_last must be false")
    require(float(cfg.training.label_smoothing) == 0.0, "label smoothing must be 0")
    require(int(cfg.training.gradient_accumulation_steps) == 1, "gradient accumulation must be 1")
    require(str(cfg.training.optimizer.name).lower() == "sgd", "optimizer must be SGD")
    require(float(cfg.training.optimizer.momentum) == 0.9, "momentum must be 0.9")
    require(float(cfg.training.optimizer.weight_decay) == 5e-4, "weight decay must be 5e-4")
    require(bool(cfg.training.optimizer.nesterov) is True, "Nesterov must be enabled")
    require(str(cfg.training.scheduler.name).lower() == ("multistep" if final_v1 else "cosine"), "unexpected scheduler")
    if final_v1:
        require(list(cfg.training.scheduler.milestones) == [100, 150], "milestones must be [100, 150]")
        require(float(cfg.training.scheduler.gamma) == 0.1, "scheduler gamma must be 0.1")
    else:
        require(float(cfg.training.scheduler.eta_min_factor) == 0.01, "eta_min_factor must be 0.01")
    require(bool(cfg.privacy.dp_sgd) is False, "CIFAR-100 pretraining is public: DP-SGD must be false")

    objective = str(cfg.objective.name).lower()
    require(objective in {"ce", "trades"}, "objective must be ce or trades")
    lr = float(cfg.training.optimizer.learning_rate)
    require(lr in ({0.1} if final_v1 else {0.05, 0.1, 0.2}), "unexpected learning rate")
    beta = cfg.objective.get("beta", None)
    if objective == "ce":
        require(beta is None, "CE runs must set objective.beta to null")
    else:
        require(float(beta) in ({3.0, 6.0} if final_v1 else {1.0, 3.0, 6.0, 9.0}), "unexpected TRADES beta")
        require(str(cfg.objective.threat_model).lower() == "linf", "TRADES threat model must be linf")
        require(_same_float(cfg.objective.epsilon, EPSILON), "TRADES epsilon must be exactly 8/255")
        require(_same_float(cfg.objective.step_size, STEP_SIZE), "TRADES step size must be exactly 2/255")
        require(int(cfg.objective.steps) == 10, "TRADES must use 10 inner steps")
        require(bool(cfg.objective.random_start) is True, "TRADES random start must be enabled")
        if final_v1:
            require(str(cfg.objective.attack_init) == "gaussian", "final_v1 requires gaussian TRADES init")
            require(_same_float(cfg.objective.gaussian_std, 0.001), "gaussian_std must be 0.001")
        require(float(cfg.objective.pixel_min) == 0.0, "pixel_min must be 0")
        require(float(cfg.objective.pixel_max) == 1.0, "pixel_max must be 1")
        require(str(cfg.objective.beta_semantics) == "kl_multiplier", "beta must multiply KL")

    require(int(cfg.validation.robust_every_epochs) == (1 if final_v1 else 5), "unexpected robust validation cadence")
    validation_attack = cfg.validation.pgd10 if final_v1 else cfg.validation.pgd20
    validation_name = "PGD-10" if final_v1 else "PGD-20"
    require(_same_float(validation_attack.epsilon, EPSILON), f"{validation_name} epsilon must be 8/255")
    require(_same_float(validation_attack.step_size, STEP_SIZE), f"{validation_name} step size must be 2/255")
    require(int(validation_attack.steps) == (10 if final_v1 else 20), f"validation attack must be {validation_name}")
    require(int(validation_attack.periodic_restarts) == 1, "periodic PGD must use one restart")
    require(int(validation_attack.final_restarts) == (1 if final_v1 else 5), "unexpected final PGD restarts")
    require(int(cfg.validation.top_k) == 3, "final robust selection must re-evaluate top three")
    require(
        _same_float(cfg.validation.clean_tie_threshold, 0.0 if final_v1 else 0.002),
        "final_v1 uses the paired one-standard-error rule; legacy tie threshold must be 0.2 pp",
    )
    if errors:
        raise ValueError("invalid CIFAR-100 pretrain config:\n- " + "\n- ".join(errors))


def scientific_config_hash(cfg: DictConfig | dict[str, Any]) -> str:
    value = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    assert isinstance(value, dict)
    value = json.loads(json.dumps(value))
    value.pop("runtime", None)
    provenance = value.get("provenance", {})
    for key in ("config_hash", "config_path", "runtime_config_path", "git_commit"):
        provenance.pop(key, None)
    wandb = value.get("logging", {}).get("wandb", {})
    wandb.pop("name", None)
    wandb.pop("run_id", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wandb_run_name(cfg: DictConfig) -> str:
    objective = str(cfg.objective.name).lower()
    beta = "na" if objective == "ce" else _format_number(float(cfg.objective.beta))
    lr = _format_number(float(cfg.training.optimizer.learning_rate))
    return (
        f"pretrain__obj-{objective}__beta-{beta}__lr-{lr}__wd-5e-4__seed-{int(cfg.experiment.seed)}__"
        f"{str(cfg.provenance.config_hash)[:8]}"
    )


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Reproducible public CIFAR-100 pretraining")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--resume", default=None, help="checkpoint path or 'auto'")
    parser.add_argument("--prepare-assets-only", action="store_true")
    return parser.parse_args(argv)


def _format_number(value: float) -> str:
    return f"{value:g}"


def _same_float(value: Any, expected: float) -> bool:
    return abs(float(value) - expected) <= 1e-15


def _project_relative(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
