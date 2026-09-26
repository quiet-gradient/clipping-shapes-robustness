from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


EPSILON = 8.0 / 255.0
STEP_SIZE = 2.0 / 255.0
FINAL_RUNS = {
    "clean_ce_s42": ("ce", None, 42),
    "trades_b3_s42": ("trades", 3.0, 42),
    "trades_b6_s42": ("trades", 6.0, 42),
    "trades_b3_s43": ("trades", 3.0, 43),
    "trades_b3_s44": ("trades", 3.0, 44),
}


def load_pretraining_config(path: str | Path) -> DictConfig:
    path = Path(path).resolve()
    raw = OmegaConf.load(path)
    base_reference = raw.pop("base_config", None)
    if base_reference is None:
        cfg = raw
    else:
        base = OmegaConf.load((path.parent / str(base_reference)).resolve())
        cfg = OmegaConf.merge(base, raw)
    OmegaConf.resolve(cfg)
    validate_pretraining_config(cfg)
    return cfg


def validate_pretraining_config(cfg: DictConfig) -> None:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    run_id = str(cfg.experiment.run_id)
    require(run_id in FINAL_RUNS, "experiment.run_id is not part of the final five-run protocol")
    expected = FINAL_RUNS.get(run_id)
    if expected is not None:
        objective, beta, seed = expected
        require(str(cfg.objective.name) == objective, "objective differs from the final manifest")
        require(_optional_float(cfg.objective.beta) == beta, "beta differs from the final manifest")
        require(int(cfg.experiment.seed) == seed, "training seed differs from the final manifest")

    seed = int(cfg.experiment.seed)
    require(str(cfg.protocol.name) == "final_public_cifar100_v1", "unexpected protocol name")
    require(int(cfg.experiment.split_seed) == 20260818, "split seed must be 20260818")
    require(str(cfg.data.name) == "cifar100", "dataset must be CIFAR-100")
    require(int(cfg.data.num_classes) == 100, "CIFAR-100 must have 100 classes")
    require(int(cfg.data.image_size) == 32, "image size must be 32")
    require(int(cfg.data.train_per_class) == 490, "train split must contain 490 examples per class")
    require(int(cfg.data.validation_per_class) == 10, "validation split must contain 10 examples per class")
    require(not bool(cfg.data.normalize_in_transform), "data transforms must emit pixel-space tensors")
    require(int(cfg.data.augmentation.random_crop_size) == 32, "random crop size must be 32")
    require(int(cfg.data.augmentation.random_crop_padding) == 4, "random crop padding must be 4")
    require(
        _same_float(cfg.data.augmentation.horizontal_flip_probability, 0.5),
        "horizontal flip probability must be 0.5",
    )

    require(str(cfg.model.architecture) == "cifar_resnet18", "architecture must be CIFAR-style ResNet-18")
    require(str(cfg.model.initialization) == "random_tensor_seed_42", "initialization identifier is invalid")
    require(str(cfg.model.trainable_scope) == "full_model", "the complete model must be trainable")
    require(str(cfg.model.normalization) == "group_norm", "normalization must be GroupNorm")
    require(int(cfg.model.group_norm_groups) == 32, "GroupNorm must use 32 groups")

    require(int(cfg.training.epochs) == 200, "training must run for 200 epochs")
    require(int(cfg.training.batch_size) == 256, "batch size must be 256")
    require(bool(cfg.training.shuffle), "training data must be shuffled")
    require(not bool(cfg.training.drop_last), "the final incomplete batch must be retained")
    require(float(cfg.training.label_smoothing) == 0.0, "label smoothing must be zero")
    require(int(cfg.training.gradient_accumulation_steps) == 1, "gradient accumulation must be disabled")
    require(str(cfg.training.precision) == "fp32", "training precision must be FP32")
    require(not bool(cfg.training.amp), "automatic mixed precision must be disabled")
    require(not bool(cfg.training.tf32), "TF32 must be disabled")
    optimizer = cfg.training.optimizer
    require(str(optimizer.name) == "sgd", "optimizer must be SGD")
    require(_same_float(optimizer.learning_rate, 0.1), "initial learning rate must be 0.1")
    require(_same_float(optimizer.momentum, 0.9), "momentum must be 0.9")
    require(_same_float(optimizer.weight_decay, 5e-4), "weight decay must be 5e-4")
    require(bool(optimizer.nesterov), "Nesterov momentum must be enabled")
    scheduler = cfg.training.scheduler
    require(str(scheduler.name) == "multistep", "scheduler must be multistep")
    require(list(scheduler.milestones) == [100, 150], "learning-rate milestones must be 100 and 150")
    require(_same_float(scheduler.gamma, 0.1), "each learning-rate drop must multiply by 0.1")

    require(_same_float(cfg.objective.epsilon, EPSILON), "TRADES radius must be 8/255")
    require(int(cfg.objective.steps) == 10, "TRADES must use 10 inner steps")
    require(_same_float(cfg.objective.step_size, STEP_SIZE), "TRADES step size must be 2/255")
    require(bool(cfg.objective.random_start), "TRADES random start must be enabled")
    require(str(cfg.objective.attack_init) == "gaussian", "TRADES must use a Gaussian start")
    require(_same_float(cfg.objective.gaussian_std, 0.001), "Gaussian start standard deviation must be 0.001")
    require(str(cfg.objective.beta_semantics) == "kl_multiplier", "beta must multiply the robust KL term")

    attack = cfg.validation.pgd10
    require(int(cfg.validation.robust_every_epochs) == 1, "PGD-10 validation must run every epoch")
    require(_same_float(attack.epsilon, EPSILON), "PGD-10 radius must be 8/255")
    require(int(attack.steps) == 10, "validation attack must be PGD-10")
    require(_same_float(attack.step_size, STEP_SIZE), "PGD-10 step size must be 2/255")
    require(int(attack.restarts) == 1, "periodic PGD-10 must use one restart")
    require(str(cfg.autoattack.version) == "standard", "AutoAttack version must be standard")
    require(int(cfg.autoattack.examples) == 1000, "AutoAttack must use the full 1,000-example validation split")
    require(_same_float(cfg.autoattack.epsilon, EPSILON), "AutoAttack radius must be 8/255")
    require(int(cfg.autoattack.seed) == seed, "AutoAttack seed must equal the training-stream seed")
    require(int(cfg.selection.pgd_top_k) == 3, "exactly three PGD-10 checkpoints must receive AutoAttack")
    require(int(cfg.selection.binomial_examples) == 1000, "selection standard error must use n=1000")
    require(
        _same_float(cfg.selection.pgd_minus_square_instability_threshold, 0.08),
        "PGD-versus-Square instability threshold must be 0.08",
    )

    reproducibility = cfg.reproducibility
    require(bool(reproducibility.deterministic_algorithms), "deterministic algorithms must be enabled")
    require(int(reproducibility.initialization_seed) == 42, "all runs must share initialization seed 42")
    require(int(reproducibility.dataloader_seed) == seed * 1000 + 1, "data-order stream seed is invalid")
    require(int(reproducibility.augmentation_seed) == seed * 1000 + 2, "augmentation stream seed is invalid")
    require(int(reproducibility.trades_seed) == seed * 1000 + 3, "TRADES stream seed is invalid")
    require(int(reproducibility.pgd_seed) == seed * 1000 + 4, "PGD stream seed is invalid")

    if errors:
        raise ValueError("invalid public-pretraining configuration:\n- " + "\n- ".join(errors))


def scientific_config_hash(cfg: DictConfig | dict[str, Any]) -> str:
    value = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_float(value: Any, expected: float) -> bool:
    return abs(float(value) - expected) <= 1e-15


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
