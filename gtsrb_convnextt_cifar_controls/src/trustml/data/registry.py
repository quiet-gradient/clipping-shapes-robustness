from __future__ import annotations

from omegaconf import DictConfig

from trustml.data.cifar10 import build_cifar10_datamodule
from trustml.data.cifar100 import build_cifar100_datamodule


def build_datamodule(cfg: DictConfig, audit_cfg: DictConfig | None = None) -> dict:
    name = str(cfg.name).lower()
    audit_enabled = audit_cfg is not None and bool(audit_cfg.get("enabled", False))
    if name == "cifar10":
        return build_cifar10_datamodule(cfg, audit_cfg=audit_cfg)
    if audit_enabled:
        raise ValueError(f"One-run epsilon audit is only implemented for cifar10, got: {cfg.name}")
    if name == "cifar100":
        return build_cifar100_datamodule(cfg)
    raise ValueError(f"Unknown dataset: {cfg.name}")

