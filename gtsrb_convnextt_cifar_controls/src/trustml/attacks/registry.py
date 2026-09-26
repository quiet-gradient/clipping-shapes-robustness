from __future__ import annotations

from omegaconf import DictConfig

from trustml.attacks.autoattack import AutoAttackAttack
from trustml.attacks.pgd import PGDAttack


def build_attacks(cfg: DictConfig | None, data_cfg=None) -> list:
    if cfg is None:
        return []
    attacks = []
    pgd_cfg = cfg.get("pgd", None)
    if pgd_cfg is not None and bool(pgd_cfg.get("enabled", False)):
        mean = tuple(data_cfg.get("mean", (0.4914, 0.4822, 0.4465))) if data_cfg is not None else None
        std = tuple(data_cfg.get("std", (0.2023, 0.1994, 0.2010))) if data_cfg is not None else None
        attacks.append(PGDAttack(pgd_cfg, mean=mean, std=std))
    autoattack_cfg = cfg.get("autoattack", None)
    if autoattack_cfg is not None and bool(autoattack_cfg.get("enabled", False)):
        mean = tuple(data_cfg.get("mean", (0.4914, 0.4822, 0.4465))) if data_cfg is not None else None
        std = tuple(data_cfg.get("std", (0.2023, 0.1994, 0.2010))) if data_cfg is not None else None
        attacks.append(AutoAttackAttack(autoattack_cfg, mean=mean, std=std))
    return attacks
