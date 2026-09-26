from __future__ import annotations

from omegaconf import ListConfig

from trustml.defenses.dp_sgd import DPSGDDefense
from trustml.defenses.igr import IGRDefense
from trustml.defenses.label_smoothing import LabelSmoothingDefense
from trustml.defenses.trades import TRADESDefense


def build_defenses(cfg: ListConfig | list) -> list:
    builders = {
        "dp_sgd": DPSGDDefense,
        "igr": IGRDefense,
        "trades": TRADESDefense,
        "label_smoothing": LabelSmoothingDefense,
    }
    defenses = []
    for item in cfg or []:
        if not bool(item.get("enabled", True)):
            continue
        name = str(item.name).lower()
        if name not in builders:
            raise ValueError(f"Unknown defense: {item.name}")
        defenses.append(builders[name](item))
    return defenses

