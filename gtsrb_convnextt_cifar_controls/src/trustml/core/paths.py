from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig


def sanitize_name(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-") or "experiment"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_dir(cfg: DictConfig) -> Path:
    base_dir = Path(str(cfg.logging.local.get("dir", "outputs")))
    exp_name = sanitize_name(str(cfg.experiment.name))
    run_dir = base_dir / exp_name / timestamp()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir

