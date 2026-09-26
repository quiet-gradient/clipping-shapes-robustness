from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from trustml.core.config import save_config
from trustml.core.paths import create_output_dir
from trustml.logging.base import Logger


class LocalLogger(Logger):
    def __init__(self, cfg: DictConfig) -> None:
        self.output_dir = create_output_dir(cfg)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        local_cfg = cfg.logging.local
        self.include_private_train_gradients = bool(
            local_cfg.get("include_private_train_gradients", True)
        )
        self.include_private_train_metrics = bool(
            local_cfg.get("include_private_train_metrics", True)
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_config(self, cfg: DictConfig) -> None:
        save_config(cfg, self.output_dir / "config.yaml")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        metrics = {
            key: value
            for key, value in metrics.items()
            if (
                self.include_private_train_gradients
                or not _is_private_train_gradient_metric(str(key))
            )
            and (
                self.include_private_train_metrics
                or not _is_private_train_metric(str(key))
            )
        }
        if not metrics:
            return
        record = {
            "time": time.time(),
            "step": step,
            "metrics": _jsonify(metrics),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        return None

    def finish(self, result: dict[str, Any] | None = None) -> None:
        if result is not None:
            self.log_metrics({"final/" + key: value for key, value in result.items()}, step=None)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _is_private_train_gradient_metric(key: str) -> bool:
    key = key.removeprefix("final/")
    return key.startswith(("A ", "B ", "C ", "D ", "H-C ", "S ", "S-C "))


def _is_private_train_metric(key: str) -> bool:
    key = key.removeprefix("final/")
    if key.startswith("train/"):
        return key != "train/lr"
    if key in {"Train Loss", "Train Accuracy", "IGR penalty", "defense/igr_penalty"}:
        return True
    if key == "defense/trades_kl":
        return True
    return key.startswith(
        ("TRADES CE", "TRADES KL", "TRADES weighted KL", "TRADES total loss")
    )
