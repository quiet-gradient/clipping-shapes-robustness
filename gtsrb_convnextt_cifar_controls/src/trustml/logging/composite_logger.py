from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from trustml.logging.base import Logger
from trustml.logging.local_logger import LocalLogger
from trustml.logging.wandb_logger import WandbLogger


class CompositeLogger(Logger):
    def __init__(self, loggers: list[Logger]) -> None:
        self.loggers = loggers
        self.output_dir = _first_attr(loggers, "output_dir")
        self.checkpoint_dir = _first_attr(loggers, "checkpoint_dir")
        self.wandb_run_id = _first_attr(loggers, "wandb_run_id")

    def log_config(self, cfg: DictConfig) -> None:
        self._call("log_config", cfg)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._call("log_metrics", metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._call("log_artifact", path, artifact_path=artifact_path)

    def finish(self, result: dict[str, Any] | None = None) -> None:
        self._call("finish", result)

    def _call(self, method_name: str, *args, **kwargs) -> None:
        for logger in self.loggers:
            method = getattr(logger, method_name, None)
            if method is None:
                continue
            try:
                method(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - one backend must not break the run.
                warnings.warn(f"{logger.__class__.__name__}.{method_name} failed: {exc}")


def build_logger(cfg: DictConfig) -> CompositeLogger:
    loggers = []
    if bool(cfg.logging.local.get("enabled", True)):
        loggers.append(LocalLogger(cfg))

    output_dir = _first_attr(loggers, "output_dir")
    if bool(cfg.logging.wandb.get("enabled", False)):
        loggers.append(WandbLogger(cfg, output_dir=output_dir))

    mlm_cfg = cfg.logging.get("mlm", None)
    if mlm_cfg is not None and bool(mlm_cfg.get("enabled", False)):
        from trustml.adapters.mlm_adapter import MLMLogger

        loggers.append(MLMLogger(cfg))

    if not loggers:
        loggers.append(LocalLogger(cfg))

    return CompositeLogger(loggers)


def _first_attr(loggers: list, attr: str):
    for logger in loggers:
        value = getattr(logger, attr, None)
        if value is not None:
            return value
    return None
