from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class Logger:
    """Backend-agnostic experiment tracker interface.

    To add a new tracker (MLflow, ClickHouse, ...), subclass this, implement
    the methods you need, and register it in ``build_logger``. Local JSONL
    logging stays on regardless of the remote backend, so runs survive a
    tracker outage.
    """

    output_dir: Path | None = None
    checkpoint_dir: Path | None = None

    def log_config(self, cfg: DictConfig) -> None:
        return None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        return None

    def finish(self, result: dict[str, Any] | None = None) -> None:
        return None
