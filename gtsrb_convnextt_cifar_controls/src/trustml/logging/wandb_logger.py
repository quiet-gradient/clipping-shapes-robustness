from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from trustml.logging.base import Logger


class WandbLogger(Logger):
    def __init__(self, cfg: DictConfig, output_dir: Path | None = None) -> None:
        self.enabled = False
        self.run = None
        self._wandb = None
        wandb_cfg = cfg.logging.wandb

        try:
            import wandb
        except ImportError:
            warnings.warn("wandb is not installed; continuing with local logging.")
            return

        mode = _none_if_empty(wandb_cfg.get("mode", "online")) or "online"
        entity = _none_if_empty(wandb_cfg.get("entity", None))
        project = _none_if_empty(wandb_cfg.get("project", cfg.experiment.project))
        tags = list(wandb_cfg.get("tags", []))
        run_id = _none_if_empty(wandb_cfg.get("run_id", None))

        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            warnings.warn("WANDB_API_KEY is not set; trying wandb.init anyway.")

        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=str(cfg.experiment.name),
                id=None if run_id is None else str(run_id),
                resume="allow" if run_id is not None else None,
                tags=tags,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=mode,
                dir=str(output_dir) if output_dir else None,
            )
        except Exception as exc:  # noqa: BLE001 - logger must not crash training.
            warnings.warn(f"wandb.init failed; continuing with local logging only: {exc}")
            return

        self.enabled = True
        self._wandb = wandb
        self.log_checkpoints = bool(wandb_cfg.get("log_checkpoints_as_artifacts", False))
        self.include_private_train_gradients = bool(
            wandb_cfg.get("include_private_train_gradients", False)
        )
        self.include_private_train_metrics = bool(
            wandb_cfg.get("include_private_train_metrics", False)
        )

    @property
    def wandb_run_id(self) -> str | None:
        if self.run is not None:
            return str(self.run.id)
        return None

    def log_config(self, cfg: DictConfig) -> None:
        if self.enabled and self.run is not None:
            self.run.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        # The trainer mixes epoch-indexed and global-step-indexed metrics, so a
        # raw step= kwarg would make W&B silently drop out-of-order rows.
        payload = {
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
        if step is not None:
            payload.setdefault("trainer/step", int(step))
        for name, (counts, edges) in _histogram_specs(payload).items():
            payload[name] = self._wandb.Histogram(
                np_histogram=(np.asarray(counts), np.asarray(edges))
            )
        self._wandb.log(payload)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self.enabled or not self.log_checkpoints or self._wandb is None:
            return
        artifact = self._wandb.Artifact(artifact_path or Path(path).stem, type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self, result: dict[str, Any] | None = None) -> None:
        if self.enabled and self.run is not None:
            if result:
                self.log_metrics({"final/" + key: value for key, value in result.items()})
            self.run.finish()


def _none_if_empty(value):
    if value is None:
        return None
    text = str(value)
    if text.strip().lower() in {"", "none", "null"}:
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
        (
            "TRADES CE",
            "TRADES KL",
            "TRADES weighted KL",
            "TRADES total loss",
        )
    )


def _histogram_specs(metrics: dict[str, Any]) -> dict[str, tuple[list[float], list[float]]]:
    grouped: dict[str, dict[str, dict[int, float]]] = {}
    for key, value in metrics.items():
        text = str(key)
        if not text.startswith("probe/hist/") or not isinstance(value, (int, float)):
            continue
        prefix, leaf = text.rsplit("/", 1)
        kind, separator, index_text = leaf.partition("_")
        if not separator or kind not in {"count", "edge"} or not index_text.isdigit():
            continue
        grouped.setdefault(prefix, {"count": {}, "edge": {}})[kind][int(index_text)] = float(value)
    result = {}
    for prefix, parts in grouped.items():
        counts = [parts["count"][index] for index in sorted(parts["count"])]
        edges = [parts["edge"][index] for index in sorted(parts["edge"])]
        if counts and len(edges) == len(counts) + 1:
            result[f"{prefix}/histogram"] = (counts, edges)
    return result
