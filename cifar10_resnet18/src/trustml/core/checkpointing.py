from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_module", model)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
    cfg: DictConfig | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if cfg is not None:
        state["config"] = OmegaConf.to_container(cfg, resolve=True)
    if extra_state:
        state.update(extra_state)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)
    return path


def load_model_checkpoint(model: torch.nn.Module, path: str | Path, strict: bool = True) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    unwrap_model(model).load_state_dict(state_dict, strict=strict)
    return checkpoint
