from __future__ import annotations

from typing import Any

import torch


class Defense:
    name = "base"
    requires_input_grad = False

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.last_metrics: dict[str, float] = {}
        self.setup_metrics: dict[str, float] = {}

    def setup(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        cfg,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
        return model, optimizer, train_loader

    def loss(
        self,
        model: torch.nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        base_loss: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        return base_loss

    def metrics(self) -> dict[str, float]:
        return dict(self.last_metrics)

