from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_module", model)

