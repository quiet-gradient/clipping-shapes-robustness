from __future__ import annotations

import torch


def evaluate_robustness(
    model: torch.nn.Module,
    dataloader,
    attacks: list,
    device: torch.device,
    progress: bool = True,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for attack in attacks:
        metrics.update(attack.run(model, dataloader, device=device, progress=progress))
    return metrics

