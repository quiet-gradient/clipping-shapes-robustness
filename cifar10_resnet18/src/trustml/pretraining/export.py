from __future__ import annotations

from pathlib import Path

import torch


def export_downstream_checkpoint(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {destination}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 170:
        raise ValueError("the downstream export must be created from epoch 170")
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError("source checkpoint has no resolved configuration")
    experiment = config.get("experiment", {})
    objective = config.get("objective", {})
    if int(experiment.get("seed", -1)) != 42:
        raise ValueError("the downstream export must use training seed 42")
    if objective.get("name") != "trades" or float(objective.get("beta", -1.0)) != 3.0:
        raise ValueError("the downstream export must use the beta=3 TRADES run")
    model = checkpoint.get("model")
    if not isinstance(model, dict) or not model or not all(torch.is_tensor(value) for value in model.values()):
        raise ValueError("source checkpoint has no tensor model state")
    payload = {
        "format_version": 1,
        "model": {key: value.detach().cpu() for key, value in model.items()},
        "epoch": 170,
        "source": {"dataset": "CIFAR-100", "objective": "TRADES", "beta": 3.0, "seed": 42},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination
