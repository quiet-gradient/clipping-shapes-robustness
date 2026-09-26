from __future__ import annotations

import torch


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def apply_adaptation(model: torch.nn.Module, cfg) -> None:
    adaptation = cfg.get("adaptation", {})
    method = str(adaptation.get("method", "last_block")).lower()
    if method != "last_block":
        raise ValueError("The released experiment matrix requires last_block adaptation")
    for param in model.parameters():
        param.requires_grad = False
    target = getattr(model, "last_block", None)
    classifier = getattr(model, "classifier", None)
    if target is None or classifier is None:
        raise ValueError("last_block adaptation requires model.last_block and model.classifier")
    for param in target.parameters():
        param.requires_grad = True
    for param in classifier.parameters():
        param.requires_grad = True


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.training.optimizer
    params = trainable_parameters(model)
    if not params:
        raise ValueError("No trainable parameters. Check adaptation config.")

    name = str(optimizer_cfg.get("name", "sgd")).lower()
    lr = float(optimizer_cfg.get("lr", 0.1))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if name != "sgd":
        raise ValueError("The released experiment matrix requires SGD")
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=float(optimizer_cfg.get("momentum", 0.9)),
        weight_decay=weight_decay,
        nesterov=bool(optimizer_cfg.get("nesterov", True)),
    )


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    scheduler_cfg = cfg.training.get("scheduler", {"name": "cosine"})
    name = str(scheduler_cfg.get("name", "cosine")).lower()
    if name != "cosine":
        raise ValueError("The released experiment matrix requires cosine scheduling")
    epochs = max(1, int(cfg.training.get("epochs", 1)))
    base_lr = float(cfg.training.optimizer.get("lr", optimizer.param_groups[0]["lr"]))
    eta_min_factor = float(scheduler_cfg.get("eta_min_factor", 0.01))
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=base_lr * eta_min_factor,
    )


def parameter_summary(model: torch.nn.Module) -> dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "params/total": float(total),
        "params/trainable": float(trainable),
        "params/trainable_fraction": float(trainable / max(1, total)),
    }
