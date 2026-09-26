from __future__ import annotations

import torch


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def apply_adaptation(model: torch.nn.Module, cfg) -> None:
    adaptation = cfg.get("adaptation", {})
    method = str(adaptation.get("method", "full_finetune")).lower()
    if method == "full_finetune":
        for param in model.parameters():
            param.requires_grad = True
        return

    if method == "last_block":
        for param in model.parameters():
            param.requires_grad = False
        target = getattr(model, "last_block", None)
        classifier = getattr(model, "classifier", None)
        if target is None or classifier is None:
            raise ValueError("last_block adaptation requires model.last_block and model.classifier.")
        for param in target.parameters():
            param.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True
        return

    if method == "lora":
        raise NotImplementedError("LoRA adaptation is intentionally separate from defenses and not wired yet.")

    raise ValueError(f"Unknown adaptation.method: {method}")


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.training.optimizer
    params = trainable_parameters(model)
    if not params:
        raise ValueError("No trainable parameters. Check adaptation config.")

    name = str(optimizer_cfg.get("name", "sgd")).lower()
    lr = float(optimizer_cfg.get("lr", 0.1))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(optimizer_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(optimizer_cfg.get("nesterov", True)),
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    scheduler_cfg = cfg.training.get("scheduler", {"name": "none"})
    name = str(scheduler_cfg.get("name", "none")).lower()
    if name in {"none", "null"}:
        return None
    if name == "cosine":
        epochs = max(1, int(cfg.training.get("epochs", 1)))
        base_lr = float(cfg.training.optimizer.get("lr", optimizer.param_groups[0]["lr"]))
        eta_min_factor = float(scheduler_cfg.get("eta_min_factor", 0.01))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=base_lr * eta_min_factor,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 30)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def parameter_summary(model: torch.nn.Module) -> dict[str, float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "params/total": float(total),
        "params/trainable": float(trainable),
        "params/trainable_fraction": float(trainable / max(1, total)),
    }

