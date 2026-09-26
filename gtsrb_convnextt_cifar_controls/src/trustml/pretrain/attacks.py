from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TradesOutput:
    total_loss: torch.Tensor
    clean_ce: torch.Tensor
    trades_kl: torch.Tensor
    adversarial: torch.Tensor
    clean_accuracy: float
    approx_adv_accuracy: float
    linf_mean: float
    linf_max: float


def trades_adversarial(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    clean_probs: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    generator: torch.Generator,
    random_start: bool = True,
    attack_init: str = "uniform",
    gaussian_std: float = 0.001,
) -> torch.Tensor:
    original = pixels.detach()
    was_training = model.training
    model.eval()
    try:
        if random_start:
            if attack_init == "uniform":
                noise = torch.rand(
                    original.shape, dtype=original.dtype, device=original.device, generator=generator
                )
                adversarial = original + (noise * 2.0 - 1.0) * float(epsilon)
            elif attack_init == "gaussian":
                noise = torch.randn(
                    original.shape, dtype=original.dtype, device=original.device, generator=generator
                )
                adversarial = original + noise * float(gaussian_std)
            else:
                raise ValueError(f"unsupported TRADES attack_init={attack_init!r}")
        else:
            adversarial = original.clone()
        adversarial = adversarial.clamp(0.0, 1.0).detach()
        for _ in range(int(steps)):
            adversarial.requires_grad_(True)
            logits = model(adversarial)
            kl = F.kl_div(
                F.log_softmax(logits, dim=1), clean_probs.detach(), reduction="batchmean"
            )
            gradient = torch.autograd.grad(kl, adversarial, only_inputs=True)[0]
            adversarial = adversarial.detach() + float(step_size) * gradient.sign()
            delta = (adversarial - original).clamp(-float(epsilon), float(epsilon))
            adversarial = (original + delta).clamp(0.0, 1.0).detach()
    finally:
        model.train(was_training)
    return adversarial.detach()


def trades_objective(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    epsilon: float,
    step_size: float,
    steps: int,
    generator: torch.Generator,
    random_start: bool = True,
    attack_init: str = "uniform",
    gaussian_std: float = 0.001,
) -> TradesOutput:
    clean_logits = model(pixels)
    clean_ce = F.cross_entropy(clean_logits, targets, reduction="mean")
    clean_probs = F.softmax(clean_logits.detach(), dim=1)
    adversarial = trades_adversarial(
        model,
        pixels,
        clean_probs,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        generator=generator,
        random_start=random_start,
        attack_init=attack_init,
        gaussian_std=gaussian_std,
    )
    adversarial_logits = model(adversarial)
    trades_kl = F.kl_div(
        F.log_softmax(adversarial_logits, dim=1), clean_probs, reduction="batchmean"
    )
    total_loss = clean_ce + float(beta) * trades_kl
    delta = (adversarial - pixels).detach().flatten(1).abs().amax(dim=1)
    return TradesOutput(
        total_loss=total_loss,
        clean_ce=clean_ce,
        trades_kl=trades_kl,
        adversarial=adversarial.detach(),
        clean_accuracy=float(clean_logits.detach().argmax(1).eq(targets).float().mean().item()),
        approx_adv_accuracy=float(adversarial_logits.detach().argmax(1).eq(targets).float().mean().item()),
        linf_mean=float(delta.mean().item()),
        linf_max=float(delta.max().item()),
    )


def pgd_linf_adversarial(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    restarts: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    original = pixels.detach()
    best_loss = torch.full((len(original),), -torch.inf, device=original.device)
    best_adversarial = original.clone()
    was_training = model.training
    model.eval()
    try:
        for _ in range(int(restarts)):
            noise = torch.rand(
                original.shape,
                dtype=original.dtype,
                device=original.device,
                generator=generator,
            )
            adversarial = (original + (noise * 2.0 - 1.0) * float(epsilon)).clamp(0.0, 1.0)
            for _ in range(int(steps)):
                adversarial = adversarial.detach().requires_grad_(True)
                loss = F.cross_entropy(model(adversarial), targets, reduction="sum")
                gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
                adversarial = adversarial.detach() + float(step_size) * gradient.sign()
                delta = (adversarial - original).clamp(-float(epsilon), float(epsilon))
                adversarial = (original + delta).clamp(0.0, 1.0).detach()
            with torch.no_grad():
                per_example = F.cross_entropy(model(adversarial), targets, reduction="none")
            update = per_example > best_loss
            best_loss = torch.where(update, per_example, best_loss)
            best_adversarial[update] = adversarial[update]
    finally:
        model.train(was_training)
    return best_adversarial.detach(), best_loss.detach()


def evaluate_pgd(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    restarts: int,
    generator: torch.Generator,
    max_batches: int | None = None,
) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total = 0
    max_linf = 0.0
    for batch_index, (pixels, targets) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        pixels = pixels.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        adversarial, losses = pgd_linf_adversarial(
            model,
            pixels,
            targets,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            restarts=restarts,
            generator=generator,
        )
        with torch.no_grad():
            predictions = model(adversarial).argmax(1)
        batch_size = int(targets.numel())
        total_loss += float(losses.sum().item())
        total_correct += int(predictions.eq(targets).sum().item())
        total += batch_size
        max_linf = max(max_linf, float((adversarial - pixels).flatten(1).abs().amax().item()))
    if total == 0:
        raise RuntimeError("PGD validation processed zero examples")
    if max_linf > float(epsilon) + 1e-6:
        raise AssertionError(f"PGD perturbation exceeded epsilon: {max_linf} > {epsilon}")
    return {
        "accuracy": total_correct / total,
        "loss": total_loss / total,
        "linf_max": max_linf,
    }


def evaluate_pgd20(*args, **kwargs) -> dict[str, float]:
    """Compatibility wrapper for the historical 20-step validation schema."""
    metrics = evaluate_pgd(*args, **kwargs)
    return {f"val/pgd20_{key}": value for key, value in metrics.items()}
