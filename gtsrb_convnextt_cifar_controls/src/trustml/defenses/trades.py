from __future__ import annotations

import torch
import torch.nn.functional as F

from trustml.defenses.base import Defense


class TRADESDefense(Defense):
    name = "trades"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.beta = float(cfg.get("beta", 3.0))
        self.epsilon = float(cfg.get("epsilon", 8.0 / 255.0))
        self.step_size = float(cfg.get("step_size", 2.0 / 255.0))
        self.steps = int(cfg.get("steps", 10))
        self.mean = tuple(cfg.get("mean", (0.4914, 0.4822, 0.4465)))
        self.std = tuple(cfg.get("std", (0.2023, 0.1994, 0.2010)))
        # Graph-attached components of the last combined loss; the trainer uses
        # them for legacy "A ..." gradient diagnostics under DP and clears them.
        self.last_ce_loss: torch.Tensor | None = None
        self.last_kl_loss: torch.Tensor | None = None

    def _bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.new_tensor(self.mean).view(1, -1, 1, 1)
        std = x.new_tensor(self.std).view(1, -1, 1, 1)
        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std
        eps = self.epsilon / std
        step = self.step_size / std
        return lower, upper, eps, step

    def _generate_adversarial(self, model, x: torch.Tensor) -> torch.Tensor:
        lower, upper, eps, step = self._bounds(x)
        was_training = model.training
        model.eval()
        _set_grad_sample_hooks(model, False)
        try:
            with torch.no_grad():
                clean_probs = F.softmax(model(x), dim=1)
                x_adv = torch.clamp(x + 0.001 * torch.randn_like(x), min=lower, max=upper)

            for _ in range(self.steps):
                x_adv.requires_grad_(True)
                adv_logits = model(x_adv)
                kl = F.kl_div(F.log_softmax(adv_logits, dim=1), clean_probs, reduction="batchmean")
                grad = torch.autograd.grad(kl, x_adv)[0]
                x_adv = x_adv.detach() + step * grad.sign()
                delta = torch.clamp(x_adv - x, min=-eps, max=eps)
                x_adv = torch.clamp(x + delta, min=lower, max=upper).detach()
        finally:
            _set_grad_sample_hooks(model, True)

        if was_training:
            model.train()
        return x_adv.detach()

    def loss(self, model, batch, base_loss, context):
        x, _ = batch
        x_detached = x.detach()
        with torch.no_grad():
            clean_probs = F.softmax(context["logits"].detach(), dim=1)

        x_adv = self._generate_adversarial(model, x_detached)
        kl_loss = self.kl_loss(model, x_adv, clean_probs)
        total_loss = base_loss + self.beta * kl_loss
        self.last_ce_loss = base_loss
        self.last_kl_loss = kl_loss
        self.last_metrics = {
            "defense/trades_kl": float(kl_loss.detach().item()),
            "defense/trades_beta": self.beta,
            "TRADES CE": float(base_loss.detach().item()),
            "TRADES KL": float(kl_loss.detach().item()),
            "TRADES weighted KL": float(self.beta * kl_loss.detach().item()),
            "TRADES total loss": float(total_loss.detach().item()),
            "TRADES weighted KL to total loss ratio": float(
                (self.beta * kl_loss.detach().item()) / (total_loss.detach().item() + 1e-12)
            ),
        }
        return total_loss

    def kl_loss(self, model, x_adv: torch.Tensor, clean_probs: torch.Tensor) -> torch.Tensor:
        adv_logits = model(x_adv)
        return F.kl_div(F.log_softmax(adv_logits, dim=1), clean_probs, reduction="batchmean")


def _set_grad_sample_hooks(model, enabled: bool) -> None:
    method_name = "enable_hooks" if enabled else "disable_hooks"
    method = getattr(model, method_name, None)
    if callable(method):
        method()
