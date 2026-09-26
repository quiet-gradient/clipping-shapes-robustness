from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


class PGDAttack:
    def __init__(self, cfg, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)) -> None:
        self.epsilon = float(cfg.get("epsilon", 8.0 / 255.0))
        self.step_size = float(cfg.get("step_size", 2.0 / 255.0))
        self.steps = int(cfg.get("steps", 20))
        self.restarts = int(cfg.get("restarts", 1))
        self.max_batches = cfg.get("max_batches", None)
        self.mean = tuple(cfg.get("mean", mean))
        self.std = tuple(cfg.get("std", std))
        if self.restarts < 1:
            raise ValueError("PGD restarts must be positive")

    def _bounds(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.new_tensor(self.mean).view(1, -1, 1, 1)
        std = x.new_tensor(self.std).view(1, -1, 1, 1)
        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std
        eps = self.epsilon / std
        step = self.step_size / std
        return lower, upper, eps, step

    def generate(self, model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lower, upper, eps, step = self._bounds(x)
        x_orig = x.detach()
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-1.0, 1.0) * eps
        x_adv = torch.clamp(x_adv, min=lower, max=upper)

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + step * grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = torch.clamp(x_orig + delta, min=lower, max=upper).detach()
        return x_adv

    def run(self, model, dataloader, device: torch.device, progress: bool = True) -> dict[str, float]:
        was_training = model.training
        model.eval()
        # Opacus per-sample hooks must not fire during attack backward passes.
        _set_grad_sample_hooks(model, False)
        clean_correct = 0
        adv_correct = 0
        attack_success = 0
        clean_correct_total = 0
        total = 0
        max_abs_delta_first_batch = None
        flip_rate_first_batch = None

        try:
            iterator = enumerate(dataloader)
            iterator = tqdm(iterator, total=len(dataloader), desc="pgd", leave=False, disable=not progress)
            for batch_idx, (x, y) in iterator:
                if self.max_batches is not None and batch_idx >= int(self.max_batches):
                    break
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    clean_pred = model(x).argmax(dim=1)
                adv_correct_mask = torch.ones_like(y, dtype=torch.bool)
                first_adv = None
                first_adv_pred = None
                for _ in range(self.restarts):
                    x_adv = self.generate(model, x, y)
                    with torch.no_grad():
                        current_adv_pred = model(x_adv).argmax(dim=1)
                    adv_correct_mask.logical_and_(current_adv_pred.eq(y))
                    if first_adv is None:
                        first_adv = x_adv
                        first_adv_pred = current_adv_pred

                clean_mask = clean_pred.eq(y)
                if max_abs_delta_first_batch is None:
                    assert first_adv is not None and first_adv_pred is not None
                    max_abs_delta_first_batch = float((first_adv - x).detach().abs().max().item())
                    flip_rate_first_batch = float(first_adv_pred.ne(clean_pred).float().mean().item())
                clean_correct += int(clean_mask.sum().item())
                adv_correct += int(adv_correct_mask.sum().item())
                attack_success += int((~adv_correct_mask & clean_mask).sum().item())
                clean_correct_total += int(clean_mask.sum().item())
                total += int(y.numel())
        finally:
            _set_grad_sample_hooks(model, True)
            if was_training:
                model.train()
        clean_acc = clean_correct / max(1, total)
        adv_acc = adv_correct / max(1, total)
        asr = attack_success / max(1, clean_correct_total)
        return {
            "attack/pgd_clean_acc": clean_acc,
            "attack/pgd_adv_acc": adv_acc,
            "attack/pgd_asr": asr,
            "CLEAN_ACC": clean_acc,
            "ADV_ACC": adv_acc,
            "ASR": asr,
            "max_abs_delta_first_batch": 0.0 if max_abs_delta_first_batch is None else max_abs_delta_first_batch,
            "flip_rate_first_batch": 0.0 if flip_rate_first_batch is None else flip_rate_first_batch,
            "attack/pgd_restarts": float(self.restarts),
        }


def _set_grad_sample_hooks(model, enabled: bool) -> None:
    method_name = "enable_hooks" if enabled else "disable_hooks"
    method = getattr(model, method_name, None)
    if callable(method):
        method()
