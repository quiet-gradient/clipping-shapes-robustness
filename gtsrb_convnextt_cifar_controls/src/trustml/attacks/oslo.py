from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _resize(model_name: str, x: torch.Tensor) -> torch.Tensor:
    name = model_name.lower()
    if name in {"inception", "inception_v3", "inceptionv3"}:
        return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    if name in {"vgg13", "densenet121", "resnext50", "shufflenet"}:
        return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x


class OsloAttack:
    def __init__(
        self,
        source_models: Sequence[tuple[str, nn.Module]],
        validation_models: Sequence[tuple[str, nn.Module]],
        eps: float,
        eps_step: float,
        max_iter: int,
        num_subprocedures: int,
        tau: float,
        device: torch.device,
    ) -> None:
        self.source_models = list(source_models)
        self.validation_models = list(validation_models)
        self.eps = float(eps)
        self.eps_step = float(eps_step)
        self.max_iter = int(max_iter)
        self.num_subprocedures = int(num_subprocedures)
        self.tau = float(tau)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

        for _, model in self.source_models + self.validation_models:
            model.eval().to(device)

    @torch.no_grad()
    def _validation_confidence(self, x_cur: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs_sum = None
        for name, model in self.validation_models:
            probs = F.softmax(model(_resize(name, x_cur)), dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs_mean = probs_sum / max(1, len(self.validation_models))
        return probs_mean.gather(1, y.view(-1, 1)).view(-1)

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)
        x_orig = x.detach().clone()
        x_old = x_orig.clone()
        finished = torch.zeros(x.size(0), dtype=torch.bool, device=self.device)

        for k in range(1, self.num_subprocedures + 1):
            x_cur = x_old.detach().clone()
            for _ in range(self.max_iter):
                x_cur.requires_grad_(True)
                grads_sum = torch.zeros_like(x_cur)
                for name, model in self.source_models:
                    loss = self.loss_fn(model(_resize(name, x_cur)), y)
                    grads_sum += torch.autograd.grad(loss, x_cur)[0]
                grads_sum /= max(1, len(self.source_models))

                radius = self.eps * float(k) / float(self.num_subprocedures)
                x_updated = x_cur.detach() + self.eps_step * grads_sum.sign()
                x_updated = torch.min(torch.max(x_updated, x_orig - radius), x_orig + radius)

                conf = self._validation_confidence(x_updated, y)
                newly_finished = (~finished) & (conf < self.tau)
                still_active = (~finished) & ~newly_finished
                x_cur = x_cur.detach()
                x_cur[still_active] = x_updated[still_active]
                x_cur[newly_finished] = x_updated[newly_finished]
                finished |= newly_finished
                if finished.all():
                    break
            x_old = x_cur.detach()
            if finished.all():
                break
        return x_old.detach()

