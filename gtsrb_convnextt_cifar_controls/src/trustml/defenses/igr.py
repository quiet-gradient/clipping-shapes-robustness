from __future__ import annotations

import torch

from trustml.defenses.base import Defense


class IGRDefense(Defense):
    name = "igr"
    requires_input_grad = True

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.lambda_igr = float(cfg.get("lambda_igr", 0.01))
        self.std = tuple(cfg.get("std", (0.2023, 0.1994, 0.2010)))

    def loss(self, model, batch, base_loss, context):
        x, _ = batch
        if not x.requires_grad:
            raise RuntimeError("IGRDefense requires input tensors with requires_grad=True.")

        # Under Opacus, per-sample hooks must not fire during the input-grad pass,
        # otherwise they consume stored activations before the real backward.
        _set_grad_sample_hooks(model, False)
        try:
            grad_x = torch.autograd.grad(
                outputs=base_loss,
                inputs=x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        finally:
            _set_grad_sample_hooks(model, True)
        batch_size = max(1, int(x.size(0)))
        std = x.new_tensor(self.std).view(1, -1, 1, 1)
        grad_x_denorm = (grad_x * batch_size) / (std + 1e-12)
        penalty = grad_x_denorm.view(batch_size, -1).pow(2).sum(dim=1).mean()
        self.last_metrics = {
            "defense/igr_penalty": float(penalty.detach().item()),
            "IGR penalty": float(penalty.detach().item()),
            "IGR lambda": self.lambda_igr,
        }
        return base_loss + self.lambda_igr * penalty


def _set_grad_sample_hooks(model, enabled: bool) -> None:
    method_name = "enable_hooks" if enabled else "disable_hooks"
    method = getattr(model, method_name, None)
    if callable(method):
        method()
