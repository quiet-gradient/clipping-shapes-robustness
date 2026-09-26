from __future__ import annotations

import torch.nn.functional as F

from trustml.defenses.base import Defense


class LabelSmoothingDefense(Defense):
    name = "label_smoothing"

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.smoothing = float(cfg.get("smoothing", 0.1))

    def loss(self, model, batch, base_loss, context):
        _, targets = batch
        logits = context["logits"]
        smoothed_ce = F.cross_entropy(logits, targets, label_smoothing=self.smoothing)
        self.last_metrics = {
            "defense/label_smoothing": self.smoothing,
            "Label smoothing": self.smoothing,
        }
        # Replace only the plain-CE component so terms added by other defenses
        # (e.g. the TRADES KL) survive regardless of defense order.
        return base_loss - context["base_loss"] + smoothed_ce
