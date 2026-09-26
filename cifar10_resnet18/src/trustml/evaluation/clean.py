from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate_clean(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
    progress: bool = False,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    iterator = enumerate(dataloader)
    iterator = tqdm(iterator, total=len(dataloader), desc="eval", leave=False, disable=not progress)
    for batch_idx, (x, y) in iterator:
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        batch_size = int(y.numel())
        total_loss += float(loss.item()) * batch_size
        correct += int(logits.argmax(dim=1).eq(y).sum().item())
        total += batch_size

    if was_training:
        model.train()
    return {
        "loss": total_loss / max(1, total),
        "clean_acc": correct / max(1, total),
    }

