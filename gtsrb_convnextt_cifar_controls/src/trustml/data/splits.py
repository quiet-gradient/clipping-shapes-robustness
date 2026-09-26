from __future__ import annotations

import torch


def train_val_indices(n_items: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")

    n_train = int(n_items * train_ratio)
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_items, generator=generator).tolist()
    return perm[:n_train], perm[n_train:]

