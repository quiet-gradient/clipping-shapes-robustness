"""Canary construction for the one-run epsilon audit (Steinke et al., 2023).

Black-box input-space canaries per Section 6.2 of the paper: a random subset
of a held-out pool (the official test set by default), either kept as-is
("in_distribution") or randomly relabeled to a wrong class ("mislabeled", the
stronger black-box variant). Each canary is included in the training set
independently with probability 1/2 (S_i = +1) or excluded (S_i = -1); the
n - m "real" training examples are always included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from torch.utils.data import Dataset


@dataclass
class CanaryPack:
    source: str  # datamodule split the canaries were drawn from ("test" or "val")
    canary_type: str  # "mislabeled" | "in_distribution"
    seed: int
    indices: list[int]  # absolute indices into the source torchvision dataset
    original_labels: list[int]
    labels: list[int]  # canary labels (flipped for mislabeled canaries)
    s: list[int]  # +1 = included in training, -1 = excluded

    @property
    def m(self) -> int:
        return len(self.indices)

    @property
    def included_positions(self) -> list[int]:
        return [i for i, coin in enumerate(self.s) if coin == 1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CanaryPack":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


def build_canary_pack(
    pool_indices: list[int],
    pool_targets: list[int],
    m: int,
    canary_type: str,
    num_classes: int,
    seed: int,
    source: str,
) -> CanaryPack:
    """Pick m canaries from the pool, relabel if needed, flip the inclusion coins.

    ``pool_indices`` are absolute indices into the underlying dataset and
    ``pool_targets[j]`` is the label of ``pool_indices[j]``. Everything is
    driven by ``seed`` alone so the same pack (canaries AND coins) can be
    reused across mechanism variants: runs then differ only by the trained
    mechanism, not by audit randomness.
    """
    canary_type = str(canary_type).lower()
    if canary_type not in {"mislabeled", "in_distribution"}:
        raise ValueError(f"Unknown canary_type: {canary_type}")
    if m > len(pool_indices):
        raise ValueError(f"audit.m={m} exceeds canary pool size {len(pool_indices)}.")

    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(pool_indices), size=int(m), replace=False)
    indices = [int(pool_indices[j]) for j in chosen]
    original_labels = [int(pool_targets[j]) for j in chosen]
    if canary_type == "mislabeled":
        # Uniform over the num_classes - 1 wrong labels (paper: "ensuring the
        # new label is not the same as the original label").
        offsets = rng.integers(1, num_classes, size=int(m))
        labels = [int((orig + off) % num_classes) for orig, off in zip(original_labels, offsets)]
    else:
        labels = list(original_labels)
    s = [int(coin) for coin in rng.choice(np.array([-1, 1]), size=int(m))]
    return CanaryPack(
        source=str(source),
        canary_type=canary_type,
        seed=int(seed),
        indices=indices,
        original_labels=original_labels,
        labels=labels,
        s=s,
    )


class CanaryView(Dataset):
    """Presents base[indices[i]] under the base's transform with canary labels."""

    def __init__(self, base: Dataset, indices: list[int], labels: list[int]) -> None:
        if len(indices) != len(labels):
            raise ValueError("indices and labels must have the same length.")
        self.base = base
        self.indices = list(indices)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, _ = self.base[self.indices[i]]
        return x, self.labels[i]
