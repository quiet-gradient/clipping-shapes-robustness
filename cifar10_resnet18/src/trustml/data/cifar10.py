from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from trustml.data.splits import train_val_indices
from trustml.data.transforms import CIFAR10_MEAN, CIFAR10_STD, cifar_transform


def build_cifar10_datamodule(cfg: DictConfig) -> dict:
    """Build the fixed 45,000/5,000 private-train/public-dev protocol.

    The official CIFAR-10 test split is intentionally never instantiated.
    """
    root = Path(str(cfg.get("root", "./data"))).expanduser()
    mean = tuple(cfg.get("mean", CIFAR10_MEAN))
    std = tuple(cfg.get("std", CIFAR10_STD))
    download = bool(cfg.get("download", False))
    augmentation = bool(cfg.get("augmentation", True))
    seed = int(cfg.get("seed", 20260820))
    batch_size = int(cfg.get("batch_size", 512))
    num_workers = int(cfg.get("num_workers", 4))
    train_ratio = float(cfg.get("train_ratio", 0.9))

    train_transform = cifar_transform(
        train=True, mean=mean, std=std, augmentation=augmentation
    )
    eval_transform = cifar_transform(
        train=False, mean=mean, std=std, augmentation=False
    )
    full_train_augmented = torchvision.datasets.CIFAR10(
        root=str(root), train=True, transform=train_transform, download=download
    )
    full_train_evaluation = torchvision.datasets.CIFAR10(
        root=str(root), train=True, transform=eval_transform, download=False
    )
    train_indices, dev_indices = train_val_indices(
        len(full_train_augmented), train_ratio, seed
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
    }
    return {
        "train": DataLoader(
            Subset(full_train_augmented, train_indices),
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            Subset(full_train_evaluation, dev_indices),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    }
