from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from trustml.data.splits import train_val_indices
from trustml.data.transforms import CIFAR100_MEAN, CIFAR100_STD, cifar_transform


def build_cifar100_datamodule(cfg: DictConfig) -> dict[str, DataLoader]:
    root = Path(str(cfg.get("root", "./data"))).expanduser()
    mean = tuple(cfg.get("mean", CIFAR100_MEAN))
    std = tuple(cfg.get("std", CIFAR100_STD))
    download = bool(cfg.get("download", True))
    augmentation = bool(cfg.get("augmentation", True))
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 4))
    train_ratio = float(cfg.get("train_ratio", 0.9))

    train_tfm = cifar_transform(train=True, mean=mean, std=std, augmentation=augmentation)
    eval_tfm = cifar_transform(train=False, mean=mean, std=std, augmentation=False)

    full_train_aug = torchvision.datasets.CIFAR100(
        root=str(root), train=True, transform=train_tfm, download=download
    )
    full_train_eval = torchvision.datasets.CIFAR100(
        root=str(root), train=True, transform=eval_tfm, download=download
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=str(root), train=False, transform=eval_tfm, download=download
    )

    train_idx, val_idx = train_val_indices(len(full_train_aug), train_ratio, seed)
    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    use_cuda = torch.cuda.is_available()
    # persistent_workers off: lingering worker pools can fail to join at
    # interpreter exit on a headless VM and flip an otherwise-finished job to
    # ERROR. See the note in cifar10.build_cifar10_datamodule.
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": False,
    }
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
    }

