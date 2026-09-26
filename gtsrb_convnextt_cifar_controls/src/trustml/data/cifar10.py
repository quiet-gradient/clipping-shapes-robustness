from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Subset

from trustml.data.splits import train_val_indices
from trustml.data.transforms import CIFAR10_MEAN, CIFAR10_STD, cifar_transform


def build_cifar10_datamodule(cfg: DictConfig, audit_cfg: DictConfig | None = None) -> dict:
    root = Path(str(cfg.get("root", "./data"))).expanduser()
    mean = tuple(cfg.get("mean", CIFAR10_MEAN))
    std = tuple(cfg.get("std", CIFAR10_STD))
    download = bool(cfg.get("download", True))
    augmentation = bool(cfg.get("augmentation", True))
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 4))
    train_ratio = float(cfg.get("train_ratio", 0.9))

    train_tfm = cifar_transform(train=True, mean=mean, std=std, augmentation=augmentation)
    eval_tfm = cifar_transform(train=False, mean=mean, std=std, augmentation=False)

    full_train_aug = torchvision.datasets.CIFAR10(
        root=str(root), train=True, transform=train_tfm, download=download
    )
    full_train_eval = torchvision.datasets.CIFAR10(
        root=str(root), train=True, transform=eval_tfm, download=download
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=str(root), train=False, transform=eval_tfm, download=download
    )

    train_idx, val_idx = train_val_indices(len(full_train_aug), train_ratio, seed)

    audit_enabled = audit_cfg is not None and bool(audit_cfg.get("enabled", False))
    audit_pack = None
    audit_canary_eval_ds = None
    test_eval_indices: list[int] | None = None
    if audit_enabled:
        from trustml.audit.canaries import CanaryView, build_canary_pack

        source = str(audit_cfg.get("canary_source", "test")).lower()
        if source == "test":
            # Paper Section 6.2: canaries are a random subset of the test set.
            pool_indices = list(range(len(test_ds)))
            pool_targets = list(test_ds.targets)
            canary_train_base = torchvision.datasets.CIFAR10(
                root=str(root), train=False, transform=train_tfm, download=download
            )
            canary_eval_base = test_ds
        elif source == "val":
            pool_indices = list(val_idx)
            pool_targets = [int(full_train_eval.targets[i]) for i in val_idx]
            canary_train_base = full_train_aug
            canary_eval_base = full_train_eval
        else:
            raise ValueError(f"Unknown audit.canary_source: {source}")

        audit_pack = build_canary_pack(
            pool_indices=pool_indices,
            pool_targets=pool_targets,
            m=int(audit_cfg.get("m", 1000)),
            canary_type=str(audit_cfg.get("canary_type", "mislabeled")),
            num_classes=10,
            seed=int(audit_cfg.get("seed", 20230508)),
            source=source,
        )
        # Canaries with S=+1 join the training set (train transform, canary
        # labels); the pool split drops all m canaries from its eval loader so
        # test/val metrics and MIA non-members stay uncontaminated.
        canary_set = set(audit_pack.indices)
        if source == "test":
            test_eval_indices = [i for i in range(len(test_ds)) if i not in canary_set]
        else:
            val_idx = [i for i in val_idx if i not in canary_set]
        included = audit_pack.included_positions
        audit_train_ds = CanaryView(
            canary_train_base,
            [audit_pack.indices[i] for i in included],
            [audit_pack.labels[i] for i in included],
        )
        audit_canary_eval_ds = CanaryView(canary_eval_base, audit_pack.indices, audit_pack.labels)

    train_ds = Subset(full_train_aug, train_idx)
    if audit_enabled:
        train_ds = ConcatDataset([train_ds, audit_train_ds])
    val_ds = Subset(full_train_eval, val_idx)
    # Members for MIA are the private training subset scored WITHOUT augmentation
    # (deterministic eval transform), so membership signal is not diluted by
    # random crops/flips. Non-members are the official test set (already eval tfm).
    mia_member_ds = Subset(full_train_eval, train_idx)
    test_eval_ds = test_ds if test_eval_indices is None else Subset(test_ds, test_eval_indices)

    use_cuda = torch.cuda.is_available()
    # persistent_workers is deliberately off: this module builds up to five
    # loaders (train/val/test/mia_members/audit_canaries), and on a headless
    # DataSphere VM the lingering worker pools of several persistent loaders
    # fail to join at interpreter exit -- the run finishes and writes all
    # metrics, then the process exits non-zero and the job is marked ERROR.
    # Respawning workers each epoch costs ~ms and is negligible for these runs.
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": False,
    }
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_eval_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        "mia_members": DataLoader(mia_member_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
    }
    if audit_enabled:
        audit_batch = int(audit_cfg.get("batch_size", batch_size))
        loaders["audit_canaries"] = DataLoader(
            audit_canary_eval_ds, batch_size=audit_batch, shuffle=False, **loader_kwargs
        )
        loaders["audit_pack"] = audit_pack
    return loaders
