from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray,
    seed: int,
    train_per_class: int = 450,
    val_per_class: int = 50,
    num_classes: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    train: list[int] = []
    val: list[int] = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels_array == class_id)
        expected = train_per_class + val_per_class
        if class_indices.size != expected:
            raise ValueError(
                f"class {class_id} has {class_indices.size} samples, expected exactly {expected}"
            )
        shuffled = rng.permutation(class_indices)
        train.extend(int(value) for value in shuffled[:train_per_class])
        val.extend(int(value) for value in shuffled[train_per_class:])
    train_array = np.asarray(sorted(train), dtype=np.int64)
    val_array = np.asarray(sorted(val), dtype=np.int64)
    if np.intersect1d(train_array, val_array).size:
        raise AssertionError("train and validation indices overlap")
    return train_array, val_array


def indices_sha256(train_indices: np.ndarray, val_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, values in (("train", train_indices), ("val", val_indices)):
        normalized = np.asarray(values, dtype="<i8")
        digest.update(name.encode("ascii"))
        digest.update(len(normalized).to_bytes(8, "little"))
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def split_metadata(
    labels: Sequence[int] | np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    overlap = np.intersect1d(train_indices, val_indices)
    return {
        "format_version": 1,
        "dataset": "CIFAR100",
        "seed": int(seed),
        "train_size": int(train_indices.size),
        "validation_size": int(val_indices.size),
        "class_distribution": {
            "train": {
                str(class_id): int(np.sum(labels_array[train_indices] == class_id))
                for class_id in range(100)
            },
            "validation": {
                str(class_id): int(np.sum(labels_array[val_indices] == class_id))
                for class_id in range(100)
            },
        },
        "indices_sha256": indices_sha256(train_indices, val_indices),
        "overlap_count": int(overlap.size),
        "no_overlap": bool(overlap.size == 0),
    }


def ensure_split_assets(cfg, raw_dataset=None) -> tuple[Path, Path, str]:
    npz_path = Path(str(cfg.data.split_npz))
    json_path = Path(str(cfg.data.split_metadata))
    if raw_dataset is None:
        raw_dataset = torchvision.datasets.CIFAR100(
            root=str(cfg.runtime.dataset_root), train=True, transform=None, download=bool(cfg.data.download)
        )
    labels = np.asarray(raw_dataset.targets, dtype=np.int64)
    if npz_path.exists() and json_path.exists():
        with np.load(npz_path, allow_pickle=False) as split:
            train_indices = split["train_indices"].astype(np.int64, copy=False)
            val_indices = split["val_indices"].astype(np.int64, copy=False)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        train_indices, val_indices = stratified_split_indices(
            labels,
            seed=int(cfg.experiment.split_seed),
            train_per_class=int(cfg.data.train_per_class),
            val_per_class=int(cfg.data.val_per_class),
        )
        metadata = split_metadata(labels, train_indices, val_indices, int(cfg.experiment.split_seed))
        archive = Path(str(cfg.runtime.dataset_root)) / "cifar-100-python.tar.gz"
        if archive.is_file():
            metadata["dataset_archive"] = {
                "path": archive.name,
                "bytes": archive.stat().st_size,
                "md5": _file_digest(archive, "md5"),
                "sha256": _file_digest(archive, "sha256"),
            }
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, train_indices=train_indices, val_indices=val_indices)
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_split(labels, train_indices, val_indices, metadata, cfg)
    return npz_path, json_path, str(metadata["indices_sha256"])


class CIFAR100PixelSubset(Dataset):
    """CIFAR-100 subset yielding [0,1] tensors with epoch-addressed augmentation RNG."""

    def __init__(self, raw_dataset, indices: Sequence[int], train: bool, augmentation_seed: int) -> None:
        self.raw_dataset = raw_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = bool(train)
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        source_index = int(self.indices[item])
        image = torch.from_numpy(self.raw_dataset.data[source_index].copy()).permute(2, 0, 1).float().div_(255.0)
        if self.train:
            generator = torch.Generator().manual_seed(
                self.augmentation_seed + self.epoch * 1_000_003 + source_index
            )
            image = F.pad(image, (4, 4, 4, 4), mode="constant", value=0.0)
            top = int(torch.randint(0, 9, (), generator=generator).item())
            left = int(torch.randint(0, 9, (), generator=generator).item())
            image = image[:, top : top + 32, left : left + 32]
            if bool(torch.rand((), generator=generator).item() < 0.5):
                image = torch.flip(image, dims=(2,))
        return image, int(self.raw_dataset.targets[source_index])


def build_pretrain_data(cfg, dataloader_generator: torch.Generator):
    raw = torchvision.datasets.CIFAR100(
        root=str(cfg.runtime.dataset_root), train=True, transform=None, download=bool(cfg.data.download)
    )
    npz_path, json_path, split_hash = ensure_split_assets(cfg, raw_dataset=raw)
    with np.load(npz_path, allow_pickle=False) as split:
        train_indices = split["train_indices"].astype(np.int64, copy=False)
        val_indices = split["val_indices"].astype(np.int64, copy=False)
    train_dataset = CIFAR100PixelSubset(
        raw, train_indices, train=True, augmentation_seed=int(cfg.reproducibility.augmentation_seed)
    )
    val_dataset = CIFAR100PixelSubset(
        raw, val_indices, train=False, augmentation_seed=int(cfg.reproducibility.augmentation_seed)
    )
    workers = int(cfg.runtime.num_workers)
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        drop_last=False,
        generator=dataloader_generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "train_dataset": train_dataset,
        "split_hash": split_hash,
        "split_npz": npz_path,
        "split_metadata": json_path,
    }


def _validate_split(labels, train_indices, val_indices, metadata, cfg) -> None:
    expected_hash = indices_sha256(train_indices, val_indices)
    errors: list[str] = []
    train_per_class = int(cfg.data.train_per_class)
    val_per_class = int(cfg.data.val_per_class)
    expected_train = train_per_class * int(cfg.data.num_classes)
    expected_val = val_per_class * int(cfg.data.num_classes)
    if len(train_indices) != expected_train:
        errors.append(f"train size={len(train_indices)}, expected {expected_train}")
    if len(val_indices) != expected_val:
        errors.append(f"validation size={len(val_indices)}, expected {expected_val}")
    if np.intersect1d(train_indices, val_indices).size:
        errors.append("train/validation overlap is non-empty")
    if metadata.get("indices_sha256") != expected_hash:
        errors.append("metadata indices_sha256 mismatch")
    if int(metadata.get("seed", -1)) != int(cfg.experiment.split_seed):
        errors.append("split seed mismatch")
    labels_array = np.asarray(labels)
    for class_id in range(100):
        if int(np.sum(labels_array[train_indices] == class_id)) != train_per_class:
            errors.append(f"class {class_id} train count is not {train_per_class}")
        if int(np.sum(labels_array[val_indices] == class_id)) != val_per_class:
            errors.append(f"class {class_id} validation count is not {val_per_class}")
    if errors:
        raise RuntimeError("invalid saved CIFAR-100 split: " + "; ".join(errors))


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
