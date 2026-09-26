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
    *,
    seed: int,
    train_per_class: int = 490,
    validation_per_class: int = 10,
    num_classes: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    random = np.random.default_rng(int(seed))
    train: list[int] = []
    validation: list[int] = []
    for class_id in range(int(num_classes)):
        class_indices = np.flatnonzero(labels_array == class_id)
        expected = int(train_per_class) + int(validation_per_class)
        if class_indices.size != expected:
            raise ValueError(f"class {class_id} contains {class_indices.size} examples; expected {expected}")
        shuffled = random.permutation(class_indices)
        train.extend(int(value) for value in shuffled[: int(train_per_class)])
        validation.extend(int(value) for value in shuffled[int(train_per_class) :])
    train_array = np.asarray(sorted(train), dtype=np.int64)
    validation_array = np.asarray(sorted(validation), dtype=np.int64)
    if np.intersect1d(train_array, validation_array).size:
        raise AssertionError("training and validation indices overlap")
    return train_array, validation_array


def indices_sha256(train: np.ndarray, validation: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, values in (("train", train), ("val", validation)):
        normalized = np.asarray(values, dtype="<i8")
        digest.update(name.encode("ascii"))
        digest.update(len(normalized).to_bytes(8, "little"))
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def ensure_split_assets(cfg, raw_dataset, asset_dir: str | Path) -> tuple[Path, Path, str]:
    asset_dir = Path(asset_dir)
    archive_path = asset_dir / "cifar100_train49k_val1k_seed20260818.npz"
    metadata_path = archive_path.with_suffix(".json")
    labels = np.asarray(raw_dataset.targets, dtype=np.int64)
    if archive_path.exists() and metadata_path.exists():
        with np.load(archive_path, allow_pickle=False) as saved:
            train = saved["train_indices"].astype(np.int64, copy=False)
            validation = saved["val_indices"].astype(np.int64, copy=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        train, validation = stratified_split_indices(
            labels,
            seed=int(cfg.experiment.split_seed),
            train_per_class=int(cfg.data.train_per_class),
            validation_per_class=int(cfg.data.validation_per_class),
            num_classes=int(cfg.data.num_classes),
        )
        metadata = split_metadata(cfg, labels, train, validation)
        dataset_archive = Path(str(raw_dataset.root)) / "cifar-100-python.tar.gz"
        if dataset_archive.is_file():
            metadata["dataset_archive"] = {
                "path": dataset_archive.name,
                "bytes": dataset_archive.stat().st_size,
                "md5": _file_digest(dataset_archive, "md5"),
                "sha256": _file_digest(dataset_archive, "sha256"),
            }
        asset_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive_path, train_indices=train, val_indices=validation)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_split(cfg, labels, train, validation, metadata)
    return archive_path, metadata_path, str(metadata["indices_sha256"])


def split_metadata(cfg, labels, train: np.ndarray, validation: np.ndarray) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    return {
        "format_version": 1,
        "dataset": "CIFAR100",
        "seed": int(cfg.experiment.split_seed),
        "train_size": int(train.size),
        "validation_size": int(validation.size),
        "class_distribution": {
            "train": {
                str(class_id): int(np.sum(labels_array[train] == class_id))
                for class_id in range(100)
            },
            "validation": {
                str(class_id): int(np.sum(labels_array[validation] == class_id))
                for class_id in range(100)
            },
        },
        "indices_sha256": indices_sha256(train, validation),
        "overlap_count": int(np.intersect1d(train, validation).size),
        "no_overlap": bool(np.intersect1d(train, validation).size == 0),
    }


class CIFAR100PixelSubset(Dataset):
    """CIFAR-100 pixels with source-index-addressed augmentation randomness."""

    def __init__(self, raw_dataset, indices: Sequence[int], *, train: bool, augmentation_seed: int) -> None:
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


def build_pretraining_data(
    cfg,
    *,
    data_root: str | Path,
    asset_dir: str | Path,
    dataloader_generator: torch.Generator,
    num_workers: int,
    download: bool,
) -> dict[str, Any]:
    raw = torchvision.datasets.CIFAR100(
        root=str(data_root), train=True, transform=None, download=bool(download)
    )
    split_path, metadata_path, split_hash = ensure_split_assets(cfg, raw, asset_dir)
    with np.load(split_path, allow_pickle=False) as saved:
        train_indices = saved["train_indices"].astype(np.int64, copy=False)
        validation_indices = saved["val_indices"].astype(np.int64, copy=False)
    train_dataset = CIFAR100PixelSubset(
        raw,
        train_indices,
        train=True,
        augmentation_seed=int(cfg.reproducibility.augmentation_seed),
    )
    validation_dataset = CIFAR100PixelSubset(
        raw,
        validation_indices,
        train=False,
        augmentation_seed=int(cfg.reproducibility.augmentation_seed),
    )
    common = {
        "num_workers": int(num_workers),
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
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return {
        "train": train_loader,
        "validation": validation_loader,
        "train_dataset": train_dataset,
        "split_path": split_path,
        "split_metadata_path": metadata_path,
        "split_hash": split_hash,
    }


def _validate_split(cfg, labels, train, validation, metadata) -> None:
    errors: list[str] = []
    if len(train) != 49_000:
        errors.append("training split does not contain 49,000 examples")
    if len(validation) != 1_000:
        errors.append("validation split does not contain 1,000 examples")
    if np.intersect1d(train, validation).size:
        errors.append("training and validation splits overlap")
    if int(metadata.get("seed", -1)) != int(cfg.experiment.split_seed):
        errors.append("saved split seed differs from the configuration")
    if metadata.get("indices_sha256") != indices_sha256(train, validation):
        errors.append("saved split hash differs from the index arrays")
    labels_array = np.asarray(labels, dtype=np.int64)
    for class_id in range(100):
        if int(np.sum(labels_array[train] == class_id)) != 490:
            errors.append(f"class {class_id} does not have 490 training examples")
        if int(np.sum(labels_array[validation] == class_id)) != 10:
            errors.append(f"class {class_id} does not have 10 validation examples")
    if errors:
        raise RuntimeError("invalid saved CIFAR-100 split: " + "; ".join(errors))


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
