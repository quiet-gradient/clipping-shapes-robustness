from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torchvision
from torchvision.datasets.utils import download_url

from trustml.pretraining.config import file_sha256
from trustml.pretraining.data import ensure_split_assets
from trustml.pretraining.model import ensure_initialization_asset


CIFAR100_ARCHIVE_BYTES = 169_001_437
CIFAR100_ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR100_ARCHIVE_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
SPLIT_INDEX_SHA256 = "5ea2a32373bd64937897b0feae09c239b131a99829a877bcd0fbefb9b0a64f66"
INITIALIZATION_STATE_SHA256 = "8e69a3f39f1a058067527565b2c86972b7f9f1fb7fc174a6d5e61aa40cffde6b"
SPLIT_FILENAME = "cifar100_train49k_val1k_seed20260818.npz"
INITIALIZATION_FILENAME = "cifar_resnet18_gn_seed42.pt"


def prepare_pretraining_assets(
    cfg,
    *,
    data_root: str | Path,
    asset_root: str | Path,
    download: bool,
) -> dict[str, Any]:
    data_root = Path(data_root)
    asset_root = Path(asset_root)
    archive = data_root / "cifar-100-python.tar.gz"
    if bool(download) and not archive.is_file():
        data_root.mkdir(parents=True, exist_ok=True)
        download_url(
            torchvision.datasets.CIFAR100.url,
            str(data_root),
            filename=torchvision.datasets.CIFAR100.filename,
            md5=CIFAR100_ARCHIVE_MD5,
        )
    raw = torchvision.datasets.CIFAR100(
        root=str(data_root), train=True, transform=None, download=bool(download)
    )
    archive_details = _verify_dataset_archive(archive)
    split_path, metadata_path, split_hash = ensure_split_assets(cfg, raw, asset_root)
    if split_hash != SPLIT_INDEX_SHA256:
        raise RuntimeError(f"fixed split hash mismatch: {split_hash}")
    initialization_path, initialization_hash = ensure_initialization_asset(
        cfg, asset_root / INITIALIZATION_FILENAME
    )
    if initialization_hash != INITIALIZATION_STATE_SHA256:
        raise RuntimeError(
            "shared initialization tensor hash differs from the pinned PyTorch 2.4.1 result"
        )
    return {
        "format_version": 1,
        "dataset_archive": archive_details,
        "split": {
            "npz": split_path.name,
            "json": metadata_path.name,
            "indices_sha256": split_hash,
        },
        "initialization": {
            "path": initialization_path.name,
            "file_sha256": file_sha256(initialization_path),
            "state_sha256": initialization_hash,
            "seed": 42,
        },
    }


def write_asset_receipt(payload: dict[str, Any], asset_root: str | Path) -> Path:
    path = Path(asset_root) / "pretraining_assets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _verify_dataset_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"CIFAR-100 archive is missing: {path}")
    observed = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "md5": _digest(path, "md5"),
        "sha256": _digest(path, "sha256"),
    }
    expected = {
        "filename": "cifar-100-python.tar.gz",
        "bytes": CIFAR100_ARCHIVE_BYTES,
        "md5": CIFAR100_ARCHIVE_MD5,
        "sha256": CIFAR100_ARCHIVE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"CIFAR-100 archive identity mismatch: {observed}")
    return observed


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
