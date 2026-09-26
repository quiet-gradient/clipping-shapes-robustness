#!/usr/bin/env python3
"""Check the environment and immutable assets before a reproduction run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "torch": "2.4.1+cu121",
    "torchvision": "0.19.1+cu121",
    "numpy": "1.26.4",
    "opacus": "1.5.4",
    "omegaconf": "2.3.1",
    "PyYAML": "6.0.3",
    "tqdm": "4.67.1",
    "scipy": "1.14.1",
    "autoattack": "0.1",
}
CHECKPOINT_SHA256 = "53f358fe8a0e7b87e0dbbf8b0814d885e6bb82ba5452baa22521a58bf2d68909"
DATA_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
CIFAR100_DATA_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
CIFAR100_SPLIT_NPZ_SHA256 = "4c106065a957559bbf80963028c5339e527f2003927dda6f3cc67e396081af05"
CIFAR100_SPLIT_JSON_SHA256 = "4da06805fddad6331b20363dd68ae7c45137f7a75fd3be04952201c015ca9951"
PRETRAIN_INITIAL_STATE_SHA256 = "8e69a3f39f1a058067527565b2c86972b7f9f1fb7fc174a6d5e61aa40cffde6b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(
    *,
    data_root: Path,
    pretraining_asset_root: Path | None,
    require_exact: bool,
    require_cuda: bool,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    versions = {}
    for distribution, expected in EXPECTED.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            observed = "missing"
        versions[distribution] = {"expected": expected, "observed": observed}
        if observed != expected:
            message = f"{distribution}: expected {expected}, observed {observed}"
            (errors if require_exact else warnings).append(message)

    python_minor = f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
    if python_minor != "3.11":
        message = f"Python: expected 3.11.x, observed {platform.python_version()}"
        (errors if require_exact else warnings).append(message)

    checkpoint = ROOT / "artifacts/public_pretrain/checkpoint_epoch_170.pt"
    if not checkpoint.is_file():
        errors.append(f"missing initialization checkpoint: {checkpoint.relative_to(ROOT)}")
        checkpoint_hash = None
    else:
        checkpoint_hash = _sha256(checkpoint)
        if checkpoint_hash != CHECKPOINT_SHA256:
            errors.append("initialization checkpoint SHA-256 mismatch")

    archive = data_root / "cifar-10-python.tar.gz"
    if archive.is_file():
        data_hash = _sha256(archive)
        if data_hash != DATA_SHA256:
            errors.append("CIFAR-10 archive SHA-256 mismatch")
    else:
        data_hash = None
        warnings.append("CIFAR-10 archive is absent; run scripts/prepare_data.py")

    cifar100_archive = data_root / "cifar-100-python.tar.gz"
    if cifar100_archive.is_file():
        cifar100_hash = _sha256(cifar100_archive)
        if cifar100_hash != CIFAR100_DATA_SHA256:
            errors.append("CIFAR-100 archive SHA-256 mismatch")
    else:
        cifar100_hash = None
        warnings.append("CIFAR-100 archive is absent; run scripts/prepare_pretraining_data.py")

    bundled_asset_root = ROOT / "artifacts/public_pretrain"
    split_npz = bundled_asset_root / "cifar100_train49k_val1k_seed20260818.npz"
    split_json = split_npz.with_suffix(".json")
    split_hashes = {
        "npz": _sha256(split_npz) if split_npz.is_file() else None,
        "json": _sha256(split_json) if split_json.is_file() else None,
    }
    if split_hashes != {
        "npz": CIFAR100_SPLIT_NPZ_SHA256,
        "json": CIFAR100_SPLIT_JSON_SHA256,
    }:
        errors.append("fixed CIFAR-100 split asset SHA-256 mismatch")

    runtime_asset_root = pretraining_asset_root or data_root / "pretraining_assets"
    common_initialization = runtime_asset_root / "cifar_resnet18_gn_seed42.pt"
    if common_initialization.is_file():
        payload = torch.load(common_initialization, map_location="cpu", weights_only=False)
        initialization_state_hash = payload.get("state_sha256")
        if initialization_state_hash != PRETRAIN_INITIAL_STATE_SHA256:
            errors.append("public-pretraining common initialization tensor hash mismatch")
    else:
        initialization_state_hash = None
        warnings.append(
            "public-pretraining common initialization is absent; "
            "run scripts/prepare_pretraining_data.py before regenerating the checkpoint"
        )

    cuda = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda.update(
            {
                "device": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(properties.total_memory),
                "runtime": torch.version.cuda,
            }
        )
        if require_exact and torch.version.cuda != "12.1":
            errors.append(f"CUDA runtime: expected 12.1, observed {torch.version.cuda}")
    elif require_cuda:
        errors.append("CUDA is required for full training and AutoAttack evaluation")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "python": platform.python_version(),
        "packages": versions,
        "cuda": cuda,
        "checkpoint_sha256": checkpoint_hash,
        "data_archive_sha256": data_hash,
        "cifar100_archive_sha256": cifar100_hash,
        "cifar100_split_sha256": split_hashes,
        "pretraining_initial_state_sha256": initialization_state_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pretraining-asset-root", type=Path, default=None)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = check(
        data_root=args.data_root,
        pretraining_asset_root=args.pretraining_asset_root,
        require_exact=args.require_exact,
        require_cuda=args.require_cuda,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
