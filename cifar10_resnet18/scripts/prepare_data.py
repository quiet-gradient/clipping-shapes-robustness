#!/usr/bin/env python3
"""Download, extract, and verify the CIFAR-10 training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torchvision


ARCHIVE_NAME = "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
EXPECTED_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 6))


def _digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    torchvision.datasets.CIFAR10(root=str(root), train=True, download=True)
    archive = root / ARCHIVE_NAME
    extracted = root / "cifar-10-batches-py"
    if not archive.is_file():
        raise FileNotFoundError(f"downloaded archive is missing: {archive}")
    md5 = _digest(archive, "md5")
    sha256 = _digest(archive, "sha256")
    if md5 != ARCHIVE_MD5 or sha256 != ARCHIVE_SHA256:
        raise RuntimeError(
            f"CIFAR-10 archive checksum mismatch: md5={md5}, sha256={sha256}"
        )
    missing = [name for name in EXPECTED_BATCHES if not (extracted / name).is_file()]
    if missing:
        raise RuntimeError(f"CIFAR-10 extraction is incomplete: {missing}")
    return {
        "status": "PASS",
        "data_root": str(root),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "training_batches": len(EXPECTED_BATCHES),
        "official_test_split_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
