"""Verify downloaded GTSRB release assets against the frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("weights/files"))
    parser.add_argument("--manifest", type=Path, default=Path("weights/manifest.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    if len(assets) != manifest["asset_count"]:
        raise SystemExit("manifest asset count mismatch")
    if sum(row["bytes"] for row in assets) != manifest["total_bytes"]:
        raise SystemExit("manifest byte count mismatch")

    seen: set[str] = set()
    checked = 0
    for row in assets:
        name = row["asset"]
        if Path(name).name != name or name in seen:
            raise SystemExit(f"unsafe or duplicate asset name: {name}")
        seen.add(name)
        path = args.directory / name
        if not path.is_file():
            if args.allow_partial:
                continue
            raise SystemExit(f"missing asset: {path}")
        if path.stat().st_size != row["bytes"]:
            raise SystemExit(f"size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["export_sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        checked += 1

    if not checked:
        raise SystemExit("no assets checked")
    print(f"PASS: {checked}/{len(assets)} release assets verified")


if __name__ == "__main__":
    main()
