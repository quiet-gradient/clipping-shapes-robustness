"""Verify downloaded evaluation weights against the release manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("weights/files"))
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifests = args.manifest or [Path("weights/manifest.json"), Path("weights/additional_manifest.json")]
    assets = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if len(manifest["assets"]) != manifest["asset_count"]:
            raise SystemExit(f"manifest asset count mismatch: {path}")
        if sum(row["bytes"] for row in manifest["assets"]) != manifest["total_bytes"]:
            raise SystemExit(f"manifest byte count mismatch: {path}")
        assets.extend(manifest["assets"])

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
