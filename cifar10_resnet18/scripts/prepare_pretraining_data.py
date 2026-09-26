#!/usr/bin/env python3
"""Download, identify, split, and validate the public CIFAR-100 training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from trustml.pretraining.assets import prepare_pretraining_assets, write_asset_receipt
    from trustml.pretraining.config import load_pretraining_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pretraining/runs/trades_b3_s42.yaml",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "data/pretraining_assets")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    cfg = load_pretraining_config(args.config)
    payload = prepare_pretraining_assets(
        cfg,
        data_root=args.data_root,
        asset_root=args.asset_root,
        download=args.download,
    )
    receipt = write_asset_receipt(payload, args.asset_root)
    payload["receipt"] = receipt.name
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
