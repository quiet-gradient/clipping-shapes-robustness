#!/usr/bin/env python3
"""Export the metadata-minimal epoch-170 checkpoint used for downstream training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from trustml.pretraining.export import export_downstream_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = export_downstream_checkpoint(args.checkpoint, args.output)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"output={path}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
