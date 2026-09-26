#!/usr/bin/env python3
"""Run one final public CIFAR-100 pretraining configuration."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    from trustml.pretraining.cli import main

    main()
