#!/usr/bin/env python3
"""List or run the five final public-pretraining configurations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/pretraining.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or len({row["run_id"] for row in rows}) != 5:
        raise RuntimeError("final pretraining manifest must contain five unique runs")
    return rows


def _validate_row(row: dict[str, str]) -> Path:
    path = ROOT / row["config_path"]
    if not path.is_file():
        raise RuntimeError(f"configuration is missing for {row['run_id']}")
    observed = _sha256(path)
    if observed != row["config_sha256"]:
        raise RuntimeError(f"configuration hash mismatch for {row['run_id']}: {observed}")
    base = ROOT / "configs/pretraining/base.yaml"
    observed_base = _sha256(base)
    if observed_base != row["base_config_sha256"]:
        raise RuntimeError(f"base configuration hash mismatch: {observed_base}")
    return path


def _select(rows: list[dict[str, str]], args) -> list[dict[str, str]]:
    if args.run_id:
        requested = set(args.run_id)
        selected = [row for row in rows if row["run_id"] in requested]
        missing = sorted(requested - {row["run_id"] for row in selected})
        if missing:
            raise SystemExit(f"unknown run IDs: {', '.join(missing)}")
        return selected
    if args.all or args.list:
        return rows
    raise SystemExit("select --all, --run-id, or --list")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "data/pretraining_assets")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/pretraining")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'auto'; only for one run.")
    parser.add_argument("--smoke-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    args = parser.parse_args()
    selected = _select(_rows(), args)
    for row in selected:
        _validate_row(row)
    if args.resume is not None and len(selected) != 1:
        parser.error("--resume requires exactly one selected run")
    if args.list:
        for row in selected:
            beta = row["beta"] or "-"
            print(
                f"{row['run_id']:<18} objective={row['objective']:<6} beta={beta:<3} "
                f"seed={row['seed']} role={row['evidence_role']}"
            )
        print(f"selected={len(selected)}")
        return

    failures: list[tuple[str, int]] = []
    for index, row in enumerate(selected, start=1):
        command = [
            sys.executable,
            str(ROOT / "scripts/train_public_pretraining.py"),
            "--config",
            row["config_path"],
            "--data-root",
            str(args.data_root),
            "--asset-root",
            str(args.asset_root),
            "--output-root",
            str(args.output_root),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--download" if args.download else "--no-download",
        ]
        for flag, value in (
            ("--resume", args.resume),
            ("--smoke-epochs", args.smoke_epochs),
            ("--max-train-batches", args.max_train_batches),
            ("--max-validation-batches", args.max_validation_batches),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        print(f"[{index}/{len(selected)}] {' '.join(command)}")
        if args.dry_run:
            continue
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = row["seed"]
        source = str(ROOT / "src")
        environment["PYTHONPATH"] = (
            source
            if not environment.get("PYTHONPATH")
            else source + os.pathsep + environment["PYTHONPATH"]
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode:
            failures.append((row["run_id"], completed.returncode))
            if not args.continue_on_error:
                break
    if failures:
        for run_id, return_code in failures:
            print(f"FAILED {run_id}: exit {return_code}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
