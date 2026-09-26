#!/usr/bin/env python3
"""List or run the exact training configurations in the experiment manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/experiments.csv"
REQUIRED_FINAL_METRICS = ("dev/clean_acc", "attack/pgd_adv_acc", "dp/epsilon")
INITIALIZATION_SHA256 = "53f358fe8a0e7b87e0dbbf8b0814d885e6bb82ba5452baa22521a58bf2d68909"
_VERIFIED_INITIALIZATIONS: set[Path] = set()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(row: dict[str, str]) -> Path:
    path = ROOT / row["config_path"]
    if not path.is_file():
        raise RuntimeError("configuration file is missing")
    observed = _sha256(path)
    if observed != row["config_sha256"]:
        raise RuntimeError(f"configuration hash mismatch: {observed}")
    return path


def _validate_initialization(row: dict[str, str]) -> Path:
    if row["source_checkpoint"] != "artifacts/public_pretrain/checkpoint_epoch_170.pt":
        raise RuntimeError("manifest source checkpoint path is not canonical")
    path = ROOT / row["source_checkpoint"]
    if path in _VERIFIED_INITIALIZATIONS:
        return path
    if not path.is_file() or _sha256(path) != INITIALIZATION_SHA256:
        raise RuntimeError("initialization checkpoint is missing or has the wrong SHA-256")
    _VERIFIED_INITIALIZATIONS.add(path)
    return path


def _validated_checkpoint(path: Path, row: dict[str, str]) -> dict[str, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise RuntimeError("final checkpoint has no model state")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("final checkpoint has no embedded configuration")
    if str(config.get("experiment", {}).get("name")) != row["run_id"]:
        raise RuntimeError("final checkpoint run ID differs from the manifest")
    if int(config.get("experiment", {}).get("seed", -1)) != int(row["seed"]):
        raise RuntimeError("final checkpoint seed differs from the manifest")
    state = payload.get("trainer_state")
    if not isinstance(state, dict):
        raise RuntimeError("final checkpoint has no trainer state")
    if int(payload.get("epoch", -1)) != 50 or int(state.get("current_epoch", -1)) != 50:
        raise RuntimeError("final checkpoint is not the locked epoch-50 iterate")
    if int(state.get("optimizer_step", -1)) != 4_400:
        raise RuntimeError("final checkpoint does not contain 4,400 optimizer steps")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("final checkpoint has no metrics")
    validated = {}
    for key in REQUIRED_FINAL_METRICS:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"final checkpoint is missing finite metric {key!r}")
        validated[key] = float(value)
    for key in ("dev/clean_acc", "attack/pgd_adv_acc"):
        if not 0.0 <= validated[key] <= 1.0:
            raise RuntimeError(f"final checkpoint metric {key!r} is outside [0, 1]")
    if validated["dp/epsilon"] <= 0.0:
        raise RuntimeError("final checkpoint epsilon must be positive")
    return validated


def _completion_path(run_id: str) -> Path:
    return ROOT / "outputs" / run_id / "completion.json"


def _write_completion(
    *, row: dict[str, str], checkpoint: Path, metrics: dict[str, float]
) -> None:
    receipt = {
        "status": "COMPLETE",
        "run_id": row["run_id"],
        "config_sha256": row["config_sha256"],
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "required_metrics": metrics,
    }
    path = _completion_path(row["run_id"])
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_completion(
    row: dict[str, str], checkpoint: Path
) -> dict[str, float]:
    _validate_config(row)
    _validate_initialization(row)
    receipt_path = _completion_path(row["run_id"])
    if not receipt_path.is_file():
        raise RuntimeError("completion receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETE" or receipt.get("run_id") != row["run_id"]:
        raise RuntimeError("completion receipt identity is invalid")
    if receipt.get("config_sha256") != row["config_sha256"]:
        raise RuntimeError("completion receipt config hash is stale")
    if receipt.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("completion receipt checkpoint hash is stale")
    metrics = _validated_checkpoint(checkpoint, row)
    if receipt.get("required_metrics") != metrics:
        raise RuntimeError("completion receipt metrics differ from the checkpoint")
    return metrics


def _rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select(rows: list[dict[str, str]], args) -> list[dict[str, str]]:
    selected = rows
    if args.run_id:
        requested = set(args.run_id)
        selected = [row for row in selected if row["run_id"] in requested]
        missing = sorted(requested - {row["run_id"] for row in selected})
        if missing:
            raise SystemExit(f"unknown run IDs: {', '.join(missing)}")
    if args.phase:
        selected = [row for row in selected if row["phase"] in set(args.phase)]
    if not (args.all or args.run_id or args.phase or args.list):
        raise SystemExit("select --all, --phase, --run-id, or --list")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--phase", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    selected = _select(_rows(), args)

    if args.list:
        for row in selected:
            print(
                f"{row['run_id']:<34} {row['phase']:<36} "
                f"AA={row['dev_autoattack']} probe={row['geometry_probe']}"
            )
        print(f"selected={len(selected)}")
        return

    failures = []
    for index, row in enumerate(selected, start=1):
        run_id = row["run_id"]
        try:
            _validate_config(row)
            _validate_initialization(row)
        except Exception as exc:
            raise SystemExit(f"invalid manifest configuration for {run_id}: {exc}") from exc
        final_checkpoint = ROOT / "outputs" / run_id / "checkpoints/final.pt"
        if final_checkpoint.exists():
            if args.skip_existing:
                try:
                    _validate_completion(row, final_checkpoint)
                except Exception as exc:
                    raise SystemExit(
                        f"refusing to skip incomplete or stale run {run_id}: {exc}"
                    ) from exc
                else:
                    print(f"[{index}/{len(selected)}] skip verified {run_id}")
                    continue
            raise SystemExit(
                f"refusing to overwrite {final_checkpoint.relative_to(ROOT)}; "
                "move the existing run or pass --skip-existing"
            )
        command = [
            sys.executable,
            str(ROOT / "scripts/train_private_finetune.py"),
            "--config",
            row["config_path"],
        ]
        print(f"[{index}/{len(selected)}] {' '.join(command)}")
        if args.dry_run:
            continue
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = row["seed"]
        source_path = str(ROOT / "src")
        environment["PYTHONPATH"] = (
            source_path
            if not environment.get("PYTHONPATH")
            else source_path + os.pathsep + environment["PYTHONPATH"]
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode:
            failures.append((run_id, completed.returncode))
            if not args.continue_on_error:
                break
        else:
            try:
                metrics = _validated_checkpoint(final_checkpoint, row)
                _write_completion(
                    row=row, checkpoint=final_checkpoint, metrics=metrics
                )
            except Exception as exc:
                print(f"FAILED {run_id}: invalid final artifact: {exc}", file=sys.stderr)
                failures.append((run_id, 2))
                if not args.continue_on_error:
                    break
    if failures:
        for run_id, returncode in failures:
            print(f"FAILED {run_id}: exit {returncode}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
