#!/usr/bin/env python3
"""Apply the predeclared one-standard-error rule across seed-42 TRADES betas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _candidate(output_root: Path, run_id: str, beta: float) -> dict:
    from trustml.pretraining.selection import select_within_run

    summary_path = output_root / run_id / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"run summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise RuntimeError(f"run is not complete: {run_id}")
    if summary.get("run_id") != run_id or int(summary.get("seed", -1)) != 42:
        raise RuntimeError(f"run identity mismatch: {run_id}")
    if summary.get("objective") != "trades" or float(summary.get("beta", -1)) != beta:
        raise RuntimeError(f"objective identity mismatch: {run_id}")
    candidates = summary.get("autoattack_candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise RuntimeError(f"run must contain exactly three AutoAttack candidates: {run_id}")
    selected = select_within_run(candidates, examples=1000)
    if int(selected["epoch"]) != int(summary["selected_autoattack_epoch"]):
        raise RuntimeError(f"within-run selection is inconsistent: {run_id}")
    return {
        "run_id": run_id,
        "beta": beta,
        "epoch": int(selected["epoch"]),
        "clean_accuracy": float(selected["clean_accuracy"]),
        "pgd10_accuracy": float(selected["pgd10_accuracy"]),
        "autoattack_accuracy": float(selected["autoattack_accuracy"]),
        "candidate_checkpoint": (
            Path(run_id) / "checkpoints" / selected["checkpoint"]
        ).as_posix(),
        "selected_checkpoint": (Path(run_id) / summary["selected_checkpoint"]).as_posix(),
        "summary": (Path(run_id) / "run_summary.json").as_posix(),
    }


def main() -> None:
    from trustml.pretraining.selection import binomial_standard_error, select_across_betas

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/pretraining")
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    candidates = [
        _candidate(args.output_root, "trades_b3_s42", 3.0),
        _candidate(args.output_root, "trades_b6_s42", 6.0),
    ]
    selected = select_across_betas(candidates, examples=1000)
    best_accuracy = max(candidate["autoattack_accuracy"] for candidate in candidates)
    payload = {
        "format_version": 1,
        "status": "COMPLETE",
        "selection_population": "fixed_1000_example_validation_split",
        "rule": "within one binomial standard error of best AutoAttack; then clean accuracy, beta, epoch",
        "best_autoattack_accuracy": best_accuracy,
        "binomial_standard_error": binomial_standard_error(best_accuracy, 1000),
        "candidates": candidates,
        "selected": selected,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        return
    destination = args.destination or args.output_root / "selection_across_beta.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
