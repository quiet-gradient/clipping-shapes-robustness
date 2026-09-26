#!/usr/bin/env python3
"""List or run the exact fixed-development AutoAttack evaluations."""

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

from run_experiments import _validate_completion


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/autoattack.csv"
EXPERIMENT_MANIFEST = ROOT / "manifests/experiments.csv"
SPLIT_MANIFEST = ROOT / "manifests/data_split.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_report(
    row: dict[str, str],
    *,
    experiments: dict[str, dict[str, str]],
    expected_ids: list[int],
    expected_id_hash: str,
) -> None:
    path = ROOT / row["output_path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    run_id = row["run_id"]
    if report.get("schema_version") != 1 or report.get("report_kind") != "fixed_dev_autoattack":
        raise RuntimeError("report schema or kind is invalid")
    if report.get("run_id") != run_id:
        raise RuntimeError("report run ID differs from the manifest")
    if report.get("split") != "fixed_public_dev" or report.get("official_test_used") is not False:
        raise RuntimeError("report split or official-test flag is invalid")
    if int(report.get("sample_count", -1)) != int(row["expected_examples"]):
        raise RuntimeError("report sample count differs from the manifest")
    if report.get("sample_id_sha256") != expected_id_hash:
        raise RuntimeError("report sample-order hash differs from the locked manifest")
    records = report.get("per_example")
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise RuntimeError("report has an incomplete per-example trace")
    observed_ids = [int(record["sample_id"]) for record in records]
    if observed_ids != expected_ids:
        raise RuntimeError("report per-example order differs from the locked manifest")
    if [int(record["position"]) for record in records] != list(range(len(expected_ids))):
        raise RuntimeError("report positions are not contiguous and ordered")
    mask_fields = (
        "clean_correct",
        "correct_after_apgd_ce",
        "correct_after_apgd_dlr",
        "correct_after_fab",
        "correct_after_square",
        "final_robust_correct",
    )
    for record in records:
        if any(type(record.get(field)) is not bool for field in mask_fields):
            raise RuntimeError("report per-example masks must be complete booleans")
        previous = record["clean_correct"]
        for field in mask_fields[1:5]:
            current = record[field]
            if current and not previous:
                raise RuntimeError("report survivor masks are not elementwise monotone")
            previous = current
        if record["final_robust_correct"] != record["correct_after_square"]:
            raise RuntimeError("report final and post-Square masks differ")

    experiment = experiments.get(run_id)
    if experiment is None or experiment["dev_autoattack"] != "true":
        raise RuntimeError("report run is not approved by the experiment manifest")
    config_path = ROOT / row["config_path"]
    checkpoint_path = ROOT / row["checkpoint_path"]
    if _sha256(config_path) != experiment["config_sha256"]:
        raise RuntimeError("current configuration hash differs from the experiment manifest")
    if report.get("config_sha256") != experiment["config_sha256"]:
        raise RuntimeError("report configuration hash is stale")
    if not checkpoint_path.is_file():
        raise RuntimeError("report checkpoint is missing")
    if report.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise RuntimeError("report checkpoint hash is stale")

    attack = report.get("attack", {})
    if (
        attack.get("name") != "AutoAttack"
        or attack.get("version") != "standard"
        or attack.get("package_version") != "0.1"
        or attack.get("norm") != "Linf"
        or attack.get("precision") != "FP32"
        or attack.get("official_component_order")
        != ["apgd-ce", "apgd-t", "fab-t", "square"]
        or int(attack.get("seed", -1)) != int(row["attack_seed"])
        or int(attack.get("batch_size", -1)) != int(row["batch_size"])
        or abs(float(attack.get("epsilon", -1.0)) - 8.0 / 255.0) > 1e-15
    ):
        raise RuntimeError("report attack protocol differs from the manifest")
    invariants = report.get("invariants", {})
    required_invariants = (
        "model_eval",
        "logits_not_probabilities",
        "single_normalization",
        "input_pixel_range_valid",
        "linf_budget_valid",
        "fp32",
        "fixed_dev_order",
    )
    if not all(invariants.get(key) is True for key in required_invariants):
        raise RuntimeError("report contains a failed required invariant")
    if invariants.get("fixed_dev_order_sha256") != expected_id_hash:
        raise RuntimeError("report invariant sample-order hash is stale")

    metrics = report.get("metrics", {})
    clean_metric = metrics.get("attack/autoattack_clean_acc")
    robust_metric = metrics.get("attack/autoattack_adv_acc")
    for name, metric in (
        ("clean", clean_metric),
        ("robust", robust_metric),
    ):
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            raise RuntimeError(f"report is missing finite AutoAttack {name} accuracy")
        if not 0.0 <= float(metric) <= 1.0:
            raise RuntimeError(f"report AutoAttack {name} accuracy is outside [0, 1]")
    clean_mean = sum(record["clean_correct"] for record in records) / len(records)
    robust_mean = sum(record["final_robust_correct"] for record in records) / len(records)
    if abs(float(clean_metric) - clean_mean) > 1e-15:
        raise RuntimeError("report clean metric differs from the per-example masks")
    if abs(float(robust_metric) - robust_mean) > 1e-15:
        raise RuntimeError("report robust metric differs from the per-example masks")
    component_fields = {
        "apgd_ce": "correct_after_apgd_ce",
        "apgd_dlr": "correct_after_apgd_dlr",
        "fab": "correct_after_fab",
        "square": "correct_after_square",
    }
    component_accuracy = report.get("component_robust_accuracy", {})
    for name, field in component_fields.items():
        observed = component_accuracy.get(name)
        expected = sum(record[field] for record in records) / len(records)
        if not isinstance(observed, (int, float)) or abs(float(observed) - expected) > 1e-15:
            raise RuntimeError(f"report component metric {name!r} differs from its masks")
    execution = report.get("execution", {})
    reached = execution.get("component_reached_examples", {})
    survivor_fields = (
        ("apgd-ce", "clean_correct"),
        ("apgd-t", "correct_after_apgd_ce"),
        ("fab-t", "correct_after_apgd_dlr"),
        ("square", "correct_after_fab"),
    )
    for component, prior_field in survivor_fields:
        expected = sum(record[prior_field] for record in records)
        if int(reached.get(component, -1)) != expected:
            raise RuntimeError(f"report reached count for {component!r} is inconsistent")
    max_perturbation = metrics.get("attack/autoattack_max_perturbation")
    if (
        not isinstance(max_perturbation, (int, float))
        or not math.isfinite(float(max_perturbation))
        or float(max_perturbation) > 8.0 / 255.0 + 1e-6
    ):
        raise RuntimeError("report maximum perturbation is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with EXPERIMENT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        experiments = {row["run_id"]: row for row in csv.DictReader(handle)}
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    expected_ids = [int(value) for value in split["public_dev_source_indices"]]
    expected_id_hash = str(split["public_dev_source_indices_sha256"])
    if _canonical_hash(expected_ids) != expected_id_hash:
        raise SystemExit("public-development manifest hash is internally inconsistent")
    if args.run_id:
        requested = set(args.run_id)
        rows = [row for row in rows if row["run_id"] in requested]
        missing = sorted(requested - {row["run_id"] for row in rows})
        if missing:
            raise SystemExit(f"unknown AutoAttack run IDs: {', '.join(missing)}")
    if not (args.all or args.run_id or args.list):
        raise SystemExit("select --all, --run-id, or --list")
    if args.list:
        for row in rows:
            print(row["run_id"])
        print(f"selected={len(rows)}")
        return

    failures = []
    for index, row in enumerate(rows, start=1):
        experiment = experiments.get(row["run_id"])
        if experiment is None:
            raise SystemExit(f"unknown experiment for {row['run_id']}")
        checkpoint = ROOT / row["checkpoint_path"]
        if not args.dry_run:
            try:
                _validate_completion(experiment, checkpoint)
            except Exception as exc:
                raise SystemExit(
                    f"training artifact for {row['run_id']} is incomplete or stale: {exc}"
                ) from exc
        output = ROOT / row["output_path"]
        if output.exists():
            if args.skip_existing:
                try:
                    _validate_report(
                        row,
                        experiments=experiments,
                        expected_ids=expected_ids,
                        expected_id_hash=expected_id_hash,
                    )
                except Exception as exc:
                    raise SystemExit(
                        f"refusing to skip incomplete or stale evaluation "
                        f"{row['run_id']}: {exc}"
                    ) from exc
                else:
                    print(f"[{index}/{len(rows)}] skip verified {row['run_id']}")
                    continue
            raise SystemExit(
                f"refusing to overwrite {output.relative_to(ROOT)}; "
                "move it or pass --skip-existing"
            )
        command = [
            sys.executable,
            "-m",
            "trustml.dev_autoattack",
            "--config",
            row["config_path"],
            "--checkpoint",
            row["checkpoint_path"],
            "--output",
            row["output_path"],
            "--run-id",
            row["run_id"],
            "--batch-size",
            row["batch_size"],
            "--expected-examples",
            row["expected_examples"],
            "--attack-seed",
            row["attack_seed"],
        ]
        print(f"[{index}/{len(rows)}] {' '.join(command)}")
        if args.dry_run:
            continue
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = row["attack_seed"]
        source_path = str(ROOT / "src")
        environment["PYTHONPATH"] = (
            source_path
            if not environment.get("PYTHONPATH")
            else source_path + os.pathsep + environment["PYTHONPATH"]
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode:
            failures.append((row["run_id"], completed.returncode))
            if not args.continue_on_error:
                break
        else:
            try:
                _validate_report(
                    row,
                    experiments=experiments,
                    expected_ids=expected_ids,
                    expected_id_hash=expected_id_hash,
                )
            except Exception as exc:
                print(
                    f"FAILED {row['run_id']}: invalid AutoAttack report: {exc}",
                    file=sys.stderr,
                )
                failures.append((row["run_id"], 2))
                if not args.continue_on_error:
                    break
    if failures:
        for run_id, returncode in failures:
            print(f"FAILED {run_id}: exit {returncode}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
