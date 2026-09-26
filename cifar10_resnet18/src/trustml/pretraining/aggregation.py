from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from trustml.pretraining.assets import INITIALIZATION_STATE_SHA256, SPLIT_INDEX_SHA256
from trustml.pretraining.config import (
    FINAL_RUNS,
    load_pretraining_config,
    scientific_config_hash,
)
from trustml.pretraining.selection import (
    binomial_standard_error,
    mark_consistency,
    select_across_betas,
    select_within_run,
)


def aggregate_pretraining_results(
    package_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    package_root = Path(package_root)
    output_root = Path(output_root)
    manifest_path = package_root / "manifests/pretraining.csv"
    rows = _manifest_rows(manifest_path)
    normalized_runs = []
    for row in rows:
        config_path = package_root / row["config_path"]
        _validate_file_hash(config_path, row["config_sha256"], "run configuration")
        _validate_file_hash(
            package_root / "configs/pretraining/base.yaml",
            row["base_config_sha256"],
            "base configuration",
        )
        cfg = load_pretraining_config(config_path)
        summary_path = output_root / row["run_id"] / "run_summary.json"
        summary = _load_json(summary_path)
        normalized_runs.append(_validate_and_normalize_run(row, cfg, summary))

    by_id = {run["run_id"]: run for run in normalized_runs}
    beta_candidates = [
        _cross_beta_candidate(by_id["trades_b3_s42"]),
        _cross_beta_candidate(by_id["trades_b6_s42"]),
    ]
    selected_beta = select_across_betas(beta_candidates, examples=1000)
    best_beta_accuracy = max(candidate["autoattack_accuracy"] for candidate in beta_candidates)
    replication_ids = ["trades_b3_s42", "trades_b3_s43", "trades_b3_s44"]
    replication_accuracies = [
        float(by_id[run_id]["selected"]["autoattack_accuracy"])
        for run_id in replication_ids
    ]
    replication_mean = statistics.mean(replication_accuracies)
    replication_sd = statistics.stdev(replication_accuracies)
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "source": "validated final run_summary.json files",
        "configuration_integrity": {
            "manifest": "manifests/pretraining.csv",
            "released_base_config_sha256": rows[0]["base_config_sha256"],
            "released_config_sha256": {
                row["run_id"]: row["config_sha256"] for row in rows
            },
            "released_scientific_config_sha256": {
                run["run_id"]: run["released_scientific_config_sha256"]
                for run in normalized_runs
            },
            "historical_source_hashes_are_not_release_hashes": True,
        },
        "runs": normalized_runs,
        "cross_beta_selection": {
            "candidates": beta_candidates,
            "best_autoattack_accuracy": best_beta_accuracy,
            "binomial_standard_error": binomial_standard_error(best_beta_accuracy, 1000),
            "selected": selected_beta,
        },
        "replication_summary": {
            "included_runs": replication_ids,
            "selected_autoattack_accuracies": replication_accuracies,
            "autoattack_mean": replication_mean,
            "autoattack_sample_sd": replication_sd,
            "autoattack_mean_percent": 100.0 * replication_mean,
            "autoattack_sample_sd_percentage_points": 100.0 * replication_sd,
        },
    }


def compare_with_paper_report(
    aggregate: dict[str, Any],
    report_path: str | Path,
) -> None:
    report = _load_json(Path(report_path))
    problems: list[str] = []
    aggregate_runs = {run["run_id"]: run for run in aggregate["runs"]}
    report_runs = {run["run_id"]: run for run in report.get("runs", [])}
    if set(report_runs) != set(FINAL_RUNS):
        problems.append("paper report does not contain the exact five final runs")
    for run_id in sorted(set(FINAL_RUNS) & set(report_runs)):
        actual = aggregate_runs[run_id]
        expected = report_runs[run_id]
        expected_objective = "cross_entropy" if actual["objective"] == "ce" else "trades"
        _equal(problems, f"{run_id}.objective", expected.get("objective"), expected_objective)
        _equal(problems, f"{run_id}.seed", expected.get("seed"), actual["seed"])
        _close(problems, f"{run_id}.beta", expected.get("beta"), actual["beta"])
        if actual["objective"] == "ce":
            _equal(
                problems,
                f"{run_id}.best_clean_epoch",
                expected.get("best_clean_epoch"),
                actual["best_clean_epoch"],
            )
            _close(
                problems,
                f"{run_id}.best_clean_percent",
                expected.get("best_clean_percent"),
                100.0 * actual["best_clean_accuracy"],
            )
            if expected.get("autoattack_candidates") != []:
                problems.append(f"{run_id} must have no AutoAttack candidates")
            continue
        _equal(
            problems,
            f"{run_id}.selected_epoch",
            expected.get("selected_epoch"),
            actual["selected"]["epoch"],
        )
        expected_candidates = {
            int(candidate["epoch"]): candidate
            for candidate in expected.get("autoattack_candidates", [])
        }
        actual_candidates = {
            int(candidate["epoch"]): candidate for candidate in actual["autoattack_candidates"]
        }
        if set(expected_candidates) != set(actual_candidates):
            problems.append(f"{run_id} candidate epoch set differs")
            continue
        for epoch, candidate in actual_candidates.items():
            expected_candidate = expected_candidates[epoch]
            for report_key, aggregate_key in (
                ("clean_percent", "clean_accuracy"),
                ("pgd10_percent", "pgd10_accuracy"),
                ("autoattack_percent", "autoattack_accuracy"),
                ("square_percent", "square_accuracy"),
            ):
                _close(
                    problems,
                    f"{run_id}.epoch{epoch}.{report_key}",
                    expected_candidate.get(report_key),
                    100.0 * float(candidate[aggregate_key]),
                )
            _equal(
                problems,
                f"{run_id}.epoch{epoch}.unstable",
                expected_candidate.get("unstable"),
                candidate["unstable"],
            )
            _equal(
                problems,
                f"{run_id}.epoch{epoch}.selected",
                expected_candidate.get("selected"),
                epoch == int(actual["selected"]["epoch"]),
            )

    cross_beta = aggregate["cross_beta_selection"]
    report_rule = report.get("selection_rule", {})
    _close(
        problems,
        "selection_rule.seed42_beta3_minus_beta6_autoattack_points",
        report_rule.get("seed42_beta3_minus_beta6_autoattack_points"),
        100.0
        * (
            aggregate_runs["trades_b3_s42"]["selected"]["autoattack_accuracy"]
            - aggregate_runs["trades_b6_s42"]["selected"]["autoattack_accuracy"]
        ),
    )
    _close(
        problems,
        "selection_rule.seed42_one_standard_error_points",
        report_rule.get("seed42_one_standard_error_points"),
        100.0 * cross_beta["binomial_standard_error"],
        tolerance=5e-4,
    )
    _equal(
        problems,
        "cross_beta selected run",
        cross_beta["selected"]["run_id"],
        "trades_b3_s42",
    )
    report_replication = report.get("replication_summary_percent", {})
    aggregate_replication = aggregate["replication_summary"]
    _equal(
        problems,
        "replication included runs",
        report_replication.get("included_runs"),
        aggregate_replication["included_runs"],
    )
    _close(
        problems,
        "replication AutoAttack mean",
        report_replication.get("autoattack_mean"),
        aggregate_replication["autoattack_mean_percent"],
    )
    _close(
        problems,
        "replication AutoAttack sample SD",
        report_replication.get("autoattack_sample_sd"),
        aggregate_replication["autoattack_sample_sd_percentage_points"],
    )
    assets = report.get("assets", {})
    _equal(
        problems,
        "split index hash",
        report.get("protocol", {}).get("split_index_sha256"),
        SPLIT_INDEX_SHA256,
    )
    _equal(
        problems,
        "initialization tensor hash",
        assets.get("common_initial_tensor_state_sha256"),
        INITIALIZATION_STATE_SHA256,
    )
    if problems:
        raise RuntimeError("paper report comparison failed:\n- " + "\n- ".join(problems))


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or {row["run_id"] for row in rows} != set(FINAL_RUNS):
        raise RuntimeError("pretraining manifest must contain the exact five final runs")
    if len({row["base_config_sha256"] for row in rows}) != 1:
        raise RuntimeError("pretraining manifest has inconsistent base configuration hashes")
    return rows


def _validate_and_normalize_run(row, cfg, summary) -> dict[str, Any]:
    run_id = row["run_id"]
    objective, beta, seed = FINAL_RUNS[run_id]
    expected_hash = scientific_config_hash(cfg)
    checks = {
        "status": (summary.get("status"), "COMPLETE"),
        "run_id": (summary.get("run_id"), run_id),
        "objective": (summary.get("objective"), objective),
        "seed": (summary.get("seed"), seed),
        "config_hash": (summary.get("config_hash"), expected_hash),
        "split_hash": (summary.get("split_hash"), SPLIT_INDEX_SHA256),
        "initialization_hash": (
            summary.get("initialization_hash"),
            INITIALIZATION_STATE_SHA256,
        ),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"{run_id} summary {name} mismatch: {actual!r} != {expected!r}")
    actual_beta = summary.get("beta")
    if (None if actual_beta is None else float(actual_beta)) != beta:
        raise RuntimeError(f"{run_id} summary beta mismatch")
    normalized: dict[str, Any] = {
        "run_id": run_id,
        "objective": objective,
        "beta": beta,
        "seed": seed,
        "released_config_sha256": row["config_sha256"],
        "released_scientific_config_sha256": expected_hash,
        "best_clean_epoch": int(summary["best_clean_epoch"]),
        "best_clean_accuracy": _accuracy(summary["best_clean_accuracy"], run_id),
        "autoattack_candidates": [],
        "selected": None,
    }
    candidates = summary.get("autoattack_candidates", [])
    if objective == "ce":
        if candidates:
            raise RuntimeError("cross-entropy reference must not contain AutoAttack candidates")
        if summary.get("selected_checkpoint") != "checkpoints/checkpoint_best_clean.pt":
            raise RuntimeError("cross-entropy reference selected checkpoint is invalid")
        return normalized
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise RuntimeError(f"{run_id} must contain exactly three AutoAttack candidates")
    epochs: set[int] = set()
    for raw in candidates:
        required = {
            "epoch",
            "clean_accuracy",
            "pgd10_accuracy",
            "pgd10_loss",
            "autoattack_accuracy",
            "autoattack_components",
            "autoattack_report",
            "unstable",
            "checkpoint",
        }
        missing = required - set(raw)
        if missing:
            raise RuntimeError(f"{run_id} has an incomplete AutoAttack candidate: {sorted(missing)}")
        epoch = int(raw["epoch"])
        if epoch in epochs:
            raise RuntimeError(f"{run_id} repeats AutoAttack candidate epoch {epoch}")
        epochs.add(epoch)
        components = raw.get("autoattack_components")
        if not isinstance(components, dict) or set(components) != {
            "apgd-ce",
            "apgd-dlr",
            "fab-t",
            "square",
        }:
            raise RuntimeError(f"{run_id} candidate epoch {epoch} has an invalid component set")
        normalized_components = {
            name: _accuracy(value, run_id) for name, value in components.items()
        }
        square = normalized_components["square"]
        pgd10_loss = float(raw["pgd10_loss"])
        if not math.isfinite(pgd10_loss) or pgd10_loss < 0.0:
            raise RuntimeError(f"{run_id} candidate epoch {epoch} has invalid PGD-10 loss")
        expected_checkpoint = f"checkpoint_pgd10_epoch_{epoch:03d}.pt"
        if str(raw["checkpoint"]) != expected_checkpoint:
            raise RuntimeError(f"{run_id} candidate epoch {epoch} checkpoint identity mismatch")
        expected_report = f"autoattack/epoch_{epoch:03d}/report.json"
        if str(raw["autoattack_report"]) != expected_report:
            raise RuntimeError(f"{run_id} candidate epoch {epoch} report identity mismatch")
        candidate = {
            "epoch": epoch,
            "clean_accuracy": _accuracy(raw["clean_accuracy"], run_id),
            "pgd10_accuracy": _accuracy(raw["pgd10_accuracy"], run_id),
            "autoattack_accuracy": _accuracy(raw["autoattack_accuracy"], run_id),
            "square_accuracy": _accuracy(square, run_id),
            "pgd10_loss": pgd10_loss,
            "autoattack_components": normalized_components,
            "autoattack_report": expected_report,
            "checkpoint": str(raw["checkpoint"]),
        }
        checked = mark_consistency(candidate, threshold=0.08)
        if bool(raw.get("unstable")) != bool(checked["unstable"]):
            raise RuntimeError(f"{run_id} candidate epoch {epoch} has a stale stability flag")
        normalized["autoattack_candidates"].append(checked)
    selected = select_within_run(normalized["autoattack_candidates"], examples=1000)
    expected_epoch = int(summary["selected_autoattack_epoch"])
    if int(selected["epoch"]) != expected_epoch:
        raise RuntimeError(f"{run_id} within-run checkpoint selection mismatch")
    if summary.get("selected_checkpoint") != "checkpoints/checkpoint_selected_autoattack.pt":
        raise RuntimeError(f"{run_id} selected checkpoint identity mismatch")
    for key in ("autoattack_accuracy", "clean_accuracy", "pgd10_accuracy"):
        summary_key = f"selected_{key}"
        if summary_key in summary and not math.isclose(
            float(summary[summary_key]), float(selected[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"{run_id} {summary_key} mismatch")
    normalized["selected"] = selected
    return normalized


def _cross_beta_candidate(run: dict[str, Any]) -> dict[str, Any]:
    selected = run["selected"]
    return {
        "run_id": run["run_id"],
        "beta": float(run["beta"]),
        "epoch": int(selected["epoch"]),
        "clean_accuracy": float(selected["clean_accuracy"]),
        "pgd10_accuracy": float(selected["pgd10_accuracy"]),
        "autoattack_accuracy": float(selected["autoattack_accuracy"]),
    }


def _validate_file_hash(path: Path, expected: str, label: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: {path.name}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def _accuracy(value: Any, run_id: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"{run_id} contains an invalid accuracy")
    return number


def _equal(problems: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        problems.append(f"{label}: {actual!r} != {expected!r}")


def _close(
    problems: list[str],
    label: str,
    actual: Any,
    expected: Any,
    *,
    tolerance: float = 1e-10,
) -> None:
    if actual is None or expected is None:
        if actual != expected:
            problems.append(f"{label}: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        problems.append(f"{label}: {actual!r} != {expected!r}")
