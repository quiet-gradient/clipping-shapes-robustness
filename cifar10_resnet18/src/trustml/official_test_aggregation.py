"""Validate and aggregate all 24 post-lock official-test outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from trustml.official_test import (
    EXPECTED_EXAMPLES,
    FROZEN_BINDINGS,
    OFFICIAL_SAMPLE_ID_SHA256,
    array_sha256,
    resolve_local_path,
    sha256,
)


BOOTSTRAP_SEED = 20260827
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_BLOCK_SIZE = 100
MASK_NAMES = (
    "clean_correct",
    "pgd20x5_correct",
    "aa8_clean_correct",
    "aa8_after_apgd_ce",
    "aa8_after_apgd_dlr",
    "aa8_after_fab",
    "aa8_after_square",
    "aa8_attack_time_final_correct",
    "aa8_final_correct",
    "aa4_clean_correct",
    "aa4_after_apgd_ce",
    "aa4_after_apgd_dlr",
    "aa4_after_fab",
    "aa4_after_square",
    "aa4_attack_time_final_correct",
    "aa4_final_correct",
)
REQUIRED_ARRAYS = {"sample_id", "target", *MASK_NAMES}
COMPARISONS = (
    ("rq1_objective_attribution", "H-Natural", "H-Joint"),
    ("rq2_strategy_selected", "H-C", "H-Joint"),
    ("rq2_query_matched", "H-C", "H-JointMatched"),
    ("rq3_objective_attribution", "S-Natural", "S-Joint"),
    ("rq3_strategy_selected", "S-C", "S-Joint"),
    ("rq3_query_matched", "S-C", "S-JointMatched"),
)
BOOTSTRAP_COMPARISONS = {
    "rq1_objective_attribution",
    "rq2_query_matched",
    "rq3_query_matched",
}


def _read_manifest(path: Path) -> tuple[Path, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {(row["paper_arm"], int(row["seed"])): row["run_id"] for row in rows}
    if len(rows) != 24 or observed != FROZEN_BINDINGS:
        raise RuntimeError("official-test manifest differs from the frozen 24-row design")
    if any(row.get("selection_locked_before_official_test") != "true" for row in rows):
        raise RuntimeError("official-test aggregation requires post-lock manifest rows")
    return path.resolve().parent.parent, rows


def _validate_stage_chain(arrays: Mapping[str, np.ndarray], prefix: str, evaluation_id: str) -> None:
    previous = arrays[f"{prefix}_clean_correct"]
    for suffix in ("after_apgd_ce", "after_apgd_dlr", "after_fab", "after_square"):
        current = arrays[f"{prefix}_{suffix}"]
        if np.any(current & ~previous):
            raise RuntimeError(f"{evaluation_id}: {prefix} stage masks are not monotone")
        previous = current
    if not np.array_equal(previous, arrays[f"{prefix}_attack_time_final_correct"]):
        raise RuntimeError(
            f"{evaluation_id}: {prefix} attack-time final mask differs from post-Square"
        )


def validate_clean_mask_identity(
    arrays: Mapping[str, np.ndarray], evaluation_id: str
) -> None:
    direct = arrays["clean_correct"]
    if not np.array_equal(direct, arrays["aa8_clean_correct"]):
        raise RuntimeError(f"{evaluation_id}: direct and AA8 clean masks differ")
    if not np.array_equal(direct, arrays["aa4_clean_correct"]):
        raise RuntimeError(f"{evaluation_id}: direct and AA4 clean masks differ")
    for radius in ("aa8", "aa4"):
        if np.any(arrays[f"{radius}_final_correct"] & ~direct):
            raise RuntimeError(
                f"{evaluation_id}: {radius} final mask is not a clean-correct subset"
            )


def _load_evaluation(root: Path, row: Mapping[str, str]) -> dict[str, Any]:
    evaluation_id = row["evaluation_id"]
    report_path = resolve_local_path(root, row["output_json"])
    compact_path = resolve_local_path(root, row["output_npz"])
    if not report_path.is_file() or not compact_path.is_file():
        raise RuntimeError(f"{evaluation_id}: JSON and compact outputs are both required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "schema_version": 1,
        "report_kind": "post_lock_official_test",
        "intended_use": "confirmation_only_not_model_selection",
        "evaluation_id": evaluation_id,
        "rq": row["rq"],
        "paper_arm": row["paper_arm"],
        "training_seed": int(row["seed"]),
        "run_id": row["run_id"],
        "sample_count": EXPECTED_EXAMPLES,
        "sample_id_sha256": OFFICIAL_SAMPLE_ID_SHA256,
        "config_sha256": row["config_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "compact_output_path": row["output_npz"],
    }
    for key, value in expected_report.items():
        if report.get(key) != value:
            raise RuntimeError(f"{evaluation_id}: report field {key!r} is invalid")
    if report.get("compact_output_sha256") != sha256(compact_path):
        raise RuntimeError(f"{evaluation_id}: compact output file hash is stale")
    invariants = report.get("invariants", {})
    required_invariants = (
        "post_lock_acknowledged",
        "not_authorized_for_tuning",
        "ordered_test_ids_0_to_9999",
        "sample_order_sha256_valid",
        "model_eval",
        "fp32",
        "logits_not_probabilities",
        "single_normalization_path",
        "clean_correctness_equal_across_evaluators",
        "finite_logits_and_adversarial_inputs",
        "pgd_per_example_worst_case",
        "pgd_linf_bound_valid",
        "autoattack_linf_bounds_valid",
        "autoattack_stage_monotonicity_valid",
        "autoattack_headline_from_posthoc_fp32_logits",
        "compact_output_complete",
    )
    if not all(invariants.get(key) is True for key in required_invariants):
        raise RuntimeError(f"{evaluation_id}: report contains a failed required invariant")
    attack_outputs = {}
    for radius in ("aa8", "aa4"):
        output = report.get("diagnostics", {}).get(radius, {}).get("attack_output", {})
        if output.get("headline_source") != "posthoc_fp32_logits":
            raise RuntimeError(f"{evaluation_id}: {radius} headline is not post-hoc FP32")
        attack_outputs[radius] = output

    with np.load(compact_path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise RuntimeError(f"{evaluation_id}: compact array inventory is invalid")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    expected_hashes = report.get("per_array_sha256", {})
    for name, value in arrays.items():
        if value.shape != (EXPECTED_EXAMPLES,):
            raise RuntimeError(f"{evaluation_id}: compact array {name!r} has an invalid shape")
        if expected_hashes.get(name) != array_sha256(value):
            raise RuntimeError(f"{evaluation_id}: compact array {name!r} hash is stale")
    if not np.array_equal(arrays["sample_id"], np.arange(EXPECTED_EXAMPLES, dtype=np.int32)):
        raise RuntimeError(f"{evaluation_id}: sample IDs are not ordered 0 through 9999")
    if arrays["target"].dtype != np.uint8 or np.any(arrays["target"] > 9):
        raise RuntimeError(f"{evaluation_id}: target array is invalid")
    if report.get("target_sha256") != expected_hashes.get("target"):
        raise RuntimeError(f"{evaluation_id}: target hash is stale")
    for name in MASK_NAMES:
        if arrays[name].dtype != np.bool_:
            raise RuntimeError(f"{evaluation_id}: mask {name!r} is not boolean")
    _validate_stage_chain(arrays, "aa8", evaluation_id)
    _validate_stage_chain(arrays, "aa4", evaluation_id)
    validate_clean_mask_identity(arrays, evaluation_id)

    counts = {
        "clean_correct": int(arrays["clean_correct"].sum()),
        "pgd20x5_correct": int(arrays["pgd20x5_correct"].sum()),
        "aa8_correct": int(arrays["aa8_final_correct"].sum()),
        "aa4_correct": int(arrays["aa4_final_correct"].sum()),
    }
    if report.get("observed_counts") != counts:
        raise RuntimeError(f"{evaluation_id}: headline counts differ from compact masks")
    disagreement_diagnostics = []
    for radius, count_key in (("aa8", "aa8_correct"), ("aa4", "aa4_correct")):
        diagnostic = attack_outputs[radius]
        posthoc_correct = int(diagnostic.get("posthoc_adv_correct", -1))
        headline_correct = int(diagnostic.get("headline_adv_correct", -1))
        if posthoc_correct != counts[count_key] or headline_correct != counts[count_key]:
            raise RuntimeError(
                f"{evaluation_id}: {radius} post-hoc diagnostic differs from headline mask"
            )
        correctness = int(
            diagnostic.get("attack_time_vs_posthoc_correctness_disagreement_count", -1)
        )
        predictions = int(
            diagnostic.get("attack_time_vs_posthoc_prediction_disagreement_count", -1)
        )
        if correctness < 0 or predictions < 0:
            raise RuntimeError(f"{evaluation_id}: {radius} disagreement diagnostics are incomplete")
        if correctness or predictions:
            disagreement_diagnostics.append(
                {
                    "evaluation_id": evaluation_id,
                    "radius": radius,
                    "correctness_disagreements": correctness,
                    "prediction_disagreements": predictions,
                    "attack_time_correct": int(diagnostic["attack_time_adv_correct"]),
                    "posthoc_correct": posthoc_correct,
                }
            )
    reference_counts = {
        "clean_correct": int(row["reference_clean_correct"]),
        "pgd20x5_correct": int(row["reference_pgd20x5_correct"]),
        "aa8_correct": int(row["reference_aa8_correct"]),
        "aa4_correct": int(row["reference_aa4_correct"]),
    }
    return {
        "evaluation_id": evaluation_id,
        "rq": row["rq"],
        "arm": row["paper_arm"],
        "seed": int(row["seed"]),
        "run_id": row["run_id"],
        "counts": counts,
        "reference_counts": reference_counts,
        "matches_reference": counts == reference_counts,
        "target_sha256": expected_hashes["target"],
        "json_sha256": sha256(report_path),
        "npz_sha256": sha256(compact_path),
        "disagreement_diagnostics": disagreement_diagnostics,
        "masks": {
            "clean": arrays["clean_correct"],
            "pgd20x5": arrays["pgd20x5_correct"],
            "aa8": arrays["aa8_final_correct"],
            "aa4": arrays["aa4_final_correct"],
        },
    }


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if left.shape != (EXPECTED_EXAMPLES,) or right.shape != (EXPECTED_EXAMPLES,):
        raise RuntimeError("paired bootstrap requires two ordered 10,000-example masks")
    difference = right.astype(np.int8) - left.astype(np.int8)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, BOOTSTRAP_BLOCK_SIZE):
        stop = min(start + BOOTSTRAP_BLOCK_SIZE, resamples)
        indices = rng.integers(
            0,
            EXPECTED_EXAMPLES,
            size=(stop - start, EXPECTED_EXAMPLES),
        )
        draws[start:stop] = difference[indices].mean(axis=1)
    interval = np.quantile(draws, [0.025, 0.975])
    return {
        "difference": float(difference.mean()),
        "ci95": [float(interval[0]), float(interval[1])],
    }


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def _comparison(
    by_arm_seed: Mapping[tuple[str, int], dict[str, Any]],
    name: str,
    left: str,
    right: str,
) -> dict[str, Any]:
    metric_fields = {
        "clean": "clean_correct",
        "pgd": "pgd20x5_correct",
        "aa8": "aa8_correct",
        "aa4": "aa4_correct",
    }
    output: dict[str, Any] = {"name": name, "left": left, "right": right}
    for label, field in metric_fields.items():
        differences = [
            (
                by_arm_seed[(right, seed)]["counts"][field]
                - by_arm_seed[(left, seed)]["counts"][field]
            )
            / 100.0
            for seed in (42, 43, 44)
        ]
        mean, sample_sd = _mean_sd(differences)
        output[f"delta_{label}_mean"] = round(mean, 4)
        output[f"delta_{label}_sd"] = round(sample_sd, 4)
    return output


def _paired_aa8(
    by_arm_seed: Mapping[tuple[str, int], dict[str, Any]],
    name: str,
    left: str,
    right: str,
) -> dict[str, Any]:
    rows = []
    for seed in (42, 43, 44):
        result = paired_bootstrap(
            by_arm_seed[(left, seed)]["masks"]["aa8"],
            by_arm_seed[(right, seed)]["masks"]["aa8"],
        )
        rows.append(
            {
                "seed": seed,
                "difference": round(100.0 * result["difference"], 2),
                "ci95": [round(100.0 * value, 2) for value in result["ci95"]],
            }
        )
    return {"name": name, "left": left, "right": right, "per_seed": rows}


def _core_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm_seed = {(item["arm"], item["seed"]): item for item in evaluations}
    comparisons = [
        _comparison(by_arm_seed, name, left, right)
        for name, left, right in COMPARISONS
    ]
    comparison_by_name = {item["name"]: item for item in comparisons}
    hard = comparison_by_name["rq2_query_matched"]
    soft = comparison_by_name["rq3_query_matched"]
    hard_per_seed = [
        (
            by_arm_seed[("H-JointMatched", seed)]["counts"]["aa8_correct"]
            - by_arm_seed[("H-C", seed)]["counts"]["aa8_correct"]
        )
        / 100.0
        for seed in (42, 43, 44)
    ]
    soft_per_seed = [
        (
            by_arm_seed[("S-JointMatched", seed)]["counts"]["aa8_correct"]
            - by_arm_seed[("S-C", seed)]["counts"]["aa8_correct"]
        )
        / 100.0
        for seed in (42, 43, 44)
    ]
    interaction = [soft - hard for soft, hard in zip(soft_per_seed, hard_per_seed, strict=True)]
    interaction_mean, interaction_sd = _mean_sd(interaction)
    bootstrap = [
        _paired_aa8(by_arm_seed, name, left, right)
        for name, left, right in COMPARISONS
        if name in BOOTSTRAP_COMPARISONS
    ]
    evaluation_rows = [
        {
            "rq": item["rq"],
            "arm": item["arm"],
            "seed": item["seed"],
            "clean": item["counts"]["clean_correct"] / 100.0,
            "pgd20x5": item["counts"]["pgd20x5_correct"] / 100.0,
            "aa8": item["counts"]["aa8_correct"] / 100.0,
            "aa4": item["counts"]["aa4_correct"] / 100.0,
        }
        for item in evaluations
    ]
    return {
        "evaluations_percent": evaluation_rows,
        "comparisons_percent": comparisons,
        "cross_map_percent": {
            "hard_exact_gap_mean": hard["delta_aa8_mean"],
            "hard_exact_gap_sd": hard["delta_aa8_sd"],
            "soft_exact_gap_mean": soft["delta_aa8_mean"],
            "soft_exact_gap_sd": soft["delta_aa8_sd"],
            "soft_minus_hard_mean": round(interaction_mean, 4),
            "soft_minus_hard_sd": round(interaction_sd, 4),
        },
        "paired_aa8_bootstrap_percent": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "comparisons": bootstrap,
        },
    }


def _validate_reference(core: Mapping[str, Any], reference_path: Path) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    for key in (
        "evaluations_percent",
        "comparisons_percent",
        "cross_map_percent",
        "paired_aa8_bootstrap_percent",
    ):
        if core[key] != reference.get(key):
            raise RuntimeError(f"aggregated section {key!r} differs from the released report")


def aggregate(
    *,
    manifest_path: Path,
    output_path: Path,
    reference_path: Path | None = None,
    require_paper_counts: bool = True,
) -> dict[str, Any]:
    root, rows = _read_manifest(manifest_path)
    output_path = output_path.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing aggregate: {output_path}")
    evaluations = [_load_evaluation(root, row) for row in rows]
    if len({item["target_sha256"] for item in evaluations}) != 1:
        raise RuntimeError("official-test target identities differ across evaluations")
    reference_mismatches = [
        item["evaluation_id"] for item in evaluations if not item["matches_reference"]
    ]
    if require_paper_counts and reference_mismatches:
        raise RuntimeError(
            "official outputs differ from paper reference counts: "
            + ", ".join(reference_mismatches)
        )
    core = _core_summary(evaluations)
    if reference_path is not None:
        _validate_reference(core, reference_path)
    output = {
        "schema_version": 1,
        "status": "COMPLETE",
        "selection_lock": "closed_before_official_test",
        "protocol": {
            "dataset": "CIFAR-10 official test",
            "examples": EXPECTED_EXAMPLES,
            "pgd": "20 steps, 5 restarts, per-example worst case, Linf 8/255",
            "autoattack": "standard, Linf, radii 8/255 and 4/255",
            "attack_seed": 42,
            "precision": "FP32",
            "model_mode": "eval",
            "autoattack_headline": "post-hoc FP32 logits on final adversarial examples",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        **core,
        "validation": {
            "complete_model_evaluations": len(evaluations),
            "common_target_sha256": evaluations[0]["target_sha256"],
            "reference_count_mismatches": reference_mismatches,
            "released_report_match_checked": reference_path is not None,
            "attack_time_posthoc_diagnostics": [
                diagnostic
                for item in evaluations
                for diagnostic in item["disagreement_diagnostics"]
            ],
        },
        "source_outputs": [
            {
                "evaluation_id": item["evaluation_id"],
                "run_id": item["run_id"],
                "json_sha256": item["json_sha256"],
                "npz_sha256": item["npz_sha256"],
            }
            for item in evaluations
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    source_root = Path(__file__).resolve().parents[2]
    working_root = Path.cwd().resolve()
    package_root = (
        working_root
        if (working_root / "manifests/official_test.csv").is_file()
        else source_root
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=package_root / "manifests/official_test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "results/official_test/aggregated_results.json",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=package_root / "reports/official_test_results.json",
    )
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="aggregate an independent rerun without requiring exact paper-reference values",
    )
    args = parser.parse_args(argv)
    try:
        output = aggregate(
            manifest_path=args.manifest,
            output_path=args.output,
            reference_path=None if args.skip_reference_check else args.reference_report,
            require_paper_counts=not args.skip_reference_check,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"validated={output['validation']['complete_model_evaluations']} "
        f"comparisons={len(output['comparisons_percent'])} "
        f"paired_aa8_intervals="
        f"{sum(len(item['per_seed']) for item in output['paired_aa8_bootstrap_percent']['comparisons'])}"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
