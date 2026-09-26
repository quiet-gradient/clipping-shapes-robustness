#!/usr/bin/env python3
"""List, dry-run, or execute the 24 predeclared official-test evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/official_test.csv"
EXPERIMENTS = ROOT / "manifests/experiments.csv"
EXPECTED_EXAMPLES = 10_000
OFFICIAL_BATCH_SIZE = 128
SAMPLE_ID_SHA256 = "a74675bfed3834ac1c3140b2d7fb7d93d90a76608cc08515a5019a1cb2075ff4"
REQUIRED_ARRAYS = {
    "sample_id",
    "target",
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
}
EXPECTED_BINDINGS = {
    ("H-Natural", 42): "hce_c030_lr0050_s42",
    ("H-Natural", 43): "hce_locked_s43",
    ("H-Natural", 44): "hce_locked_s44",
    ("H-Joint", 42): "ha_b1_c030_lr0050_s42",
    ("H-Joint", 43): "ha_locked_s43",
    ("H-Joint", 44): "ha_locked_s44",
    ("H-C", 42): "hc_q1of3_c100_lr0025_s42",
    ("H-C", 43): "hc_locked_s43",
    ("H-C", 44): "hc_locked_s44",
    ("H-JointMatched", 42): "ha_b1_c100_lr0025_s42",
    ("H-JointMatched", 43): "ha_jointmatched_s43",
    ("H-JointMatched", 44): "ha_jointmatched_s44",
    ("S-Natural", 42): "sce_g1000_lr050_s42",
    ("S-Natural", 43): "sce_g1000_lr050_s43",
    ("S-Natural", 44): "sce_g1000_lr050_s44",
    ("S-Joint", 42): "sa_lr100_s42",
    ("S-Joint", 43): "sa_g0010_lr100_s43",
    ("S-Joint", 44): "sa_g0010_lr100_s44",
    ("S-C", 42): "sc_g0100_q025_lr100_s42",
    ("S-C", 43): "sc_locked_s43",
    ("S-C", 44): "sc_locked_s44",
    ("S-JointMatched", 42): "sa_g0100_lr100_s42",
    ("S-JointMatched", 43): "sa_g0100_lr100_s43",
    ("S-JointMatched", 44): "sa_g0100_lr100_s44",
}
RQ_BY_ARM = {
    "H-Natural": "RQ1",
    "H-Joint": "RQ1",
    "H-C": "RQ2",
    "H-JointMatched": "RQ2",
    "S-Natural": "RQ3",
    "S-Joint": "RQ3",
    "S-C": "RQ3",
    "S-JointMatched": "RQ3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _resolve_local(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"manifest path must be package-relative: {relative!r}")
    resolved = (ROOT / candidate).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"manifest path escapes the package: {relative!r}")
    return resolved


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validate_manifest() -> list[dict[str, str]]:
    rows = _read_csv(MANIFEST)
    experiments = {row["run_id"]: row for row in _read_csv(EXPERIMENTS)}
    if len(rows) != 24 or len(EXPECTED_BINDINGS) != 24:
        raise RuntimeError("official-test manifest must contain exactly 24 rows")
    if len({row["evaluation_id"] for row in rows}) != 24:
        raise RuntimeError("official-test evaluation IDs are not unique")
    if len({row["output_json"] for row in rows}) != 24:
        raise RuntimeError("official-test JSON output paths are not unique")
    if len({row["output_npz"] for row in rows}) != 24:
        raise RuntimeError("official-test compact output paths are not unique")

    observed_bindings: dict[tuple[str, int], str] = {}
    for row in rows:
        seed = int(row["seed"])
        key = (row["paper_arm"], seed)
        observed_bindings[key] = row["run_id"]
        if seed not in {42, 43, 44}:
            raise RuntimeError(f"{row['evaluation_id']}: unexpected training seed")
        if row["rq"] != RQ_BY_ARM.get(row["paper_arm"]):
            raise RuntimeError(f"{row['evaluation_id']}: paper arm/RQ binding is invalid")
        if row["selection_locked_before_official_test"] != "true":
            raise RuntimeError(f"{row['evaluation_id']}: post-lock declaration is missing")
        expected_json = f"results/official_test/{row['evaluation_id']}.json"
        expected_npz = f"results/official_test/{row['evaluation_id']}.npz"
        expected_checkpoint = f"outputs/{row['run_id']}/checkpoints/final.pt"
        if row["output_json"] != expected_json:
            raise RuntimeError(f"{row['evaluation_id']}: JSON output path is not canonical")
        if row["output_npz"] != expected_npz:
            raise RuntimeError(f"{row['evaluation_id']}: compact output path is not canonical")
        if row["checkpoint_path"] != expected_checkpoint:
            raise RuntimeError(f"{row['evaluation_id']}: checkpoint output path is not canonical")
        checkpoint_hash = row.get("checkpoint_sha256", "")
        if len(checkpoint_hash) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_hash
        ):
            raise RuntimeError(f"{row['evaluation_id']}: frozen checkpoint hash is invalid")
        for field in (
            "reference_clean_correct",
            "reference_pgd20x5_correct",
            "reference_aa8_correct",
            "reference_aa4_correct",
        ):
            count = int(row[field])
            if not 0 <= count <= EXPECTED_EXAMPLES:
                raise RuntimeError(f"{row['evaluation_id']}: {field} is outside the valid range")

        experiment = experiments.get(row["run_id"])
        if experiment is None:
            raise RuntimeError(f"{row['evaluation_id']}: training run is not in experiments.csv")
        if experiment["config_path"] != row["config_path"]:
            raise RuntimeError(f"{row['evaluation_id']}: training config path differs from the run ledger")
        if experiment["config_sha256"] != row["config_sha256"]:
            raise RuntimeError(f"{row['evaluation_id']}: training config hash differs from the run ledger")
        if int(experiment["seed"]) != seed:
            raise RuntimeError(f"{row['evaluation_id']}: training seed differs from the run ledger")
        config_path = _resolve_local(row["config_path"])
        if _sha256(config_path) != row["config_sha256"]:
            raise RuntimeError(f"{row['evaluation_id']}: current training config hash is stale")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["experiment"]["name"] != row["run_id"]:
            raise RuntimeError(f"{row['evaluation_id']}: config run identity is invalid")
        if int(config["experiment"]["seed"]) != seed:
            raise RuntimeError(f"{row['evaluation_id']}: config seed is invalid")
        if config["protocol"].get("test_locked") is not True:
            raise RuntimeError(f"{row['evaluation_id']}: training config does not lock the test")

    if observed_bindings != EXPECTED_BINDINGS:
        raise RuntimeError("official-test paper-arm/run mapping differs from the frozen 24-row design")
    return rows


def _validate_stage_arrays(arrays: Mapping[str, np.ndarray], prefix: str) -> None:
    previous = arrays[f"{prefix}_clean_correct"].astype(np.bool_)
    for suffix in ("after_apgd_ce", "after_apgd_dlr", "after_fab", "after_square"):
        current = arrays[f"{prefix}_{suffix}"].astype(np.bool_)
        if np.any(current & ~previous):
            raise RuntimeError(f"existing {prefix} compact survivor masks are not monotone")
        previous = current
    if not np.array_equal(previous, arrays[f"{prefix}_attack_time_final_correct"]):
        raise RuntimeError(
            f"existing {prefix} attack-time final and post-Square masks differ"
        )


def _validate_existing(row: Mapping[str, str]) -> None:
    report_path = _resolve_local(row["output_json"])
    compact_path = _resolve_local(row["output_npz"])
    if not report_path.is_file() or not compact_path.is_file():
        raise RuntimeError("both JSON and compact outputs are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("report_kind") != "post_lock_official_test":
        raise RuntimeError("existing official-test report schema is invalid")
    if report.get("evaluation_id") != row["evaluation_id"]:
        raise RuntimeError("existing report evaluation identity differs from the manifest")
    if report.get("run_id") != row["run_id"]:
        raise RuntimeError("existing report training identity differs from the manifest")
    if report.get("sample_id_sha256") != SAMPLE_ID_SHA256:
        raise RuntimeError("existing report sample-order hash is invalid")
    if report.get("config_sha256") != row["config_sha256"]:
        raise RuntimeError("existing report config hash is stale")
    checkpoint = _resolve_local(row["checkpoint_path"])
    if not checkpoint.is_file() or _sha256(checkpoint) != row["checkpoint_sha256"]:
        raise RuntimeError("current checkpoint differs from the frozen manifest hash")
    if report.get("checkpoint_sha256") != row["checkpoint_sha256"]:
        raise RuntimeError("existing report checkpoint hash is stale")
    if report.get("compact_output_sha256") != _sha256(compact_path):
        raise RuntimeError("existing compact output file hash is stale")

    with np.load(compact_path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise RuntimeError("existing compact output has an unexpected array inventory")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name, value in arrays.items():
        if value.shape != (EXPECTED_EXAMPLES,):
            raise RuntimeError(f"existing compact array {name!r} has an invalid shape")
        expected_hash = report.get("per_array_sha256", {}).get(name)
        if expected_hash != _array_sha256(value):
            raise RuntimeError(f"existing compact array {name!r} has a stale hash")
    if not np.array_equal(arrays["sample_id"], np.arange(EXPECTED_EXAMPLES, dtype=np.int32)):
        raise RuntimeError("existing compact output has an invalid sample order")
    if np.any(arrays["target"] > 9):
        raise RuntimeError("existing compact output has invalid labels")
    for name in REQUIRED_ARRAYS - {"sample_id", "target"}:
        if arrays[name].dtype != np.bool_:
            raise RuntimeError(f"existing compact mask {name!r} is not boolean")
    _validate_stage_arrays(arrays, "aa8")
    _validate_stage_arrays(arrays, "aa4")
    if not np.array_equal(arrays["clean_correct"], arrays["aa8_clean_correct"]):
        raise RuntimeError("existing direct and AA8 clean masks differ")
    if not np.array_equal(arrays["clean_correct"], arrays["aa4_clean_correct"]):
        raise RuntimeError("existing direct and AA4 clean masks differ")
    for radius in ("aa8", "aa4"):
        if np.any(arrays[f"{radius}_final_correct"] & ~arrays["clean_correct"]):
            raise RuntimeError(f"existing {radius} final mask is not a clean-correct subset")

    observed = report.get("observed_counts", {})
    expected_observed = {
        "clean_correct": int(arrays["clean_correct"].sum()),
        "pgd20x5_correct": int(arrays["pgd20x5_correct"].sum()),
        "aa8_correct": int(arrays["aa8_final_correct"].sum()),
        "aa4_correct": int(arrays["aa4_final_correct"].sum()),
    }
    if observed != expected_observed:
        raise RuntimeError("existing report metrics differ from compact per-example outputs")
    for radius, count_key in (("aa8", "aa8_correct"), ("aa4", "aa4_correct")):
        diagnostic = report.get("diagnostics", {}).get(radius, {}).get("attack_output", {})
        if diagnostic.get("headline_source") != "posthoc_fp32_logits":
            raise RuntimeError(f"existing {radius} headline is not post-hoc FP32")
        if int(diagnostic.get("headline_adv_correct", -1)) != expected_observed[count_key]:
            raise RuntimeError(f"existing {radius} headline diagnostic count is stale")
        if int(diagnostic.get("posthoc_adv_correct", -1)) != expected_observed[count_key]:
            raise RuntimeError(f"existing {radius} post-hoc diagnostic count is stale")
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
    if not all(invariants.get(name) is True for name in required_invariants):
        raise RuntimeError("existing report contains a failed required invariant")


def _select_rows(
    rows: list[dict[str, str]],
    evaluation_ids: Sequence[str] | None,
) -> list[dict[str, str]]:
    if not evaluation_ids:
        return rows
    requested = set(evaluation_ids)
    selected = [row for row in rows if row["evaluation_id"] in requested]
    missing = sorted(requested - {row["evaluation_id"] for row in selected})
    if missing:
        raise RuntimeError(f"unknown official-test evaluation IDs: {', '.join(missing)}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-id", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--acknowledge-official-test", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(OFFICIAL_BATCH_SIZE,),
        default=OFFICIAL_BATCH_SIZE,
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    try:
        rows = _validate_manifest()
        rows = _select_rows(rows, args.evaluation_id)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not (args.all or args.evaluation_id or args.list):
        parser.error("select --all, --evaluation-id, or --list")
    if args.all and args.evaluation_id:
        parser.error("--all and --evaluation-id are mutually exclusive")
    if args.list:
        for row in rows:
            print(
                "\t".join(
                    (
                        row["evaluation_id"],
                        row["rq"],
                        row["paper_arm"],
                        row["seed"],
                        row["run_id"],
                    )
                )
            )
        print(f"selected={len(rows)}")
        return 0
    if not args.dry_run and not args.acknowledge_official_test:
        parser.error(
            "--acknowledge-official-test is required and confirms that model selection "
            "and evaluation settings were frozen before official-test access"
        )

    failures: list[tuple[str, int]] = []
    for index, row in enumerate(rows, start=1):
        output_json = _resolve_local(row["output_json"])
        output_npz = _resolve_local(row["output_npz"])
        if output_json.exists() or output_npz.exists():
            if args.skip_existing and output_json.exists() and output_npz.exists():
                try:
                    _validate_existing(row)
                except Exception as exc:
                    parser.error(
                        f"refusing to skip incomplete or stale evaluation "
                        f"{row['evaluation_id']}: {exc}"
                    )
                print(f"[{index}/{len(rows)}] skip verified {row['evaluation_id']}")
                continue
            parser.error(
                f"refusing to overwrite outputs for {row['evaluation_id']}; "
                "move both output files or pass --skip-existing"
            )

        command = [
            sys.executable,
            "-m",
            "trustml.official_test",
            "--manifest",
            str(MANIFEST.relative_to(ROOT)),
            "--evaluation-id",
            row["evaluation_id"],
            "--batch-size",
            str(args.batch_size),
            "--acknowledge-official-test",
        ]
        if args.data_root is not None:
            command.extend(("--data-root", str(args.data_root)))
        print(f"[{index}/{len(rows)}] {' '.join(command)}")
        if args.dry_run:
            continue
        checkpoint = _resolve_local(row["checkpoint_path"])
        if not checkpoint.is_file():
            parser.error(
                f"training checkpoint for {row['evaluation_id']} is missing: "
                f"{row['checkpoint_path']}"
            )
        if _sha256(checkpoint) != row["checkpoint_sha256"]:
            parser.error(
                f"training checkpoint for {row['evaluation_id']} differs from the frozen SHA-256"
            )
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "42"
        source_path = str(ROOT / "src")
        environment["PYTHONPATH"] = (
            source_path
            if not environment.get("PYTHONPATH")
            else source_path + os.pathsep + environment["PYTHONPATH"]
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode:
            failures.append((row["evaluation_id"], completed.returncode))
            if not args.continue_on_error:
                break
            continue
        try:
            _validate_existing(row)
        except Exception as exc:
            print(
                f"FAILED {row['evaluation_id']}: output validation failed: {exc}",
                file=sys.stderr,
            )
            failures.append((row["evaluation_id"], 2))
            if not args.continue_on_error:
                break
    if failures:
        for evaluation_id, returncode in failures:
            print(f"FAILED {evaluation_id}: exit {returncode}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
