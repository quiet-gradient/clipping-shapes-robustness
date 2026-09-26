"""Sealed remaining official-test suite, including secondary AA at 4/255."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from trustml.attacks.autoattack import AutoAttackAttack
from trustml.blind_official_test import (
    ATTACK_SEED,
    COMPONENTS,
    EPSILON,
    EXPECTED_EXAMPLES,
    clean_mask,
    pgd_mask,
    sha256,
    validate_trace,
)
from trustml.core.config import load_config
from trustml.core.model_loading import load_checkpoint_model
from trustml.data.registry import build_datamodule


SECONDARY_EPSILON = 4.0 / 255.0
PROFILES = {"secondary_4_only", "full_primary_secondary"}


def authorized_row(manifest: Path, eval_id: str) -> dict[str, str]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["eval_id"] == eval_id]
    if len(matches) != 1:
        raise RuntimeError(f"remaining-suite manifest does not authorize {eval_id!r} exactly once")
    if matches[0]["suite_profile"] not in PROFILES:
        raise RuntimeError(f"{eval_id}: unknown suite profile")
    return matches[0]


def run_autoattack(
    *,
    model,
    loader,
    device: torch.device,
    cfg,
    epsilon: float,
    output: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    aa = AutoAttackAttack(
        OmegaConf.create(
            {
                "epsilon": epsilon,
                "norm": "Linf",
                "version": "standard",
                "batch_size": 128,
                "seed": ATTACK_SEED,
                "expected_examples": EXPECTED_EXAMPLES,
                "require_full_test": True,
                "verbose": False,
                "log_path": str(output.with_suffix(f".{label}.autoattack.log")),
                "capture_component_trace": True,
            }
        ),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    metrics = aa.run(model, loader, device=device, progress=False)
    trace = aa.last_component_trace
    if trace is None:
        raise RuntimeError(f"{label}: AutoAttack component trace is missing")
    validate_trace(trace)
    diagnostics = aa.last_diagnostics or {}
    if not diagnostics.get("validation_passed", False):
        raise RuntimeError(f"{label}: AutoAttack normalization diagnostics failed")
    if float(metrics["attack/autoattack_max_perturbation"]) > epsilon + 1e-6:
        raise RuntimeError(f"{label}: AutoAttack exceeded Linf bound")
    return metrics, trace, {
        "normalization": diagnostics,
        "output": aa.last_output_diagnostics or {},
    }


def trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    masks = trace["correct_after_component"]
    denominator = len(trace["final_robust_correct"])
    if denominator == 0:
        raise RuntimeError("AutoAttack trace is empty")
    return {
        "accuracy": sum(bool(value) for value in trace["final_robust_correct"])
        / denominator,
        "components": {
            component: sum(bool(value) for value in masks[component]) / denominator
            for component in COMPONENTS
        },
        "component_runtimes_seconds": trace["component_durations_seconds"],
        "component_reached_examples": trace["component_reached_examples"],
    }


def evaluate(manifest: Path, eval_id: str, output: Path) -> None:
    if os.environ.get("BLIND_TEST_AUTHORIZATION") != "full_suite_predeclared_v1":
        raise RuntimeError("missing immutable remaining-suite authorization token")
    row = authorized_row(manifest, eval_id)
    config = Path(row["config_path"])
    checkpoint = Path(row["checkpoint_path"])
    if sha256(config) != row["config_sha256"]:
        raise RuntimeError("official-test config SHA-256 mismatch")
    if sha256(checkpoint) != row["checkpoint_sha256"]:
        raise RuntimeError("official-test checkpoint SHA-256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("official attack suite requires CUDA")

    torch.manual_seed(ATTACK_SEED)
    torch.cuda.manual_seed_all(ATTACK_SEED)
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = load_config(config)
    cfg.data.download = False
    cfg.data.augmentation = False
    cfg.data.batch_size = 128
    cfg.data.num_workers = min(2, int(cfg.data.get("num_workers", 2)))
    loader = build_datamodule(cfg.data)["test"]
    if len(loader.dataset) != EXPECTED_EXAMPLES:
        raise RuntimeError("official CIFAR-10 test must contain exactly 10000 examples")

    model, saved_cfg = load_checkpoint_model(str(checkpoint), cfg_override=cfg)
    if str(saved_cfg.experiment.name) != row["source_run_id"]:
        raise RuntimeError("checkpoint run identity mismatch")
    model = model.float().to(device).eval()
    if model.training or any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("FP32/eval invariant failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    profile = row["suite_profile"]
    direct_clean: list[bool] | None = None
    pgd_correct: list[bool] | None = None
    pgd_max_delta: float | None = None
    primary_metrics: dict[str, Any] | None = None
    primary_trace: dict[str, Any] | None = None
    primary_diagnostics: dict[str, Any] | None = None
    if profile == "full_primary_secondary":
        direct_clean = clean_mask(model, loader, device)
        pgd_correct, pgd_max_delta = pgd_mask(model, loader, device, cfg)
        primary_metrics, primary_trace, primary_diagnostics = run_autoattack(
            model=model,
            loader=loader,
            device=device,
            cfg=cfg,
            epsilon=EPSILON,
            output=output,
            label="eps8",
        )
        if len(direct_clean) != EXPECTED_EXAMPLES or len(pgd_correct) != EXPECTED_EXAMPLES:
            raise RuntimeError("incomplete clean or PGD correctness vector")
        if pgd_max_delta > EPSILON + 1e-6:
            raise RuntimeError("PGD exceeded Linf bound")

    secondary_metrics, secondary_trace, secondary_diagnostics = run_autoattack(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        epsilon=SECONDARY_EPSILON,
        output=output,
        label="eps4",
    )

    per_example: list[dict[str, Any]] = []
    secondary_masks = secondary_trace["correct_after_component"]
    primary_masks = primary_trace["correct_after_component"] if primary_trace else None
    for index in range(EXPECTED_EXAMPLES):
        record: dict[str, Any] = {
            "sample_id": index,
            "aa4_clean_correct": bool(secondary_trace["clean_correct"][index]),
            "aa4_after_apgd_ce": bool(secondary_masks["apgd-ce"][index]),
            "aa4_after_apgd_dlr": bool(secondary_masks["apgd-t"][index]),
            "aa4_after_fab": bool(secondary_masks["fab-t"][index]),
            "aa4_after_square": bool(secondary_masks["square"][index]),
            "aa4_final_correct": bool(secondary_trace["final_robust_correct"][index]),
        }
        if primary_trace and primary_masks and direct_clean is not None and pgd_correct is not None:
            record.update(
                {
                    "clean_correct": bool(direct_clean[index]),
                    "pgd20_5r_correct": bool(pgd_correct[index]),
                    "aa8_after_apgd_ce": bool(primary_masks["apgd-ce"][index]),
                    "aa8_after_apgd_dlr": bool(primary_masks["apgd-t"][index]),
                    "aa8_after_fab": bool(primary_masks["fab-t"][index]),
                    "aa8_after_square": bool(primary_masks["square"][index]),
                    "aa8_final_correct": bool(primary_trace["final_robust_correct"][index]),
                }
            )
        per_example.append(record)

    warnings: list[str] = []
    metrics: dict[str, Any] = {"aa4_standard": trace_summary(secondary_trace)}
    invariants: dict[str, Any] = {
        "ordered_test_ids_0_to_9999": True,
        "secondary_autoattack_linf_bound": secondary_metrics[
            "attack/autoattack_max_perturbation"
        ],
        "secondary_autoattack_diagnostics": secondary_diagnostics,
        "stage_monotonic": True,
        "finite": True,
        "model_eval": True,
        "fp32": True,
    }
    if primary_trace and primary_metrics and primary_diagnostics and pgd_correct is not None:
        clean_acc = sum(bool(value) for value in direct_clean or []) / EXPECTED_EXAMPLES
        pgd_acc = sum(bool(value) for value in pgd_correct) / EXPECTED_EXAMPLES
        aa8_summary = trace_summary(primary_trace)
        metrics.update(
            {
                "clean_accuracy": clean_acc,
                "pgd20_5restart_worst_case_accuracy": pgd_acc,
                "aa8_standard": aa8_summary,
            }
        )
        if pgd_acc - float(aa8_summary["accuracy"]) > 0.05:
            warnings.append("PGD-AA gap exceeds 5 pp")
        if (
            float(aa8_summary["components"]["fab-t"])
            - float(aa8_summary["accuracy"])
            > 0.03
        ):
            warnings.append("Square removes more than 3 pp after white-box stages")
        invariants.update(
            {
                "pgd_linf_bound": pgd_max_delta,
                "primary_autoattack_linf_bound": primary_metrics[
                    "attack/autoattack_max_perturbation"
                ],
                "primary_autoattack_diagnostics": primary_diagnostics,
            }
        )

    report = {
        "schema_version": 1,
        "report_kind": "sealed_remaining_official_test_suite",
        "sealed": True,
        "reveal_state": "do_not_inspect_before_final_confirmation_lock",
        "suite_profile": profile,
        "eval_id": eval_id,
        "source_run_id": row["source_run_id"],
        "rq": row["rq"],
        "arm": row["arm"],
        "training_seed": int(row["seed"]),
        "attack_seed": ATTACK_SEED,
        "sample_count": EXPECTED_EXAMPLES,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "config_sha256": row["config_sha256"],
        "source_git_commit": os.environ.get("TRUSTML_GIT_COMMIT", "unknown"),
        "transport_bundle_sha256": os.environ.get("TRUSTML_BUNDLE_HASH", "unknown"),
        "metrics": metrics,
        "attack": {
            "norm": "Linf",
            "primary_epsilon": EPSILON if profile == "full_primary_secondary" else None,
            "secondary_epsilon": SECONDARY_EPSILON,
            "precision": "FP32",
            "model_mode": "eval",
            "pgd_steps": 20 if profile == "full_primary_secondary" else None,
            "pgd_restarts": 5 if profile == "full_primary_secondary" else None,
            "autoattack_version": "standard",
            "component_order": list(COMPONENTS),
        },
        "invariants": invariants,
        "warnings": warnings,
        "per_example": per_example,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.eval_id, args.output)


if __name__ == "__main__":
    main()
