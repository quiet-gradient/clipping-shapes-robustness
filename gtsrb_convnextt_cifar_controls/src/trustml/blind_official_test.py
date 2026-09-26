"""Sealed official-test evaluation for predeclared downstream checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from trustml.attacks.autoattack import AutoAttackAttack
from trustml.attacks.pgd import PGDAttack, _set_grad_sample_hooks
from trustml.core.config import load_config
from trustml.core.model_loading import load_checkpoint_model
from trustml.data.registry import build_datamodule


EXPECTED_EXAMPLES = 10_000
ATTACK_SEED = 42
EPSILON = 8.0 / 255.0
COMPONENTS = ("apgd-ce", "apgd-t", "fab-t", "square")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authorized_row(manifest: Path, eval_id: str) -> dict[str, str]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["eval_id"] == eval_id]
    if len(matches) != 1:
        raise RuntimeError(f"blind manifest does not authorize exactly one row for {eval_id}")
    return matches[0]


@torch.no_grad()
def clean_mask(model, loader, device: torch.device) -> list[bool]:
    result: list[bool] = []
    for x, y in loader:
        logits = model(x.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite clean logits")
        result.extend(logits.argmax(dim=1).cpu().eq(y).tolist())
    return result


def pgd_mask(model, loader, device: torch.device, cfg) -> tuple[list[bool], float]:
    attack = PGDAttack(
        OmegaConf.create({"epsilon": EPSILON, "step_size": 2.0 / 255.0,
                          "steps": 20, "restarts": 5}),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    std = torch.tensor(cfg.data.std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    result: list[bool] = []
    max_pixel_delta = 0.0
    _set_grad_sample_hooks(model, False)
    try:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            robust = torch.ones_like(y, dtype=torch.bool)
            for _ in range(5):
                x_adv = attack.generate(model, x, y)
                max_pixel_delta = max(
                    max_pixel_delta,
                    float(((x_adv - x).abs() * std).max().item()),
                )
                with torch.no_grad():
                    pred = model(x_adv).argmax(dim=1)
                robust.logical_and_(pred.eq(y))
            result.extend(robust.cpu().tolist())
    finally:
        _set_grad_sample_hooks(model, True)
    return result, max_pixel_delta


def validate_trace(trace: dict[str, Any]) -> None:
    if trace.get("official_component_order") != list(COMPONENTS):
        raise RuntimeError("unexpected AutoAttack component order")
    masks = trace.get("correct_after_component", {})
    previous = sum(bool(value) for value in trace["clean_correct"])
    for component in COMPONENTS:
        values = masks.get(component)
        if not isinstance(values, list) or len(values) != EXPECTED_EXAMPLES:
            raise RuntimeError(f"incomplete AutoAttack mask for {component}")
        if int(trace["component_reached_examples"][component]) != previous:
            raise RuntimeError(f"AutoAttack reach mismatch for {component}")
        current = sum(bool(value) for value in values)
        if current > previous:
            raise RuntimeError(f"AutoAttack survivor count increased after {component}")
        previous = current
    if list(masks["square"]) != list(trace["final_robust_correct"]):
        raise RuntimeError("post-Square and final masks differ")


def evaluate(manifest: Path, eval_id: str, output: Path) -> None:
    if os.environ.get("BLIND_TEST_AUTHORIZATION") != "rq1_rq2_predeclared_v1":
        raise RuntimeError("missing immutable blind-test authorization token")
    row = authorized_row(manifest, eval_id)
    config = Path(row["config_path"])
    checkpoint = Path(row["checkpoint_path"])
    if sha256(config) != row["config_sha256"] or sha256(checkpoint) != row["checkpoint_sha256"]:
        raise RuntimeError("blind-test input SHA-256 mismatch")
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

    direct_clean = clean_mask(model, loader, device)
    pgd_correct, pgd_max_delta = pgd_mask(model, loader, device, cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    aa = AutoAttackAttack(
        OmegaConf.create({
            "epsilon": EPSILON,
            "norm": "Linf",
            "version": "standard",
            "batch_size": 128,
            "seed": ATTACK_SEED,
            "expected_examples": EXPECTED_EXAMPLES,
            "require_full_test": True,
            "verbose": False,
            "log_path": str(output.with_suffix(".autoattack.log")),
            "capture_component_trace": True,
        }),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )
    aa_metrics = aa.run(model, loader, device=device, progress=False)
    trace = aa.last_component_trace
    if trace is None:
        raise RuntimeError("AutoAttack component trace is missing")
    validate_trace(trace)
    if len(direct_clean) != EXPECTED_EXAMPLES or len(pgd_correct) != EXPECTED_EXAMPLES:
        raise RuntimeError("incomplete clean or PGD correctness vector")
    if pgd_max_delta > EPSILON + 1e-6:
        raise RuntimeError("PGD exceeded Linf bound")
    diagnostics = aa.last_diagnostics or {}
    if not diagnostics.get("validation_passed", False):
        raise RuntimeError("AutoAttack normalization diagnostics failed")

    stage_masks = trace["correct_after_component"]
    final_aa = [bool(value) for value in trace["final_robust_correct"]]
    per_example = [
        {
            "sample_id": index,
            "clean_correct": bool(direct_clean[index]),
            "pgd20_5r_correct": bool(pgd_correct[index]),
            "after_apgd_ce": bool(stage_masks["apgd-ce"][index]),
            "after_apgd_dlr": bool(stage_masks["apgd-t"][index]),
            "after_fab": bool(stage_masks["fab-t"][index]),
            "after_square": bool(stage_masks["square"][index]),
            "final_autoattack_correct": final_aa[index],
        }
        for index in range(EXPECTED_EXAMPLES)
    ]
    clean_acc = sum(direct_clean) / EXPECTED_EXAMPLES
    pgd_acc = sum(pgd_correct) / EXPECTED_EXAMPLES
    aa_acc = sum(final_aa) / EXPECTED_EXAMPLES
    warnings = []
    if pgd_acc - aa_acc > 0.05:
        warnings.append("PGD-AA gap exceeds 5 pp")
    fab_acc = sum(bool(value) for value in stage_masks["fab-t"]) / EXPECTED_EXAMPLES
    if fab_acc - aa_acc > 0.03:
        warnings.append("Square removes more than 3 pp after white-box stages")
    report = {
        "schema_version": 1,
        "report_kind": "sealed_rq1_rq2_official_test",
        "sealed": True,
        "reveal_state": "do_not_inspect_before_final_confirmation_lock",
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
        "metrics": {
            "clean_accuracy": clean_acc,
            "pgd20_5restart_worst_case_accuracy": pgd_acc,
            "autoattack_standard_accuracy": aa_acc,
            "autoattack_components": {
                component: sum(bool(value) for value in stage_masks[component]) / EXPECTED_EXAMPLES
                for component in COMPONENTS
            },
        },
        "attack": {
            "norm": "Linf", "epsilon": EPSILON, "precision": "FP32",
            "model_mode": "eval", "pgd_steps": 20, "pgd_restarts": 5,
            "pgd_restart_reduction": "per-example logical AND",
            "autoattack_version": "standard",
        },
        "invariants": {
            "ordered_test_ids_0_to_9999": True,
            "pgd_linf_bound": pgd_max_delta,
            "autoattack_linf_bound": aa_metrics["attack/autoattack_max_perturbation"],
            "autoattack_normalization": diagnostics,
            "autoattack_output": aa.last_output_diagnostics,
            "stage_monotonic": True,
            "finite": True,
        },
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
