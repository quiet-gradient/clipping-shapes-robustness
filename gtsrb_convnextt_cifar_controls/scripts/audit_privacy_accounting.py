#!/usr/bin/env python3
"""Pre-submit sampled-Gaussian accounting audit for downstream configs."""

from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

from omegaconf import OmegaConf
from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier


def _defense(cfg, name: str):
    for item in cfg.defenses:
        if bool(item.get("enabled", True)) and str(item.name).lower() == name:
            return item
    return None


def epsilon_for(mechanism: str, sigma: float, q: float, steps: int, delta: float) -> float:
    accountant = create_accountant(mechanism=mechanism)
    accountant.history = [(float(sigma), float(q), int(steps))]
    if mechanism.lower() == "prv":
        # Opacus 1.5.x can reject the default very fine discretization for this
        # long composition. A coarser *upper-bound* cross-check is stable; the
        # numerical epsilon error is logged and covered by the 10%-of-target
        # disagreement gate below.
        return float(
            accountant.get_epsilon(
                delta=float(delta), eps_error=0.1, delta_error=float(delta) / 10.0
            )
        )
    return float(accountant.get_epsilon(delta=float(delta)))


def audit_config(path: Path) -> dict[str, object]:
    cfg = OmegaConf.load(path)
    dp = _defense(cfg, "dp_sgd")
    if dp is None:
        raise RuntimeError(f"{path}: DP-SGD must be enabled")
    if not bool(dp.get("poisson_sampling", False)):
        raise RuntimeError(f"{path}: Poisson sampling is required")
    if cfg.training.get("max_train_batches", None) is not None:
        raise RuntimeError(f"{path}: capped training batches do not match the locked accountant")
    if str(dp.get("aggregation_denominator", "")).lower() != "logical_batch_size":
        raise RuntimeError(f"{path}: aggregation must divide by the locked logical batch size")

    dataset_size = 50_000 if str(cfg.data.name).lower() == "cifar10" else None
    if dataset_size is None:
        raise RuntimeError(f"{path}: unsupported dataset for an exact campaign audit")
    train_size = int(dataset_size * float(cfg.data.get("train_ratio", 0.9)))
    batches_per_epoch = math.ceil(train_size / int(cfg.data.batch_size))
    # Opacus' DPDataLoader uses one Poisson draw per original DataLoader batch.
    sample_probability = 1.0 / batches_per_epoch
    steps = int(cfg.training.epochs) * batches_per_epoch
    target_epsilon = float(dp.epsilon)
    delta = float(dp.delta)
    primary = str(dp.get("accountant", "prv"))
    crosscheck = str(dp.get("accountant_crosscheck", "rdp"))
    calibrated_sigma = float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_probability,
            steps=steps,
            accountant=primary,
            epsilon_tolerance=0.005,
        )
    )
    configured_sigma = dp.get("noise_multiplier", None)
    if configured_sigma is None:
        raise RuntimeError(f"{path}: noise_multiplier must be pinned in the resolved config")
    sigma = float(configured_sigma)
    epsilon_primary = epsilon_for(primary, sigma, sample_probability, steps, delta)
    epsilon_crosscheck = epsilon_for(crosscheck, sigma, sample_probability, steps, delta)
    target_tolerance = float(dp.get("achieved_epsilon_tolerance", 0.02))
    crosscheck_tolerance_fraction = float(
        dp.get("accountant_crosscheck_tolerance_fraction", 0.1)
    )
    crosscheck_tolerance = crosscheck_tolerance_fraction * target_epsilon

    if bool(dp.get("componentwise_one_release", False)):
        sensitivity = float(dp.sensitivity_bound)
    elif bool(dp.get("auto_s", False)):
        sensitivity = 1.0
    else:
        sensitivity = float(dp.max_grad_norm)

    errors = []
    trades = _defense(cfg, "trades")
    if bool(dp.get("auto_s", False)) and bool(dp.get("componentwise_one_release", False)):
        strategy = "S-C"
    elif bool(dp.get("auto_s", False)) and trades is not None:
        strategy = "S-A"
    elif bool(dp.get("auto_s", False)):
        strategy = "S-CE"
    else:
        strategy = str((cfg.get("protocol", {}) or {}).get("arm", "hard"))
    q_regularizer = float(
        dp.get("q_regularizer", dp.get("componentwise_q_r", 0.5))
    )
    if strategy == "S-C":
        if abs(sensitivity - 1.0) > 1e-12:
            errors.append("S-C must use unit total sensitivity")
        if not 0.0 < q_regularizer < 1.0:
            errors.append("S-C q_regularizer must lie strictly between zero and one")
        if abs(float(dp.get("task_sensitivity_cap", 1.0 - q_regularizer)) - (1.0 - q_regularizer)) > 1e-12:
            errors.append("S-C task sensitivity cap must equal 1-q")
        if abs(float(dp.get("robust_sensitivity_cap", q_regularizer)) - q_regularizer) > 1e-12:
            errors.append("S-C robust sensitivity cap must equal q")
    if bool(dp.get("componentwise_one_release", False)) and not bool(dp.get("auto_s", False)):
        strategy = "H-C"
        if trades is None or str(trades.get("dp_strategy", "")).upper() != "C":
            errors.append("H-C one-release requires TRADES dp_strategy=C")
        beta = float(trades.get("beta", 0.0)) if trades is not None else 0.0
        if beta <= 0.0:
            errors.append("H-C beta must be positive")
        if not 0.0 < q_regularizer < 1.0:
            errors.append("H-C q_regularizer must lie strictly between zero and one")
        expected_task = (1.0 - q_regularizer) * sensitivity
        expected_robust = q_regularizer * sensitivity / max(beta, 1e-30)
        if abs(float(dp.max_grad_norm) - expected_task) > 1e-12:
            errors.append("H-C task clip must equal (1-q)*sensitivity")
        if abs(float(dp.c_regularizer_clip_norm) - expected_robust) > 1e-12:
            errors.append("H-C robust clip must equal q*sensitivity/beta")
        if abs(float(dp.max_grad_norm) + beta * float(dp.c_regularizer_clip_norm) - sensitivity) > 1e-12:
            errors.append("H-C component bounds do not sum to total sensitivity")
        configured_absolute = float(dp.get("absolute_noise_std", sigma * sensitivity))
        if abs(configured_absolute - sigma * sensitivity) > 1e-12:
            errors.append("H-C absolute noise std must equal sigma*sensitivity")
    if abs(epsilon_primary - target_epsilon) > target_tolerance:
        errors.append("primary achieved epsilon misses the target tolerance")
    if abs(sigma - calibrated_sigma) > 1e-12:
        errors.append("configured noise multiplier differs from the audited binary search")
    if abs(epsilon_primary - epsilon_crosscheck) > crosscheck_tolerance:
        errors.append("PRV/RDP cross-check exceeds the locked tolerance")
    report = {
        "config": path.as_posix(),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "target_epsilon": target_epsilon,
        "strategy": strategy,
        "q_regularizer": q_regularizer if strategy in {"S-C", "H-C"} else None,
        "gamma": float(dp.get("auto_s_gamma", 0.01)) if strategy.startswith("S-") else None,
        "achieved_epsilon_primary": epsilon_primary,
        "achieved_epsilon_crosscheck": epsilon_crosscheck,
        "delta": delta,
        "noise_multiplier": sigma,
        "binary_search_noise_multiplier": calibrated_sigma,
        "sensitivity_bound": sensitivity,
        "absolute_noise_std_before_averaging": sigma * sensitivity,
        "sampling": "poisson_add_remove",
        "sampling_probability": sample_probability,
        "logical_batch_size": int(cfg.data.batch_size),
        "aggregation_denominator": int(cfg.data.batch_size),
        "expected_poisson_batch_size": sample_probability * train_size,
        "train_size": train_size,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps": steps,
        "primary_accountant": primary,
        "crosscheck_accountant": crosscheck,
        "crosscheck_epsilon_error_bound": 0.1 if crosscheck.lower() == "prv" else None,
        "crosscheck_tolerance_fraction_of_target": crosscheck_tolerance_fraction,
        "crosscheck_tolerance_absolute": crosscheck_tolerance,
        "opacus_version": version("opacus"),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [audit_config(path) for path in args.configs]
    sa_reports = [report for report in reports if report["strategy"] == "S-A"]
    sc_reports = [report for report in reports if report["strategy"] == "S-C"]
    if sa_reports and sc_reports:
        reference_sigmas = {float(report["noise_multiplier"]) for report in sa_reports}
        sc_sigmas = {float(report["noise_multiplier"]) for report in sc_reports}
        if len(reference_sigmas) != 1 or sc_sigmas != reference_sigmas:
            for report in sa_reports + sc_reports:
                report["errors"].append("S-A and S-C must use the identical noise multiplier")
                report["status"] = "FAIL"
    payload = {"status": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL", "runs": reports}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
