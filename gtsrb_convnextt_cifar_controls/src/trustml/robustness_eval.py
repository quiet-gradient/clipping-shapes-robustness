"""Standalone clean + PGD + AutoAttack evaluation for saved checkpoints.

The command deliberately rebuilds each model from the config embedded in its
checkpoint.  This preserves DP architecture surgery (BatchNorm -> GroupNorm)
and avoids the silent mismatch in older standalone evaluators.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import OmegaConf

from trustml.attacks.autoattack import AutoAttackAttack
from trustml.attacks.pgd import PGDAttack
from trustml.core.model_loading import load_checkpoint_model
from trustml.core.seed import seed_everything
from trustml.data.registry import build_datamodule
from trustml.evaluation.clean import evaluate_clean


AUTOATTACK_COMMIT = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"


class RobustnessGuardError(RuntimeError):
    """Post-evaluation consistency failure with preserved count evidence."""

    def __init__(self, message: str, evidence: dict) -> None:
        super().__init__(message)
        self.evidence = evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_paths(explicit: list[str], patterns: list[str]) -> list[Path]:
    found = [Path(value) for value in explicit]
    for pattern in patterns:
        found.extend(Path(value) for value in glob.glob(pattern))
    unique = sorted({path.resolve() for path in found})
    if not unique:
        raise ValueError("No checkpoints were supplied or matched.")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    return unique


def _device(spec: str) -> torch.device:
    requested = str(spec).lower()
    if requested == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Full robustness evaluation requires CUDA; no CUDA device is available.")
        return torch.device("cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda but torch.cuda.is_available() is False.")
    return torch.device(requested)


def _target_metadata(checkpoint: dict, cfg, path: Path) -> dict:
    defenses = cfg.get("defenses", []) or []
    dp = next(
        (
            item
            for item in defenses
            if str(item.get("name", "")).lower() == "dp_sgd" and bool(item.get("enabled", True))
        ),
        None,
    )
    trades = next(
        (
            item
            for item in defenses
            if str(item.get("name", "")).lower() == "trades" and bool(item.get("enabled", True))
        ),
        None,
    )
    metrics = checkpoint.get("metrics", {}) or {}
    attained = metrics.get("dp/epsilon", metrics.get("DP epsilon"))
    strategy = None if trades is None else str(trades.get("dp_strategy", "A")).upper()
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "experiment": str(cfg.get("experiment", {}).get("name", path.stem)),
        "seed": int(cfg.get("experiment", {}).get("seed", 42)),
        "strategy": strategy,
        "beta": None if trades is None else float(trades.get("beta", 0.0)),
        "target_epsilon": None if dp is None else float(dp.get("epsilon")),
        "attained_epsilon_at_checkpoint": None if attained is None else float(attained),
        # Strategy B adds an unclipped, unnoised KL gradient after the private
        # task update. Its task channel has an accountant value, but the full
        # model update is not covered by that DP claim.
        "dp_claim_valid": None if dp is None else strategy != "B",
    }


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def evaluate_checkpoint(path: Path, args, overrides: list[str]) -> dict:
    # PGD uses a random start. Seed before building the loader/model so the
    # complete clean/PGD/AutoAttack evaluation is reproducible per checkpoint.
    seed_everything(args.seed)
    model, cfg = load_checkpoint_model(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        OmegaConf.resolve(cfg)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = _device(args.device)
    model.to(device).eval()
    data = build_datamodule(cfg.data)
    test_loader = data["test"]
    if len(test_loader.dataset) != args.expected_examples:
        raise ValueError(
            f"Expected the full {args.expected_examples}-example test set, "
            f"got {len(test_loader.dataset)} examples."
        )

    selected = {name.strip().lower() for name in args.attacks.split(",") if name.strip()}
    unknown = selected - {"pgd", "autoattack"}
    if unknown:
        raise ValueError(f"Unknown robustness attack(s): {sorted(unknown)}")

    started = time.time()
    record = {
        "schema_version": 2,
        "status": "running",
        "dataset": str(cfg.data.get("name", "cifar10")),
        "n_examples": int(len(test_loader.dataset)),
        "evaluation_mode": {
            "attacks": sorted(selected),
            "pixel_wrapper_diagnostics": bool(args.diagnose_pixel_wrapper or "autoattack" in selected),
        },
        "autoattack": {
            "upstream_commit": AUTOATTACK_COMMIT,
            "version": "standard",
            "attacks": ["apgd-ce", "apgd-t", "fab-t", "square"],
            "norm": "Linf",
            "epsilon": float(args.epsilon),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        "pixel_wrapper": {
            "pixel_quantization_levels": int(args.pixel_quantization_levels),
            "pixel_lattice_tolerance": float(args.pixel_lattice_tolerance),
            "roundtrip_tolerance": float(args.pixel_roundtrip_tolerance),
            "max_prediction_disagreements": int(args.max_wrapper_prediction_disagreements),
        },
        "pgd": {
            "norm": "Linf",
            "epsilon": float(args.epsilon),
            "step_size": float(args.pgd_step_size),
            "steps": int(args.pgd_steps),
            "seed": int(args.seed),
            "max_batches": None,
        },
        **_target_metadata(checkpoint, cfg, path),
    }

    clean = evaluate_clean(model, test_loader, device, max_batches=None, progress=True)
    metrics: dict[str, float] = {
        "test/clean_acc": float(clean["clean_acc"]),
        "test/loss": float(clean["loss"]),
    }

    if "pgd" in selected:
        pgd = PGDAttack(
            {
                "epsilon": args.epsilon,
                "step_size": args.pgd_step_size,
                "steps": args.pgd_steps,
                "max_batches": None,
            },
            mean=tuple(cfg.data.get("mean", (0.4914, 0.4822, 0.4465))),
            std=tuple(cfg.data.get("std", (0.2023, 0.1994, 0.2010))),
        )
        metrics.update(pgd.run(model, test_loader, device=device, progress=True))

    autoattack = None
    if "autoattack" in selected or args.diagnose_pixel_wrapper:
        aa_log = args.log_path
        if aa_log is None:
            aa_log = str(args.output.parent / f"{path.stem}.autoattack.log")
        autoattack = AutoAttackAttack(
            {
                "epsilon": args.epsilon,
                "norm": "Linf",
                "version": "standard",
                "batch_size": args.batch_size,
                "seed": args.seed,
                "max_batches": None,
                "expected_examples": args.expected_examples,
                "require_full_test": True,
                "verbose": True,
                "log_path": aa_log,
                "pixel_quantization_levels": args.pixel_quantization_levels,
                "pixel_lattice_tolerance": args.pixel_lattice_tolerance,
                "roundtrip_tolerance": args.pixel_roundtrip_tolerance,
                "max_prediction_disagreements": args.max_wrapper_prediction_disagreements,
            },
            mean=tuple(cfg.data.get("mean", (0.4914, 0.4822, 0.4465))),
            std=tuple(cfg.data.get("std", (0.2023, 0.1994, 0.2010))),
        )

    if "autoattack" in selected:
        assert autoattack is not None
        metrics.update(autoattack.run(model, test_loader, device=device, progress=True))
        record["pixel_wrapper_diagnostics"] = autoattack.last_diagnostics
        record["autoattack_output_diagnostics"] = autoattack.last_output_diagnostics

    if args.diagnose_pixel_wrapper and "autoattack" not in selected:
        assert autoattack is not None
        record["pixel_wrapper_diagnostics"] = autoattack.diagnose(model, test_loader, device=device)

    diagnostics = record.get("pixel_wrapper_diagnostics")
    if isinstance(diagnostics, dict):
        direct_count = int(diagnostics["direct_clean_correct"])
        standalone_count = int(round(metrics["test/clean_acc"] * args.expected_examples))
        direct_evidence = {
            "standalone_clean_correct": standalone_count,
            "diagnostic_direct_clean_correct": direct_count,
            "n_examples": int(args.expected_examples),
        }
        if abs(standalone_count - direct_count) > 1:
            raise RobustnessGuardError(
                "Standalone clean evaluation disagrees with the diagnostic direct path.",
                direct_evidence,
            )

    if "autoattack" in selected:
        assert isinstance(diagnostics, dict)
        wrapper_count = int(diagnostics["wrapper_clean_correct"])
        autoattack_count = int(round(metrics["AUTOATTACK_CLEAN_ACC"] * args.expected_examples))
        wrapper_evidence = {
            "autoattack_clean_correct": autoattack_count,
            "diagnostic_wrapper_clean_correct": wrapper_count,
            "diagnostic_direct_clean_correct": int(diagnostics["direct_clean_correct"]),
            "prediction_disagreement_count": int(diagnostics["prediction_disagreement_count"]),
            "roundtrip_max_abs_error": float(diagnostics["roundtrip_max_abs_error"]),
            "input_lattice_max_abs_error": float(diagnostics["input_lattice_max_abs_error"]),
            "n_examples": int(args.expected_examples),
        }
        if abs(autoattack_count - wrapper_count) > 1:
            raise RobustnessGuardError(
                "AutoAttack clean evaluation disagrees with its diagnostic wrapper path.",
                wrapper_evidence,
            )
        if metrics["AUTOATTACK_ADV_ACC"] > metrics["AUTOATTACK_CLEAN_ACC"] + 1e-12:
            raise RobustnessGuardError(
                "AutoAttack adversarial accuracy exceeds its clean accuracy.",
                {
                    "autoattack_clean_acc": float(metrics["AUTOATTACK_CLEAN_ACC"]),
                    "autoattack_adv_acc": float(metrics["AUTOATTACK_ADV_ACC"]),
                },
            )

    record["metrics"] = metrics
    record["status"] = "complete"
    record["elapsed_seconds"] = float(time.time() - started)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="checkpoint path (repeatable)")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="checkpoint glob (repeatable)")
    parser.add_argument("--output", type=Path, required=True, help="append-only JSONL result path")
    parser.add_argument("--log-path", default=None, help="AutoAttack text log (one-checkpoint jobs)")
    parser.add_argument("--attacks", default="pgd,autoattack", help="comma list: pgd,autoattack")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--pgd-step-size", type=float, default=2.0 / 255.0)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-examples", type=int, default=10_000)
    parser.add_argument(
        "--diagnose-pixel-wrapper",
        action="store_true",
        help="record direct/legacy/fixed wrapper diagnostics without requiring AutoAttack",
    )
    parser.add_argument("--pixel-quantization-levels", type=int, default=256)
    parser.add_argument("--pixel-lattice-tolerance", type=float, default=1e-6)
    parser.add_argument("--pixel-roundtrip-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-wrapper-prediction-disagreements", type=int, default=16)
    args, overrides = parser.parse_known_args(argv)

    paths = _checkpoint_paths(args.checkpoint, args.checkpoint_glob)
    for path in paths:
        try:
            record = evaluate_checkpoint(path, args, overrides)
        except Exception as exc:
            error_record = {
                "schema_version": 2,
                "status": "error",
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            diagnostics = getattr(exc, "diagnostics", None)
            if isinstance(diagnostics, dict):
                error_record["pixel_wrapper_diagnostics"] = diagnostics
            evidence = getattr(exc, "evidence", None)
            if isinstance(evidence, dict):
                error_record["guard_evidence"] = evidence
            _write_record(
                args.output,
                error_record,
            )
            raise
        _write_record(args.output, record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
