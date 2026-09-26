"""Auditable fixed-dev AutoAttack evaluation for the downstream RQ1 gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset

from trustml.attacks.autoattack import AutoAttackAttack
from trustml.core.config import load_config
from trustml.core.model_loading import load_checkpoint_model
from trustml.data.registry import build_datamodule


AA_SEED = 42
AA_EPSILON = 8.0 / 255.0
AA_COMPONENTS = ("apgd-ce", "apgd-t", "fab-t", "square")
COMPONENT_OUTPUT_NAMES = {
    "apgd-ce": "correct_after_apgd_ce",
    "apgd-t": "correct_after_apgd_dlr",
    "fab-t": "correct_after_fab",
    "square": "correct_after_square",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _subset_sample_ids(dataset) -> list[int]:
    if not isinstance(dataset, Subset):
        raise RuntimeError("fixed public dev must be a torch.utils.data.Subset")
    ids = [int(value) for value in dataset.indices]
    if len(ids) != len(set(ids)):
        raise RuntimeError("fixed public dev sample IDs are not unique")
    return ids


def _autoattack_version() -> str:
    try:
        return importlib.metadata.version("autoattack")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _gpu_metadata(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "model": torch.cuda.get_device_name(index),
        "device_index": int(index),
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _validate_trace(trace: dict[str, Any], expected_examples: int) -> None:
    order = trace.get("official_component_order")
    if order != list(AA_COMPONENTS):
        raise RuntimeError(f"unexpected AutoAttack component order: {order}")
    clean = trace.get("clean_correct")
    final = trace.get("final_robust_correct")
    masks = trace.get("correct_after_component", {})
    for name, values in [("clean_correct", clean), ("final_robust_correct", final)]:
        if not isinstance(values, list) or len(values) != expected_examples:
            raise RuntimeError(f"{name} does not contain exactly {expected_examples} flags")
    previous = sum(bool(value) for value in clean)
    for component in AA_COMPONENTS:
        values = masks.get(component)
        if not isinstance(values, list) or len(values) != expected_examples:
            raise RuntimeError(f"missing complete component mask for {component}")
        current = sum(bool(value) for value in values)
        reached = int(trace["component_reached_examples"][component])
        if reached != previous:
            raise RuntimeError(
                f"component {component} reached count {reached} != prior survivor count {previous}"
            )
        if current > previous:
            raise RuntimeError(f"component {component} increased robust survivor count")
        previous = current
    if [bool(value) for value in masks["square"]] != [bool(value) for value in final]:
        raise RuntimeError("final robust mask differs from the post-Square mask")


def evaluate(
    *,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    run_id: str,
    batch_size: int,
    expected_examples: int,
    attack_seed: int,
    public_dev_only: bool = False,
) -> dict[str, Any]:
    if expected_examples != 5_000:
        raise RuntimeError("RQ1 dev AutoAttack is locked to all 5,000 fixed public dev examples")
    if attack_seed != AA_SEED:
        raise RuntimeError(f"RQ1 dev AutoAttack seed is locked to {AA_SEED}")
    if not torch.cuda.is_available():
        raise RuntimeError("dev AutoAttack is an A100 CUDA-only job")

    # PyTorch 2.4 rejects an unindexed ``torch.device("cuda")`` in
    # ``set_device`` even when exactly one GPU is visible (the DataSphere A100
    # setup). Resolve the visible current index explicitly and then use that
    # same indexed device throughout the evaluation.
    device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    torch.set_grad_enabled(True)
    torch.backends.cudnn.benchmark = False

    cfg = load_config(config_path)
    cfg.data.download = False
    cfg.data.augmentation = False
    cfg.data.batch_size = int(batch_size)
    cfg.data.num_workers = min(2, int(cfg.data.get("num_workers", 2)))
    if public_dev_only:
        # Same fixed dev transform/order, but never instantiate train=False.
        from torchvision.datasets import CIFAR10
        from torch.utils.data import DataLoader
        from trustml.data.splits import train_val_indices
        from trustml.data.transforms import cifar_transform
        data = CIFAR10(str(cfg.data.root), train=True, download=False,
                       transform=cifar_transform(train=False, mean=cfg.data.mean,
                                                 std=cfg.data.std, augmentation=False))
        _, indices = train_val_indices(len(data), float(cfg.data.train_ratio), int(cfg.data.seed))
        loader = DataLoader(Subset(data, indices), batch_size=batch_size, shuffle=False,
                            num_workers=int(cfg.data.num_workers), pin_memory=True)
    else:
        loader = build_datamodule(cfg.data)["val"]
    sample_ids = _subset_sample_ids(loader.dataset)
    if len(sample_ids) != expected_examples:
        raise RuntimeError(
            f"fixed public dev contains {len(sample_ids)} examples, expected {expected_examples}"
        )

    checkpoint_hash = sha256(checkpoint_path)
    model, saved_cfg = load_checkpoint_model(str(checkpoint_path), cfg_override=cfg)
    checkpoint_run_id = str(saved_cfg.experiment.name)
    if checkpoint_run_id != run_id:
        raise RuntimeError(f"checkpoint run ID {checkpoint_run_id!r} != requested {run_id!r}")
    model = model.float().to(device).eval()
    if model.training:
        raise RuntimeError("model.eval() invariant failed")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("FP32 model invariant failed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".autoattack.log")
    attack = AutoAttackAttack(
        OmegaConf.create(
            {
                "epsilon": AA_EPSILON,
                "norm": "Linf",
                "version": "standard",
                "batch_size": int(batch_size),
                "seed": int(attack_seed),
                "expected_examples": int(expected_examples),
                "require_full_test": False,
                "verbose": True,
                "log_path": str(log_path),
                "capture_component_trace": True,
            }
        ),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    metrics = attack.run(model, loader, device=device, progress=True)
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    trace = attack.last_component_trace
    if trace is None:
        raise RuntimeError("AutoAttack component trace was not captured")
    _validate_trace(trace, expected_examples)

    diagnostics = attack.last_diagnostics or {}
    output_diagnostics = attack.last_output_diagnostics or {}
    if not diagnostics.get("validation_passed", False):
        raise RuntimeError("pixel/normalization diagnostics did not pass")
    if int(diagnostics.get("n_examples", -1)) != expected_examples:
        raise RuntimeError("pixel diagnostics example count mismatch")
    if float(metrics["attack/autoattack_max_perturbation"]) > AA_EPSILON + 1e-6:
        raise RuntimeError("Linf attack budget invariant failed")

    masks = trace["correct_after_component"]
    per_example: list[dict[str, Any]] = []
    for position, sample_id in enumerate(sample_ids):
        record = {
            "position": position,
            "sample_id": sample_id,
            "clean_correct": bool(trace["clean_correct"][position]),
            "final_robust_correct": bool(trace["final_robust_correct"][position]),
        }
        for component, output_name in COMPONENT_OUTPUT_NAMES.items():
            record[output_name] = bool(masks[component][position])
        per_example.append(record)

    component_accuracy = {
        COMPONENT_OUTPUT_NAMES[component].removeprefix("correct_after_"): (
            sum(bool(value) for value in masks[component]) / expected_examples
        )
        for component in AA_COMPONENTS
    }
    command = [sys.executable, "-m", "trustml.dev_autoattack", *sys.argv[1:]]
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_kind": "rq1_fixed_dev_autoattack",
        "run_id": run_id,
        "split": "fixed_public_dev",
        "official_test_used": False,
        "sample_count": expected_examples,
        "sample_id_sha256": canonical_hash(sample_ids),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "source_git_commit": os.environ.get("TRUSTML_GIT_COMMIT", "unknown"),
        "transport_bundle_sha256": os.environ.get("TRUSTML_BUNDLE_HASH", "unknown"),
        "attack": {
            "name": "AutoAttack",
            "version": "standard",
            "package_version": _autoattack_version(),
            "norm": "Linf",
            "epsilon": AA_EPSILON,
            "seed": attack_seed,
            "precision": "FP32",
            "batch_size": batch_size,
            "official_component_order": list(AA_COMPONENTS),
        },
        "execution": {
            "command": command,
            "command_shell": " ".join(command),
            "python": platform.python_version(),
            "gpu": _gpu_metadata(device),
            "peak_gpu_memory_allocated_bytes": peak_allocated,
            "peak_gpu_memory_reserved_bytes": peak_reserved,
            "total_wall_clock_seconds": float(trace["total_seconds"]),
            "component_durations_seconds": trace["component_durations_seconds"],
            "component_reached_examples": trace["component_reached_examples"],
        },
        "metrics": metrics,
        "component_robust_accuracy": component_accuracy,
        "invariants": {
            "model_eval": True,
            "logits_not_probabilities": True,
            "single_normalization": True,
            "input_pixel_range_valid": True,
            "linf_budget_valid": True,
            "fp32": True,
            "fixed_dev_order": True,
            "pixel_wrapper_diagnostics": diagnostics,
            "attack_output_diagnostics": output_diagnostics,
        },
        "per_example": per_example,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_example"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--expected-examples", type=int, default=5_000)
    parser.add_argument("--attack-seed", type=int, default=AA_SEED)
    args = parser.parse_args()
    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        run_id=args.run_id,
        batch_size=args.batch_size,
        expected_examples=args.expected_examples,
        attack_seed=args.attack_seed,
    )


if __name__ == "__main__":
    main()
