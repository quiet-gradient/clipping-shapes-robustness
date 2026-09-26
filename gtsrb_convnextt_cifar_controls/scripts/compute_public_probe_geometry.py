#!/usr/bin/env python3
"""Compute aggregate gradient geometry on the locked public CIFAR-10 dev probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from audit_privacy_accounting import audit_config
from trustml.core.config import load_config
from trustml.core.model_loading import load_checkpoint_model
from trustml.data.registry import build_datamodule
from trustml.defenses.dp_sgd import componentwise_clip_bounds
from trustml.training.gradient_probe import run_gradient_probe
from trustml.training.optimizers import apply_adaptation


def _defense(cfg, name: str):
    for item in cfg.defenses:
        if str(item.get("name", "")).lower() == name:
            return item
    raise KeyError(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_probe(dataset, *, size: int, seed: int):
    if size <= 0 or size > len(dataset):
        raise ValueError(f"probe size must be in [1, {len(dataset)}]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    subset_positions = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    samples = [dataset[int(index)] for index in subset_positions]
    x = torch.stack([sample[0] for sample in samples])
    y = torch.as_tensor([int(sample[1]) for sample in samples], dtype=torch.long)
    parent_indices = getattr(dataset, "indices", None)
    source_indices = (
        [int(parent_indices[position]) for position in subset_positions]
        if parent_indices is not None
        else [int(position) for position in subset_positions]
    )
    return x, y, subset_positions, source_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--snapshot-fraction", type=float, required=True)
    args = parser.parse_args()

    if not 0.0 <= args.snapshot_fraction <= 1.0:
        raise ValueError("snapshot fraction must be between zero and one")
    cfg = load_config(args.config)
    protocol = cfg.get("protocol", {}) or {}
    if str(protocol.get("selection_split", "val")).lower() != "val":
        raise RuntimeError("campaign geometry must use the public development split")

    probe_cfg = (cfg.get("diagnostics", {}) or {}).get("gradient_probe", {}) or {}
    if str(probe_cfg.get("source", "val")).lower() != "val":
        raise RuntimeError("official test-set probes are forbidden for campaign geometry")
    size = int(args.size or probe_cfg.get("size", 512))
    seed = int(probe_cfg.get("seed", 20260820))
    microbatch_size = int(args.microbatch_size or probe_cfg.get("microbatch_size", 4))

    data = build_datamodule(cfg.data)
    x, y, positions, source_indices = _fixed_probe(data["val"].dataset, size=size, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("full public gradient geometry is a GPU-only job")

    model, saved_cfg = load_checkpoint_model(str(args.checkpoint), cfg_override=cfg)
    apply_adaptation(model, cfg)
    model.to(device)
    dp = _defense(cfg, "dp_sgd")
    trades = _defense(cfg, "trades")
    beta = float(trades.get("beta", 3.0))
    sensitivity = float(dp.get("sensitivity_bound", dp.get("max_grad_norm", 1.0)))
    if bool(dp.get("componentwise_one_release", False)):
        task_clip, robust_clip = componentwise_clip_bounds(
            beta=beta,
            sensitivity_bound=sensitivity,
            q_r=float(dp.get("componentwise_q_r", 0.5)),
        )
        joint_clip = sensitivity
    else:
        joint_clip = float(dp.get("max_grad_norm", sensitivity))
        task_clip = float(dp.get("max_grad_norm", sensitivity))
        robust_clip = float(dp.get("c_regularizer_clip_norm", task_clip / max(beta, 1.0)))

    accounting = audit_config(args.config)
    if accounting["status"] != "PASS":
        raise RuntimeError("privacy accounting must pass before public geometry")
    sigma = float(accounting["noise_multiplier"])
    metrics = run_gradient_probe(
        model,
        x.to(device),
        y.to(device),
        beta=beta,
        joint_clip_norm=joint_clip,
        task_clip_norm=task_clip,
        robust_clip_norm=robust_clip,
        noise_multiplier_a=sigma,
        noise_multiplier_c_task=sigma,
        noise_multiplier_c_robust=sigma,
        noise_multiplier_d=sigma,
        expected_batch_size=int(accounting["aggregation_denominator"]),
        epsilon=float(trades.get("epsilon", 8.0 / 255.0)),
        step_size=float(trades.get("step_size", 2.0 / 255.0)),
        steps=int(trades.get("steps", 10)),
        mean=tuple(trades.get("mean", cfg.data.mean)),
        std=tuple(trades.get("std", cfg.data.std)),
        microbatch_size=microbatch_size,
        seed=seed,
        auto_s_gamma=float(dp.get("auto_s_gamma", 0.01)),
        auto_s_componentwise_q=float(
            dp.get("q_regularizer", dp.get("componentwise_q_r", 0.5))
        ),
        learning_rate=float(cfg.training.optimizer.lr),
    )
    report = {
        "format_version": 1,
        "public_source": "cifar10_train_dev",
        "data_split_seed": int(cfg.data.seed),
        "probe_seed": seed,
        "probe_size": size,
        "probe_subset_positions": positions,
        "probe_source_indices": source_indices,
        "snapshot_fraction": args.snapshot_fraction,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_seed": int(saved_cfg.experiment.seed),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "accounting": accounting,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
