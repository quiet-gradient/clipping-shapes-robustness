"""Glue for running the one-run epsilon audit inside a training run.

``prepare_audit`` runs before training: it persists the canary pack and scores
the canaries under the initial weights w0. ``finalize_audit`` runs after
training: it scores the canaries under the final weights, forms the black-box
score of Algorithm 3 (Score(x) = loss(w0, x) - loss(w_final, x)), sweeps the
number of guesses, and converts correct-guess counts into 95%-confidence lower
bounds on epsilon via Theorem 5.2.

Everything needed to recompute the bound offline (coins, losses, scores,
per-k results) is written to ``audit_results.json`` in the run directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from trustml.audit.canaries import CanaryPack
from trustml.audit.steinke import audit_epsilon_sweep, score_auc


@dataclass
class AuditState:
    pack: CanaryPack
    init_losses: np.ndarray


@torch.no_grad()
def per_example_losses(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    """Per-example cross-entropy on the canary labels, in loader order."""
    was_training = model.training
    model.eval()
    chunks = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        chunks.append(F.cross_entropy(logits, y, reduction="none").cpu())
    if was_training:
        model.train()
    return torch.cat(chunks).numpy()


def _dp_claim(cfg) -> dict:
    for item in cfg.get("defenses", []):
        if str(item.get("name", "")).lower() == "dp_sgd" and bool(item.get("enabled", True)):
            return {
                "epsilon": float(item.get("epsilon", 8.0)),
                "delta": float(item.get("delta", 1e-5)),
                "poisson_sampling": bool(item.get("poisson_sampling", False)),
            }
    return {}


def prepare_audit(cfg, trainer, data, pack: CanaryPack) -> AuditState:
    """Score canaries under w0 and persist the pack before training starts.

    w0 is the model exactly as built (pretrained fine-tune weights included) but
    before Opacus' BatchNorm->GroupNorm conversion; the baseline only has to be
    a fixed function of the example, independent of the inclusion coins, so
    this does not affect the validity of the bound.
    """
    trainer.model.to(trainer.device)
    init_losses = per_example_losses(trainer.model, data["audit_canaries"], trainer.device)

    run_dir = getattr(trainer.logger, "output_dir", None)
    if run_dir is not None:
        payload = {**pack.to_dict(), "init_losses": [float(x) for x in init_losses]}
        pack_path = Path(run_dir) / "audit_pack.json"
        pack_path.write_text(json.dumps(payload))

    trainer.logger.log_metrics(
        {
            "audit/enabled": 1.0,
            "audit/m": float(pack.m),
            "audit/num_included": float(len(pack.included_positions)),
        },
        step=0,
    )
    return AuditState(pack=pack, init_losses=init_losses)


def finalize_audit(cfg, trainer, data, state: AuditState) -> dict[str, float]:
    audit_cfg = cfg.get("audit", {})
    pack = state.pack
    claim = _dp_claim(cfg)
    delta = float(audit_cfg.get("delta", claim.get("delta", 1e-5)))
    confidence = float(audit_cfg.get("confidence", 0.95))
    p = 1.0 - confidence
    score_kind = str(audit_cfg.get("score", "loss_delta")).lower()
    target_epsilon = claim.get("epsilon", None)

    final_losses = per_example_losses(trainer.model, data["audit_canaries"], trainer.device)
    if score_kind == "loss_delta":
        scores = state.init_losses - final_losses
    elif score_kind == "neg_final_loss":
        scores = -final_losses
    else:
        raise ValueError(f"Unknown audit score: {score_kind}")

    s_true = np.asarray(pack.s, dtype=np.int64)
    k_values = audit_cfg.get("k_values", None)
    k_values = [int(k) for k in k_values] if k_values else None
    results = audit_epsilon_sweep(
        scores, s_true, delta=delta, p=p, k_values=k_values, target_epsilon=target_epsilon
    )
    best = max(results, key=lambda res: (res.eps_lb, -res.r))

    included = s_true == 1
    metrics: dict[str, float] = {
        "audit/eps_lb": float(best.eps_lb),
        "audit/eps_lb_r": float(best.r),
        "audit/eps_lb_v": float(best.v),
        "audit/eps_lb_guess_acc": float(best.accuracy),
        "audit/delta": delta,
        "audit/confidence": confidence,
        "audit/score_auc": float(score_auc(scores, s_true)),
        "audit/score_mean_included": float(scores[included].mean()) if included.any() else float("nan"),
        "audit/score_mean_excluded": float(scores[~included].mean()) if (~included).any() else float("nan"),
        "Audit epsilon lower bound": float(best.eps_lb),
    }
    if target_epsilon is not None:
        metrics["audit/target_epsilon"] = float(target_epsilon)
        metrics["audit/p_value_at_target"] = float(min(res.p_value_at_target for res in results))
        metrics["audit/eps_lb_over_target"] = float(best.eps_lb) / float(target_epsilon)

    run_dir = getattr(trainer.logger, "output_dir", None)
    if run_dir is not None:
        report = {
            "method": "steinke_one_run",
            "paper": "arXiv:2305.08846 (NeurIPS 2023)",
            "score": score_kind,
            "delta": delta,
            "confidence": confidence,
            "target_epsilon": target_epsilon,
            "poisson_sampling": claim.get("poisson_sampling", None),
            "pack": pack.to_dict(),
            "init_losses": [float(x) for x in state.init_losses],
            "final_losses": [float(x) for x in final_losses],
            "scores": [float(x) for x in scores],
            "sweep": [res.to_dict() for res in results],
            "best": best.to_dict(),
        }
        report_path = Path(run_dir) / "audit_results.json"
        report_path.write_text(json.dumps(report, indent=2))
        trainer.logger.log_artifact(report_path, artifact_path="audit-results")

    return metrics
