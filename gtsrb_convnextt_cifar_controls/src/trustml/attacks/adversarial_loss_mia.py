"""Adversarial-loss membership inference attack (Song et al., CCS'19).

Threshold attack, no shadow models. For each example we score how confidently
the model fits it, both on the clean input and on a PGD-perturbed version. The
intuition (Song, Shokri, Mittal 2019, "Privacy Risks of Securing Machine
Learning Models against Adversarial Examples"): robust training flattens the
loss in a neighbourhood of each *training* point, so members look easy not only
at x but also around x, which widens the member/non-member gap that an
adversarial-loss score exploits.

Membership ground truth here:
  - members     = the private training subset (train_idx from the seeded split),
  - non-members = the held-out CIFAR-10 test set.

We report full-ROC metrics with an emphasis on the low-FPR regime (TPR@0.1%/1%),
which is the standard lens since Carlini et al. (LiRA, 2022); a single AUC or
balanced accuracy hides whether an attack confidently re-identifies *any*
members, which is what actually matters for a privacy claim.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from trustml.attacks.pgd import PGDAttack, _set_grad_sample_hooks


def _confidence_scores(logits: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-sample membership signals; higher => looks more like a member.

    - neg_loss: -CE(logits, y). Members are fit with lower loss.
    - correct_conf: softmax prob of the true class.
    - hinge: true-class logit minus the max other logit (Carlini's stable margin).
    """
    log_probs = F.log_softmax(logits, dim=1)
    neg_loss = log_probs.gather(1, y.view(-1, 1)).squeeze(1)  # = -CE per sample
    probs = log_probs.exp()
    correct_conf = probs.gather(1, y.view(-1, 1)).squeeze(1)
    true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y.view(-1, 1), float("-inf"))
    max_other = other.max(dim=1).values
    hinge = true_logit - max_other
    return {"neg_loss": neg_loss, "correct_conf": correct_conf, "hinge": hinge}


class AdversarialLossMIA:
    """Song-style adversarial-loss membership inference over a fixed member split."""

    def __init__(self, cfg, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)) -> None:
        self.max_batches = cfg.get("max_batches", None)
        self.class_calibrated = bool(cfg.get("class_calibrated", True))
        self.fpr_targets = list(cfg.get("fpr_targets", [0.001, 0.01, 0.05]))
        # Adversarial score uses the same PGD as robustness eval unless overridden.
        attack_cfg = dict(cfg.get("pgd", {}) or {})
        attack_cfg.setdefault("epsilon", cfg.get("epsilon", 8.0 / 255.0))
        attack_cfg.setdefault("step_size", cfg.get("step_size", 2.0 / 255.0))
        attack_cfg.setdefault("steps", cfg.get("steps", 20))
        self.pgd = PGDAttack(attack_cfg, mean=mean, std=std)

    def _selection_fpr(self) -> float:
        """FPR at which signals are ranked: 1% when configured, else the closest."""
        targets = [float(t) for t in self.fpr_targets] or [0.01]
        return min(targets, key=lambda t: abs(t - 0.01))

    def _collect(self, model, dataloader, device, with_adv: bool, desc: str):
        scores: dict[str, list[np.ndarray]] = {}
        labels: list[np.ndarray] = []
        iterator = enumerate(dataloader)
        iterator = tqdm(iterator, total=len(dataloader), desc=desc, leave=False)
        for batch_idx, (x, y) in iterator:
            if self.max_batches is not None and batch_idx >= int(self.max_batches):
                break
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                clean = _confidence_scores(model(x), y)
            batch = {f"clean_{k}": v for k, v in clean.items()}
            if with_adv:
                x_adv = self.pgd.generate(model, x, y)
                with torch.no_grad():
                    adv = _confidence_scores(model(x_adv), y)
                batch.update({f"adv_{k}": v for k, v in adv.items()})
            for k, v in batch.items():
                scores.setdefault(k, []).append(v.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
        merged = {k: np.concatenate(v) for k, v in scores.items()}
        return merged, np.concatenate(labels)

    def run(self, model, member_loader, nonmember_loader, device: torch.device) -> dict[str, float]:
        was_training = model.training
        model.eval()
        _set_grad_sample_hooks(model, False)
        try:
            mem, mem_y = self._collect(model, member_loader, device, with_adv=True, desc="mia-member")
            non, non_y = self._collect(model, nonmember_loader, device, with_adv=True, desc="mia-nonmember")
        finally:
            _set_grad_sample_hooks(model, True)
            if was_training:
                model.train()

        metrics: dict[str, float] = {}
        best = None
        for signal in sorted(set(mem.keys()) & set(non.keys())):
            m_raw, n_raw = mem[signal], non[signal]
            if self.class_calibrated:
                m_score = _class_calibrate(m_raw, mem_y, n_raw, non_y)
                n_score = _class_calibrate(n_raw, non_y, n_raw, non_y)
            else:
                m_score, n_score = m_raw, n_raw
            res = _roc_metrics(m_score, n_score, self.fpr_targets)
            for k, v in res.items():
                metrics[f"mia/{signal}/{k}"] = float(v)
            # Rank by TPR at the low-FPR operating point, tie-broken by AUC:
            # at 1% FPR many signals land on the same handful of samples, and
            # an unbroken tie would silently keep whichever came first.
            rank = (res.get(tpr_key(self._selection_fpr()), 0.0), res.get("auc", 0.0))
            if best is None or rank > best[1]:
                best = (signal, rank, res)

        if best is not None:
            signal, _, res = best
            metrics["mia/best_signal_code"] = float(_signal_code(signal))
            for k, v in res.items():
                metrics[f"mia/best/{k}"] = float(v)
        metrics["mia/n_members"] = float(len(mem_y))
        metrics["mia/n_nonmembers"] = float(len(non_y))
        return metrics


def tpr_key(fpr_target: float) -> str:
    """Metric key for TPR at a given FPR.

    Single source of truth for the format. It used to be spelled out twice --
    emitted as "tpr@1%fpr" by _roc_metrics but looked up as "tpr@1.0%fpr" by the
    best-signal selection, so the lookup always missed, every signal ranked 0.0,
    and "mia/best/*" silently reported whichever signal sorted first
    alphabetically instead of the strongest one.
    """
    return f"tpr@{fpr_target * 100:g}%fpr"


def _class_calibrate(raw: np.ndarray, raw_y: np.ndarray, ref: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    """Subtract the per-class non-member mean so a single global threshold is comparable across classes.

    Song et al. threshold per class; centering by the non-member class mean is an
    equivalent monotone reparametrisation that lets us pool all classes into one ROC.
    """
    per_class_mean = {}
    for c in np.unique(ref_y):
        per_class_mean[int(c)] = float(ref[ref_y == c].mean()) if np.any(ref_y == c) else 0.0
    offsets = np.array([per_class_mean.get(int(c), 0.0) for c in raw_y])
    return raw - offsets


def _roc_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr_targets) -> dict[str, float]:
    scores = np.concatenate([member_scores, nonmember_scores])
    is_member = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    order = np.argsort(-scores, kind="mergesort")  # high score => predict member
    y = is_member[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    n_pos = float(member_scores.size)
    n_neg = float(nonmember_scores.size)
    tpr = tp / max(1.0, n_pos)
    fpr = fp / max(1.0, n_neg)

    # np.trapz was removed in numpy 2.0 (np.trapezoid is its replacement);
    # local env runs 1.26 while the DataSphere VM resolves an unpinned numpy 2.x.
    _trapezoid = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trapezoid(tpr, fpr)) if n_neg > 0 and n_pos > 0 else 0.0
    out = {"auc": auc, "attack_advantage": float(np.max(tpr - fpr))}
    for target in fpr_targets:
        idx = np.searchsorted(fpr, target, side="right") - 1
        out[tpr_key(target)] = float(tpr[idx]) if idx >= 0 else 0.0
    # Balanced accuracy at the best threshold, for comparison with older tables.
    out["balanced_acc"] = float(0.5 * (1.0 + np.max(tpr - fpr)))
    return out


def _signal_code(name: str) -> int:
    order = ["clean_neg_loss", "clean_correct_conf", "clean_hinge", "adv_neg_loss", "adv_correct_conf", "adv_hinge"]
    return order.index(name) if name in order else -1
