"""Robust Membership Inference Attack (RMIA), offline variant.

Zarifzadeh, Liu & Shokri, "Low-Cost High-Power Membership Inference Attacks"
(ICML'24, arXiv:2312.03262). RMIA is a per-example likelihood-ratio test that
reaches LiRA-level power at low FPR with only a handful of reference models,
instead of the 64+ shadow models LiRA needs.

Core idea. For a target sample x with true label y and target model theta, the
membership signal is

    LR(x) = Pr(x | theta) / Pr(x),

where Pr(x | theta) is the model's softmax probability on the true label and
Pr(x) is the same probability marginalised over models that did NOT train on x.
The online attack estimates Pr(x) = 0.5*(Pr(x|IN) + Pr(x|OUT)) from reference
models trained with and without x. The *offline* attack (what we implement) only
uses OUT reference models and approximates the IN branch with a linear
correction controlled by a single coefficient a in [0, 1]:

    Pr(x) ~= 0.5 * ((1 + a) * p_out(x) + (1 - a)),      # eq. (offline)

with p_out(x) the mean OUT-reference probability. a=1 trusts the OUT estimate
directly (Pr = p_out); a=0 assumes membership pushes the probability to 1
(Pr = (p_out + 1)/2). a is a hyperparameter tuned on the reference models
(~0.3 for CIFAR-10 in the paper).

Membership score. RMIA does not threshold LR(x) directly; it counts, over a
population Z of fresh samples z, the fraction whose pairwise ratio clears a
threshold gamma:

    score(x) = (1/|Z|) * #{ z in Z : LR(x) / LR(z) >= gamma }.

At gamma=1 this is the empirical CDF of LR(x) under the population, so its ROC
matches ranking by LR(x); gamma>1 sharpens the low-FPR regime. We report both
the population-count score (`rmia/offline/*`) and the raw LR (`rmia/lr/*`).

This module is pure scoring: it consumes a target model and already-trained
reference models (see `shadow_models.py`) and never trains anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# ROC/TPR-at-low-FPR reducer shared with the adversarial-loss MIA so both attacks
# report an identical metric surface (auc, tpr@x%fpr, attack_advantage, ...).
from trustml.attacks.adversarial_loss_mia import _roc_metrics


@torch.no_grad()
def collect_true_class_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probability of the true label for every sample, in loader order.

    The loader must be un-shuffled so positions line up across the target and
    reference models (and, for members, with the reference OUT-mask).
    """
    was_training = model.training
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    try:
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            x = x.to(device)
            y = y.to(device)
            p = torch.softmax(model(x), dim=1)
            true_p = p.gather(1, y.view(-1, 1)).squeeze(1)
            probs.append(true_p.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    finally:
        if was_training:
            model.train()
    if not probs:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    return np.concatenate(probs).astype(np.float64), np.concatenate(labels).astype(np.int64)


def out_reference_mean(ref_probs: np.ndarray, out_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Mean reference probability over OUT models, per sample.

    ref_probs, out_mask: [num_refs, num_samples]. Samples with no OUT reference
    (member seen by every reference) fall back to the all-reference mean; the
    count of such samples is returned so the caller can flag a too-small/too-full
    reference set.
    """
    if ref_probs.ndim != 2 or out_mask.shape != ref_probs.shape:
        raise ValueError(f"shape mismatch: ref_probs {ref_probs.shape}, out_mask {out_mask.shape}")
    mask = out_mask.astype(np.float64)
    denom = mask.sum(axis=0)
    numer = (ref_probs * mask).sum(axis=0)
    has_out = denom > 0
    all_mean = ref_probs.mean(axis=0)
    p_out = np.where(has_out, numer / np.where(has_out, denom, 1.0), all_mean)
    return p_out, int((~has_out).sum())


def offline_pr(p_out: np.ndarray, a: float) -> np.ndarray:
    """Offline RMIA estimate of the marginal Pr(x) from OUT-reference means."""
    a = float(a)
    return 0.5 * ((1.0 + a) * p_out + (1.0 - a))


def likelihood_ratio(target_probs: np.ndarray, pr_x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """LR(x) = Pr(x | theta) / Pr(x), guarded against division by ~0."""
    return target_probs / np.clip(pr_x, eps, None)


def rmia_population_score(lr_query: np.ndarray, lr_population: np.ndarray, gamma: float) -> np.ndarray:
    """Fraction of population z with LR(query)/LR(z) >= gamma, per query.

    Equivalent to the fraction of population LRs at or below LR(query)/gamma,
    computed in O((Nq + Np) log Np) via a single sort + searchsorted.
    """
    gamma = float(gamma)
    if lr_population.size == 0:
        return np.zeros_like(lr_query)
    sorted_pop = np.sort(lr_population)
    thresh = lr_query / gamma
    counts = np.searchsorted(sorted_pop, thresh, side="right")
    return counts.astype(np.float64) / float(sorted_pop.size)


def collect_reference_probs(
    reference_models: Sequence[torch.nn.Module],
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> np.ndarray:
    """Stack true-class probabilities from every reference model: [num_refs, N]."""
    rows = [collect_true_class_probs(m, loader, device, max_batches=max_batches)[0] for m in reference_models]
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0)


def evaluate_rmia(
    target_model: torch.nn.Module,
    reference_models: Sequence[torch.nn.Module],
    member_loader,
    nonmember_loader,
    out_mask_members: np.ndarray,
    device: torch.device,
    *,
    a: float = 0.3,
    gamma: float = 1.0,
    fpr_targets: Iterable[float] = (0.001, 0.01, 0.05),
    max_batches: int | None = None,
    population_loader=None,
) -> dict[str, float]:
    """Score members vs non-members with offline RMIA and return ROC metrics.

    out_mask_members[r, i] is True when member i was NOT in reference r's training
    set. Non-members (the test set) are OUT of every reference by construction, so
    their marginal uses all references. The population Z for the pairwise ratio is
    the non-member set by default, or `population_loader` if given (which must be
    disjoint from every reference's training data for its OUT means to be valid).
    """
    fpr_targets = list(fpr_targets)
    if len(reference_models) == 0:
        raise ValueError("RMIA needs at least one reference model.")

    target_mem, mem_y = collect_true_class_probs(target_model, member_loader, device, max_batches)
    target_non, non_y = collect_true_class_probs(target_model, nonmember_loader, device, max_batches)
    ref_mem = collect_reference_probs(reference_models, member_loader, device, max_batches)
    ref_non = collect_reference_probs(reference_models, nonmember_loader, device, max_batches)

    n_mem = target_mem.shape[0]
    n_non = target_non.shape[0]
    # collect_true_class_probs and the mask both follow loader order, so aligning
    # by truncation is safe when max_batches shortens the member stream.
    mask = np.asarray(out_mask_members, dtype=bool)[:, :n_mem]
    if mask.shape != ref_mem.shape:
        raise ValueError(f"member mask {mask.shape} does not match reference probs {ref_mem.shape}")

    p_out_mem, members_without_out = out_reference_mean(ref_mem, mask)
    p_out_non = ref_non.mean(axis=0)

    pr_mem = offline_pr(p_out_mem, a)
    pr_non = offline_pr(p_out_non, a)
    lr_mem = likelihood_ratio(target_mem, pr_mem)
    lr_non = likelihood_ratio(target_non, pr_non)

    if population_loader is not None:
        pop_target, _ = collect_true_class_probs(target_model, population_loader, device, max_batches)
        pop_ref = collect_reference_probs(reference_models, population_loader, device, max_batches)
        lr_pop = likelihood_ratio(pop_target, offline_pr(pop_ref.mean(axis=0), a))
    else:
        lr_pop = lr_non

    score_mem = rmia_population_score(lr_mem, lr_pop, gamma)
    score_non = rmia_population_score(lr_non, lr_pop, gamma)

    metrics: dict[str, float] = {}
    for key, res in (("offline", _roc_metrics(score_mem, score_non, fpr_targets)),
                     ("lr", _roc_metrics(lr_mem, lr_non, fpr_targets))):
        for name, value in res.items():
            metrics[f"rmia/{key}/{name}"] = float(value)
    metrics["rmia/n_members"] = float(n_mem)
    metrics["rmia/n_nonmembers"] = float(n_non)
    metrics["rmia/n_references"] = float(len(reference_models))
    metrics["rmia/offline_a"] = float(a)
    metrics["rmia/gamma"] = float(gamma)
    metrics["rmia/members_without_out_ref"] = float(members_without_out)
    return metrics


def run_rmia_from_split(
    model, data: dict, cfg, device: torch.device, references_dir: str | Path | None = None
) -> dict[str, float]:
    """Opt-in RMIA for a live training run; no-op unless configured and prepared.

    Returns {} (rather than raising) when RMIA is disabled, the reference dir is
    absent, or the member/test loaders are missing, so wiring this into
    Trainer.evaluate never disturbs a normal run. The member OUT-mask is rebuilt
    from the datamodule's seeded split, matching what `trustml.rmia_eval` does.

    ``references_dir`` overrides ``attacks.rmia.references_dir`` from the config
    (used by the standalone CLI to point at a reference dir without editing the
    checkpoint's embedded config).
    """
    rmia_cfg = (cfg.get("attacks", {}) or {}).get("rmia", None)
    if not rmia_cfg or not bool(rmia_cfg.get("enabled", False)):
        return {}
    if "mia_members" not in data or "test" not in data:
        return {}
    ref_dir = references_dir or rmia_cfg.get("references_dir", None)
    if not ref_dir or not (Path(ref_dir) / "manifest.json").exists():
        return {}

    from trustml.attacks.shadow_models import load_shadow_models, out_mask_for_indices
    from trustml.data.splits import train_val_indices

    # Reference checkpoints are self-describing. Applying the target config to
    # them changes non-DP BatchNorm shadows into DP GroupNorm models and
    # invalidates their scores.
    reference_models, masks, manifest = load_shadow_models(
        ref_dir, device, max_models=rmia_cfg.get("max_models", None)
    )
    pool_size = int(manifest["pool_size"])
    data_seed = int(cfg.data.get("seed", 42))
    train_ratio = float(cfg.data.get("train_ratio", 0.9))
    train_idx, _ = train_val_indices(pool_size, train_ratio, data_seed)
    out_mask = out_mask_for_indices(masks, train_idx)
    return evaluate_rmia(
        model,
        reference_models,
        data["mia_members"],
        data["test"],
        out_mask,
        device,
        a=float(rmia_cfg.get("offline_a", 0.3)),
        gamma=float(rmia_cfg.get("gamma", 1.0)),
        fpr_targets=list(rmia_cfg.get("fpr_targets", [0.001, 0.01, 0.05])),
        max_batches=rmia_cfg.get("max_batches", None),
    )


class RMIAAttack:
    """Config-driven wrapper around :func:`evaluate_rmia`.

    Held separate from the input-space attacks in `attacks.registry` (PGD/OSLO):
    RMIA scores a whole member/non-member split against reference models rather
    than perturbing a single batch, so it is driven by `trustml.rmia_eval` (or the
    opt-in Trainer hook) instead of the robustness eval loop.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.a = float(cfg.get("offline_a", 0.3))
        self.gamma = float(cfg.get("gamma", 1.0))
        self.max_batches = cfg.get("max_batches", None)
        self.fpr_targets = list(cfg.get("fpr_targets", [0.001, 0.01, 0.05]))

    def run(
        self,
        target_model,
        reference_models,
        member_loader,
        nonmember_loader,
        out_mask_members,
        device,
        population_loader=None,
    ) -> dict[str, float]:
        return evaluate_rmia(
            target_model,
            reference_models,
            member_loader,
            nonmember_loader,
            out_mask_members,
            device,
            a=self.a,
            gamma=self.gamma,
            fpr_targets=self.fpr_targets,
            max_batches=self.max_batches,
            population_loader=population_loader,
        )
