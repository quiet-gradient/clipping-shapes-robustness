"""LiRA -- the Likelihood Ratio Attack (Carlini et al., IEEE S&P 2022).

"Membership Inference Attacks From First Principles" (arXiv:2112.03570). LiRA is
the reference membership-inference attack in the literature and the one that
established low-FPR reporting as the standard. Its power comes from *per-example
difficulty calibration*: instead of thresholding a raw loss (where intrinsically
hard examples masquerade as non-members), it models, for each example x, the
distribution of the model's confidence when x was in training (IN) and when it
was not (OUT), then scores the target by the likelihood ratio between them.

Confidence is measured in logit space,

    phi(x) = log( p_y / (1 - p_y) ),

computed stably from raw logits as ``logit_y - logsumexp_{j != y} logit_j``.
Carlini et al. show this transform makes the IN/OUT distributions approximately
Gaussian, which is what licenses the parametric test:

    online:   Lambda(x) = log N(phi_target; mu_in, s_in^2) - log N(phi_target; mu_out, s_out^2)
    offline:  Lambda(x) = (phi_target - mu_out) / s_out          # one-sided, OUT models only

Querying each model on several augmentations of x (identity, flip, shifts) and
summing the per-augmentation log-densities sharpens the test further; we treat
augmentations as independent Gaussian coordinates (diagonal covariance), as in
the paper's multi-query variant.

Variance handling matters when shadows are few: with ``global_variance=True``
the per-example variances are pooled across the dataset, which is far more
stable than estimating a variance per example from a handful of shadows.

IMPORTANT -- the evaluation split. Online LiRA needs both IN and OUT shadows for
*every* scored example, so members and non-members must both come from the pool
the shadows were sampled from. Using the CIFAR-10 test set as non-members (as
the offline RMIA path does) leaves them OUT of every shadow and makes mu_in
undefined; the correct non-member set is the held-out slice of the train pool
(the datamodule's ``val`` split). Offline LiRA has no such constraint.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# Shared ROC/TPR-at-low-FPR reducer, so LiRA, RMIA and the adversarial-loss MIA
# all report an identical metric surface (auc, tpr@x%fpr, attack_advantage, ...).
from trustml.attacks.adversarial_loss_mia import _roc_metrics

_VARIANCE_FLOOR = 1e-8


def augmentation_ops(num_augmentations: int) -> list[str]:
    """Query-time augmentations, in a fixed order: identity, flip, then shifts."""
    ops = ["identity", "flip", "shift_+2_0", "shift_-2_0", "shift_0_+2", "shift_0_-2",
           "shift_+2_+2", "shift_-2_-2"]
    n = max(1, int(num_augmentations))
    if n > len(ops):
        raise ValueError(f"num_augmentations={n} exceeds the {len(ops)} defined ops.")
    return ops[:n]


def apply_augmentation(x: torch.Tensor, op: str) -> torch.Tensor:
    """Apply one query-time augmentation to an already-normalised batch."""
    if op == "identity":
        return x
    if op == "flip":
        return torch.flip(x, dims=[3])
    if op.startswith("shift"):
        _, dh, dw = op.split("_")
        return torch.roll(x, shifts=(int(dh), int(dw)), dims=(2, 3))
    raise ValueError(f"Unknown augmentation op: {op}")


def logit_scaled_confidence(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """phi = log(p_y / (1 - p_y)), computed stably from raw logits.

    Equivalent to logit(softmax(logits)[y]) but avoids forming p_y and then
    dividing by 1 - p_y, which loses all precision once p_y -> 1 (exactly the
    regime memorised training points live in).
    """
    true_logit = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y.view(-1, 1), float("-inf"))
    return true_logit - torch.logsumexp(other, dim=1)


@torch.no_grad()
def collect_phis(model: torch.nn.Module, loader, device: torch.device, ops: Sequence[str],
                 max_batches: int | None = None) -> np.ndarray:
    """Logit-scaled confidences for one model: [num_augmentations, num_samples].

    The loader must be un-shuffled so positions line up across models and with
    the membership mask.
    """
    was_training = model.training
    model.eval()
    per_op: list[list[np.ndarray]] = [[] for _ in ops]
    try:
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            x = x.to(device)
            y = y.to(device)
            for i, op in enumerate(ops):
                phi = logit_scaled_confidence(model(apply_augmentation(x, op)), y)
                per_op[i].append(phi.detach().cpu().numpy())
    finally:
        if was_training:
            model.train()
    if not per_op[0]:
        return np.zeros((len(ops), 0), dtype=np.float64)
    return np.stack([np.concatenate(chunks).astype(np.float64) for chunks in per_op], axis=0)


def collect_shadow_phis(models: Sequence[torch.nn.Module], loader, device: torch.device,
                        ops: Sequence[str], max_batches: int | None = None) -> np.ndarray:
    """Stack per-model confidences: [num_shadows, num_augmentations, num_samples]."""
    rows = [collect_phis(m, loader, device, ops, max_batches=max_batches) for m in models]
    if not rows:
        return np.zeros((0, len(ops), 0), dtype=np.float64)
    return np.stack(rows, axis=0)


def _masked_moments(phi: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-example mean/variance over the selected shadows.

    phi: [K, A, N]; mask: [K, N] bool. Returns (mu, var, count) shaped
    [A, N], [A, N], [N]. Entries with count 0 come back as NaN.
    """
    weights = mask[:, None, :].astype(np.float64)  # [K, 1, N]
    count = weights.sum(axis=0)[0]  # [N]
    safe = np.where(count > 0, count, 1.0)
    mu = (phi * weights).sum(axis=0) / safe
    var = (((phi - mu[None, :, :]) ** 2) * weights).sum(axis=0) / safe
    empty = count == 0
    if empty.any():
        mu[:, empty] = np.nan
        var[:, empty] = np.nan
    return mu, var, count


def _pooled_sigma(var: np.ndarray, global_variance: bool) -> np.ndarray:
    """Standard deviation per augmentation (pooled) or per example.

    Examples with no shadow in the group carry NaN variance; they are excluded
    from the pooled estimate here and their scores are marked NaN by the
    callers, so an all-NaN slice is an expected degenerate case, not an error.
    """
    if global_variance:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice
            pooled = np.nanmean(var, axis=1, keepdims=True)  # [A, 1]
        pooled = np.where(np.isfinite(pooled), pooled, _VARIANCE_FLOOR)
        return np.sqrt(np.maximum(pooled, _VARIANCE_FLOOR))
    return np.sqrt(np.maximum(var, _VARIANCE_FLOOR))


def _log_normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2.0 * np.pi * sigma**2) - ((x - mu) ** 2) / (2.0 * sigma**2)


def lira_offline_scores(phi_target: np.ndarray, phi_shadow: np.ndarray, out_mask: np.ndarray,
                        global_variance: bool = True) -> np.ndarray:
    """One-sided OUT-only score; higher => more member-like.

    Summed over augmentations, this is the z-score of the target's confidence
    under the OUT distribution -- monotone in the one-sided p-value, so the ROC
    is identical to thresholding 1 - Phi(z).
    """
    mu_out, var_out, count = _masked_moments(phi_shadow, out_mask)
    sigma_out = _pooled_sigma(var_out, global_variance)
    z = (phi_target - mu_out) / sigma_out
    scores = np.nansum(z, axis=0)
    scores[count == 0] = np.nan
    return scores


def lira_online_scores(phi_target: np.ndarray, phi_shadow: np.ndarray, in_mask: np.ndarray,
                       global_variance: bool = True) -> tuple[np.ndarray, int]:
    """Likelihood-ratio score against per-example IN and OUT Gaussians.

    Returns (scores, num_undefined) where undefined entries (an example with no
    IN or no OUT shadow) are NaN and must be handled by the caller.
    """
    mu_in, var_in, count_in = _masked_moments(phi_shadow, in_mask)
    mu_out, var_out, count_out = _masked_moments(phi_shadow, ~in_mask)
    sigma_in = _pooled_sigma(var_in, global_variance)
    sigma_out = _pooled_sigma(var_out, global_variance)

    log_in = _log_normal_pdf(phi_target, mu_in, sigma_in)
    log_out = _log_normal_pdf(phi_target, mu_out, sigma_out)
    scores = np.nansum(log_in - log_out, axis=0)
    undefined = (count_in == 0) | (count_out == 0)
    scores[undefined] = np.nan
    return scores, int(undefined.sum())


def _finite_pair(member: np.ndarray, nonmember: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return member[np.isfinite(member)], nonmember[np.isfinite(nonmember)]


def evaluate_lira(
    target_model: torch.nn.Module,
    shadow_models: Sequence[torch.nn.Module],
    member_loader,
    nonmember_loader,
    in_mask_members: np.ndarray,
    device: torch.device,
    *,
    in_mask_nonmembers: np.ndarray | None = None,
    num_augmentations: int = 2,
    global_variance: bool = True,
    fpr_targets: Iterable[float] = (0.001, 0.01, 0.05),
    max_batches: int | None = None,
) -> dict[str, float]:
    """Score members vs non-members with LiRA and return ROC metrics.

    ``in_mask_members[r, i]`` is True when shadow r trained on member i. Pass
    ``in_mask_nonmembers`` when the non-members also come from the shadow pool
    (the ``val`` split); without it only the offline attack is reported, since
    the online test needs an IN distribution for every scored example.
    """
    if len(shadow_models) == 0:
        raise ValueError("LiRA needs at least one shadow model.")
    fpr_targets = list(fpr_targets)
    ops = augmentation_ops(num_augmentations)

    phi_t_mem = collect_phis(target_model, member_loader, device, ops, max_batches)
    phi_t_non = collect_phis(target_model, nonmember_loader, device, ops, max_batches)
    phi_s_mem = collect_shadow_phis(shadow_models, member_loader, device, ops, max_batches)
    phi_s_non = collect_shadow_phis(shadow_models, nonmember_loader, device, ops, max_batches)

    n_mem = phi_t_mem.shape[1]
    n_non = phi_t_non.shape[1]
    # collect_phis and the masks both follow loader order, so truncating is safe
    # when max_batches shortens the stream.
    mask_mem = np.asarray(in_mask_members, dtype=bool)[:, :n_mem]
    if mask_mem.shape != (phi_s_mem.shape[0], n_mem):
        raise ValueError(
            f"member mask {mask_mem.shape} does not match shadow phis "
            f"(expected {(phi_s_mem.shape[0], n_mem)})"
        )

    metrics: dict[str, float] = {}

    # Offline: OUT models only, valid for any non-member set.
    off_mem = lira_offline_scores(phi_t_mem, phi_s_mem, ~mask_mem, global_variance)
    if in_mask_nonmembers is not None:
        mask_non = np.asarray(in_mask_nonmembers, dtype=bool)[:, :n_non]
        off_non = lira_offline_scores(phi_t_non, phi_s_non, ~mask_non, global_variance)
    else:
        mask_non = None
        off_non = lira_offline_scores(phi_t_non, phi_s_non, np.ones((len(shadow_models), n_non), dtype=bool),
                                      global_variance)
    m_off, n_off = _finite_pair(off_mem, off_non)
    for key, value in _roc_metrics(m_off, n_off, fpr_targets).items():
        metrics[f"lira/offline/{key}"] = float(value)

    # Online: needs an IN distribution on both sides.
    if mask_non is not None:
        on_mem, undef_mem = lira_online_scores(phi_t_mem, phi_s_mem, mask_mem, global_variance)
        on_non, undef_non = lira_online_scores(phi_t_non, phi_s_non, mask_non, global_variance)
        m_on, n_on = _finite_pair(on_mem, on_non)
        for key, value in _roc_metrics(m_on, n_on, fpr_targets).items():
            metrics[f"lira/online/{key}"] = float(value)
        metrics["lira/online_undefined_members"] = float(undef_mem)
        metrics["lira/online_undefined_nonmembers"] = float(undef_non)
    else:
        metrics["lira/online_skipped_no_pool_nonmembers"] = 1.0

    metrics["lira/n_members"] = float(n_mem)
    metrics["lira/n_nonmembers"] = float(n_non)
    metrics["lira/n_shadows"] = float(len(shadow_models))
    metrics["lira/num_augmentations"] = float(len(ops))
    metrics["lira/global_variance"] = 1.0 if global_variance else 0.0
    return metrics


def run_lira_from_split(model, data: dict, cfg, device: torch.device,
                        shadow_dir: str | Path | None = None) -> dict[str, float]:
    """Opt-in LiRA for a live run or the standalone CLI; no-op unless prepared.

    Members are the target's seeded training subset. Non-members are the
    held-out ``val`` slice of the same pool, so both sides have IN and OUT
    shadows and the online attack is well defined.
    """
    lira_cfg = (cfg.get("attacks", {}) or {}).get("lira", None)
    if not lira_cfg or not bool(lira_cfg.get("enabled", False)):
        return {}
    if "mia_members" not in data or "val" not in data:
        return {}
    directory = shadow_dir or lira_cfg.get("shadow_dir", None)
    if not directory or not (Path(directory) / "manifest.json").exists():
        return {}

    from trustml.attacks.shadow_models import in_mask_for_indices, load_shadow_models
    from trustml.data.splits import train_val_indices

    # Each shadow checkpoint embeds the recipe and architecture it was trained
    # with. Never merge the target config here: doing so can silently turn a
    # non-DP BatchNorm shadow into the target's DP GroupNorm architecture.
    shadow_models, masks, manifest = load_shadow_models(
        directory, device, max_models=lira_cfg.get("max_models", None)
    )
    pool_size = int(manifest["pool_size"])
    data_seed = int(cfg.data.get("seed", 42))
    train_ratio = float(cfg.data.get("train_ratio", 0.9))
    train_idx, val_idx = train_val_indices(pool_size, train_ratio, data_seed)
    return evaluate_lira(
        model,
        shadow_models,
        data["mia_members"],
        data["val"],
        in_mask_for_indices(masks, train_idx),
        device,
        in_mask_nonmembers=in_mask_for_indices(masks, val_idx),
        num_augmentations=int(lira_cfg.get("num_augmentations", 2)),
        global_variance=bool(lira_cfg.get("global_variance", True)),
        fpr_targets=list(lira_cfg.get("fpr_targets", [0.001, 0.01, 0.05])),
        max_batches=lira_cfg.get("max_batches", None),
    )
