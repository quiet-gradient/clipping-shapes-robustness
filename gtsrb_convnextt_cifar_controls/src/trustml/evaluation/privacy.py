"""Membership-inference privacy checks against a trained model.

One entry point -- :func:`run_privacy_suite` -- runs every configured MIA
(adversarial-loss, OSLO, offline RMIA) against a model over the datamodule's
seeded member/non-member split. It is shared by ``Trainer.evaluate`` (end of a
training run) and the standalone ``trustml.privacy_eval`` CLI (scoring an
existing checkpoint), so both report an identical metric surface. Each attack
self-guards to ``{}`` unless enabled, and a failure in one is isolated as
``eval/<name>_error=1`` without blocking the others.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from trustml.attacks.oslo import OsloAttack
from trustml.attacks.oslo_surrogates import DEFAULT_WEIGHTS_DIR, load_source_and_validation_models


def evaluate_adversarial_loss_mia(model, data: dict, cfg, device: torch.device) -> dict[str, float]:
    """Song-style adversarial-loss MIA; no-op unless attacks.mia.enabled."""
    mia_cfg = (cfg.get("attacks", {}) or {}).get("mia", None)
    if not mia_cfg or not bool(mia_cfg.get("enabled", False)):
        return {}
    if "mia_members" not in data or "test" not in data:
        return {}
    from trustml.attacks.adversarial_loss_mia import AdversarialLossMIA

    mean = tuple(cfg.data.get("mean", (0.4914, 0.4822, 0.4465)))
    std = tuple(cfg.data.get("std", (0.2023, 0.1994, 0.2010)))
    attack = AdversarialLossMIA(mia_cfg, mean=mean, std=std)
    return attack.run(model, data["mia_members"], data["test"], device)


def run_privacy_suite(
    model,
    data: dict,
    cfg,
    device: torch.device,
    references_dir: str | None = None,
    shadow_dir: str | None = None,
) -> dict[str, float]:
    """Run every enabled membership-inference attack and merge their metrics.

    ``references_dir`` overrides ``attacks.rmia.references_dir`` and
    ``shadow_dir`` overrides ``attacks.lira.shadow_dir`` (both used by the CLI).
    RMIA and LiRA consume the same shadow-model artifacts, so passing one
    directory for both is the normal case. Attacks are optional diagnostics: a
    broken one is recorded as ``eval/<name>_error=1`` and never blocks the others.
    """
    from trustml.attacks.lira import run_lira_from_split
    from trustml.attacks.rmia import run_rmia_from_split

    metrics: dict[str, float] = {}

    def _safe(name: str, fn) -> None:
        try:
            metrics.update(fn())
        except Exception:
            print(f"{name} evaluation failed:\n{traceback.format_exc()}", file=sys.stderr)
            metrics[f"eval/{name}_error"] = 1.0

    _safe("mia", lambda: evaluate_adversarial_loss_mia(model, data, cfg, device))
    _safe("oslo", lambda: evaluate_oslo_mia(model, data, cfg, device))
    _safe("rmia", lambda: run_rmia_from_split(model, data, cfg, device, references_dir=references_dir))
    _safe("lira", lambda: run_lira_from_split(model, data, cfg, device, shadow_dir=shadow_dir))
    return metrics


def _parse_tau_list(raw) -> list[float]:
    """OSLO confidence thresholds from a comma-string, a scalar, or a sequence."""
    if isinstance(raw, str):
        return [float(value.strip()) for value in raw.split(",") if value.strip()]
    if isinstance(raw, (int, float)):
        return [float(raw)]
    return [float(value) for value in raw]


def evaluate_oslo_mia(model, data: dict, cfg, device: torch.device) -> dict[str, float]:
    attacks_cfg = cfg.get("attacks", None)
    oslo_cfg = attacks_cfg.get("oslo", None) if attacks_cfg is not None else None
    if oslo_cfg is None or not bool(oslo_cfg.get("enabled", False)):
        return {}

    taus = _parse_tau_list(oslo_cfg.get("tau_list", "0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001,0.00001,0.00003"))

    source_models, validation_models = load_source_and_validation_models(
        device=device,
        weights_dir=oslo_cfg.get("weights_dir", str(DEFAULT_WEIGHTS_DIR)),
    )
    # Members: the same seeded private training subset the other MIAs use,
    # under the deterministic eval transform ("mia_members"); random
    # crops/flips from the train loader would perturb the very inputs whose
    # adversarial neighbourhood the attack measures. "train" stays as a
    # fallback for callers without the member split.
    member_source = data.get("mia_members", data["train"])
    member_ds = member_source.dataset if hasattr(member_source, "dataset") else member_source
    non_member_ds = data["test"].dataset if hasattr(data["test"], "dataset") else data["test"]

    n_pos = len(member_ds)
    n_neg = len(non_member_ds)
    n = min(n_pos, n_neg)
    sample_size = oslo_cfg.get("sample_size", 100)
    if sample_size is not None and int(sample_size) < n:
        n = int(sample_size)

    generator = torch.Generator().manual_seed(int(oslo_cfg.get("balance_seed", 42)))
    pos_idx = torch.randperm(n_pos, generator=generator)[:n].tolist()
    neg_idx = torch.randperm(n_neg, generator=generator)[:n].tolist()
    attack_dataset = ConcatDataset([Subset(member_ds, pos_idx), Subset(non_member_ds, neg_idx)])
    dataloader = DataLoader(
        attack_dataset,
        batch_size=int(oslo_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(oslo_cfg.get("num_workers", 0)),
    )

    metrics: dict[str, float] = {}
    was_training = model.training
    model.eval()
    model.to(device)

    for tau in taus:
        attack = OsloAttack(
            source_models=source_models,
            validation_models=validation_models,
            eps=float(oslo_cfg.get("eps", 80.0 / 255)),
            eps_step=float(oslo_cfg.get("eps_step", 2.0 / 255)),
            max_iter=int(oslo_cfg.get("max_iter", 10)),
            num_subprocedures=int(oslo_cfg.get("num_subprocedures", 80)),
            tau=tau,
            device=device,
        )
        pred_parts: list[np.ndarray] = []
        true_parts: list[np.ndarray] = []
        processed = 0

        for images, labels in dataloader:
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device).long()
            x_adv = attack.generate(images, labels)
            with torch.no_grad():
                preds = model(x_adv).argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            true_labels = labels.detach().cpu().numpy().astype(np.int64)
            pred_parts.append((preds == true_labels).astype(np.int8))
            idx = np.arange(processed, processed + batch_size)
            true_parts.append((idx < n).astype(np.int8))
            processed += batch_size

        if not pred_parts:
            continue
        membership_preds = np.concatenate(pred_parts, axis=0)
        true_membership = np.concatenate(true_parts, axis=0)
        tp = int(((membership_preds == 1) & (true_membership == 1)).sum())
        tn = int(((membership_preds == 0) & (true_membership == 0)).sum())
        fp = int(((membership_preds == 1) & (true_membership == 0)).sum())
        fn = int(((membership_preds == 0) & (true_membership == 1)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        tau_str = str(tau).replace(".", "_")
        raw_metrics = {
            "tau": tau,
            "processed": int(membership_preds.shape[0]),
            "balanced_n_per_class": int(n),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "tpr": float(recall),
            "fpr": float(fpr),
            "f1_score": float(f1),
        }
        for key, value in raw_metrics.items():
            metrics[f"OSLO_tau{tau_str}_{key}"] = float(value)

    if was_training:
        model.train()
    return metrics
