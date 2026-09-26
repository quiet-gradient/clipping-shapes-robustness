from __future__ import annotations

import math
from typing import Any

import torch


def grad_norm(grads: list[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum().item())
    return total**0.5


def param_grad_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return total**0.5


def trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def clear_private_optimizer_state(optimizer) -> None:
    for param in optimizer.params:
        param.grad = None
        if hasattr(param, "grad_sample"):
            param.grad_sample = None
        if hasattr(param, "summed_grad"):
            param.summed_grad = None


def per_sample_clip_diagnostics(optimizer) -> dict[str, float]:
    per_param_norms = []
    with torch.no_grad():
        for param in optimizer.params:
            grad_sample = getattr(param, "grad_sample", None)
            if grad_sample is None:
                continue
            if isinstance(grad_sample, torch.Tensor):
                flat_grad_sample = grad_sample
            elif isinstance(grad_sample, list) and grad_sample:
                flat_grad_sample = torch.cat(grad_sample, dim=0)
            else:
                continue

            per_param_norms.append(flat_grad_sample.reshape(len(flat_grad_sample), -1).norm(2, dim=-1))

        if not per_param_norms:
            return {
                "preclip_sample_count": 0.0,
                "preclip_norm_mean": 0.0,
                "preclip_norm_p50": 0.0,
                "preclip_norm_p90": 0.0,
                "preclip_norm_p99": 0.0,
                "preclip_norm_max": 0.0,
                "preclip_clip_fraction": 0.0,
                "preclip_mean_clip_factor": 0.0,
                "preclip_min_clip_factor": 0.0,
            }

        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        max_grad_norm = float(optimizer.max_grad_norm)
        clip_factors = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)
        quantiles = torch.quantile(
            per_sample_norms.float(),
            torch.tensor([0.5, 0.9, 0.99], device=per_sample_norms.device),
        )

        return {
            "preclip_sample_count": float(per_sample_norms.numel()),
            "preclip_norm_mean": float(per_sample_norms.mean().item()),
            "preclip_norm_p50": float(quantiles[0].item()),
            "preclip_norm_p90": float(quantiles[1].item()),
            "preclip_norm_p99": float(quantiles[2].item()),
            "preclip_norm_max": float(per_sample_norms.max().item()),
            "preclip_clip_fraction": float((per_sample_norms > max_grad_norm).float().mean().item()),
            "preclip_mean_clip_factor": float(clip_factors.mean().item()),
            "preclip_min_clip_factor": float(clip_factors.min().item()),
        }


def dp_grads_from_current_samples(
    optimizer,
    run_step_hook: bool = True,
    noise_multiplier: float | None = None,
    max_grad_norm: float | None = None,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    previous_noise_multiplier = optimizer.noise_multiplier
    previous_max_grad_norm = optimizer.max_grad_norm
    if noise_multiplier is not None:
        optimizer.noise_multiplier = float(noise_multiplier)
    if max_grad_norm is not None:
        optimizer.max_grad_norm = float(max_grad_norm)

    try:
        preclip = per_sample_clip_diagnostics(optimizer)
        optimizer.clip_and_accumulate()
        accumulated_iterations = max(1, int(optimizer.accumulated_iterations))
        grad_scale = 1.0
        if getattr(optimizer, "loss_reduction", "mean") == "mean":
            grad_scale = float(optimizer.expected_batch_size) * float(accumulated_iterations)

        signal_sq = 0.0
        param_count = 0
        for param in optimizer.params:
            param_count += param.numel()
            if getattr(param, "summed_grad", None) is not None:
                summed_grad = param.summed_grad.detach()
                signal_sq += float(summed_grad.pow(2).sum().item()) / (grad_scale * grad_scale)

        optimizer.add_noise()
        optimizer.scale_grad()

        noise_sq = 0.0
        noised_sq = 0.0
        for param in optimizer.params:
            if param.grad is None:
                continue
            noised_grad = param.grad.detach()
            noised_sq += float(noised_grad.pow(2).sum().item())
            if getattr(param, "summed_grad", None) is not None:
                signal_grad = param.summed_grad.detach().view_as(noised_grad) / grad_scale
                noise_grad = noised_grad - signal_grad
                noise_sq += float(noise_grad.pow(2).sum().item())

        max_norm = float(optimizer.max_grad_norm)
        expected_noise_norm = (
            float(optimizer.noise_multiplier) * max_norm * (float(param_count) ** 0.5) / grad_scale
        )
        signal_norm = signal_sq**0.5
        noise_norm = noise_sq**0.5
        noised_norm = noised_sq**0.5
        diagnostics = {
            "signal_norm": signal_norm,
            "noise_norm": noise_norm,
            "expected_noise_norm": expected_noise_norm,
            "noised_norm": noised_norm,
            "snr": signal_norm / (noise_norm + 1e-12),
            "signal_to_noised_ratio": signal_norm / (noised_norm + 1e-12),
        }
        diagnostics.update(preclip)

        step_hook = getattr(optimizer, "step_hook", None)
        if run_step_hook and step_hook is not None:
            step_hook(optimizer)

        grads = [param.grad.detach().clone() if param.grad is not None else None for param in optimizer.params]
        clear_private_optimizer_state(optimizer)
        return grads, diagnostics
    finally:
        optimizer.noise_multiplier = previous_noise_multiplier
        optimizer.max_grad_norm = previous_max_grad_norm


def clipped_grad_sums_from_current_samples(
    optimizer,
    max_grad_norm: float,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Per-sample clip at ``max_grad_norm`` and return summed clipped grads.

    Unlike :func:`dp_grads_from_current_samples` this adds no noise, records no
    accountant step, and returns the *unscaled* sums (strategy D noises the
    combined channels once, later).
    """
    previous_max_grad_norm = optimizer.max_grad_norm
    optimizer.max_grad_norm = float(max_grad_norm)
    try:
        preclip = per_sample_clip_diagnostics(optimizer)
        optimizer.clip_and_accumulate()
        sums = [
            param.summed_grad.detach().clone() if getattr(param, "summed_grad", None) is not None else None
            for param in optimizer.params
        ]
        clear_private_optimizer_state(optimizer)
        return sums, preclip
    finally:
        optimizer.max_grad_norm = previous_max_grad_norm


def auto_s_transform(vectors: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply AUTO-S independently to the leading (per-example) dimension."""
    if vectors.ndim < 2:
        raise ValueError("AUTO-S expects a leading sample dimension and at least one feature dimension")
    if gamma <= 0.0:
        raise ValueError("AUTO-S gamma must be positive")
    norms = vectors.reshape(vectors.shape[0], -1).norm(2, dim=1)
    shape = (vectors.shape[0],) + (1,) * (vectors.ndim - 1)
    return vectors / (norms + float(gamma)).reshape(shape)


def auto_s_componentwise_transform(
    task_vectors: torch.Tensor,
    robust_vectors: torch.Tensor,
    *,
    beta: float,
    q_regularizer: float,
    gamma: float,
) -> torch.Tensor:
    """Reference/vectorized AUTO-S componentwise TRADES transformation.

    ``robust_vectors`` contains the unweighted per-example KL gradients
    ``r_i``.  The weighted branch ``h_i = beta * r_i`` is formed *before* its
    norm and denominator are evaluated.  Each row is an independent example;
    no averaging or batch-level normalization occurs here.
    """
    if task_vectors.shape != robust_vectors.shape:
        raise ValueError(
            "AUTO-S componentwise task/robust gradient shapes must match: "
            f"{task_vectors.shape} != {robust_vectors.shape}"
        )
    if task_vectors.ndim < 2:
        raise ValueError(
            "AUTO-S componentwise expects a leading sample dimension and at least one feature dimension"
        )
    beta = float(beta)
    q_regularizer = float(q_regularizer)
    gamma = float(gamma)
    if beta <= 0.0:
        raise ValueError("AUTO-S componentwise beta must be positive")
    if not 0.0 < q_regularizer < 1.0:
        raise ValueError("AUTO-S componentwise q_regularizer must be between 0 and 1")
    if gamma <= 0.0:
        raise ValueError("AUTO-S componentwise gamma must be positive")

    task_cap = 1.0 - q_regularizer
    robust_cap = q_regularizer
    weighted_robust = robust_vectors * beta
    task_norms = task_vectors.reshape(task_vectors.shape[0], -1).norm(2, dim=1)
    robust_norms = weighted_robust.reshape(weighted_robust.shape[0], -1).norm(2, dim=1)
    shape = (task_vectors.shape[0],) + (1,) * (task_vectors.ndim - 1)
    task_factors = task_cap / (task_norms + task_cap * gamma)
    robust_factors = robust_cap / (robust_norms + robust_cap * gamma)
    return task_vectors * task_factors.reshape(shape) + weighted_robust * robust_factors.reshape(shape)


def auto_s_component_grad_sums_from_current_samples(
    optimizer,
    *,
    gamma: float,
    sensitivity_cap: float,
    vector_scale: float = 1.0,
    clear_state: bool = True,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Transform one per-example gradient branch and return unaveraged sums.

    The query is ``cap * (scale*v) / (||scale*v|| + cap*gamma)`` where the norm
    is the full concatenated gradient over all trainable parameters.  No noise,
    averaging, accountant hook, momentum, or optimizer step is performed.  The
    caller may therefore evaluate the second branch and combine both returned
    sums before making one Gaussian release.
    """
    gamma = float(gamma)
    sensitivity_cap = float(sensitivity_cap)
    vector_scale = float(vector_scale)
    if gamma <= 0.0:
        raise ValueError("AUTO-S component gamma must be positive")
    if not 0.0 < sensitivity_cap < 1.0:
        raise ValueError("AUTO-S component sensitivity_cap must be between 0 and 1")
    if vector_scale == 0.0:
        raise ValueError("AUTO-S component vector_scale must be non-zero")

    samples: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    sample_count: int | None = None
    norm_sq: torch.Tensor | None = None
    for param in optimizer.params:
        grad_sample = getattr(param, "grad_sample", None)
        if isinstance(grad_sample, list):
            grad_sample = torch.cat(grad_sample, dim=0) if grad_sample else None
        if grad_sample is None:
            continue
        if sample_count is None:
            sample_count = int(grad_sample.shape[0])
            norm_sq = torch.zeros(sample_count, device=grad_sample.device, dtype=torch.float64)
        if int(grad_sample.shape[0]) != sample_count:
            raise RuntimeError("inconsistent Opacus grad_sample batch dimensions")
        assert norm_sq is not None
        scaled = grad_sample.detach().double().mul(vector_scale)
        norm_sq.add_(scaled.reshape(sample_count, -1).pow(2).sum(dim=1))
        samples.append((param, grad_sample.detach()))

    if sample_count is None or norm_sq is None:
        raise RuntimeError("AUTO-S component requires populated per-example grad_sample tensors")
    input_norms = norm_sq.sqrt()
    factors = sensitivity_cap / (input_norms + sensitivity_cap * gamma)
    output_norms = input_norms * factors

    sums_by_id: dict[int, torch.Tensor] = {}
    for param, grad_sample in samples:
        factor_shape = (sample_count,) + (1,) * (grad_sample.ndim - 1)
        transformed = grad_sample.mul(vector_scale) * factors.to(
            device=grad_sample.device, dtype=grad_sample.dtype
        ).reshape(factor_shape)
        sums_by_id[id(param)] = transformed.sum(dim=0)
    sums = [sums_by_id.get(id(param)) for param in optimizer.params]
    if clear_state:
        clear_private_optimizer_state(optimizer)
    else:
        # The final branch may keep grad_sample alive until Opacus has used it
        # to determine one accumulated iteration for scaling/accounting.
        for param in optimizer.params:
            param.grad = None
            if hasattr(param, "summed_grad"):
                param.summed_grad = None
    return sums, {
        "sample_count": float(sample_count),
        "input_norm_mean": float(input_norms.mean().item()),
        "input_norm_max": float(input_norms.max().item()),
        "output_norm_mean": float(output_norms.mean().item()),
        "output_norm_max": float(output_norms.max().item()),
        "gamma": gamma,
        "sensitivity_cap": sensitivity_cap,
        "vector_scale": vector_scale,
    }


def auto_s_grad_sums_from_current_samples(optimizer, gamma: float) -> dict[str, float]:
    """Set summed grads to the sum of per-example AUTO-S contributions.

    No averaging or noise happens here. The caller must add one Gaussian draw
    with sensitivity one, divide once by the logical batch size, invoke the
    accountant hook once, and only then apply optimizer momentum.
    """
    if gamma <= 0.0:
        raise ValueError("AUTO-S gamma must be positive")

    samples: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    sample_count: int | None = None
    norm_sq: torch.Tensor | None = None
    for param in optimizer.params:
        grad_sample = getattr(param, "grad_sample", None)
        if isinstance(grad_sample, list):
            grad_sample = torch.cat(grad_sample, dim=0) if grad_sample else None
        if grad_sample is None:
            continue
        if sample_count is None:
            sample_count = int(grad_sample.shape[0])
            norm_sq = torch.zeros(sample_count, device=grad_sample.device, dtype=torch.float64)
        if int(grad_sample.shape[0]) != sample_count:
            raise RuntimeError("inconsistent Opacus grad_sample batch dimensions")
        assert norm_sq is not None
        norm_sq.add_(grad_sample.detach().reshape(sample_count, -1).double().pow(2).sum(dim=1))
        samples.append((param, grad_sample.detach()))

    if sample_count is None or norm_sq is None:
        raise RuntimeError("AUTO-S requires populated per-example grad_sample tensors")
    norms = norm_sq.sqrt()
    factors = 1.0 / (norms + float(gamma))
    output_norms = norms * factors

    for param in optimizer.params:
        param.summed_grad = None
    for param, grad_sample in samples:
        factor_shape = (sample_count,) + (1,) * (grad_sample.ndim - 1)
        transformed = grad_sample * factors.to(
            device=grad_sample.device, dtype=grad_sample.dtype
        ).reshape(factor_shape)
        param.summed_grad = transformed.sum(dim=0)

    return {
        "sample_count": float(sample_count),
        "input_norm_mean": float(norms.mean().item()),
        "input_norm_max": float(norms.max().item()),
        "output_norm_mean": float(output_norms.mean().item()),
        "output_norm_max": float(output_norms.max().item()),
        "gamma": float(gamma),
    }


def combined_grad_stats(
    ce_grads: list[torch.Tensor | None],
    kl_grads: list[torch.Tensor | None],
    beta: float,
) -> dict[str, float]:
    ce_sq = 0.0
    kl_sq = 0.0
    combined_sq = 0.0
    dot = 0.0
    beta = float(beta)

    for ce_grad, kl_grad in zip(ce_grads, kl_grads):
        if ce_grad is None and kl_grad is None:
            continue
        if ce_grad is None:
            kl_det = kl_grad.detach()
            kl_sq += float(kl_det.pow(2).sum().item())
            combined_sq += float((beta * kl_det).pow(2).sum().item())
            continue
        if kl_grad is None:
            ce_det = ce_grad.detach()
            ce_sq += float(ce_det.pow(2).sum().item())
            combined_sq += float(ce_det.pow(2).sum().item())
            continue
        ce_det = ce_grad.detach()
        kl_det = kl_grad.detach()
        ce_sq += float(ce_det.pow(2).sum().item())
        kl_sq += float(kl_det.pow(2).sum().item())
        dot += float((ce_det * kl_det).sum().item())
        combined_sq += float((ce_det + beta * kl_det).pow(2).sum().item())

    ce_norm = ce_sq**0.5
    kl_norm = kl_sq**0.5
    combined_norm = combined_sq**0.5
    trades_contrib_norm = abs(beta) * kl_norm
    return {
        "ce_norm": ce_norm,
        "kl_norm": kl_norm,
        "trades_contrib_norm": trades_contrib_norm,
        "combined_norm": combined_norm,
        "cosine": dot / (ce_norm * kl_norm + 1e-12),
        "trades_to_ce_ratio": trades_contrib_norm / (ce_norm + 1e-12),
        "trades_to_combined_ratio": trades_contrib_norm / (combined_norm + 1e-12),
    }


def trades_a_raw_grad_diagnostics(
    ce_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    beta: float,
) -> dict[str, float]:
    """Raw (pre-DP) gradient stats for the legacy DP+TRADES strategy-A path.

    The caller must disable Opacus grad-sample hooks around this call.
    """
    ce_grads = torch.autograd.grad(ce_loss, params, retain_graph=True, allow_unused=True)
    kl_grads = torch.autograd.grad(kl_loss, params, retain_graph=True, allow_unused=True)
    return combined_grad_stats(list(ce_grads), list(kl_grads), beta)


def assign_combined_dp_grads(
    params: list[torch.nn.Parameter],
    task_grads: list[torch.Tensor | None],
    regularizer_grads: list[torch.Tensor | None],
    regularizer_weight: float,
) -> None:
    for param, task_grad, regularizer_grad in zip(params, task_grads, regularizer_grads):
        combined = None
        if task_grad is not None:
            combined = task_grad.to(device=param.device, dtype=param.dtype)
        if regularizer_grad is not None:
            scaled = regularizer_grad.to(device=param.device, dtype=param.dtype).mul(float(regularizer_weight))
            combined = scaled if combined is None else combined.add(scaled)
        param.grad = combined


def accountant_step_counts(
    accountant,
    task_noise_multiplier: float | None,
    regularizer_noise_multiplier: float | None,
    regularizer_name: str,
    manual_task_passes: int | None = None,
    manual_regularizer_passes: int | None = None,
) -> dict[str, Any]:
    history = getattr(accountant, "history", [])
    task_steps = 0
    regularizer_steps = 0
    other_steps = 0
    total_steps = 0
    sample_rate_sum = 0.0
    unique_noise_multipliers = set()
    task_noise_min_abs_diff = None
    regularizer_noise_min_abs_diff = None
    ambiguous_same_sigma = (
        task_noise_multiplier is not None
        and regularizer_noise_multiplier is not None
        and math.isclose(float(task_noise_multiplier), float(regularizer_noise_multiplier), rel_tol=0.0, abs_tol=1e-12)
    )

    for entry in history:
        if len(entry) != 3:
            continue
        noise_multiplier, sample_rate, num_steps = entry
        noise_multiplier = float(noise_multiplier)
        num_steps = int(num_steps)
        total_steps += num_steps
        sample_rate_sum += float(sample_rate) * num_steps
        unique_noise_multipliers.add(noise_multiplier)

        if task_noise_multiplier is not None:
            diff = abs(noise_multiplier - float(task_noise_multiplier))
            task_noise_min_abs_diff = diff if task_noise_min_abs_diff is None else min(task_noise_min_abs_diff, diff)
        if regularizer_noise_multiplier is not None:
            diff = abs(noise_multiplier - float(regularizer_noise_multiplier))
            regularizer_noise_min_abs_diff = (
                diff if regularizer_noise_min_abs_diff is None else min(regularizer_noise_min_abs_diff, diff)
            )

        if ambiguous_same_sigma:
            continue
        if task_noise_multiplier is not None and math.isclose(
            noise_multiplier, float(task_noise_multiplier), rel_tol=0.0, abs_tol=1e-12
        ):
            task_steps += num_steps
        elif regularizer_noise_multiplier is not None and math.isclose(
            noise_multiplier, float(regularizer_noise_multiplier), rel_tol=0.0, abs_tol=1e-12
        ):
            regularizer_steps += num_steps
        else:
            other_steps += num_steps

    if ambiguous_same_sigma:
        task_steps = int(manual_task_passes or 0)
        regularizer_steps = int(manual_regularizer_passes or 0)
        other_steps = max(0, total_steps - task_steps - regularizer_steps)

    return {
        "history_entries": len(history),
        "total_steps": total_steps,
        "task_steps": task_steps,
        f"{regularizer_name.lower()}_steps": regularizer_steps,
        "other_steps": other_steps,
        "ambiguous_same_sigma": 1.0 if ambiguous_same_sigma else 0.0,
        "mean_sample_rate": sample_rate_sum / max(1, total_steps),
        "unique_noise_multipliers": len(unique_noise_multipliers),
        "task_noise_min_abs_diff": 0.0 if task_noise_min_abs_diff is None else task_noise_min_abs_diff,
        f"{regularizer_name.lower()}_noise_min_abs_diff": (
            0.0 if regularizer_noise_min_abs_diff is None else regularizer_noise_min_abs_diff
        ),
    }


def preclip_metric_names(prefix: str, diagnostics: dict[str, float]) -> dict[str, float]:
    mapping = {
        "preclip sample count": "preclip_sample_count",
        "preclip grad norm mean": "preclip_norm_mean",
        "preclip grad norm p50": "preclip_norm_p50",
        "preclip grad norm p90": "preclip_norm_p90",
        "preclip grad norm p99": "preclip_norm_p99",
        "preclip grad norm max": "preclip_norm_max",
        "preclip clipped fraction": "preclip_clip_fraction",
        "preclip mean clip factor": "preclip_mean_clip_factor",
        "preclip min clip factor": "preclip_min_clip_factor",
    }
    return {f"{prefix} {metric_suffix}": float(diagnostics[key]) for metric_suffix, key in mapping.items()}
