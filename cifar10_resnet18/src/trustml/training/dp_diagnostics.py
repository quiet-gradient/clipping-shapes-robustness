from __future__ import annotations

import torch



def param_grad_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return total**0.5



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



def clipped_grad_sums_from_current_samples(
    optimizer,
    max_grad_norm: float,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Per-sample clip at ``max_grad_norm`` and return summed clipped grads.

    This adds no noise, records no accountant step, and returns the unscaled
    sums so the componentwise mechanism can noise the combined channels once.
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
