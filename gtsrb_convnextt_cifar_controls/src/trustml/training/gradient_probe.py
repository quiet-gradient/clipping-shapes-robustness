"""Public-only gradient-geometry diagnostics for DP-SGD + TRADES.

The functions in this module never inspect a private training batch and never
sample DP noise.  They evaluate a fixed public probe batch at the current model
state, construct per-example CE and TRADES-KL gradients, and simulate the
deterministic clipping/noise geometry of strategies A, C and D.

Only aggregate scalar metrics are returned.  Per-example gradient vectors,
noise tensors and DP RNG state are neither logged nor persisted.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


class GradientProbeStateRestorationError(RuntimeError):
    """The probe could not restore training-critical model/Opacus state."""


def compute_gradient_probe_metrics(
    task_grads: torch.Tensor,
    robust_grads: torch.Tensor,
    *,
    beta: float,
    joint_clip_norm: float,
    task_clip_norm: float,
    robust_clip_norm: float,
    noise_multiplier_a: float,
    noise_multiplier_c_task: float,
    noise_multiplier_c_robust: float,
    noise_multiplier_d: float,
    expected_batch_size: int | float | None = None,
    auto_s_gamma: float = 0.01,
    auto_s_componentwise_q: float = 0.5,
    learning_rate: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute aggregate P0 metrics from flattened per-example gradients.

    ``robust_grads`` is the *unweighted* KL gradient ``r_i``.  The function
    applies ``beta`` exactly once.  ``expected_batch_size`` is the denominator
    used by the real DP optimizer; it intentionally need not equal probe size.
    """
    accumulator = _GradientProbeAccumulator(
        beta=beta,
        joint_clip_norm=joint_clip_norm,
        task_clip_norm=task_clip_norm,
        robust_clip_norm=robust_clip_norm,
        noise_multiplier_a=noise_multiplier_a,
        noise_multiplier_c_task=noise_multiplier_c_task,
        noise_multiplier_c_robust=noise_multiplier_c_robust,
        noise_multiplier_d=noise_multiplier_d,
        expected_batch_size=expected_batch_size,
        auto_s_gamma=auto_s_gamma,
        auto_s_componentwise_q=auto_s_componentwise_q,
        learning_rate=learning_rate,
        eps=eps,
    )
    accumulator.add(task_grads, robust_grads)
    return accumulator.finalize()


def validate_componentwise_geometry_identities(
    task_grads: torch.Tensor,
    weighted_robust_grads: torch.Tensor,
    *,
    q: float,
    sensitivity_bound: float,
    eps: float = 1e-14,
) -> dict[str, float]:
    """Validate the hard componentwise identities in FP64.

    ``weighted_robust_grads`` is ``h_i = beta r_i`` already.  The projection
    statistic is the maximum *negative* slack: positive slack is valid.
    """
    if task_grads.ndim != 2 or task_grads.shape != weighted_robust_grads.shape:
        raise ValueError("gradient tensors must have the same [examples, parameters] shape")
    if not 0.0 < float(q) < 1.0:
        raise ValueError("q must be strictly between zero and one")
    if float(sensitivity_bound) <= 0.0:
        raise ValueError("sensitivity_bound must be positive")

    g = task_grads.detach().to(dtype=torch.float64)
    h = weighted_robust_grads.detach().to(dtype=torch.float64)
    c_sum = float(sensitivity_bound)
    c_g = (1.0 - float(q)) * c_sum
    c_h = float(q) * c_sum
    g_norm = g.norm(2, dim=1)
    h_norm = h.norm(2, dim=1)
    s = g + h
    s_norm = s.norm(2, dim=1)
    alpha_g = (c_g / g_norm.clamp_min(eps)).clamp(max=1.0)
    alpha_h = (c_h / h_norm.clamp_min(eps)).clamp(max=1.0)
    alpha_j = (c_sum / s_norm.clamp_min(eps)).clamp(max=1.0)
    u = alpha_g[:, None] * g + alpha_h[:, None] * h
    j = alpha_j[:, None] * s
    u_norm = u.norm(2, dim=1)

    projection_slack = (
        (s - u).square().sum(dim=1)
        - (s - j).square().sum(dim=1)
        - (u - j).square().sum(dim=1)
    )

    nonzero = (g_norm > eps) & (h_norm > eps)
    rotation_valid = nonzero & (s_norm > eps) & (u_norm > eps)
    rho = torch.zeros_like(g_norm)
    rho[nonzero] = (
        (g[nonzero] * h[nonzero]).sum(dim=1)
        / (g_norm[nonzero] * h_norm[nonzero])
    ).clamp(-1.0, 1.0)
    cos_theta = torch.zeros_like(g_norm)
    cos_theta[rotation_valid] = (
        (u[rotation_valid] * s[rotation_valid]).sum(dim=1)
        / (u_norm[rotation_valid] * s_norm[rotation_valid])
    ).clamp(-1.0, 1.0)
    sin_theta = torch.sqrt((1.0 - cos_theta.square()).clamp_min(0.0))
    rotation_rhs = torch.zeros_like(g_norm)
    rotation_rhs[rotation_valid] = (
        (alpha_g[rotation_valid] - alpha_h[rotation_valid]).abs()
        * g_norm[rotation_valid]
        * h_norm[rotation_valid]
        * torch.sqrt((1.0 - rho[rotation_valid].square()).clamp_min(0.0))
        / (u_norm[rotation_valid] * s_norm[rotation_valid])
    )
    rotation_residual = (sin_theta - rotation_rhs).abs()[rotation_valid]

    utilization_sq = (
        (1.0 - float(q)) ** 2
        + float(q) ** 2
        + 2.0 * float(q) * (1.0 - float(q)) * rho
    )
    saturated = (
        nonzero
        & (g_norm > c_g)
        & (h_norm > c_h)
        & (utilization_sq > eps)
    )
    utilization_observed = u_norm[saturated] / c_sum
    utilization_expected = torch.sqrt(utilization_sq[saturated])
    utilization_residual = (utilization_observed - utilization_expected).abs()

    def _maximum(value: torch.Tensor) -> float:
        return 0.0 if value.numel() == 0 else float(value.max().item())

    return {
        "dtype_is_float64": 1.0 if g.dtype == torch.float64 else 0.0,
        "example_count": float(g.shape[0]),
        "projection_max_negative_slack": _maximum((-projection_slack).clamp_min(0.0)),
        "projection_min_slack": float(projection_slack.min().item()),
        "rotation_valid_count": float(rotation_valid.sum().item()),
        "rotation_excluded_count": float((~rotation_valid).sum().item()),
        "rotation_residual_max": _maximum(rotation_residual),
        "saturated_valid_count": float(saturated.sum().item()),
        "saturated_excluded_count": float((~saturated).sum().item()),
        "saturated_utilization_residual_max": _maximum(utilization_residual),
    }


def auto_s_analytic_jvp(
    x: torch.Tensor,
    direction: torch.Tensor,
    *,
    gamma: float,
    eps: float = 1e-14,
) -> torch.Tensor:
    """Analytic JVP of ``x / (||x|| + gamma)`` in ``direction``."""
    if x.shape != direction.shape:
        raise ValueError("x and direction must have identical shapes")
    if float(gamma) <= 0.0:
        raise ValueError("gamma must be positive")
    norm = x.norm()
    if float(norm.item()) <= eps:
        return direction / float(gamma)
    unit = x / norm
    parallel = torch.dot(direction.reshape(-1), unit.reshape(-1)) * unit
    orthogonal = direction - parallel
    return (
        float(gamma) * parallel / (norm + float(gamma)).square()
        + orthogonal / (norm + float(gamma))
    )


def extract_per_sample_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float = 8.0 / 255.0,
    step_size: float = 2.0 / 255.0,
    steps: int = 10,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    microbatch_size: int = 4,
    seed: int = 20260814,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return public per-example CE/KL gradients plus inner-PGD aggregates.

    This lower-level API is useful for unit tests and small probes.  Large real
    probes should use :func:`run_gradient_probe`, which streams microbatches and
    never materializes ``probe_size x parameter_count`` tensors at once.
    """
    _validate_probe_batch(x, y)
    chunks_g: list[torch.Tensor] = []
    chunks_r: list[torch.Tensor] = []
    with _preserve_probe_state(model, device=x.device, seed=seed):
        base_model = _unwrap_model(model)
        x_adv, clean_probs, attack_metrics = _generate_probe_adversarial(
            base_model,
            x,
            y,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            mean=mean,
            std=std,
        )
        chunk_size = max(1, int(microbatch_size))
        for start in range(0, int(x.shape[0]), chunk_size):
            stop = min(int(x.shape[0]), start + chunk_size)
            g_chunk, r_chunk = _extract_gradient_chunk(
                base_model,
                x[start:stop],
                y[start:stop],
                x_adv[start:stop],
                clean_probs[start:stop],
            )
            chunks_g.append(g_chunk.detach())
            chunks_r.append(r_chunk.detach())
    return torch.cat(chunks_g, dim=0), torch.cat(chunks_r, dim=0), attack_metrics


def run_gradient_probe(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    beta: float,
    joint_clip_norm: float,
    task_clip_norm: float,
    robust_clip_norm: float,
    noise_multiplier_a: float,
    noise_multiplier_c_task: float,
    noise_multiplier_c_robust: float,
    noise_multiplier_d: float,
    expected_batch_size: int | float,
    epsilon: float = 8.0 / 255.0,
    step_size: float = 2.0 / 255.0,
    steps: int = 10,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    microbatch_size: int = 4,
    seed: int = 20260814,
    auto_s_gamma: float = 0.01,
    auto_s_componentwise_q: float = 0.5,
    learning_rate: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Run memory-bounded public-probe diagnostics and return scalar metrics."""
    _validate_probe_batch(x, y)
    accumulator = _GradientProbeAccumulator(
        beta=beta,
        joint_clip_norm=joint_clip_norm,
        task_clip_norm=task_clip_norm,
        robust_clip_norm=robust_clip_norm,
        noise_multiplier_a=noise_multiplier_a,
        noise_multiplier_c_task=noise_multiplier_c_task,
        noise_multiplier_c_robust=noise_multiplier_c_robust,
        noise_multiplier_d=noise_multiplier_d,
        expected_batch_size=expected_batch_size,
        auto_s_gamma=auto_s_gamma,
        auto_s_componentwise_q=auto_s_componentwise_q,
        learning_rate=learning_rate,
        eps=eps,
    )
    with _preserve_probe_state(model, device=x.device, seed=seed):
        base_model = _unwrap_model(model)
        x_adv, clean_probs, attack_metrics = _generate_probe_adversarial(
            base_model,
            x,
            y,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            mean=mean,
            std=std,
        )
        chunk_size = max(1, int(microbatch_size))
        for start in range(0, int(x.shape[0]), chunk_size):
            stop = min(int(x.shape[0]), start + chunk_size)
            g_chunk, r_chunk = _extract_gradient_chunk(
                base_model,
                x[start:stop],
                y[start:stop],
                x_adv[start:stop],
                clean_probs[start:stop],
            )
            accumulator.add(g_chunk, r_chunk)
            del g_chunk, r_chunk

    metrics = accumulator.finalize()
    metrics.update(attack_metrics)
    return _finite_scalar_metrics(metrics)


class _GradientProbeAccumulator:
    """Streaming sufficient statistics for the public probe."""

    def __init__(
        self,
        *,
        beta: float,
        joint_clip_norm: float,
        task_clip_norm: float,
        robust_clip_norm: float,
        noise_multiplier_a: float,
        noise_multiplier_c_task: float,
        noise_multiplier_c_robust: float,
        noise_multiplier_d: float,
        expected_batch_size: int | float | None,
        auto_s_gamma: float,
        auto_s_componentwise_q: float,
        learning_rate: float,
        eps: float,
    ) -> None:
        for name, value in {
            "joint_clip_norm": joint_clip_norm,
            "task_clip_norm": task_clip_norm,
            "robust_clip_norm": robust_clip_norm,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        self.beta = float(beta)
        self.joint_clip_norm = float(joint_clip_norm)
        self.task_clip_norm = float(task_clip_norm)
        self.robust_clip_norm = float(robust_clip_norm)
        self.noise_multiplier_a = float(noise_multiplier_a)
        self.noise_multiplier_c_task = float(noise_multiplier_c_task)
        self.noise_multiplier_c_robust = float(noise_multiplier_c_robust)
        self.noise_multiplier_d = float(noise_multiplier_d)
        if float(auto_s_gamma) <= 0.0:
            raise ValueError("auto_s_gamma must be positive")
        if not 0.0 < float(auto_s_componentwise_q) < 1.0:
            raise ValueError("auto_s_componentwise_q must be between zero and one")
        if float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.auto_s_gamma = float(auto_s_gamma)
        self.auto_s_componentwise_q = float(auto_s_componentwise_q)
        self.learning_rate = float(learning_rate)
        if expected_batch_size is not None and float(expected_batch_size) <= 0.0:
            raise ValueError("expected_batch_size must be positive")
        self.expected_batch_size = None if expected_batch_size is None else float(expected_batch_size)
        self.eps = float(eps)
        self.count = 0
        self.param_count: int | None = None
        self.values: dict[str, list[torch.Tensor]] = {}
        self.vector_sums: dict[str, torch.Tensor] = {}

    def _append(self, name: str, value: torch.Tensor) -> None:
        self.values.setdefault(name, []).append(value.detach().float().cpu())

    def _add_vector(self, name: str, value: torch.Tensor) -> None:
        summed = value.detach().sum(dim=0)
        if name not in self.vector_sums:
            self.vector_sums[name] = summed.clone()
        else:
            self.vector_sums[name].add_(summed)

    def add(self, task_grads: torch.Tensor, robust_grads: torch.Tensor) -> None:
        if task_grads.ndim != 2 or robust_grads.ndim != 2:
            raise ValueError("task_grads and robust_grads must have shape [examples, parameters]")
        if task_grads.shape != robust_grads.shape:
            raise ValueError(f"gradient shapes differ: {task_grads.shape} vs {robust_grads.shape}")
        if task_grads.shape[0] == 0 or task_grads.shape[1] == 0:
            raise ValueError("gradient probe requires at least one example and one trainable parameter")
        if self.param_count is None:
            self.param_count = int(task_grads.shape[1])
        elif self.param_count != int(task_grads.shape[1]):
            raise ValueError("parameter dimension changed between probe microbatches")

        g = task_grads.detach().float()
        r = robust_grads.detach().float()
        h = r * self.beta
        s = g + h
        g_norm = g.norm(2, dim=1)
        r_norm = r.norm(2, dim=1)
        h_norm = h.norm(2, dim=1)
        s_norm = s.norm(2, dim=1)
        dot = (g * h).sum(dim=1)

        alpha_a = (self.joint_clip_norm / (s_norm + self.eps)).clamp(max=1.0)
        alpha_g = (self.task_clip_norm / (g_norm + self.eps)).clamp(max=1.0)
        alpha_r = (self.robust_clip_norm / (r_norm + self.eps)).clamp(max=1.0)
        delta_d = self.task_clip_norm + abs(self.beta) * self.robust_clip_norm
        alpha_a_delta = (delta_d / (s_norm + self.eps)).clamp(max=1.0)

        u_a = alpha_a.unsqueeze(1) * s
        u_sep = alpha_g.unsqueeze(1) * g + alpha_r.unsqueeze(1) * h
        u_a_delta = alpha_a_delta.unsqueeze(1) * s
        u_sep_norm = u_sep.norm(2, dim=1)
        distortion_sep = (u_sep - s).square().sum(dim=1)
        distortion_a_delta = (u_a_delta - s).square().sum(dim=1)
        matched_distortion_gain = distortion_sep - distortion_a_delta
        commutation_defect = (u_sep - u_a_delta).norm(2, dim=1)
        projection_slack = (
            distortion_sep
            - distortion_a_delta
            - (u_sep - u_a_delta).square().sum(dim=1)
        )

        auto_s_scale = 1.0 / (s_norm + self.auto_s_gamma)
        auto_s_update = auto_s_scale.unsqueeze(1) * s
        sc_q = self.auto_s_componentwise_q
        sc_task_cap = 1.0 - sc_q
        sc_robust_cap = sc_q
        sc_task_scale = sc_task_cap / (
            g_norm + sc_task_cap * self.auto_s_gamma
        )
        sc_robust_scale = sc_robust_cap / (
            h_norm + sc_robust_cap * self.auto_s_gamma
        )
        sc_task_update = sc_task_scale.unsqueeze(1) * g
        sc_robust_update = sc_robust_scale.unsqueeze(1) * h
        sc_update = sc_task_update + sc_robust_update
        sc_task_output_norm = sc_task_update.norm(2, dim=1)
        sc_robust_output_norm = sc_robust_update.norm(2, dim=1)
        sc_output_norm = sc_update.norm(2, dim=1)
        sc_beta_eff_ratio = sc_robust_scale / (sc_task_scale + self.eps)
        sc_valid_cos = (sc_output_norm * auto_s_update.norm(2, dim=1)) > self.eps
        sc_sa_cos = torch.where(
            sc_valid_cos,
            (sc_update * auto_s_update).sum(dim=1)
            / (sc_output_norm * auto_s_update.norm(2, dim=1) + self.eps),
            torch.zeros_like(sc_output_norm),
        ).clamp(-1.0, 1.0)
        h_parallel_signed = (h * s).sum(dim=1) / (s_norm + self.eps)
        h_parallel = h_parallel_signed.abs()
        h_orthogonal = torch.sqrt((h_norm.square() - h_parallel.square()).clamp(min=0.0))

        valid_cos = (g_norm * h_norm) > self.eps
        task_robust_cos = torch.where(
            valid_cos,
            dot / (g_norm * h_norm + self.eps),
            torch.zeros_like(dot),
        ).clamp(-1.0, 1.0)
        cancellation = (s_norm / (g_norm + h_norm + self.eps)).clamp(0.0, 1.0)
        interaction = (2.0 * dot / (g_norm.square() + h_norm.square() + self.eps)).clamp(-1.0, 1.0)

        sep_valid = (u_sep_norm * s_norm) > self.eps
        sep_cos = torch.where(
            sep_valid,
            (u_sep * s).sum(dim=1) / (u_sep_norm * s_norm + self.eps),
            torch.zeros_like(s_norm),
        ).clamp(-1.0, 1.0)
        angle_degrees = torch.rad2deg(torch.acos(sep_cos.clamp(-1.0, 1.0)))
        beta_eff_ratio = alpha_r / (alpha_g + self.eps)
        beta_eff_log2 = torch.log2(beta_eff_ratio + self.eps)
        robust_to_task = h_norm / (g_norm + self.eps)
        robust_to_task_log10 = torch.log10((h_norm + self.eps) / (g_norm + self.eps))

        task_clipped = g_norm > self.task_clip_norm
        robust_clipped = r_norm > self.robust_clip_norm
        q_global = abs(self.beta) * self.robust_clip_norm / delta_d
        allocation_valid = (
            (g_norm > self.eps)
            & (h_norm > self.eps)
            & task_clipped
            & robust_clipped
        )
        q_star = h_norm / (g_norm + h_norm + self.eps)
        allocation_mismatch = (q_star - q_global).abs()
        abs_log_beta_eff = torch.log(beta_eff_ratio.clamp_min(self.eps)).abs()
        under_utilization = 1.0 - (u_sep_norm / (delta_d + self.eps)).clamp(0.0, 1.0)
        self._append("task_norm", g_norm)
        self._append("robust_unscaled_norm", r_norm)
        self._append("robust_norm", h_norm)
        self._append("total_norm", s_norm)
        self._append("robust_to_task", robust_to_task)
        self._append("robust_to_task_log10", robust_to_task_log10)
        self._append("task_robust_cos", task_robust_cos)
        self._append("task_robust_cos_valid", valid_cos.float())
        self._append("task_robust_dot", dot)
        self._append("task_robust_norm_product", g_norm * h_norm)
        self._append("cancellation", cancellation)
        self._append("interaction", interaction)
        self._append("alpha_a", alpha_a)
        self._append("alpha_g", alpha_g)
        self._append("alpha_r", alpha_r)
        self._append("joint_clipped", (s_norm > self.joint_clip_norm).float())
        self._append("task_clipped", task_clipped.float())
        self._append("robust_clipped", robust_clipped.float())
        self._append("clip_none", (~task_clipped & ~robust_clipped).float())
        self._append("clip_task_only", (task_clipped & ~robust_clipped).float())
        self._append("clip_robust_only", (~task_clipped & robust_clipped).float())
        self._append("clip_both", (task_clipped & robust_clipped).float())
        self._append("beta_eff_ratio", beta_eff_ratio)
        self._append("beta_eff_log2", beta_eff_log2)
        self._append("sep_cos", sep_cos)
        self._append("sep_cos_valid", sep_valid.float())
        self._append("sep_angle_degrees", angle_degrees)
        self._append("alpha_gap", (alpha_g - alpha_r).abs())
        self._append("retention_a", u_a.norm(2, dim=1) / (s_norm + self.eps))
        self._append("retention_sep", u_sep_norm / (s_norm + self.eps))
        a_valid = (u_a.norm(2, dim=1) * s_norm) > self.eps
        a_cos = torch.where(
            a_valid,
            (u_a * s).sum(dim=1) / (u_a.norm(2, dim=1) * s_norm + self.eps),
            torch.zeros_like(s_norm),
        ).clamp(-1.0, 1.0)
        self._append("a_cos", a_cos)
        self._append("a_cos_valid", a_valid.float())
        self._append("d_utilization", (u_sep_norm / (delta_d + self.eps)).clamp(0.0, 1.0))
        self._append("matched_distortion_gain", matched_distortion_gain)
        self._append("projection_slack", projection_slack)
        self._append("commutation_defect", commutation_defect)
        self._append("allocation_valid", allocation_valid.float())
        self._append("q_star", q_star)
        self._append("allocation_mismatch", allocation_mismatch)
        self._append("abs_log_beta_eff", abs_log_beta_eff)
        self._append("under_utilization", under_utilization)
        self._append("auto_s_output_norm", auto_s_update.norm(2, dim=1))
        self._append("auto_s_effective_lr", self.learning_rate * auto_s_scale)
        self._append("auto_s_radial_response", self.auto_s_gamma * auto_s_scale)
        self._append("auto_s_robust_parallel_fraction", h_parallel / (h_norm + self.eps))
        self._append("auto_s_robust_orthogonal_fraction", h_orthogonal / (h_norm + self.eps))
        self._append(
            "auto_s_robust_parallel_signed_fraction",
            h_parallel_signed / (h_norm + self.eps),
        )
        self._append("sc_task_scale", sc_task_scale)
        self._append("sc_robust_scale", sc_robust_scale)
        self._append("sc_beta_eff_ratio", sc_beta_eff_ratio)
        self._append("sc_task_output_norm", sc_task_output_norm)
        self._append("sc_robust_output_norm", sc_robust_output_norm)
        self._append("sc_output_norm", sc_output_norm)
        self._append("sc_sensitivity_utilization", sc_output_norm)
        self._append("sc_sa_cos", sc_sa_cos)
        self._append("sc_sa_cos_valid", sc_valid_cos.float())
        self._append("sc_distortion", (sc_update - s).square().sum(dim=1))
        self._append("sa_distortion", (auto_s_update - s).square().sum(dim=1))

        self._add_vector("G", g)
        self._add_vector("H", h)
        self._add_vector("m_A", u_a)
        self._add_vector("m_sep", u_sep)
        self._add_vector("m_A_delta", u_a_delta)
        self._add_vector("m_SA", auto_s_update)
        self._add_vector("m_SC", sc_update)
        self.count += int(g.shape[0])

    def _cat(self, name: str) -> torch.Tensor:
        return torch.cat(self.values[name], dim=0)

    def finalize(self) -> dict[str, float]:
        if self.count <= 0 or self.param_count is None:
            raise ValueError("cannot finalize an empty gradient probe")
        metrics: dict[str, float] = {
            "probe/meta/sample_count": float(self.count),
            "probe/meta/trainable_parameter_count": float(self.param_count),
            "probe/meta/expected_batch_size": float(self.expected_batch_size or self.count),
            "probe/meta/beta": self.beta,
            "probe/meta/joint_clip_norm": self.joint_clip_norm,
            "probe/meta/task_clip_norm": self.task_clip_norm,
            "probe/meta/robust_clip_norm": self.robust_clip_norm,
            "probe/meta/noise_multiplier_A": self.noise_multiplier_a,
            "probe/meta/noise_multiplier_C_task": self.noise_multiplier_c_task,
            "probe/meta/noise_multiplier_C_robust": self.noise_multiplier_c_robust,
            "probe/meta/noise_multiplier_D": self.noise_multiplier_d,
            "probe/meta/auto_s_gamma": self.auto_s_gamma,
            "probe/meta/auto_s_componentwise_q": self.auto_s_componentwise_q,
            "probe/meta/auto_s_componentwise_task_cap": 1.0
            - self.auto_s_componentwise_q,
            "probe/meta/auto_s_componentwise_robust_cap": self.auto_s_componentwise_q,
            "probe/meta/auto_s_componentwise_sensitivity": 1.0,
            "probe/meta/learning_rate": self.learning_rate,
        }

        for source, prefix in (
            ("task_norm", "probe/raw/task_norm"),
            ("robust_unscaled_norm", "probe/raw/robust_unscaled_norm"),
            ("robust_norm", "probe/raw/robust_norm"),
            ("total_norm", "probe/raw/total_norm"),
        ):
            metrics.update(_distribution_metrics(prefix, self._cat(source), eps=self.eps, near_zero=True))
        for source, label in (
            ("task_norm", "task_norm"),
            ("robust_norm", "robust_norm"),
            ("total_norm", "total_norm"),
            ("task_robust_cos", "task_robust_cos"),
            ("beta_eff_ratio", "beta_eff_ratio"),
            ("matched_distortion_gain", "matched_distortion_gain"),
            ("auto_s_effective_lr", "auto_s_effective_lr"),
            ("auto_s_radial_response", "auto_s_radial_response"),
            ("sc_beta_eff_ratio", "sc_beta_eff_ratio"),
            ("sc_task_output_norm", "sc_task_output_norm"),
            ("sc_robust_output_norm", "sc_robust_output_norm"),
            ("sc_output_norm", "sc_output_norm"),
            ("sc_sa_cos", "sc_sa_cos"),
        ):
            metrics.update(_histogram_metrics(f"probe/hist/{label}", self._cat(source)))

        ratio = self._cat("robust_to_task")
        ratio_log = self._cat("robust_to_task_log10")
        metrics.update(_selected_quantiles("probe/raw/robust_to_task_ratio", ratio, (0.10, 0.50, 0.90)))
        metrics.update(_selected_quantiles("probe/raw/robust_to_task_log10_ratio", ratio_log, (0.10, 0.50, 0.90)))
        metrics["probe/raw/robust_dominates_frac"] = _mean_bool(ratio > 1.0)
        metrics["probe/raw/task_dominates_frac"] = _mean_bool(ratio < 1.0)

        cos = self._cat("task_robust_cos")
        cos_valid = self._cat("task_robust_cos_valid").bool()
        valid_cos = cos[cos_valid]
        metrics.update(_selected_quantiles("probe/geometry/task_robust_cos", valid_cos, (0.10, 0.50, 0.90)))
        metrics["probe/geometry/task_robust_cos/mean"] = _safe_mean(valid_cos)
        metrics["probe/geometry/task_robust_cos/valid_frac"] = _mean_bool(cos_valid)
        metrics["probe/geometry/task_robust_cos/negative_frac"] = _safe_fraction(valid_cos < 0.0)
        metrics["probe/geometry/task_robust_cos/below_minus_half_frac"] = _safe_fraction(valid_cos < -0.5)
        metrics["probe/geometry/task_robust_cos/above_half_frac"] = _safe_fraction(valid_cos > 0.5)
        dots = self._cat("task_robust_dot")
        norm_products = self._cat("task_robust_norm_product")
        metrics["probe/geometry/task_robust_cos/weighted"] = float(
            dots.sum().item() / (norm_products.sum().item() + self.eps)
        )

        cancellation = self._cat("cancellation")
        metrics.update(_selected_quantiles("probe/geometry/cancellation_ratio", cancellation, (0.10, 0.50, 0.90)))
        metrics["probe/geometry/cancellation_ratio/mean"] = _safe_mean(cancellation)
        metrics["probe/geometry/cancellation_ratio/strong_cancellation_frac"] = _mean_bool(cancellation < 0.5)
        metrics["probe/geometry/strong_cancellation_frac"] = metrics[
            "probe/geometry/cancellation_ratio/strong_cancellation_frac"
        ]
        metrics.update(_distribution_metrics("probe/geometry/interaction", self._cat("interaction"), eps=self.eps))

        alpha_a = self._cat("alpha_a")
        alpha_g = self._cat("alpha_g")
        alpha_r = self._cat("alpha_r")
        metrics["probe/clip/A/clip_frac"] = _safe_mean(self._cat("joint_clipped"))
        metrics.update(_alpha_metrics("probe/clip/A", alpha_a))
        metrics["probe/clip/A/alpha_p10"] = metrics["probe/clip/A/alpha/p10"]
        metrics["probe/clip/A/alpha_p50"] = metrics["probe/clip/A/alpha/p50"]
        a_valid = self._cat("a_cos_valid").bool()
        valid_a_cos = self._cat("a_cos")[a_valid]
        metrics["probe/clip/A/cos_to_raw/mean"] = _safe_mean(valid_a_cos)
        metrics["probe/clip/A/cos_to_raw/max_abs_error_from_one"] = (
            0.0 if valid_a_cos.numel() == 0 else float((valid_a_cos - 1.0).abs().max().item())
        )
        metrics["probe/clip/separate/task_clip_frac"] = _safe_mean(self._cat("task_clipped"))
        metrics["probe/clip/separate/robust_clip_frac"] = _safe_mean(self._cat("robust_clipped"))
        metrics.update(_alpha_metrics("probe/clip/separate/task", alpha_g))
        metrics.update(_alpha_metrics("probe/clip/separate/robust", alpha_r))
        metrics["probe/clip/separate/task_alpha_mean"] = metrics["probe/clip/separate/task/alpha_mean"]
        metrics["probe/clip/separate/robust_alpha_mean"] = metrics["probe/clip/separate/robust/alpha_mean"]
        metrics["probe/clip/separate/task_alpha_p10"] = metrics["probe/clip/separate/task/alpha/p10"]
        metrics["probe/clip/separate/task_alpha_p50"] = metrics["probe/clip/separate/task/alpha/p50"]
        metrics["probe/clip/separate/robust_alpha_p10"] = metrics["probe/clip/separate/robust/alpha/p10"]
        metrics["probe/clip/separate/robust_alpha_p50"] = metrics["probe/clip/separate/robust/alpha/p50"]
        for source, suffix in (
            ("clip_none", "none_frac"),
            ("clip_task_only", "task_only_frac"),
            ("clip_robust_only", "robust_only_frac"),
            ("clip_both", "both_frac"),
        ):
            metrics[f"probe/clip/separate/{suffix}"] = _safe_mean(self._cat(source))

        beta_ratio = self._cat("beta_eff_ratio")
        metrics.update(_selected_quantiles("probe/clip/separate/beta_eff_ratio", beta_ratio, (0.10, 0.50, 0.90)))
        metrics.update(
            _selected_quantiles("probe/clip/separate/beta_eff_log2", self._cat("beta_eff_log2"), (0.10, 0.50, 0.90))
        )
        metrics["probe/clip/separate/beta_eff_lt_half_frac"] = _mean_bool(beta_ratio < 0.5)
        metrics["probe/clip/separate/beta_eff_gt_two_frac"] = _mean_bool(beta_ratio > 2.0)
        metrics["probe/clip/separate/beta_eff_unchanged_frac"] = _mean_bool((beta_ratio - 1.0).abs() <= 1e-6)

        sep_cos = self._cat("sep_cos")
        sep_valid = self._cat("sep_cos_valid").bool()
        valid_sep_cos = sep_cos[sep_valid]
        metrics.update(_selected_quantiles("probe/clip/separate/cos_to_raw", valid_sep_cos, (0.10, 0.50, 0.90)))
        metrics["probe/clip/separate/cos_to_raw/mean"] = _safe_mean(valid_sep_cos)
        metrics["probe/clip/separate/cos_to_raw/valid_frac"] = _mean_bool(sep_valid)
        metrics["probe/clip/separate/angle_error_mean"] = _safe_mean(self._cat("sep_angle_degrees")[sep_valid])
        metrics["probe/clip/separate/angle_error_degrees_mean"] = metrics[
            "probe/clip/separate/angle_error_mean"
        ]
        metrics["probe/clip/separate/large_rotation_frac"] = _safe_fraction(valid_sep_cos < 0.9)
        metrics["probe/clip/separate/alpha_gap/mean"] = _safe_mean(self._cat("alpha_gap"))
        metrics.update(_selected_quantiles("probe/clip/separate/alpha_gap", self._cat("alpha_gap"), (0.90,)))
        metrics.update(_selected_quantiles("probe/clip/A/norm_retention", self._cat("retention_a"), (0.10, 0.50)))
        metrics.update(
            _selected_quantiles("probe/clip/separate/norm_retention", self._cat("retention_sep"), (0.10, 0.50))
        )

        inv_n = 1.0 / float(self.count)
        G = self.vector_sums["G"] * inv_n
        H = self.vector_sums["H"] * inv_n
        S = G + H
        updates = {
            "A": self.vector_sums["m_A"] * inv_n,
            "C": self.vector_sums["m_sep"] * inv_n,
            "D": self.vector_sums["m_sep"] * inv_n,
        }
        m_a_delta = self.vector_sums["m_A_delta"] * inv_n
        m_sep = updates["D"]
        s_sq = float(S.square().sum().item())
        s_norm = math.sqrt(max(0.0, s_sq))

        metrics.update(_counterfactual_metrics("probe/counterfactual/A", updates["A"], S, self.eps))
        metrics.update(_counterfactual_metrics("probe/counterfactual/separate", m_sep, S, self.eps))
        metrics["probe/counterfactual/A_vs_separate/cos"] = _vector_cos(updates["A"], m_sep, self.eps)
        metrics.update(_counterfactual_metrics("probe/matched_sensitivity/A_delta", m_a_delta, S, self.eps))
        metrics.update(_counterfactual_metrics("probe/matched_sensitivity/separate", m_sep, S, self.eps))
        metrics["probe/matched_sensitivity/A_delta_vs_separate/cos"] = _vector_cos(
            m_a_delta, m_sep, self.eps
        )
        matched_gain = self._cat("matched_distortion_gain")
        metrics.update(
            _distribution_metrics(
                "probe/matched_sensitivity/A_delta_vs_separate/distortion_gain",
                matched_gain,
                eps=self.eps,
            )
        )
        metrics["probe/matched_sensitivity/A_delta_vs_separate/numerical_violation_frac"] = (
            _mean_bool(matched_gain < -1e-6)
        )
        projection_slack = self._cat("projection_slack")
        metrics.update(
            _distribution_metrics(
                "probe/matched_sensitivity/A_delta_vs_separate/projection_slack",
                projection_slack,
                eps=self.eps,
            )
        )
        metrics[
            "probe/matched_sensitivity/A_delta_vs_separate/projection_max_negative_slack"
        ] = float((-projection_slack).clamp(min=0.0).max().item())
        metrics[
            "probe/matched_sensitivity/A_delta_vs_separate/full_numerical_violation_frac"
        ] = _mean_bool(projection_slack < -1e-6)

        allocation_valid = self._cat("allocation_valid").bool()
        allocation_count = int(allocation_valid.sum().item())
        metrics["probe/per_example_chain/valid_both_saturated_count"] = float(allocation_count)
        metrics["probe/per_example_chain/excluded_count"] = float(self.count - allocation_count)
        if allocation_count:
            chain_values = {
                "q_star": self._cat("q_star")[allocation_valid],
                "allocation_mismatch": self._cat("allocation_mismatch")[allocation_valid],
                "abs_log_beta_eff": self._cat("abs_log_beta_eff")[allocation_valid],
                "rotation_degrees": self._cat("sep_angle_degrees")[allocation_valid],
                "commutation_defect": self._cat("commutation_defect")[allocation_valid],
                "under_utilization": self._cat("under_utilization")[allocation_valid],
            }
            for name, values in chain_values.items():
                metrics.update(
                    _distribution_metrics(
                        f"probe/per_example_chain/{name}", values, eps=self.eps
                    )
                )
            mismatch = chain_values["allocation_mismatch"]
            reweight = chain_values["abs_log_beta_eff"]
            metrics["probe/per_example_chain/spearman/mismatch_vs_reweight"] = _spearman(
                mismatch, reweight
            )
            metrics["probe/per_example_chain/spearman/reweight_vs_rotation"] = _spearman(
                reweight, chain_values["rotation_degrees"]
            )
            for name in ("rotation_degrees", "commutation_defect", "under_utilization"):
                metrics[f"probe/per_example_chain/spearman/mismatch_vs_{name}"] = _spearman(
                    mismatch, chain_values[name]
                )
        metrics.update(_distortion_metrics("probe/distortion/A", updates["A"], S, self.eps))
        metrics.update(_distortion_metrics("probe/distortion/separate", m_sep, S, self.eps))
        m_sa = self.vector_sums["m_SA"] * inv_n
        m_sc = self.vector_sums["m_SC"] * inv_n
        metrics.update(_counterfactual_metrics("probe/counterfactual/S-A", m_sa, S, self.eps))
        metrics.update(_counterfactual_metrics("probe/counterfactual/S-C", m_sc, S, self.eps))
        metrics["probe/counterfactual/S-A_vs_S-C/cos"] = _vector_cos(
            m_sa, m_sc, self.eps
        )

        batch_denominator = float(self.expected_batch_size or self.count)
        tau = {
            "A": abs(self.noise_multiplier_a) * self.joint_clip_norm / batch_denominator,
            "C": math.sqrt(
                (self.noise_multiplier_c_task * self.task_clip_norm) ** 2
                + (self.beta * self.noise_multiplier_c_robust * self.robust_clip_norm) ** 2
            )
            / batch_denominator,
            "D": abs(self.noise_multiplier_d)
            * (self.task_clip_norm + abs(self.beta) * self.robust_clip_norm)
            / batch_denominator,
        }
        for mechanism, update in updates.items():
            clip_error = float((update - S).square().sum().item())
            noise_error = float(self.param_count) * tau[mechanism] ** 2
            total_error = clip_error + noise_error
            prefix = f"probe/mse/{mechanism}"
            metrics[f"{prefix}/clipping"] = clip_error
            metrics[f"{prefix}/noise"] = noise_error
            metrics[f"{prefix}/total"] = total_error
            metrics[f"{prefix}/relative_clipping"] = clip_error / (s_sq + self.eps)
            metrics[f"{prefix}/relative_noise"] = noise_error / (s_sq + self.eps)
            metrics[f"{prefix}/relative_total"] = total_error / (s_sq + self.eps)
            metrics[f"probe/noise/{mechanism}/tau"] = tau[mechanism]
            metrics[f"probe/noise/{mechanism}/expected_norm"] = tau[mechanism] * math.sqrt(self.param_count)
            update_norm = float(update.norm().item())
            metrics[f"probe/noise/{mechanism}/snr_norm"] = update_norm / (
                tau[mechanism] * math.sqrt(self.param_count) + self.eps
            )
            metrics.update(_descent_projection_metrics(mechanism, update, G, H, S, tau[mechanism], self.eps))

        utilization = self._cat("d_utilization")
        metrics.update(_selected_quantiles("probe/sensitivity/D/utilization", utilization, (0.50, 0.90, 0.99)))
        metrics["probe/sensitivity/D/utilization/mean"] = _safe_mean(utilization)
        metrics["probe/sensitivity/D/utilization/max"] = float(utilization.max().item())
        for source, prefix in (
            ("auto_s_output_norm", "probe/auto_s/output_norm"),
            ("auto_s_effective_lr", "probe/auto_s/effective_lr"),
            ("auto_s_radial_response", "probe/auto_s/radial_response"),
            ("auto_s_robust_parallel_fraction", "probe/auto_s/robust_parallel_fraction"),
            ("auto_s_robust_orthogonal_fraction", "probe/auto_s/robust_orthogonal_fraction"),
            (
                "auto_s_robust_parallel_signed_fraction",
                "probe/auto_s/robust_parallel_signed_fraction",
            ),
        ):
            metrics.update(_distribution_metrics(prefix, self._cat(source), eps=self.eps))
        metrics["probe/invariant/auto_s_output_norm_strictly_below_one"] = (
            1.0 if float(self._cat("auto_s_output_norm").max().item()) < 1.0 else 0.0
        )
        sc_distributions = (
            ("sc_task_scale", "probe/auto_s_componentwise/task_scale"),
            ("sc_robust_scale", "probe/auto_s_componentwise/robust_scale"),
            ("sc_beta_eff_ratio", "probe/auto_s_componentwise/beta_eff_ratio"),
            ("sc_task_output_norm", "probe/auto_s_componentwise/task_output_norm"),
            ("sc_robust_output_norm", "probe/auto_s_componentwise/robust_output_norm"),
            ("sc_output_norm", "probe/auto_s_componentwise/output_norm"),
            (
                "sc_sensitivity_utilization",
                "probe/auto_s_componentwise/sensitivity_utilization",
            ),
            ("sc_distortion", "probe/auto_s_componentwise/distortion"),
            ("sa_distortion", "probe/auto_s_componentwise/counterfactual_sa_distortion"),
        )
        for source, prefix in sc_distributions:
            metrics.update(_distribution_metrics(prefix, self._cat(source), eps=self.eps))
        sc_beta_ratio = self._cat("sc_beta_eff_ratio")
        metrics["probe/auto_s_componentwise/beta_eff_ratio/iqr"] = (
            metrics["probe/auto_s_componentwise/beta_eff_ratio/p75"]
            - metrics["probe/auto_s_componentwise/beta_eff_ratio/p25"]
        )
        metrics["probe/auto_s_componentwise/beta_eff_lt_half_frac"] = _mean_bool(
            sc_beta_ratio < 0.5
        )
        metrics["probe/auto_s_componentwise/beta_eff_gt_two_frac"] = _mean_bool(
            sc_beta_ratio > 2.0
        )
        sc_cos_valid = self._cat("sc_sa_cos_valid").bool()
        valid_sc_cos = self._cat("sc_sa_cos")[sc_cos_valid]
        metrics.update(
            _distribution_metrics(
                "probe/auto_s_componentwise/counterfactual_sa_cos",
                valid_sc_cos,
                eps=self.eps,
            )
        )
        task_cap = 1.0 - self.auto_s_componentwise_q
        robust_cap = self.auto_s_componentwise_q
        metrics["probe/invariant/sc_task_output_within_cap"] = (
            1.0
            if float(self._cat("sc_task_output_norm").max().item())
            <= task_cap + 1e-6
            else 0.0
        )
        metrics["probe/invariant/sc_robust_output_within_cap"] = (
            1.0
            if float(self._cat("sc_robust_output_norm").max().item())
            <= robust_cap + 1e-6
            else 0.0
        )
        metrics["probe/invariant/sc_output_norm_at_most_one"] = (
            1.0
            if float(self._cat("sc_output_norm").max().item()) <= 1.0 + 1e-6
            else 0.0
        )
        metrics["probe/clip/A/effective_sample_size"] = _effective_sample_size(alpha_a, self.eps)
        metrics["probe/clip/separate/task_effective_sample_size"] = _effective_sample_size(alpha_g, self.eps)
        metrics["probe/clip/separate/robust_effective_sample_size"] = _effective_sample_size(alpha_r, self.eps)
        metrics["probe/clip/A/effective_sample_fraction"] = metrics[
            "probe/clip/A/effective_sample_size"
        ] / self.count
        metrics["probe/clip/separate/task_effective_sample_fraction"] = metrics[
            "probe/clip/separate/task_effective_sample_size"
        ] / self.count
        metrics["probe/clip/separate/robust_effective_sample_fraction"] = metrics[
            "probe/clip/separate/robust_effective_sample_size"
        ] / self.count
        # This is an analytical construction identity: the public
        # counterfactual intentionally assigns the same separately clipped
        # mean to C and D.  It is not an integration check of Trainer's two
        # real update paths (that requires a separate training-path test).
        metrics["probe/invariant/analytical_C_D_share_deterministic_update"] = 1.0
        metrics["probe/invariant/analytical_C_D_clipping_mse_abs_diff"] = abs(
            metrics["probe/mse/C/clipping"] - metrics["probe/mse/D/clipping"]
        )
        return _finite_scalar_metrics(metrics)


def _extract_gradient_chunk(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_adv: torch.Tensor,
    clean_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not params:
        raise ValueError("gradient probe found no trainable parameters")
    buffers = {name: buffer for name, buffer in model.named_buffers()}
    names = tuple(params)

    def ce_single(current_params, sample, target):
        logits = functional_call(
            model,
            (current_params, buffers),
            (sample.unsqueeze(0),),
            strict=False,
        )
        return F.cross_entropy(logits, target.unsqueeze(0), reduction="sum")

    def kl_single(current_params, adv_sample, target_probs):
        logits = functional_call(
            model,
            (current_params, buffers),
            (adv_sample.unsqueeze(0),),
            strict=False,
        )
        return F.kl_div(
            F.log_softmax(logits, dim=1),
            target_probs.detach().unsqueeze(0),
            reduction="sum",
        )

    task_tree = vmap(grad(ce_single), in_dims=(None, 0, 0))(params, x, y)
    robust_tree = vmap(grad(kl_single), in_dims=(None, 0, 0))(params, x_adv, clean_probs)
    task_flat = torch.cat([task_tree[name].reshape(x.shape[0], -1) for name in names], dim=1)
    robust_flat = torch.cat([robust_tree[name].reshape(x.shape[0], -1) for name in names], dim=1)
    return task_flat, robust_flat


def _generate_probe_adversarial(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    mean,
    std,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    mean_t = x.new_tensor(mean).view(1, -1, 1, 1)
    std_t = x.new_tensor(std).view(1, -1, 1, 1)
    lower = (0.0 - mean_t) / std_t
    upper = (1.0 - mean_t) / std_t
    eps_t = float(epsilon) / std_t
    step_t = float(step_size) / std_t
    with torch.no_grad():
        clean_logits = model(x)
        clean_probs = F.softmax(clean_logits, dim=1).detach()
        x_adv = torch.clamp(x + 0.001 * torch.randn_like(x), min=lower, max=upper)

    trajectory: list[float] = []
    with torch.no_grad():
        initial_logits = model(x_adv)
        initial_kl = F.kl_div(
            F.log_softmax(initial_logits, dim=1), clean_probs, reduction="none"
        ).sum(dim=1)
        trajectory.append(float(initial_kl.mean().item()))

    for _ in range(max(0, int(steps))):
        x_adv.requires_grad_(True)
        adv_logits = model(x_adv)
        per_example_kl = F.kl_div(
            F.log_softmax(adv_logits, dim=1), clean_probs, reduction="none"
        ).sum(dim=1)
        grad_x = torch.autograd.grad(per_example_kl.sum(), x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + step_t * grad_x.sign()
        delta = torch.clamp(x_adv - x, min=-eps_t, max=eps_t)
        x_adv = torch.clamp(x + delta, min=lower, max=upper).detach()
        with torch.no_grad():
            step_logits = model(x_adv)
            step_kl = F.kl_div(
                F.log_softmax(step_logits, dim=1), clean_probs, reduction="none"
            ).sum(dim=1)
            trajectory.append(float(step_kl.mean().item()))

    with torch.no_grad():
        adv_logits = model(x_adv)
        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)
        clean_correct = clean_pred.eq(y)
        attack_success = (clean_correct & adv_pred.ne(y)).sum().item() / max(1, int(clean_correct.sum().item()))
        pixel_delta = (x_adv - x) * std_t
        linf = pixel_delta.abs().flatten(1).max(dim=1).values
        boundary = linf >= (0.99 * float(epsilon))
        correct_logits = adv_logits.gather(1, y.unsqueeze(1)).squeeze(1)
        masked = adv_logits.clone()
        masked.scatter_(1, y.unsqueeze(1), -torch.inf)
        adv_margin = correct_logits - masked.max(dim=1).values

    metrics = {
        "probe/attack/kl_step_0": trajectory[0],
        "probe/attack/kl_step_final": trajectory[-1],
        "probe/attack/kl_improvement": trajectory[-1] - trajectory[0],
        "probe/attack/linf_norm_mean": float(linf.mean().item()),
        "probe/attack/linf_boundary_frac": float(boundary.float().mean().item()),
        "probe/attack/attack_success_rate": float(attack_success),
        "probe/attack/clean_accuracy": float(clean_correct.float().mean().item()),
        "probe/attack/adversarial_accuracy": float(adv_pred.eq(y).float().mean().item()),
        "probe/attack/adv_margin": float(adv_margin.mean().item()),
    }
    for index, value in enumerate(trajectory):
        metrics[f"probe/attack/kl_step_{index}"] = float(value)
    return x_adv.detach(), clean_probs, metrics


@contextmanager
def _preserve_probe_state(
    model: torch.nn.Module,
    *,
    device: torch.device,
    seed: int,
) -> Iterator[None]:
    modules = list(model.modules())
    training_flags = [module.training for module in modules]
    parameters = list(model.parameters())
    parameter_values = [parameter.detach().clone() for parameter in parameters]
    buffers = list(model.buffers())
    buffer_values = [buffer.detach().clone() for buffer in buffers]
    grad_refs = [parameter.grad for parameter in parameters]
    grad_sample_refs = [getattr(parameter, "grad_sample", None) for parameter in parameters]
    summed_grad_refs = [getattr(parameter, "summed_grad", None) for parameter in parameters]
    had_grad_sample = [hasattr(parameter, "grad_sample") for parameter in parameters]
    had_summed_grad = [hasattr(parameter, "summed_grad") for parameter in parameters]
    hook_owners = _opacus_hook_owners(model)
    hook_snapshots = [
        {
            "owner": owner,
            "enabled": bool(getattr(owner, "hooks_enabled", False)),
            "installed": hasattr(owner, "autograd_grad_sample_hooks"),
            "loss_reduction": getattr(owner, "loss_reduction", "mean"),
            "batch_first": bool(getattr(owner, "batch_first", True)),
            "force_functorch": bool(getattr(owner, "force_functorch", False)),
            "removed": False,
        }
        for owner in hook_owners
    ]
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    # Opacus' ``remove_hooks`` also destroys per-layer DP-DDP hooks, while
    # ``add_hooks`` cannot recreate them.  Refuse the diagnostic before any
    # mutation in that unsupported setup; the Trainer will record probe/error.
    if hook_snapshots and any(getattr(parameter, "ddp_hooks", None) for parameter in parameters):
        raise RuntimeError(
            "gradient probe cannot temporarily remove Opacus hooks while "
            "DP-DDP per-layer hooks are installed"
        )

    try:
        # ``disable_hooks()`` is insufficient here: PyTorch still inserts an
        # autograd ``BackwardHookFunction`` for a registered full backward
        # hook, and that Function is incompatible with ``torch.func.vmap``.
        # There is no live training graph at any supported probe call site, so
        # temporarily remove Opacus' handles and recreate them afterwards.
        for snapshot in hook_snapshots:
            owner = snapshot["owner"]
            if snapshot["installed"]:
                # Mark before calling: if Opacus fails halfway through removal,
                # the finally block must attempt restoration and, if that also
                # fails, raise the fatal restoration error rather than resume
                # DP training in an unknown hook state.
                snapshot["removed"] = True
                owner.remove_hooks()
            else:
                disable_hooks = getattr(owner, "disable_hooks", None)
                if callable(disable_hooks):
                    disable_hooks()
        for module in modules:
            module.training = False
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            yield
    finally:
        try:
            with torch.no_grad():
                for parameter, value in zip(parameters, parameter_values):
                    if not torch.equal(parameter, value):
                        parameter.copy_(value)
                for buffer, value in zip(buffers, buffer_values):
                    if not torch.equal(buffer, value):
                        buffer.copy_(value)
            for parameter, grad_ref in zip(parameters, grad_refs):
                parameter.grad = grad_ref
            for parameter, existed, value in zip(parameters, had_grad_sample, grad_sample_refs):
                if existed:
                    parameter.grad_sample = value
                elif hasattr(parameter, "grad_sample"):
                    delattr(parameter, "grad_sample")
            for parameter, existed, value in zip(parameters, had_summed_grad, summed_grad_refs):
                if existed:
                    parameter.summed_grad = value
                elif hasattr(parameter, "summed_grad"):
                    delattr(parameter, "summed_grad")
            for module, training in zip(modules, training_flags):
                module.training = training
            for snapshot in reversed(hook_snapshots):
                owner = snapshot["owner"]
                if snapshot["removed"]:
                    owner.add_hooks(
                        loss_reduction=snapshot["loss_reduction"],
                        batch_first=snapshot["batch_first"],
                        force_functorch=snapshot["force_functorch"],
                    )
                if snapshot["enabled"]:
                    owner.enable_hooks()
                else:
                    owner.disable_hooks()
        except Exception as exc:
            # Continuing DP training after a failed hook/state restoration can
            # silently remove clipping/accounting.  This error is always fatal
            # in Trainer, independently of diagnostics.fail_on_error.
            raise GradientProbeStateRestorationError(
                "public gradient probe failed to restore training-critical state"
            ) from exc


def _opacus_hook_owners(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Find GradSampleModule-like wrappers without importing Opacus here."""
    owners: list[torch.nn.Module] = []
    seen: set[int] = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if all(
            callable(getattr(module, name, None))
            for name in ("add_hooks", "remove_hooks", "enable_hooks", "disable_hooks")
        ) and hasattr(module, "hooks_enabled"):
            owners.append(module)
    return owners


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    seen: set[int] = set()
    while hasattr(current, "_module") and id(current) not in seen:
        seen.add(id(current))
        current = getattr(current, "_module")
    return current


def _validate_probe_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim < 2 or y.ndim != 1 or int(x.shape[0]) != int(y.shape[0]):
        raise ValueError(f"invalid probe shapes: x={tuple(x.shape)}, y={tuple(y.shape)}")
    if int(x.shape[0]) <= 0:
        raise ValueError("gradient probe batch is empty")
    if x.device != y.device:
        raise ValueError("probe inputs and labels must be on the same device")


def _distribution_metrics(
    prefix: str,
    values: torch.Tensor,
    *,
    eps: float,
    near_zero: bool = False,
) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        result = {
            f"{prefix}/{key}": 0.0
            for key in (
                "mean",
                "std",
                "p5",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
            )
        }
        if near_zero:
            result[f"{prefix}/near_zero_frac"] = 0.0
        return result
    quantiles = torch.quantile(values, values.new_tensor(_QUANTILES))
    result = {
        f"{prefix}/mean": float(values.mean().item()),
        f"{prefix}/std": float(values.std(unbiased=False).item()),
        f"{prefix}/max": float(values.max().item()),
    }
    for quantile, value in zip(_QUANTILES, quantiles):
        result[f"{prefix}/p{int(round(100 * quantile))}"] = float(value.item())
    if near_zero:
        result[f"{prefix}/near_zero_frac"] = _mean_bool(values <= max(eps, 1e-12))
    return result


def _selected_quantiles(prefix: str, values: torch.Tensor, quantiles: tuple[float, ...]) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {f"{prefix}/p{int(round(100 * q))}": 0.0 for q in quantiles}
    result_values = torch.quantile(values, values.new_tensor(quantiles))
    return {
        f"{prefix}/p{int(round(100 * quantile))}": float(value.item())
        for quantile, value in zip(quantiles, result_values)
    }


def _histogram_metrics(prefix: str, values: torch.Tensor, bins: int = 20) -> dict[str, float]:
    """Return aggregate bin edges/counts; never persist the underlying examples."""
    values = values.detach().float().flatten().cpu()
    if values.numel() == 0:
        values = torch.zeros(1)
    low = float(values.min().item())
    high = float(values.max().item())
    if not math.isfinite(low) or not math.isfinite(high):
        values = torch.nan_to_num(values)
        low = float(values.min().item())
        high = float(values.max().item())
    if high <= low:
        width = max(1e-6, abs(low) * 1e-6)
        low -= width
        high += width
    counts, edges = torch.histogram(values, bins=int(bins), range=(low, high))
    metrics = {
        f"{prefix}/count_{index:02d}": float(value.item())
        for index, value in enumerate(counts)
    }
    metrics.update(
        {
            f"{prefix}/edge_{index:02d}": float(value.item())
            for index, value in enumerate(edges)
        }
    )
    return metrics


def _alpha_metrics(prefix: str, alpha: torch.Tensor) -> dict[str, float]:
    metrics = {f"{prefix}/alpha_mean": _safe_mean(alpha)}
    for quantile, value in _selected_quantiles(f"{prefix}/alpha", alpha, (0.10, 0.50, 0.90)).items():
        metrics[quantile] = value
    return metrics


def _counterfactual_metrics(prefix: str, update: torch.Tensor, raw: torch.Tensor, eps: float) -> dict[str, float]:
    raw_sq = float(raw.square().sum().item())
    return {
        f"{prefix}/cos_to_raw": _vector_cos(update, raw, eps),
        f"{prefix}/norm_ratio_to_raw": float(update.norm().item()) / (math.sqrt(max(0.0, raw_sq)) + eps),
        f"{prefix}/relative_sq_error": float((update - raw).square().sum().item()) / (raw_sq + eps),
    }


def _distortion_metrics(prefix: str, update: torch.Tensor, raw: torch.Tensor, eps: float) -> dict[str, float]:
    raw_sq = float(raw.square().sum().item())
    parallel = float((update * raw).sum().item()) / (raw_sq + eps)
    orthogonal = update - parallel * raw
    return {
        f"{prefix}/parallel_retention": parallel,
        f"{prefix}/orthogonal_relative_norm": float(orthogonal.norm().item()) / (math.sqrt(max(0.0, raw_sq)) + eps),
    }


def _descent_projection_metrics(
    mechanism: str,
    update: torch.Tensor,
    task: torch.Tensor,
    robust: torch.Tensor,
    total: torch.Tensor,
    tau: float,
    eps: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    positives: dict[str, bool] = {}
    for label, direction in (("task", task), ("robust", robust), ("total", total)):
        dot = float((direction * update).sum().item())
        direction_norm = float(direction.norm().item())
        z = dot / (abs(tau) * direction_norm + eps)
        probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        metrics[f"probe/descent/{mechanism}/{label}_z"] = z
        metrics[f"probe/descent/{mechanism}/{label}_probability"] = probability
        positives[label] = dot > 0.0
        if label != "total":
            metrics[f"probe/projection/{mechanism}/{label}"] = dot
            metrics[f"probe/projection/{mechanism}/{label}_normalized"] = dot / (
                float(direction.square().sum().item()) + eps
            )
            metrics[f"probe/descent/{mechanism}/{label}_positive"] = 1.0 if positives[label] else 0.0
    metrics[f"probe/descent/{mechanism}/pareto_positive"] = (
        1.0 if positives["task"] and positives["robust"] else 0.0
    )
    return metrics


def _vector_cos(left: torch.Tensor, right: torch.Tensor, eps: float) -> float:
    denominator = float(left.norm().item()) * float(right.norm().item())
    if denominator <= eps:
        return 0.0
    return float(((left * right).sum().item()) / (denominator + eps))


def _effective_sample_size(alpha: torch.Tensor, eps: float) -> float:
    return float(alpha.sum().square().item() / (alpha.square().sum().item() + eps))


def _safe_mean(values: torch.Tensor) -> float:
    return 0.0 if values.numel() == 0 else float(values.float().mean().item())


def _mean_bool(values: torch.Tensor) -> float:
    return 0.0 if values.numel() == 0 else float(values.float().mean().item())


def _safe_fraction(values: torch.Tensor) -> float:
    return _mean_bool(values)


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks with deterministic tie handling, using only torch."""
    values = values.detach().flatten().double().cpu()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        stop = start + 1
        while stop < values.numel() and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2 or left.shape != right.shape:
        return 0.0
    x = _rankdata(left)
    y = _rankdata(right)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float(torch.dot(x, y).item() / denominator.item())


def _finite_scalar_metrics(metrics: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if not str(key).startswith("probe/"):
            raise ValueError(f"unsafe gradient-probe metric key: {key}")
        scalar = float(value)
        if not math.isfinite(scalar):
            scalar = 0.0
        result[str(key)] = scalar
    return result
