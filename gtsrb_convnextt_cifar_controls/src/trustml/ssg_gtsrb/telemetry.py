"""Scalars only, computed exclusively from the sanitized gradient and DP state."""
import math

import torch


def cosine(a, b):
    denominator = a.double().norm() * b.double().norm()
    return float((a.double().dot(b.double()) / denominator).clamp(-1, 1)) if denominator > 0 else None


def optimizer_step(optimizer, parameters):
    """Record theta_after-theta_before, so descent normally gives negative cosine.

    SGD coupled decay contribution is the current gradient's instantaneous
    decay effect; historical decay is already contained in the momentum buffer.
    AdamW decay is decoupled. No full vectors are returned or persisted.
    """
    parameters = list(parameters)
    if len(optimizer.param_groups) != 1 or any(p.grad is None for p in parameters):
        raise ValueError("one parameter group and complete sanitized gradient required")
    group = optimizer.param_groups[0]
    if group.get("maximize", False):
        raise ValueError("maximize optimizer not supported")
    before = torch.cat([p.detach().flatten() for p in parameters])
    gradient = torch.cat([p.grad.detach().flatten() for p in parameters])
    lr, wd = group["lr"], group["weight_decay"]
    optimizer.step()
    after = torch.cat([p.detach().flatten() for p in parameters])
    update = after - before
    report = {"sanitized_gradient_norm": float(gradient.double().norm()),
              "parameter_update_norm": float(update.double().norm()),
              "gradient_update_cosine": cosine(gradient, update), "learning_rate": lr,
              "update_sign_convention": "theta_after_minus_theta_before"}
    if isinstance(optimizer, torch.optim.SGD):
        if not group["nesterov"] or group["dampening"] != 0:
            raise ValueError("locked Nesterov SGD required")
        buffer = torch.cat([optimizer.state[p]["momentum_buffer"].detach().flatten() for p in parameters])
        direction = gradient + wd * before + group["momentum"] * buffer
        report.update(momentum_buffer_norm=float(buffer.double().norm()),
                      nesterov_direction_norm=float(direction.double().norm()),
                      weight_decay_contribution_norm=float((lr * (1 + group["momentum"]) * wd * before).double().norm()),
                      weight_decay_convention="current_step_coupled_decay_including_current_momentum_injection_not_history")
    elif isinstance(optimizer, torch.optim.AdamW):
        first = torch.cat([optimizer.state[p]["exp_avg"].detach().flatten() for p in parameters])
        second = torch.cat([optimizer.state[p]["exp_avg_sq"].detach().flatten() for p in parameters])
        steps = {int(optimizer.state[p]["step"].item()) for p in parameters}
        if len(steps) != 1:
            raise RuntimeError("Adam optimizer counters differ")
        step = next(iter(steps))
        b1, b2 = group["betas"]
        direction = (first / (1 - b1**step)) / ((second / (1 - b2**step)).sqrt() + group["eps"])
        report.update(first_moment_norm=float(first.double().norm()),
                      second_moment_rms=float(second.double().square().mean().sqrt()),
                      sqrt_mean_second_moment=float(second.double().mean().sqrt()),
                      preconditioned_step_norm=float((lr * direction).double().norm()),
                      moment_gradient_cosine=cosine(first, gradient),
                      preconditioned_gradient_cosine=cosine(direction, gradient),
                      weight_decay_contribution_norm=float((lr * wd * before).double().norm()),
                      weight_decay_convention="decoupled_parameter_update")
    else:
        raise ValueError("only SGD and AdamW are authorized")
    if any(isinstance(v, float) and not math.isfinite(v) for v in report.values()):
        raise RuntimeError("nonfinite optimizer telemetry")
    return report
