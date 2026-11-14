"""Gradient normalization controls and the single aggregate Gaussian release."""
from __future__ import annotations

import math
import torch

from trustml.cifar_interventions.mechanism import contribution as historical_contribution
from trustml.cifar_interventions.mechanism import maps

METHODS = ("J", "U", "D", "M", "R", "N", "nonprivate", "M_scaled")


def exact_unit(values):
    """unit(0)=0; every representable nonzero direction has unit norm.

    Scaling before computing the norm prevents squared-norm underflow. There
    is deliberately no additive epsilon or positive lower clamp.
    """
    if not torch.isfinite(values).all():
        raise ValueError("unit expects finite vectors")
    scale = values.abs().amax(dim=-1, keepdim=True)
    scaled = values / torch.where(scale > 0, scale, torch.ones_like(scale))
    length = scaled.norm(dim=-1, keepdim=True)
    return scaled / torch.where(length > 0, length, torch.ones_like(length))


def contribution(g, h, *, method, public_scale=None, **options):
    if method not in METHODS:
        raise ValueError(f"unsupported gradient control: {method}")
    if method == "nonprivate":
        if g.shape != h.shape or not torch.isfinite(g+h).all():
            raise ValueError("invalid unclipped branches")
        return g+h
    if method == "M_scaled":
        if public_scale is None or not math.isfinite(float(public_scale)) or not 0<float(public_scale)<=1:
            raise ValueError("M_scaled requires fixed public scale in (0,1]")
        value = historical_contribution(g, h, arm="M", **options)*float(public_scale)
    elif method == "N":
        _, component, _ = maps(g, h, **options)
        value = float(options["bound"])*exact_unit(component)
    else:
        value = historical_contribution(g, h, arm=method, **options)
    if not torch.isfinite(value).all() or (value.norm(dim=-1) > options["bound"]*(1+2e-6)).any():
        raise RuntimeError("query norm/finite invariant failed")
    return value




def release(summed, *, sigma, bound, denominator, generator, nonprivate=False):
    if denominator <= 0 or bound <= 0 or not torch.isfinite(summed).all():
        raise ValueError("invalid aggregate release")
    mean = summed/denominator
    if nonprivate:
        if sigma != 0:
            raise ValueError("nonprivate control must remove BOTH clipping and noise")
        return mean
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("private release requires positive noise multiplier")
    noise = torch.randn(summed.shape, device=summed.device, dtype=summed.dtype, generator=generator)
    return (summed+sigma*bound*noise)/denominator
