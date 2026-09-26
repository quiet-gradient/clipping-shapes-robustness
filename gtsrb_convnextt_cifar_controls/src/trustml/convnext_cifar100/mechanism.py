"""Global record queries; a single summed Gaussian release; diagnostic controls."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def clip(values, bound):
    norms = values.norm(dim=-1, keepdim=True)
    return values * (bound / norms.clamp_min(torch.finfo(values.dtype).tiny)).clamp(max=1.)


def query(g, h, *, bound, q, arm):
    joint = clip(g + h, bound)
    if arm == "J":
        return joint
    component = clip(g, (1-q)*bound) + clip(h, q*bound)
    if arm == "U":
        return component
    jnorm = joint.norm(dim=-1, keepdim=True)
    unorm = component.norm(dim=-1, keepdim=True)
    tiny = torch.finfo(g.dtype).tiny
    if arm == "D":
        return joint * (unorm / jnorm.clamp_min(tiny))
    if arm == "M":
        return component * (jnorm / unorm.clamp_min(tiny))
    if arm == "R":
        denom = g.norm(dim=-1, keepdim=True) + h.norm(dim=-1, keepdim=True)
        return (g+h) * (bound / denom.clamp_min(tiny)).clamp(max=1.)
    raise ValueError(arm)


def private_release(summed, *, sigma, bound, generator, denominator=512):
    noise = torch.randn(summed.shape, dtype=summed.dtype, device=summed.device, generator=generator)
    return (summed + noise * (sigma * bound)) / denominator


def lr_at(step, peak, *, total=1760, warmup=176):
    """Step is 1-based: first update peak/176, update 176 peak, final update zero."""
    if not 1 <= step <= total:
        raise ValueError(step)
    if step <= warmup:
        return peak * step / warmup
    return peak * .5 * (1 + math.cos(math.pi * (step-warmup)/(total-warmup)))


def gradient_branches(model, x, y, adversarial, clean_probs):
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    buffers = dict(model.named_buffers())
    def ce(params, sample, label):
        logits = functional_call(model, (params, buffers), (sample.unsqueeze(0),), strict=False)
        return F.cross_entropy(logits, label.unsqueeze(0), reduction="sum")
    def kl(params, sample, probs):
        logits = functional_call(model, (params, buffers), (sample.unsqueeze(0),), strict=False)
        return F.kl_div(F.log_softmax(logits, dim=-1), probs.detach().unsqueeze(0), reduction="sum")
    gt = vmap(grad(ce), in_dims=(None,0,0), randomness="error")(parameters,x,y)
    ht = vmap(grad(kl), in_dims=(None,0,0), randomness="error")(parameters,adversarial.detach(),clean_probs.detach())
    g = torch.cat([gt[n].reshape(len(x), -1) for n in parameters], dim=1)
    h = torch.cat([ht[n].reshape(len(x), -1) for n in parameters], dim=1)
    return g.detach(), h.detach()


def assign_gradient(model, flat):
    position = 0
    for p in model.parameters():
        if p.requires_grad:
            p.grad = flat[position:position+p.numel()].reshape_as(p)
            position += p.numel()
        elif p.grad is not None:
            raise RuntimeError("frozen parameter received gradient")
    if position != flat.numel():
        raise ValueError("flat gradient dimension mismatch")
