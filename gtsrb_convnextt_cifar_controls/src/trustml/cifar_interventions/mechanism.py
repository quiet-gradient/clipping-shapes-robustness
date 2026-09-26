"""Per-record maps; h is already beta*r. No averaging or randomness here."""
from __future__ import annotations

import hashlib
import hmac
from contextlib import contextmanager

import torch

ARMS = ("J", "U", "D", "M", "R")


def unit(x):
    n = x.norm(dim=-1, keepdim=True)
    # Direction of zero is defined as zero (including cancellation cases).
    return x / torch.where(n > 0, n, torch.ones_like(n))


def maps(g, h, *, map_name, bound, q, gamma):
    if g.shape != h.shape or g.ndim != 2:
        raise ValueError("expected aligned [records, trainable_parameters] branches")
    if map_name not in {"hard", "autos"} or not 0 < q < 1 or bound <= 0 or gamma <= 0:
        raise ValueError("invalid map parameters")
    if map_name == "autos" and bound != 1:
        raise ValueError("AUTO-S uses unit sensitivity")
    if not torch.isfinite(g).all() or not torch.isfinite(h).all():
        raise RuntimeError("nonfinite branch gradient")
    s = g + h
    ng, nh = g.norm(dim=-1, keepdim=True), h.norm(dim=-1, keepdim=True)
    ns = s.norm(dim=-1, keepdim=True)
    if map_name == "hard":
        def factor(n, c):
            return (c / n.clamp_min(torch.finfo(n.dtype).tiny)).clamp(max=1)
        j = s * factor(ns, bound)
        u = g * factor(ng, (1-q)*bound) + h * factor(nh, q*bound)
        r = s * factor(ng+nh, bound)
    else:
        j = s / (ns + gamma)
        u = (1-q)*g/(ng+(1-q)*gamma) + q*h/(nh+q*gamma)
        r = s / (ng+nh+gamma)
    return j, u, r


def contribution(g, h, *, arm, **kwargs):
    if arm not in ARMS:
        raise ValueError(f"unknown intervention {arm}")
    j, u, r = maps(g, h, **kwargs)
    if arm == "J":
        result = j
    elif arm == "U":
        result = u
    elif arm == "D":
        result = unit(j) * u.norm(dim=-1, keepdim=True)
    elif arm == "M":
        result = unit(u) * j.norm(dim=-1, keepdim=True)
    else:
        result = r
    if not torch.isfinite(result).all() or (result.norm(dim=-1) > kwargs["bound"] * (1+2e-6)).any():
        raise RuntimeError("query sensitivity/finite invariant failed")
    return result


def release(summed, *, sigma, bound, denominator, generator):
    if denominator <= 0 or sigma <= 0 or bound <= 0:
        raise ValueError("invalid release parameters")
    z = torch.randn(summed.shape, device=summed.device, dtype=summed.dtype, generator=generator)
    return (summed + sigma * bound * z) / denominator


class Streams:
    """RAM-only private root; reset per arm, never store seeds/states in artifacts.

    The five outputs have correlated noise. Individual-run accounting is NOT
    a DP guarantee for jointly releasing this mechanistic diagnostic bundle.
    """
    def __init__(self, secret: bytes, device="cpu"):
        if len(secret) < 32:
            raise ValueError("private root needs at least 256 bits")
        self._secret = secret
        self.generators = {}
        for name in ("sampling", "augmentation", "attack", "noise"):
            seed = int.from_bytes(hmac.digest(secret, name.encode(), "sha256")[:8], "big") % (2**63-1)
            self.generators[name] = torch.Generator(device=device if name in {"attack", "noise"} else "cpu").manual_seed(seed)

    def __getitem__(self, name):
        return self.generators[name]

    def commitment(self):
        return hashlib.sha256(self._secret).hexdigest()

    def positions(self):
        return {name: hmac.digest(self._secret, gen.get_state().cpu().numpy().tobytes(), "sha256").hex()
                for name, gen in self.generators.items()}

    @contextmanager
    def augmentation(self):
        # torchvision random crops/flips use CPU torch RNG. No worker races.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self["augmentation"].get_state())
            try:
                yield
            finally:
                self["augmentation"].set_state(torch.get_rng_state())
