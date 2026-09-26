"""Raw-pixel attacks. Fixed per-record starts for the public geometry stream."""
import hashlib

import torch
import torch.nn.functional as F


def public_uniform(pixels, sample_ids, seed, stream):
    noise = []
    for sample_id in sample_ids:
        value = int.from_bytes(hashlib.sha256(f"{seed}:{stream}:{sample_id}".encode()).digest()[:8], "big") % (2**63)
        generator = torch.Generator(device=pixels.device).manual_seed(value)
        noise.append(torch.rand(pixels.shape[1:], dtype=pixels.dtype, device=pixels.device, generator=generator))
    return torch.stack(noise)


def attack(model, pixels, labels=None, *, objective, steps=10, epsilon=8/255, step_size=2/255,
           generator=None, uniform=None, track_best=False):
    if model.training or pixels.dtype != torch.float32:
        raise ValueError("attack requires FP32 model.eval()")
    original = pixels.detach()
    if original.min() < 0 or original.max() > 1:
        raise ValueError("attack input must be raw [0,1] pixels")
    if uniform is None:
        uniform = torch.rand(pixels.shape, device=pixels.device, dtype=pixels.dtype, generator=generator)
    with torch.no_grad():
        clean_logits = model(original)
        probs = clean_logits.softmax(-1)
        robust = clean_logits.argmax(-1).eq(labels) if labels is not None else None
    adversarial = (original + (2*uniform-1)*epsilon).clamp(0, 1).detach()
    for iteration in range(steps + 1):
        adversarial.requires_grad_(True)
        logits = model(adversarial)
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite attack logits")
        if track_best:
            robust &= logits.argmax(-1).eq(labels).detach()
        if iteration == steps:
            break
        if objective == "kl":
            losses = F.kl_div(logits.log_softmax(-1), probs, reduction="none").sum(-1)
        elif objective == "ce":
            losses = F.cross_entropy(logits, labels, reduction="none")
        else:
            raise ValueError(objective)
        gradient = torch.autograd.grad(losses.sum(), adversarial)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite attack gradient")
        adversarial = adversarial.detach() + step_size * gradient.sign()
        adversarial = (original + (adversarial-original).clamp(-epsilon, epsilon)).clamp(0, 1).detach()
    if (adversarial-original).abs().max() > epsilon + 1e-6:
        raise RuntimeError("Linf bound violated")
    return adversarial.detach(), robust
