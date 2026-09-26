"""Per-image CE / adversarial CE / explicit outer-KL gradients, one release."""
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from trustml.convnext_cifar100.mechanism import clip, private_release, assign_gradient, lr_at


def validate_outer_kl_target(value):
    """The historical GTSRB default differentiates both model predictions."""
    if value not in ("full", "detached_clean"):
        raise ValueError(f"unknown outer_kl_target: {value!r}")
    return value


def gradients(model, clean, labels, trades_adv=None, ce_adv=None, *, beta=1., need_g=True,
              stable_public_kl=False, outer_kl_target="full"):
    validate_outer_kl_target(outer_kl_target)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fixed = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = dict(model.named_buffers())
    def predict(params, sample):
        return functional_call(model, ({**fixed, **params}, buffers), (sample.unsqueeze(0),), strict=True)
    def ce(params, sample, label):
        return F.cross_entropy(predict(params, sample), label.unsqueeze(0), reduction="sum")
    def kl(params, sample, adversarial):
        clean_logits = predict(params, sample)
        logq = predict(params, adversarial).log_softmax(-1)
        # The ablation changes only the outer target; x_adv remains fixed in both cases.
        if outer_kl_target == "detached_clean":
            clean_logits = clean_logits.detach()
        if stable_public_kl:
            # Public diagnostics only. Avoid differentiating xlogy(p, p) at
            # p=0 after FP32 softmax underflow. Same KL and chosen outer gradient;
            # no clamp, smoothing, or change to the private numerical path.
            logp = clean_logits.log_softmax(-1)
            return beta * (logp.exp() * (logp - logq)).sum()
        p = clean_logits.softmax(-1)
        return beta * F.kl_div(logq, p, reduction="sum")
    def flatten(values):
        result = torch.cat([values[n].reshape(len(clean), -1) for n in trainable], dim=1).detach()
        if not torch.isfinite(result).all():
            raise RuntimeError("nonfinite per-record gradient")
        return result
    result = {}
    if need_g:
        result["g"] = flatten(vmap(grad(ce), in_dims=(None, 0, 0), randomness="error")(trainable, clean, labels))
    if trades_adv is not None:
        result["h"] = flatten(vmap(grad(kl), in_dims=(None, 0, 0), randomness="error")(trainable, clean, trades_adv.detach()))
    if ce_adv is not None:
        result["a"] = flatten(vmap(grad(ce), in_dims=(None, 0, 0), randomness="error")(trainable, ce_adv.detach(), labels))
    return result


def contribution(values, role, C, q=None):
    if role == "DPOnly":
        return clip(values["g"], C)
    if role == "DPPGD":
        return clip(values["a"], C)
    if role == "Joint":
        return clip(values["g"] + values["h"], C)
    if role == "Componentwise":
        if not 0 < q < 1:
            raise ValueError("invalid allocation")
        return clip(values["g"], (1-q)*C) + clip(values["h"], q*C)
    raise ValueError("non-private roles do not call the private query")
