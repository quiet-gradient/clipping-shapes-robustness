"""Tiny CPU-only regression checks for the explicit outer-KL ablation."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from trustml.convnext_gtsrb.mechanism import gradients, contribution
from trustml.convnext_gtsrb import probe
from trustml.ssg_gtsrb.private import train


def fixture(dtype=torch.float64):
    torch.manual_seed(20260923)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 5),
                                torch.nn.Tanh(), torch.nn.Linear(5, 3)).to(dtype).eval()
    for parameter in model[1].parameters():
        parameter.requires_grad_(False)
    clean = torch.rand(3, 3, 2, 2, dtype=dtype)
    adversarial = (clean + .2).clamp(0, 1)
    return model, clean, adversarial, torch.tensor([0, 1, 2])


@pytest.mark.parametrize("target", ["full", "detached_clean"])
@pytest.mark.parametrize("stable_public_kl", [False, True])
def test_outer_gradient_matches_independent_per_record_autograd(target, stable_public_kl):
    model, clean, adversarial, labels = fixture()
    beta = 2.3
    values = gradients(model, clean, labels, adversarial, beta=beta,
                       outer_kl_target=target, stable_public_kl=stable_public_kl)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    for i in range(len(clean)):
        logits = model(clean[i:i+1])
        p = logits.softmax(-1)
        logq = model(adversarial[i:i+1]).log_softmax(-1)
        full_loss = F.kl_div(logq, p, reduction="sum")
        detached_loss = F.kl_div(logq, p.detach(), reduction="sum")
        torch.testing.assert_close(full_loss, detached_loss, rtol=0, atol=0)
        loss = full_loss if target == "full" else detached_loss
        expected_h = torch.cat([g.flatten() for g in torch.autograd.grad(beta * loss, parameters,
                                                                        retain_graph=True)])
        expected_g = torch.cat([g.flatten() for g in torch.autograd.grad(
            F.cross_entropy(logits, labels[i:i+1]), parameters)])
        torch.testing.assert_close(values["h"][i], expected_h, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(values["g"][i], expected_g, atol=1e-12, rtol=1e-10)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_default_full_is_exact_and_missing_term_is_the_clean_branch():
    model, clean, adversarial, labels = fixture()
    default = gradients(model, clean, labels, adversarial)
    full = gradients(model, clean, labels, adversarial, outer_kl_target="full")
    stopped = gradients(model, clean, labels, adversarial, outer_kl_target="detached_clean")
    for name in default:
        torch.testing.assert_close(default[name], full[name], rtol=0, atol=0)
    torch.testing.assert_close(full["g"], stopped["g"], rtol=0, atol=0)
    assert (full["h"] - stopped["h"]).norm() > 1e-5
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    for i in range(len(clean)):
        p = model(clean[i:i+1]).softmax(-1)
        logq = model(adversarial[i:i+1]).log_softmax(-1).detach()
        # Fixed log-ratio weights give sum_j log(p_j/q_j) grad_theta p_j.
        missing = (p * (p.detach().log() - logq)).sum()
        reference = torch.cat([g.flatten() for g in torch.autograd.grad(missing, parameters)])
        torch.testing.assert_close(full["h"][i] - stopped["h"][i], reference,
                                   rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("target", ["full", "detached_clean"])
def test_each_convention_preserves_record_bound_and_microbatch_aggregation(target):
    model, clean, adversarial, labels = fixture()
    values = gradients(model, clean, labels, adversarial, outer_kl_target=target)
    for role, q in [("Joint", None), ("Componentwise", .25)]:
        query = contribution(values, role, .03, q)
        assert torch.all(query.norm(dim=1) <= .03 + 1e-12)
        parts = []
        for i in range(len(clean)):
            single = gradients(model, clean[i:i+1], labels[i:i+1], adversarial[i:i+1],
                               outer_kl_target=target)
            parts.append(contribution(single, role, .03, q))
        torch.testing.assert_close(torch.cat(parts).sum(0), query.sum(0), rtol=1e-10, atol=1e-12)


def test_probe_routes_convention_but_scalar_outputs_stay_equal(monkeypatch):
    model, clean, adversarial, labels = fixture(torch.float32)
    identities = [dict(image_id=str(i), track_id=str(i), label=i) for i in range(3)]
    data = SimpleNamespace(split={"indices": {"probe": [0, 1, 2]}, "records": identities},
                           batch=lambda ids: (clean[ids], labels[ids]))
    monkeypatch.setattr(probe, "attack", lambda model, x, y, **kwargs: ((x + .2).clamp(0, 1), None))
    full = list(probe.probe_batches(model, data, microbatch=3, seed=42))
    stopped = list(probe.probe_batches(model, data, microbatch=3, seed=42,
                                      outer_kl_target="detached_clean"))
    for name in full[0][2]:
        torch.testing.assert_close(full[0][2][name], stopped[0][2][name], rtol=0, atol=0)
    expected = gradients(model, clean, labels, adversarial, outer_kl_target="detached_clean",
                         stable_public_kl=True)
    torch.testing.assert_close(stopped[0][0]["h"], expected["h"], rtol=0, atol=0)
    assert not torch.equal(full[0][0]["h"], stopped[0][0]["h"])


@pytest.mark.parametrize("target", [None, "full", "detached_clean"])
def test_geometry_uses_training_convention_and_retains_explicit_provenance(monkeypatch, tmp_path, target):
    model, clean, adversarial, labels = fixture(torch.float32)
    identities = [dict(image_id=str(i), track_id=str(i), label=i) for i in range(3)]
    data = SimpleNamespace(split={"indices": {"probe": [0, 1, 2]}, "records": identities},
                           batch=lambda ids: (clean[ids], labels[ids]))
    monkeypatch.setattr(probe, "attack", lambda model, x, y, **kwargs: ((x + .2).clamp(0, 1), None))
    seen = {}

    class CaptureAccumulator:
        def __init__(self, *args, provenance, **kwargs):
            seen["provenance"] = provenance

        def update(self, g, h, a, identity, outputs):
            seen["h"] = h

        def finalize(self, *args, **kwargs):
            return {"provenance": seen["provenance"]}

    monkeypatch.setattr(probe, "GeometryAccumulator", CaptureAccumulator)
    config = {"private": False, "geometry_q_grid": [.25], "geometry_attack_seed": 42}
    if target is not None:
        config["outer_kl_target"] = target
    provenance = {"run_id": "tiny_public_test"}
    result = probe.audit_geometry(model, data, {"blocks": {}, "trainable_parameter_count": 18},
                                  {"C_low": .1, "C_mid": 1., "C_high": 4.}, config,
                                  tmp_path, provenance, microbatch=3)
    expected = gradients(model, clean, labels, adversarial, outer_kl_target=target or "full",
                         stable_public_kl=True)
    torch.testing.assert_close(seen["h"], expected["h"], rtol=0, atol=0)
    assert result["provenance"] == (provenance if target is None else
                                     {**provenance, "outer_kl_target": target})
    assert "outer_kl_target" not in provenance


def test_invalid_convention_rejected_before_training_output(tmp_path):
    model, clean, adversarial, labels = fixture()
    with pytest.raises(ValueError, match="outer_kl_target"):
        gradients(model, clean, labels, adversarial, outer_kl_target="typo")
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="outer_kl_target"):
        train({"outer_kl_target": "typo"}, {}, None, {}, {}, None, None, None, output)
    assert not output.exists()
