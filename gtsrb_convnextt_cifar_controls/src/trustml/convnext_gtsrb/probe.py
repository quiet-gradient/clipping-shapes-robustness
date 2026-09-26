"""Public calibration and counterfactual geometry on a fixed 512-record probe."""
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attacks import attack, public_uniform
from .geometry import GeometryAccumulator, summary
from .mechanism import gradients, validate_outer_kl_target
from trustml.convnext_cifar100.preflight import write_json


def margin(logits, labels):
    truth = logits.gather(1, labels[:, None]).flatten()
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    return truth-other.max(-1).values


def probe_batches(model, data, *, microbatch, seed, beta=1., outer_kl_target="full"):
    validate_outer_kl_target(outer_kl_target)
    device = next(model.parameters()).device
    ids = data.split["indices"]["probe"]
    for begin in range(0, len(ids), microbatch):
        identity = [data.split["records"][i] for i in ids[begin:begin+microbatch]]
        x, y = data.batch(ids[begin:begin+microbatch])
        x, y = x.to(device), y.to(device)
        image_ids = [r["image_id"] for r in identity]
        trades_x, _ = attack(model, x, y, objective="kl", uniform=public_uniform(x, image_ids, seed, "trades"))
        ce_x, _ = attack(model, x, y, objective="ce", uniform=public_uniform(x, image_ids, seed, "pgd_at"))
        values = gradients(model, x, y, trades_x, ce_x, beta=beta, stable_public_kl=True,
                           outer_kl_target=outer_kl_target)
        with torch.no_grad():
            z, zt, za = model(x), model(trades_x), model(ce_x)
            outputs = {"clean_ce": F.cross_entropy(z, y, reduction="none"),
                "trades_kl": F.kl_div(zt.log_softmax(-1), z.softmax(-1), reduction="none").sum(-1),
                "adversarial_ce": F.cross_entropy(za, y, reduction="none"),
                "clean_correct": z.argmax(-1).eq(y), "adversarial_correct": za.argmax(-1).eq(y),
                "trades_adversarial_correct": zt.argmax(-1).eq(y),
                "clean_margin": margin(z, y), "adversarial_margin": margin(za, y), "trades_adversarial_margin": margin(zt, y),
                "pgd_realized_linf": (ce_x-x).flatten(1).abs().max(-1).values,
                "trades_realized_linf": (trades_x-x).flatten(1).abs().max(-1).values}
        if begin % 64 == 0:
            print(json.dumps({"stage": "public_geometry", "records": min(begin+microbatch, len(ids))}), flush=True)
        yield values, identity, outputs


def calibrate_clipping(model, data, output, *, microbatch, seed):
    started = time.monotonic()
    collected = []
    for values, _, _ in probe_batches(model, data, microbatch=microbatch, seed=seed):
        g, h, a = (values[k] for k in ["g", "h", "a"])
        collected.extend(torch.stack([v.double().norm(dim=-1) for v in [g, h, g+h, a]], dim=1).cpu().tolist())
    values = np.asarray(collected)
    med = np.median(values, axis=0)
    if values.shape != (512, 4) or not np.isfinite(values).all() or np.any(med[[0, 2, 3]] <= 0):
        raise RuntimeError("invalid public calibration")
    bounds = {"C_low": float(med[2]/4), "C_mid": float(med[2]), "C_high": float(med[2]*4),
              "C_CE": float(med[0]), "C_ADV": float(med[3])}
    result = {"status": "PASS", **bounds, "gradient_norms": {k: summary(values[:, i]) for i, k in enumerate(["g", "h", "s", "a"])},
        "clipping_fractions": {c: {k: float(np.mean(values[:, i] > C)) for i, k in enumerate(["g", "h", "s", "a"])} for c, C in bounds.items()},
        "qstar": summary(values[:, 1]/(values[:, 0]+values[:, 1])), "probe_hash": data.split["hashes"]["probe"],
        "elapsed_seconds": time.monotonic()-started, "records": 512, "private_calibration_records": 0}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"initial_gradient_norms.npz", norms=values, probe_indices=data.split["indices"]["probe"])
    write_json(output/"calibration.json", result)
    return result


def audit_geometry(model, data, architecture, calibration, config, output, provenance, *, microbatch):
    outer_kl_target = validate_outer_kl_target(config.get("outer_kl_target", "full"))
    if "outer_kl_target" in config:
        provenance = {**provenance, "outer_kl_target": outer_kl_target}
    bounds = {k: calibration[k] for k in ["C_low", "C_mid", "C_high"]}
    actual = ({k: config[k] for k in ["role", "C", "q"]}
              if config.get("bounded_query", config.get("private")) else None)
    if actual is not None and config.get('clipping_map') == 'auto_s':
        actual.update({k: config[k] for k in ['clipping_map', 'gamma', 'lr']})
    if actual is not None and 'intervention' in config:
        actual.update({k: config[k] for k in ['intervention', 'intervention_q']})
    if actual is not None and 'uniform_joint_scale' in config:
        actual['uniform_joint_scale'] = config['uniform_joint_scale']
    accumulator = GeometryAccumulator(bounds, config["geometry_q_grid"], architecture["blocks"],
        architecture["trainable_parameter_count"], device=next(model.parameters()).device, provenance=provenance, actual=actual)
    started = time.monotonic()
    for values, identity, outputs in probe_batches(model, data, microbatch=microbatch,
            seed=config["geometry_attack_seed"], outer_kl_target=outer_kl_target):
        accumulator.update(values["g"], values["h"], values["a"], identity, outputs)
    expected = [data.split["records"][i] for i in data.split["indices"]["probe"]]
    result = accumulator.finalize(output, expected, sigma=config.get("sigma", 0))
    result["elapsed_seconds"] = time.monotonic()-started
    write_json(Path(output)/"gtsrb_geometry_aggregate.json", result)
    return result
