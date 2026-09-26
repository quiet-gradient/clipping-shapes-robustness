"""A100-only public architecture/gradient/geometry audit; zero training runs."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .accounting import calibrate
from .attacks import attack, public_uniform
from .data import GTSRBTrainOnly, digest
from .evaluation import evaluate
from .mechanism import gradients, contribution
from .model import build_model
from .probe import calibrate_clipping, audit_geometry
from .common import tensor_digest, provenance, code_identity
from trustml.convnext_cifar100.model import sha256
from trustml.convnext_cifar100.preflight import configure_gpu, write_json


def gradient_audit(model, data):
    x, y = data.batch(data.split["indices"]["probe"][:2])
    x, y = x.cuda(), y.cuda()
    ids = ["audit0", "audit1"]
    t, _ = attack(model, x, y, objective="kl", uniform=public_uniform(x, ids, 42, "audit_kl"))
    a, _ = attack(model, x, y, objective="ce", uniform=public_uniform(x, ids, 42, "audit_ce"))
    values = gradients(model, x, y, t, a)
    params = [p for p in model.parameters() if p.requires_grad]
    def flat(loss):
        return torch.cat([v.flatten() for v in torch.autograd.grad(loss, params)])
    refs = {"g": flat(F.cross_entropy(model(x[:1]), y[:1])),
            "h": flat(F.kl_div(model(t[:1]).log_softmax(-1), model(x[:1]).softmax(-1), reduction="sum")),
            "a": flat(F.cross_entropy(model(a[:1]), y[:1]))}
    errors = {}
    for name, ref in refs.items():
        torch.testing.assert_close(values[name][0], ref, rtol=5e-4, atol=1e-5)
        errors[name] = float((values[name][0]-ref).abs().max())
    composite = flat(F.cross_entropy(model(x[:1]), y[:1]) + F.kl_div(model(t[:1]).log_softmax(-1), model(x[:1]).softmax(-1), reduction="sum"))
    torch.testing.assert_close(values["g"][0]+values["h"][0], composite, rtol=5e-4, atol=2e-5)
    singleton = gradients(model, x[:1], y[:1], t[:1], a[:1])
    for name in values:
        torch.testing.assert_close(values[name][:1], singleton[name], rtol=5e-4, atol=1e-5)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("missing/nonfinite intended trainable gradient")
            names.append(name)
        elif p.grad is not None:
            raise RuntimeError("frozen tensor received parameter gradient")
    model.zero_grad(set_to_none=True)
    # Per-record attack construction must not depend on companion records.
    t1, _ = attack(model, x[:1], y[:1], objective="kl", uniform=public_uniform(x[:1], ids[:1], 42, "audit_kl"))
    torch.testing.assert_close(t[:1], t1, atol=1e-5, rtol=1e-4)
    return {"status": "PASS", "per_record_max_abs_errors": errors, "trainable_gradients_checked": names,
            "frozen_parameter_gradients": 0, "batch_independence": "PASS", "outer_kl_clean_target_detached": False}


def run(assets_root, output):
    device = configure_gpu()
    if "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("A100 required for the pilot")
    root, output = Path(assets_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = json.loads((root/"assets.json").read_text())
    protocol = json.loads(Path("configs/convnext_gtsrb/protocol.json").read_text())
    metadata = assets["convnext_tiny_cvst_clean.pt"]
    if sha256(root/"GTSRB_Final_Training_Images.zip") != assets["GTSRB_Final_Training_Images.zip"]["sha256"]:
        raise ValueError("dataset hash mismatch")
    model, architecture = build_model(root/"convnext_tiny_cvst_clean.pt", metadata)
    model.to(device)
    write_json(output/"architecture.json", architecture)
    initial_digest = tensor_digest({n: p for n, p in model.named_parameters() if p.requires_grad})
    data = GTSRBTrainOnly(root/"GTSRB_Final_Training_Images.zip", root/"splits.json", protocol["augmentation"])
    write_json(output/"gradient_audit.json", gradient_audit(model, data))
    privacy = calibrate(len(data.split["indices"]["private"]))
    write_json(output/"gtsrb_privacy_accounting.json", privacy)
    microbatch = 8
    calibration = calibrate_clipping(model, data, output, microbatch=microbatch, seed=protocol["geometry_attack_seed"])
    initial_path = output/"initial_trainable.pt"
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, initial_path)
    config = {**protocol, "private": False, "sigma": privacy["noise_multiplier"], "source_metadata": metadata,
              "data_sha256": assets["GTSRB_Final_Training_Images.zip"]["sha256"], "sample_rate": privacy["sample_rate"]}
    identity = provenance(config, data.split, architecture, initial_path, step=0, epoch=0, lr=0, code=code_identity())
    identity.update(phase="public_preflight", model_tensor_digest=initial_digest)
    result = audit_geometry(model, data, architecture, calibration, config, output/"geometry_epoch000", identity, microbatch=microbatch)
    evaluation = evaluate(model, data, output/"evaluation_epoch000")
    report = {"status": "PASS", "training_performed": False, "official_test_loaded": False,
        "initial_trainable_digest": initial_digest, "calibration": calibration, "privacy": privacy,
        "architecture": architecture, "microbatch": microbatch, "initial_evaluation": evaluation,
        "provenance": identity, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "geometry_seconds": result["elapsed_seconds"], "gpu": torch.cuda.get_device_name(),
        "estimated_private_training_hours": calibration["elapsed_seconds"]*privacy["steps"]/3600,
        "timing_note": "conservative upper proxy: calibration includes both attacks and all 3 gradients; training roles need fewer"}
    write_json(output/"PREFLIGHT_PASS.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args()
    run(a.assets_root, a.output)
