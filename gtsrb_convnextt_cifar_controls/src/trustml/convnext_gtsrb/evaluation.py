"""Immediate full-dev clean/CE and fixed-1024 attack trajectories. No test."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attacks import attack, public_uniform
from trustml.convnext_cifar100.preflight import write_json


def autoattack_inputs(images, labels, device):
    # Pinned AA's APGD-T zero-gradient check uses grad.view(N, -1).
    # Dataset NHWC->NCHW views retain channels-last strides. Materialize NCHW
    # at the AA boundary only; values, model, training and attack stay unchanged.
    return images.to(device).contiguous(), labels.to(device)


def evaluate(model, data, output, *, seed=42, final=False, batch_size=64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    dev = data.split["indices"]["dev"]
    subset = data.split["indices"]["trajectory"]
    device = next(model.parameters()).device
    started = time.monotonic()
    clean, loss = [], []
    for begin in range(0, len(dev), batch_size):
        x, y = data.batch(dev[begin:begin+batch_size])
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite evaluation logits")
            clean.extend(logits.argmax(-1).eq(y).cpu().tolist())
            loss.extend(F.cross_entropy(logits, y, reduction="none").cpu().tolist())
    robust, subset_clean = [], []
    for begin in range(0, len(subset), batch_size):
        ids = subset[begin:begin+batch_size]
        x, y = data.batch(ids)
        x, y = x.to(device), y.to(device)
        uniform = public_uniform(x, [data.split["records"][i]["image_id"] for i in ids], seed, "pgd_trajectory")
        _, mask = attack(model, x, y, objective="ce", steps=20, uniform=uniform, track_best=True)
        robust.extend(mask.cpu().tolist())
        with torch.no_grad():
            subset_clean.extend(model(x).argmax(-1).eq(y).cpu().tolist())
    if np.any(np.asarray(robust) & ~np.asarray(subset_clean)):
        raise RuntimeError("PGD monotonicity violation")
    arrays = {"dev_ids": np.asarray(dev), "clean_correct": np.asarray(clean), "clean_ce": np.asarray(loss),
              "trajectory_ids": np.asarray(subset), "trajectory_clean_correct": np.asarray(subset_clean), "pgd20_correct": np.asarray(robust)}
    metrics = {"dev_clean": float(100*np.mean(clean)), "dev_ce_loss": float(np.mean(loss)),
               "dev_examples": len(dev), "pgd20": float(100*np.mean(robust)), "attack_examples": len(subset),
               "split_hash": data.split["hashes"]["dev"], "trajectory_hash": data.split["hashes"]["trajectory"],
               "attack_seed": seed, "precision": "FP32", "model_mode": "eval", "official_test_loaded": False,
               "clean_pgd_seconds": time.monotonic()-started, "epsilon": 8/255}
    if final:
        from autoattack import AutoAttack
        from autoattack.state import EvaluationState
        xs, ys = data.batch(subset)
        xs, ys = autoattack_inputs(xs, ys, device)
        metrics['autoattack_input_layout'] = 'contiguous NCHW (values unchanged)'
        surviving = np.asarray(subset_clean, dtype=bool)
        for name in ["apgd-ce", "apgd-t"]:
            begin = time.monotonic()
            adversary = AutoAttack(model, norm="Linf", eps=8/255, seed=seed, version="standard", device=str(device),
                                  log_path=str(output/f"{name}.log"), verbose=True)
            adversary.attacks_to_run = [name]
            state_path = output/f'{name}_state.json'
            adv = adversary.run_standard_evaluation(xs, ys, bs=batch_size, state_path=state_path)
            if not torch.isfinite(adv).all() or (adv-xs).abs().max() > 8/255+1e-6 or adv.min() < 0 or adv.max() > 1:
                raise RuntimeError("APGD returned invalid images")
            mask = []
            for k in range(0, len(subset), batch_size):
                with torch.no_grad():
                    logits = model(adv[k:k+batch_size])
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("nonfinite APGD posthoc logits")
                    mask.extend(logits.argmax(-1).eq(ys[k:k+batch_size]).cpu().tolist())
            attack_time = EvaluationState.from_disk(state_path).robust_flags.cpu().numpy()
            posthoc = np.asarray(mask) & np.asarray(subset_clean)
            if attack_time.shape != (1024,) or np.any(attack_time & ~np.asarray(subset_clean)):
                raise RuntimeError('invalid APGD attack-time correctness mask')
            mismatch = int(np.count_nonzero(attack_time != posthoc))
            arrays[name+'_attack_time_correct'] = attack_time
            arrays[name+'_posthoc_correct'] = posthoc
            metrics[name+'_attack_time_posthoc_mismatches'] = mismatch
            # Conservative conjunction; preserve and report any numerical warning.
            individual = attack_time & posthoc
            surviving &= individual
            arrays[name+"_correct"] = individual
            arrays["after_"+name] = surviving.copy()
            metrics[name] = float(100*individual.mean())
            metrics[name+"_seconds"] = time.monotonic()-begin
        arrays["apgd_union_correct"] = surviving
        metrics["apgd_union"] = float(100*surviving.mean())
        metrics["full_autoattack"] = False
        metrics["autoattack_version"] = "0.1 / a39220048b3c9f2cca9a4d3a54604793c68eca7e"
    np.savez_compressed(output/"public_dev_predictions.npz", **arrays)
    write_json(output/"metrics.json", metrics)
    return metrics
