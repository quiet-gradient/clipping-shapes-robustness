"""Public-only capacity/equivalence audit, zero optimizer/accountant updates."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import tarfile
import time

import torch

from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.data import GTSRBTrainOnly
from trustml.convnext_gtsrb.mechanism import gradients, contribution
from trustml.convnext_gtsrb.model import sha256
from .private import load_selected
from .rectangular import SOURCE_SHA, fetch
from .persistence import TransactionStore

CANDIDATES = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
ROLES = ["DPOnly", "Joint", "Componentwise"]


def choose(rows):
    safe = {role: {r["microbatch"] for r in rows if r["role"] == role and r["status"] == "PASS"}
            for role in ROLES}
    common = safe["Joint"] & safe["Componentwise"]
    if not safe["DPOnly"] or not common:
        raise RuntimeError("no common validated safe microbatch")
    return {"CleanDP": max(safe["DPOnly"]), "Joint": max(common), "Componentwise": max(common)}


def choose_fastest_common(rows):
    """Derived execution policy; do not alter the original maximum-capacity report."""
    import math
    timings = {}
    for role in ROLES:
        safe = [r for r in rows if r["role"] == role and r["status"] == "PASS"
                and r.get("memory_headroom_pass") and r.get("gradient_equivalent")]
        timings[role] = {r["microbatch"]: r["elapsed_compute_seconds"] for r in safe}
        if len(timings[role]) != len(safe) or any(not math.isfinite(t) or t <= 0 for t in timings[role].values()):
            raise ValueError("invalid public timing evidence")
    common = set.intersection(*(set(timings[role]) for role in ROLES))
    if not common:
        raise ValueError("no common safe timed microbatch")
    size = min(common, key=lambda b: (sum(timings[role][b] for role in ROLES), b))
    return dict(CleanDP=size, Joint=size, Componentwise=size)


def relative_error(actual, expected):
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-12))


def measure(model, x, y, fixed_adv, uniform, size, role, reference):
    """Time actual attack plus gradient work, compare at fixed adversarial inputs.

    Changing convolution batch shape can perturb PGD trajectories; using the
    same fixed x_adv separates gradient equivalence from sign-attack roundoff.
    """
    total = None
    sentinels = None
    elapsed = 0.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(x), size):
        xb, yb = x[start:start+size], y[start:start+size]
        torch.cuda.synchronize()
        t = time.monotonic()
        if role != "DPOnly":
            actual_adv, _ = attack(model, xb, yb, objective="kl", uniform=uniform[start:start+size])
            # attack() already checks FP32, eval(), finiteness and Linf bound.
            del actual_adv
        values = gradients(model, xb, yb, trades_adv=None if role == "DPOnly" else fixed_adv[start:start+size])
        update = contribution(values, role, .3, 1/3 if role == "Componentwise" else None)
        if not torch.isfinite(update).all() or (update.norm(dim=-1) > .3 * (1 + 2e-5)).any():
            raise RuntimeError("nonfinite contribution or sensitivity violation")
        if total is None:
            total = torch.zeros_like(update[0])
        total.add_(update.sum(0))
        torch.cuda.synchronize()
        elapsed += time.monotonic() - t
        if start == 0:
            sentinels = {key: value[:2].cpu() for key, value in values.items()}
        del values, update
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    total = total.cpu()
    errors = {} if reference is None else {key: relative_error(value, reference["sentinels"][key])
                                           for key, value in sentinels.items()}
    sum_error = 0. if reference is None else relative_error(total, reference["sum"])
    memory = torch.cuda.get_device_properties(0).total_memory
    # Additional SGD/noise/telemetry tensors aren't in this zero-update audit.
    # Explicit allowance plus20% card headroom, not merely "didn't OOM".
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    projected_peak = peak + 16 * parameter_bytes
    equivalent = max(errors.values(), default=0.) <= 5e-4 and sum_error <= 3e-3
    safe = projected_peak <= .8 * memory
    row = dict(role=role, microbatch=size, public_records=len(x), elapsed_compute_seconds=elapsed,
               estimated_train_only_hours=elapsed / len(x) * 512 * 3436 / 3600,
               peak_allocated_bytes=peak, peak_reserved_bytes=reserved, device_memory_bytes=memory,
               extra_training_allowance_bytes=16*parameter_bytes, projected_peak_bytes=projected_peak,
               per_example_relative_l2=errors, sum_relative_l2=sum_error,
               gradient_equivalent=equivalent, memory_headroom_pass=safe,
               status="PASS" if equivalent and safe else "FAIL_EQUIVALENCE" if not equivalent else "HEADROOM_LIMIT")
    return row, dict(sum=total, sentinels=sentinels)


def run(args):
    from scripts.platform.gtsrb_inputs import materialize
    configure_gpu()
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one A100 required")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    payload = json.loads(args.routes.read_text())
    inputs = payload["inputs"]
    materialize(args.input_archive, inputs["training_sha256"])
    archive = Path("selected_archive.tar.gz")
    fetch(inputs["selected_url"], archive, inputs["selected_archive_sha256"])
    selected = Path("gtsrb_inputs/public_pretrain_selected.pt")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember("public_pretrain_selected.pt")
        if not member.isfile():
            raise ValueError("unsafe selected member")
        with tar.extractfile(member) as src, selected.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    if sha256(selected) != SOURCE_SHA:
        raise ValueError("selected source mismatch")
    p = json.loads(Path("configs/convnext_gtsrb/protocol.json").read_text())
    data = GTSRBTrainOnly(Path("gtsrb_inputs/GTSRB_Final_Training_Images.zip"),
                         Path("gtsrb_inputs/splits.json"), p["augmentation"])
    locked = json.loads(Path("reports/ssg_gtsrb_v1/gtsrb_private_split_manifest.json").read_text())
    if data.split["split_hash"] != locked["split_hash"]:
        raise ValueError("split mismatch")
    ids = data.split["indices"]["probe"]
    assert len(ids) == 512 and set(ids).isdisjoint(data.split["indices"]["private"])
    model, architecture = load_selected(Path("gtsrb_inputs"), selected, SOURCE_SHA)
    model.cuda().eval()
    x, y = data.batch(ids)
    x, y = x.cuda(), y.cuda()
    uniform = public_uniform(x, [data.split["records"][i]["image_id"] for i in ids], 20260921, "microbatch_audit")
    # Common adversarial inputs for all numerical comparisons; no private RNG.
    fixed_adv = torch.cat([attack(model, x[i:i+8], y[i:i+8], objective="kl", uniform=uniform[i:i+8])[0]
                           for i in range(0, len(x), 8)])
    report = dict(status="RUNNING", gpu=torch.cuda.get_device_name(), source_checkpoint_sha256=SOURCE_SHA,
                  split_hash=data.split["split_hash"], probe_hash=data.split["hashes"]["probe"],
                  public_records=512, private_records_used=0, optimizer_steps=0, accountant_steps=0,
                  logical_batch=512, official_test_loaded=False, rows=[],
                  trainable_parameter_count=architecture["trainable_parameter_count"],
                  caveat="fixed-x_adv numerical equivalence; not bitwise PGD trajectory or final accuracy guarantee")
    for role in ROLES:
        reference = None
        for size in CANDIDATES:
            try:
                row, values = measure(model, x, y, fixed_adv, uniform, size, role, reference)
                if reference is None:
                    reference = values
                else:
                    del values
            except torch.cuda.OutOfMemoryError:
                row = dict(role=role, microbatch=size, status="PUBLIC_CAPACITY_OOM")
            report["rows"].append(row)
            write_json(output / "microbatch_audit.json", report)
            print(json.dumps({"phase": "public_microbatch_candidate", **row}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            if row["status"] == "FAIL_EQUIVALENCE":
                raise RuntimeError("public gradient equivalence failed; not silently accepting larger batch")
            if row["status"] in {"PUBLIC_CAPACITY_OOM", "HEADROOM_LIMIT"}:
                break
        del reference
    report.update(status="PASS", selected_microbatch_by_role=choose(report["rows"]))
    write_json(output / "microbatch_audit.json", report)
    TransactionStore(payload["routes"], output.parent / "transactions").commit("microbatch_probe", output)
    print(json.dumps({"phase": "public_microbatch_complete", "selected": report["selected_microbatch_by_role"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/microbatch_probe"))
    run(parser.parse_args())
