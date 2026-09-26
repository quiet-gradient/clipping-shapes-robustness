"""Six sequential MLM blocks, reusing CIFAR model/optimizer/gradient primitives.

No official test, private-gradient telemetry, persisted private RNG, or implicit
restart. A lost parent secret cannot be reconstructed from published artifacts.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import secrets
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from trustml.convnext_cifar100.mechanism import gradient_branches, assign_gradient
from trustml.data.splits import train_val_indices
from trustml.data.transforms import cifar_transform
from trustml.defenses.dp_sgd import _disable_inplace_relu, _replace_batchnorm_with_groupnorm
from trustml.models.registry import build_model
from trustml.training.optimizers import apply_adaptation, build_optimizer, build_scheduler
from trustml.ssg_gtsrb.persistence import transaction_files
from .mechanism import ARMS, Streams, contribution, maps, release
from .protocol import DATA, DATA_SHA, PRETRAIN, PRETRAIN_SHA, DEV_SHA, digest, load_parent, sha


def publish_strict(paths, *, publisher=None):
    if publisher is None:
        from ML_management.mlmanagement.log_api import get_debug
        from scripts.platform.mlm.publish_job_artifacts import publish_job_artifacts
        if get_debug():
            raise RuntimeError("MLM debug/local artifact mode is forbidden")
        publisher = publish_job_artifacts
    with warnings.catch_warnings():
        # SDK otherwise swallows ServerUnavailableError and silently queues locally.
        warnings.filterwarnings("error", message="Server unavailable, log:.*")
        return publisher(paths)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def accounting():
    import opacus
    from scripts.audit_privacy_accounting import epsilon_for
    e = epsilon_for("rdp", 1.123046875, 1/88, 4400, 1e-5)
    p = epsilon_for("prv", 1.123046875, 1/88, 4400, 1e-5)
    # Inherit the recorded CIFAR .4 tolerance, not the superseded .1 proposal.
    if abs(e-4) > .02 or abs(e-p) > .4:
        raise RuntimeError("frozen CIFAR privacy accounting gate failed")
    return dict(target_epsilon=4., achieved_epsilon_primary=e, achieved_epsilon_crosscheck=p,
                delta=1e-5, noise_multiplier=1.123046875, sampling_probability=1/88,
                steps=4400, aggregation_denominator=512, expected_poisson_batch_size=45000/88,
                sensitivity_bound=1., absolute_noise_std_before_averaging=1.123046875,
                primary="rdp", crosscheck="prv", crosscheck_eps_error=.1, crosscheck_delta_error=1e-6,
                crosscheck_tolerance_absolute=.4, opacus_version=opacus.__version__,
                adjacency="add_remove", release_count_per_step=1,
                joint_release_dp_claim=False, status="PASS")


def model_for(cfg, device):
    # Exactly the historical ordering: load/reinitialize head, last-block freeze,
    # BatchNorm->GroupNorm retaining affine parameters and requires_grad flags.
    torch.manual_seed(int(cfg.experiment.seed))
    model = build_model(cfg.model, root_cfg=cfg)
    apply_adaptation(model, cfg)
    _disable_inplace_relu(model)
    _replace_batchnorm_with_groupnorm(model)
    if any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)) for m in model.modules()):
        raise RuntimeError("batch coupling/dropout not allowed in this frozen model")
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 8398858:
        raise RuntimeError("last-block trainable parameter count mismatch")
    return model.to(device).float().eval()


def datasets(cfg):
    from torchvision.datasets import CIFAR10
    mean, std = tuple(cfg.data.mean), tuple(cfg.data.std)
    # Never instantiate train=False. No official-test evaluation in this module.
    aug = CIFAR10(str(cfg.data.root), train=True, download=False,
                  transform=cifar_transform(train=True, mean=mean, std=std, augmentation=True))
    dev = CIFAR10(str(cfg.data.root), train=True, download=False,
                  transform=cifar_transform(train=False, mean=mean, std=std, augmentation=False))
    train_idx, dev_idx = train_val_indices(len(dev), .9, 20260820)
    if len(train_idx) != 45000 or len(dev_idx) != 5000 or set(train_idx) & set(dev_idx):
        raise RuntimeError("CIFAR split invariant failed")
    if digest(dev_idx) != DEV_SHA:
        raise RuntimeError("ordered dev hash differs from locked RQ1")
    return aug, dev, train_idx, dev_idx


def bounds(x, cfg):
    mean = x.new_tensor(list(cfg.data.mean)).view(1, 3, 1, 1)
    std = x.new_tensor(list(cfg.data.std)).view(1, 3, 1, 1)
    return -mean/std, (1-mean)/std, (8/255)/std, (2/255)/std


def attack(model, x, cfg, generator, *, labels=None, steps=10):
    """TRADES matches existing normalized-space Gaussian initialization.

    For dev PGD, use a uniform Linf start and CE. All outputs detached.
    """
    lo, hi, eps, step = bounds(x, cfg)
    with torch.no_grad():
        probs = model(x).softmax(1)
        if labels is None:
            start = .001 * torch.randn(x.shape, device=x.device, generator=generator)
        else:
            start = (2*torch.rand(x.shape, device=x.device, generator=generator)-1)*eps
        adv = (x + start.clamp(-eps, eps)).clamp(lo, hi)
    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = (F.kl_div(logits.log_softmax(1), probs, reduction="batchmean") if labels is None
                else F.cross_entropy(logits, labels))
        grad = torch.autograd.grad(loss, adv)[0]
        if not torch.isfinite(grad).all():
            raise RuntimeError("nonfinite attack gradient")
        adv = (x + (adv.detach()+step*grad.sign()-x).clamp(-eps, eps)).clamp(lo, hi).detach()
    if not torch.isfinite(adv).all() or ((adv-x).abs() > eps+1e-6).any():
        raise RuntimeError("attack Linf/finite invariant failed")
    return adv.detach(), probs.detach()


def batch(dataset, indices, device):
    records = [dataset[int(i)] for i in indices]
    return (torch.stack([r[0] for r in records]).to(device),
            torch.tensor([r[1] for r in records], device=device))


def options(cfg):
    i = cfg.intervention
    return dict(map_name=str(i.map_name), bound=float(i.bound), q=float(i.q), gamma=float(i.gamma))


def preflight(model, dev, dev_idx, cfg, output):
    """Public-only CUDA audit, before any private access or optimizer step."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(19092026)
    x, y = batch(dev, dev_idx[:2], device)
    adv, probs = attack(model, x, cfg, gen)
    g, h = gradient_branches(model, x, y, adv, probs)
    params = [p for p in model.parameters() if p.requires_grad]
    for i in range(2):
        ce = F.cross_entropy(model(x[i:i+1]), y[i:i+1])
        kl = F.kl_div(model(adv[i:i+1]).log_softmax(1), probs[i:i+1], reduction="sum")
        grad = torch.cat([t.flatten() for t in torch.autograd.grad(ce+kl, params)])
        torch.testing.assert_close(g[i]+h[i], grad, rtol=3e-4, atol=2e-5)
    checked = []
    for m, q in [("hard", 1/3), ("autos", .25)]:
        for arm in ARMS:
            value = contribution(g, h, arm=arm, map_name=m, bound=1., q=q, gamma=.1)
            checked.append(dict(map=m, arm=arm, max_norm=float(value.norm(dim=1).max())))
    # Public one-step optimizer smoke for each map. Restore parameters after it.
    initial = [p.detach().clone() for p in params]
    for m, q in [("hard", 1/3), ("autos", .25)]:
        for arm in ARMS:
            opt = build_optimizer(cfg, model)
            update = contribution(g, h, arm=arm, map_name=m, bound=1., q=q, gamma=.1).sum(0)
            assign_gradient(model, release(update, sigma=1.123046875, bound=1., denominator=512, generator=gen))
            opt.step()
            if any(not torch.isfinite(p).all() for p in params):
                raise RuntimeError("public optimizer smoke failed")
            with torch.no_grad():
                for p, v in zip(params, initial):
                    p.copy_(v)
            opt.zero_grad(set_to_none=True)
    # Exercise the configured physical batch, not only the two-record audit.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    x, y = batch(dev, dev_idx[:int(cfg.intervention.microbatch_size)], device)
    adv, probs = attack(model, x, cfg, gen)
    g, h = gradient_branches(model, x, y, adv, probs)
    contribution(g, h, arm="D", **options(cfg)).sum(0)
    del g, h
    # Exercise the real update and reporting/serialization path, not just maps.
    # All inputs are public; restore initialization between each diagnostic arm.
    smoke = output.parent / "public_pipeline_smoke"
    smoke.mkdir(parents=True, exist_ok=False)
    for m, q in [("hard", 1/3), ("autos", .25)]:
        case = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        case.intervention.map_name, case.intervention.q = m, q
        for arm in ARMS:
            case.intervention.arm = arm
            opt = build_optimizer(case, model)
            private_update(model, opt, x, y, adv, probs, case, gen)
            with torch.no_grad():
                for p, v in zip(params, initial):
                    p.copy_(v)
            opt.zero_grad(set_to_none=True)
        geometry(model, dev, dev_idx[:8], case, 0, smoke / f"{m}_geometry.json")
        clean, pgd = evaluate(model, dev, dev_idx[:8], case, pgd=True)
        if len(clean) != 8 or len(pgd) != 8 or any(r and not c for c, r in zip(clean, pgd)):
            raise RuntimeError("public evaluation serialization smoke failed")
        np.savez_compressed(smoke / f"{m}_predictions.npz", sample_ids=dev_idx[:8], clean=clean, pgd20=pgd)
    checkpoint = smoke / "public_smoke.pt"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored["model"][name], tensor.detach().cpu(), rtol=0, atol=0)
    archive = output.parent / "public_pipeline_smoke.tar.gz"
    archive_child(smoke, archive)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    write_json(output, dict(status="PASS", public_only=True, private_training_steps=0,
                            public_optimizer_smoke_steps=20, maps=checked,
                            pipeline_smoke="10 actual updates; both geometry maps; PGD; checkpoint roundtrip; archive",
                            pipeline_archive_sha256=sha(archive),
                            physical_batch=int(cfg.intervention.microbatch_size), peak_allocated_bytes=peak,
                            gradient_equivalence="per-example composite autograd", device=str(device)))


def summarize_probe(values):
    """Keep NPZ flags boolean; use numeric 0/1 for descriptive quantiles.

    NumPy linear quantiles subtract adjacent values, which is undefined for
    bool arrays even when all flags are False. Mean of a flag is its rate.
    """
    summary = {}
    for key, values_for_key in values.items():
        numeric = np.asarray(values_for_key, dtype=np.float64)
        if numeric.ndim != 1 or not numeric.size or not np.isfinite(numeric).all():
            raise RuntimeError(f"invalid public-probe values: {key}")
        summary[key] = dict(mean=float(numeric.mean()), p05=float(np.quantile(numeric, .05)),
                            median=float(np.median(numeric)), p95=float(np.quantile(numeric, .95)))
    return summary


def geometry(model, dev, indices, cfg, step, path):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(20260820)
    values = {key: [] for key in ["g_norm", "h_norm", "cos_g_h", "qstar", "zero_J", "zero_U"]}
    for a in ARMS:
        values.update({f"{a}_{suffix}": [] for suffix in ("norm", "error_to_raw", "cos_to_raw")})
    sums = {}
    for offset in range(0, len(indices), 4):
        x, y = batch(dev, indices[offset:offset+4], device)
        adv, probs = attack(model, x, cfg, generator)
        g, h = gradient_branches(model, x, y, adv, probs)
        ng, nh = g.norm(dim=1), h.norm(dim=1)
        values["g_norm"].extend(ng.cpu().tolist())
        values["h_norm"].extend(nh.cpu().tolist())
        values["qstar"].extend((nh/(ng+nh).clamp_min(1e-30)).cpu().tolist())
        values["cos_g_h"].extend(F.cosine_similarity(g, h, dim=1).cpu().tolist())
        j, u, _ = maps(g, h, **options(cfg))
        values["zero_J"].extend((j.norm(dim=1)==0).cpu().tolist())
        values["zero_U"].extend((u.norm(dim=1)==0).cpu().tolist())
        for arm in ARMS:
            v = contribution(g, h, arm=arm, **options(cfg))
            values[f"{arm}_norm"].extend(v.norm(dim=1).cpu().tolist())
            values[f"{arm}_error_to_raw"].extend((v-g-h).norm(dim=1).cpu().tolist())
            values[f"{arm}_cos_to_raw"].extend(F.cosine_similarity(v, g+h, dim=1).cpu().tolist())
            sums[arm] = sums.get(arm, 0) + v.sum(0)
        del g, h, j, u
    summary = summarize_probe(values)
    summary["mean_update_norms"] = {a: float(v.norm()/len(indices)) for a, v in sums.items()}
    summary["mean_update_cos_to_J"] = {a: float(F.cosine_similarity(v[None], sums["J"][None])) for a, v in sums.items()}
    write_json(path, dict(step=step, public_only=True, count=len(indices), summary=summary))
    np.savez_compressed(Path(path).with_suffix(".npz"), sample_ids=np.asarray(indices),
                        **{key: np.asarray(v) for key, v in values.items()})


def private_update(model, optimizer, x, y, adv, probs, cfg, generator):
    """All physical chunks contribute to exactly ONE release and optimizer step."""
    params = [p for p in model.parameters() if p.requires_grad]
    summed = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=params[0].dtype)
    for off in range(0, len(y), int(cfg.intervention.microbatch_size)):
        sl = slice(off, off+int(cfg.intervention.microbatch_size))
        g, h = gradient_branches(model, x[sl], y[sl], adv[sl], probs[sl])
        summed.add_(contribution(g, h, arm=str(cfg.intervention.arm), **options(cfg)).sum(0))
    optimizer.zero_grad(set_to_none=True)
    assign_gradient(model, release(summed, sigma=1.123046875, bound=1., denominator=512, generator=generator))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("nonfinite model after private update")


def evaluate(model, dev, indices, cfg, *, pgd=False):
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(42)
    correct, robust = [], []
    for offset in range(0, len(indices), 128):
        x, y = batch(dev, indices[offset:offset+128], device)
        with torch.no_grad():
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite dev logits")
            correct.extend(logits.argmax(1).eq(y).cpu().tolist())
        if pgd:
            adv, _ = attack(model, x, cfg, gen, labels=y, steps=20)
            with torch.no_grad():
                robust.extend((model(adv).argmax(1).eq(y) & logits.argmax(1).eq(y)).cpu().tolist())
    return correct, robust


def run_child(raw_cfg, secret, output, audit):
    cfg = OmegaConf.create(raw_cfg)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml")
    device = torch.device("cuda")
    model = model_for(cfg, device)
    train, dev, train_idx, dev_idx = datasets(cfg)
    streams = Streams(secret, device)
    opt, scheduler = build_optimizer(cfg, model), None
    scheduler = build_scheduler(cfg, opt)
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    private_ids = np.asarray(train_idx)
    probe_order = torch.randperm(5000, generator=torch.Generator().manual_seed(20260820))[:512].tolist()
    probe_idx = [dev_idx[i] for i in probe_order]
    positions, history = [], []
    started = time.monotonic()
    import wandb
    wandb_run = wandb.init(project="dp-robustness", name=str(cfg.experiment.name), mode="offline",
                           dir=str(output), config=raw_cfg, reinit=True)
    try:
        geometry(model, dev, probe_idx, cfg, 0, output / "geometry/step_0000.json")
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for step in range(1, 4401):
            selected = torch.where(torch.rand(45000, generator=streams["sampling"]) < 1/88)[0].numpy()
            # No selected IDs, private losses, gradients, or RNG states are logged.
            if len(selected):
                with streams.augmentation():
                    x, y = batch(train, private_ids[selected], device)
                adv, probs = attack(model, x, cfg, streams["attack"])
            else:
                x = adv = torch.empty((0, 3, 32, 32), device=device)
                y = torch.empty(0, dtype=torch.long, device=device)
                probs = torch.empty((0, 10), device=device)
            private_update(model, opt, x, y, adv, probs, cfg, streams["noise"])
            accountant.step(noise_multiplier=1.123046875, sample_rate=1/88)
            if step % 20 == 0:
                print(json.dumps(dict(run=str(cfg.experiment.name), step=step, total=4400,
                                      lr=opt.param_groups[0]["lr"])), flush=True)
            if step in {1100, 2200, 3300, 4400}:
                geometry(model, dev, probe_idx, cfg, step, output / f"geometry/step_{step:04d}.json")
            if step % 88 == 0:
                clean, _ = evaluate(model, dev, dev_idx, cfg)
                row = dict(step=step, epoch=step//88, lr=opt.param_groups[0]["lr"],
                           dev_clean=100*float(np.mean(clean)))
                history.append(row)
                wandb_run.log(row, step=step)
                with (output / "history.jsonl").open("a") as f:
                    f.write(json.dumps(row)+"\n")
                positions.append(dict(step=step, positions=streams.positions()))
                scheduler.step()
        achieved = accountant.get_epsilon(1e-5)
        if abs(achieved-audit["achieved_epsilon_primary"]) > 1e-8:
            raise RuntimeError("actual accountant history does not match planned schedule")
        checkpoint = output / "checkpoints/final.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                        config=raw_cfg, epoch=50, optimizer_step=4400, privacy_accountant=accountant.state_dict(),
                        source_pretrain_sha256=PRETRAIN_SHA, private_rng_persisted=False), checkpoint)
        # Weights saved BEFORE the final PGD evaluation.
        clean, pgd = evaluate(model, dev, dev_idx, cfg, pgd=True)
        np.savez_compressed(output / "dev_predictions.npz", sample_ids=np.asarray(dev_idx),
                            clean=np.asarray(clean), pgd20=np.asarray(pgd))
        write_json(output / "privacy_accounting.json", {**audit, "actual_steps": 4400})
        write_json(output / "rng_audit.json", dict(commitment=streams.commitment(), positions=positions,
                                                    private_rng_persisted=False, jointly_dp=False))
        summary = dict(run_id=str(cfg.experiment.name), status="VERIFIED_LOCAL", steps=4400,
                       checkpoint_sha256=sha(checkpoint), config_sha256=digest(raw_cfg),
                       source_sha256=PRETRAIN_SHA, ordered_dev_sha256=digest(dev_idx),
                       probe_ids_sha256=digest(probe_idx), dev_count=5000, trainable_parameters=nparams,
                       dev_clean=100*float(np.mean(clean)), dev_pgd20=100*float(np.mean(pgd)),
                       autoattack_status="NOT_RUN", official_test_accessed=False,
                       wandb_run_id=wandb_run.id, runtime_seconds=time.monotonic()-started,
                       git_commit=os.environ.get("TRUSTML_GIT_COMMIT"),
                       bundle_sha256=os.environ.get("TRUSTML_BUNDLE_HASH"))
        write_json(output / "summary.json", summary)
        return summary, positions
    finally:
        wandb_run.finish()
        del model, opt, scheduler
        gc.collect()
        torch.cuda.empty_cache()


def archive_child(folder, archive):
    if archive.exists():
        raise RuntimeError("refusing to overwrite an existing child archive")
    files = transaction_files(folder, sorted(folder.rglob("*")))
    with tarfile.open(archive, "x:gz") as tf:
        for path in files:
            tf.add(path, arcname=path.relative_to(folder).as_posix(), recursive=False)
    checksum = Path(str(archive)+".sha256")
    checksum.write_text(f"{sha(archive)}  {archive.name}\n")
    return checksum


def run_parent(parent):
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("CIFAR intervention training requires A100; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rows = load_parent(parent)
    if sha(PRETRAIN) != PRETRAIN_SHA or sha(DATA) != DATA_SHA:
        raise RuntimeError("immutable runtime input hash mismatch")
    output = Path("outputs/cifar_interventions_v1") / parent
    output.mkdir(parents=True, exist_ok=False)  # Never silently rerun a partial parent.
    audit = accounting()
    write_json(output / "accounting_preflight.json", audit)
    # Server ACK required before training. This is NOT a remote-byte readback.
    publish_strict([output / "accounting_preflight.json"])
    cfg = OmegaConf.create(rows[0])
    _, dev, _, dev_idx = datasets(cfg)
    model = model_for(cfg, "cuda")
    preflight(model, dev, dev_idx, cfg, output / "public_preflight.json")
    publish_strict([output / "public_preflight.json", output / "public_pipeline_smoke.tar.gz",
                    output / "public_pipeline_smoke.tar.gz.sha256"])
    del model, dev
    torch.cuda.empty_cache()
    secret = secrets.token_bytes(32)
    reference = None
    for row in rows:
        run_id = row["experiment"]["name"]
        folder = output / run_id
        summary, positions = run_child(row, secret, folder, audit)
        if reference is None:
            reference = positions
        elif positions != reference:
            raise RuntimeError("matched sampling/augmentation/attack/noise RNG positions diverged")
        archive = output / f"intervention_result__{run_id}.tar.gz"
        checksum = archive_child(folder, archive)
        uploaded = publish_strict([archive, checksum])
        receipt = output / f"{run_id}.COMMITTED.json"
        write_json(receipt, dict(status="COMMITTED", run_id=run_id, files=uploaded,
                                rng_positions_match=True, remote_readback_verified=False,
                                commit_evidence="server_upload_ack; verify bytes on collection",
                                backend="MLM_native_artifact_api"))
        publish_strict([receipt])
        print(json.dumps(dict(run_id=run_id, status="COMMITTED")), flush=True)
    del secret
    write_json(output / "parent_completed.json", dict(status="COMPLETED", parent=parent, runs=5))
    publish_strict([output / "parent_completed.json"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    run_parent(parser.parse_args().parent)


if __name__ == "__main__":
    main()
