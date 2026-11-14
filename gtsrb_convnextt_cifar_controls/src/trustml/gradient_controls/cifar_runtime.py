"""Resumable CIFAR gradient-control training with private state kept outside artifacts.

The scientific path reuses the frozen CIFAR model, data, attack, gradient and
epoch scheduler. A 100-update pilot is a prefix of the 4400-update run, not a
shortened schedule. Checkpoints of RNG/optimizer/accountant state are private.
This module performs no upload and never evaluates the official test.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import secrets
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from trustml.cifar_interventions.mechanism import Streams
from trustml.cifar_interventions.protocol import PRETRAIN_SHA, DEV_SHA, DATA, DATA_SHA, digest, sha
from trustml.cifar_interventions.runtime import model_for, datasets, attack, batch, options, evaluate, write_json
from trustml.convnext_cifar100.mechanism import gradient_branches, assign_gradient
from trustml.training.optimizers import build_optimizer, build_scheduler
from .mechanism import METHODS, contribution, release

TOTAL_STEPS = 4400
STEPS_PER_EPOCH = 88
PROBE_STEPS = (0, 1100, 2200, 3300, 4400)
PROBE_SIZE = 512
STATE_VERSION = 1


def validate_config(raw):
    """Reject scientific drift instead of silently accepting unused overrides."""
    cfg = OmegaConf.create(raw)
    r, i = cfg.control, cfg.intervention
    required = dict(steps=TOTAL_STEPS, batches_per_epoch=88, denominator=512,
                    sampling_probability=1/88, beta=1, outer_kl_target="detached_clean")
    for key, expected in required.items():
        if i.get(key) != expected:
            raise ValueError(f"frozen CIFAR intervention.{key} must be {expected!r}")
    if int(r.get("planned_steps", TOTAL_STEPS)) != TOTAL_STEPS or cfg.training.epochs != 50:
        raise ValueError("prefix runs must retain the full 4400-update/50-epoch schedule")
    if cfg.training.scheduler.name != "cosine" or float(cfg.training.scheduler.eta_min_factor) != .01:
        raise ValueError("frozen cosine epoch schedule mismatch")
    if cfg.training.optimizer.name != "sgd" or float(cfg.training.optimizer.momentum) != .9:
        raise ValueError("frozen SGD optimizer mismatch")
    if float(cfg.training.optimizer.weight_decay) != .0005 or not cfg.training.optimizer.nesterov:
        raise ValueError("frozen SGD weight decay/Nesterov mismatch")
    if cfg.data.name != "cifar10" or cfg.model.name != "resnet18" or cfg.adaptation.method != "last_block":
        raise ValueError("runner only implements frozen last-block CIFAR ResNet-18")
    if tuple(cfg.data.mean) != (.4914, .4822, .4465) or tuple(cfg.data.std) != (.2023, .1994, .201):
        raise ValueError("frozen input normalization mismatch")
    if cfg.model.num_classes != 10 or int(cfg.data.batch_size) != 512:
        raise ValueError("frozen class count/logical batch mismatch")
    if float(cfg.data.train_ratio) != .9 or int(cfg.data.seed) != 20260820:
        raise ValueError("frozen CIFAR split mismatch")
    if int(cfg.data.num_workers) != 0 or not cfg.data.augmentation:
        raise ValueError("resume requires the frozen single-process augmentation path")
    if int(i.microbatch_size) <= 0 or int(i.microbatch_size) > 32:
        raise ValueError("physical gradient batch must be in 1..32")
    if r.method not in METHODS:
        raise ValueError("unknown gradient control")
    if float(r.get("delta", 1e-5)) != 1e-5:
        raise ValueError("frozen delta must be 1e-5")
    if r.method == "nonprivate":
        if float(r.sigma) != 0 or r.get("target_epsilon") is not None:
            raise ValueError("unclipped/no-noise control cannot carry a DP guarantee")
    elif float(r.sigma) <= 0 or float(r.target_epsilon) not in (2., 4.):
        raise ValueError("private configs need positive sigma and epsilon 2 or 4")
    if i.map_name == "autos" and (float(i.bound), float(i.gamma), float(i.q)) != (1., .1, .25):
        raise ValueError("CIFAR AUTO-S matched anchor is B=1, gamma=.1, q=.25")
    if i.map_name == "hard" and (float(i.bound), float(i.q)) != (1., 1/3):
        raise ValueError("CIFAR hard matched anchor is C=1, q=1/3")
    if i.map_name not in ("autos", "hard"):
        raise ValueError("unsupported map")
    if r.method != "nonprivate":
        expected_lr = .01510270856068702 if i.map_name == "autos" else .025
        if float(cfg.training.optimizer.lr) != expected_lr:
            raise ValueError("matched intervention LR must equal the frozen anchor")
    if any(cfg.training.get(k) is not None for k in ("max_train_batches", "max_eval_batches")):
        raise ValueError("batch truncation is forbidden")
    if r.method == "M_scaled":
        if i.map_name != "autos" or float(r.target_epsilon)!=4:
            raise ValueError("this frozen M_scaled campaign is AUTO-S epsilon4 only")
        scale=float(r.get("fixed_M_scale",0))
        if not np.isfinite(scale) or not 0<scale<=1:
            raise ValueError("M_scaled requires fixed attenuation in (0,1]")
        spec=r.get("scale_calibration")
        if spec is None or sha(str(spec.path))!=str(spec.sha256):raise ValueError("public scale calibration hash mismatch")
        calibration=json.loads(Path(str(spec.path)).read_text())
        if not (calibration['source']=='public_dev' and calibration['frozen_before_private_training'] is True and calibration['seed']==int(cfg.experiment.seed) and calibration['source_step']==0 and calibration['c']==scale and calibration['sample_count']==512 and calibration['source_pretrain_sha256']==PRETRAIN_SHA):raise ValueError("scale calibration identity/public-source mismatch")
    return cfg


def planned_accounting(cfg):
    if cfg.control.method == "nonprivate":
        return dict(status="NOT_APPLICABLE", reason="unclipped_no_noise", dp_claim=False,
                    planned_steps=TOTAL_STEPS, aggregation_denominator=512)
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    accountant.history = [(float(cfg.control.sigma), 1/88, TOTAL_STEPS)]
    eps = accountant.get_epsilon(1e-5)
    if abs(eps-float(cfg.control.target_epsilon)) > .02:
        raise ValueError("sigma does not attain the declared epsilon on the full schedule")
    return dict(status="PASS", target_epsilon=float(cfg.control.target_epsilon),
                achieved_epsilon_primary=eps, delta=1e-5, primary="rdp",
                noise_multiplier=float(cfg.control.sigma), sampling_probability=1/88,
                planned_steps=TOTAL_STEPS, aggregation_denominator=512,
                sensitivity_bound=float(cfg.intervention.bound), release_count_per_step=1,
                joint_release_dp_claim=False)


def source_identity():
    import trustml.cifar_interventions.runtime as old_runtime
    import trustml.cifar_interventions.mechanism as old_mechanism
    import trustml.convnext_cifar100.mechanism as gradients
    import trustml.training.optimizers as optimizers
    from . import mechanism, query_geometry
    paths = [Path(__file__), Path(mechanism.__file__), Path(old_runtime.__file__),
             Path(old_mechanism.__file__), Path(gradients.__file__), Path(optimizers.__file__), Path(query_geometry.__file__)]
    return dict(files={str(p.relative_to(Path(__file__).parents[2])): sha(p) for p in paths},
                torch_version=torch.__version__, numpy_version=np.__version__)


@contextmanager
def isolated_public_rng():
    """Public diagnostics may not advance any training/global random stream."""
    py, npstate = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(npstate)


def state_payload(model, optimizer, scheduler, accountant, streams, *, step, identity, history):
    return dict(version=STATE_VERSION, step=step, identity=identity, history=history,
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict() if scheduler is not None else None,
                accountant=accountant.state_dict() if accountant is not None else None,
                stream_secret=streams._secret,
                streams={name: gen.get_state() for name, gen in streams.generators.items()},
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_payload(payload, model, optimizer, scheduler, accountant, *, identity, device):
    if payload["version"] != STATE_VERSION or payload["identity"] != identity:
        raise RuntimeError("resume configuration/source/input/environment identity mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if accountant is not None:
        accountant.load_state_dict(payload["accountant"])
    streams = Streams(payload["stream_secret"], device)
    for name, state in payload["streams"].items():
        streams[name].set_state(state.cpu())
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    return streams


def assert_state_equal(left, right, path="state"):
    if torch.is_tensor(left):
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise RuntimeError(f"state round-trip changed {path}")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise RuntimeError(f"state round-trip changed {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"state keys differ at {path}")
        for key in left:
            assert_state_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"state lengths differ at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_state_equal(a, b, f"{path}.{index}")
    elif left != right:
        raise RuntimeError(f"state round-trip changed {path}")


def write_private_snapshot(directory, payload):
    """Immutable generations plus an atomic pointer; no partial .pt is resumable."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    path = directory / f"state_{payload['step']:07d}.pt"
    if path.exists():
        raise RuntimeError("private snapshot generation already exists")
    temp = path.with_name(path.name+"."+secrets.token_hex(8)+".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    receipt = dict(file=path.name, sha256=sha(path), step=payload["step"])
    pointer = directory / "latest.tmp"
    fd = os.open(pointer, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(receipt, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pointer, directory / "latest.json")
    return receipt


def read_private_snapshot(directory):
    receipt = json.loads((directory / "latest.json").read_text())
    if Path(receipt["file"]).name != receipt["file"]:
        raise RuntimeError("invalid private snapshot name")
    path = directory / receipt["file"]
    if sha(path) != receipt["sha256"]:
        raise RuntimeError("private resume snapshot SHA-256 mismatch")
    # Only our own hash-verified protected snapshot, never an external artifact.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["step"] != receipt["step"]:
        raise RuntimeError("private resume step mismatch")
    return payload, receipt


def private_root(path, *, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise RuntimeError("private random root has unsafe permissions")
        secret = path.read_bytes()
    else:
        secret = secrets.token_bytes(32) if expected is None else expected
        if len(secret) != 32:
            raise RuntimeError("private random root must have exactly 32 bytes")
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
            handle.write(secret)
            handle.flush()
            os.fsync(handle.fileno())
    if len(secret) != 32:
        raise RuntimeError("private random root must have exactly 32 bytes")
    if expected is not None and not secrets.compare_digest(secret, expected):
        raise RuntimeError("protected resume secret differs from the shared sibling root")
    return secret




def update(model, optimizer, x, y, adversarial, clean_probs, cfg, noise_generator):
    device = next(model.parameters()).device
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summed = torch.zeros(nparams, device=device)
    microbatch = int(cfg.intervention.microbatch_size)
    for start in range(0, len(y), microbatch):
        end = start+microbatch
        g, h = gradient_branches(model, x[start:end], y[start:end], adversarial[start:end], clean_probs[start:end])
        value = contribution(g, h, method=str(cfg.control.method), public_scale=float(cfg.control.fixed_M_scale) if cfg.control.method=="M_scaled" else None, **options(cfg))
        summed.add_(value.sum(0))
    released = release(summed, sigma=float(cfg.control.sigma), bound=float(cfg.intervention.bound),
                       denominator=512, generator=noise_generator,
                       nonprivate=cfg.control.method == "nonprivate")
    optimizer.zero_grad(set_to_none=True)
    assign_gradient(model, released)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("nonfinite model after update")


def public_startup(model, dev, indices, cfg, output):
    """Configured gradient/optimizer path on public data, restoring all weights."""
    device = next(model.parameters()).device
    with isolated_public_rng():
        generator = torch.Generator(device=device).manual_seed(19092026)
        x, y = batch(dev, indices[:int(cfg.intervention.microbatch_size)], device)
        adv, probs = attack(model, x, cfg, generator)
        g, h = gradient_branches(model, x[:2], y[:2], adv[:2], probs[:2])
        params = [p for p in model.parameters() if p.requires_grad]
        for idx in range(2):
            loss = F.cross_entropy(model(x[idx:idx+1]), y[idx:idx+1])
            loss = loss+F.kl_div(model(adv[idx:idx+1]).log_softmax(1), probs[idx:idx+1].detach(), reduction="sum")
            expected = torch.cat([grad.flatten() for grad in torch.autograd.grad(loss, params)])
            torch.testing.assert_close(g[idx]+h[idx], expected, rtol=3e-4, atol=2e-5)
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        optimizer = build_optimizer(cfg, model)
        update(model, optimizer, x, y, adv, probs, cfg, generator)
        model.load_state_dict(initial, strict=True)
        optimizer.zero_grad(set_to_none=True)
        clean, pgd = evaluate(model, dev, indices[:2], cfg, pgd=True)
        if len(clean) != 2 or len(pgd) != 2:
            raise RuntimeError("public evaluation smoke failed")
    write_json(output / "public_startup.json", dict(status="PASS", method=str(cfg.control.method),
               public_only=True, private_updates=0, fp32=True, eval_mode=True,
               physical_batch=int(cfg.intervention.microbatch_size), outer_kl="detached_clean",
               gradient_equivalence="CE + stop-target KL autograd", pgd20_smoke_examples=2))


def public_probe(model, dev, indices, cfg, step, output):
    """Same-theta rules, evaluated along THIS method's own trajectory."""
    device = next(model.parameters()).device
    arms = ("J", "U", "D", "M", "R", "N") + (("M_scaled",) if cfg.control.method=="M_scaled" else ())
    sums, norms = {}, {arm: [] for arm in arms}
    zeros = {arm: 0 for arm in arms}
    ngs, nhs = [], []
    with isolated_public_rng():
        generator = torch.Generator(device=device).manual_seed(20260820)
        for start in range(0, len(indices), 4):
            x, y = batch(dev, indices[start:start+4], device)
            adversarial, probs = attack(model, x, cfg, generator)
            g, h = gradient_branches(model, x, y, adversarial, probs)
            ngs.extend(g.norm(dim=1).cpu().tolist())
            nhs.extend(h.norm(dim=1).cpu().tolist())
            for arm in arms:
                values = contribution(g, h, method=arm, public_scale=float(cfg.control.fixed_M_scale) if arm=="M_scaled" else None, **options(cfg))
                vn = values.norm(dim=1)
                norms[arm].extend(vn.cpu().tolist())
                zeros[arm] += int((vn == 0).sum())
                if arm not in sums:
                    sums[arm] = torch.zeros(values.shape[1], device=device)
                sums[arm].add_(values.sum(0))
    means = {arm: total/len(indices) for arm, total in sums.items()}
    j, r = means["J"], means["R"]
    j2 = j.dot(j)
    fitted_c = float(j.dot(r)/j2) if j2 > 0 else None
    row = dict(step=step, count=len(indices), trajectory_method=str(cfg.control.method),
               comparison="same_theta_counterfactual_queries_on_own_trajectory",
               ordered_probe_ids_sha256=digest(list(map(int, indices))), public_only=True,
               aggregate_denominator=len(indices), training_denominator=512,
               mean_contribution_norm={a: float(m.norm()) for a, m in means.items()},
               mean_individual_norm={a: float(np.mean(n)) for a, n in norms.items()},
               zero_fraction={a: z/len(indices) for a, z in zeros.items()},
               at_bound_fraction={a: float(np.mean(np.asarray(n) >= float(cfg.intervention.bound)*(1-2e-6))) for a, n in norms.items()},
               mean_g_norm=float(np.mean(ngs)), mean_h_norm=float(np.mean(nhs)),
               fitted_aggregate_cJ_to_R=fitted_c,
               fitted_aggregate_residual_norm=float((r-fitted_c*j).norm()) if fitted_c is not None else None)
    from .query_geometry import aggregate_diagnostics
    scale=float(cfg.control.fixed_M_scale) if cfg.control.method=="M_scaled" else None
    row['aggregate_geometry']=aggregate_diagnostics(means,scale)
    if scale is not None:
        calibration=json.loads(Path(str(cfg.control.scale_calibration.path)).read_text())
        row['fixed_M_scale_calibration_sha256']=sha(str(cfg.control.scale_calibration.path))
        if step==0:
            if row['ordered_probe_ids_sha256']!=calibration['ordered_probe_ids_sha256']:raise RuntimeError('public calibration probe IDs mismatch')
            observed=row['mean_contribution_norm']
            for arm in ['U','M']:
                if abs(observed[arm]/calibration['public_aggregate_norms'][arm]-1)>5e-4:raise RuntimeError('public initialization aggregate differs from frozen calibration')
            if abs(row['aggregate_geometry']['fixed_scale']['scaled_M_over_U']-1)>5e-4:raise RuntimeError('initial public norm matching failed')
            row['initial_public_scale_gate']='PASS'
    folder = output / "geometry"
    write_json(folder / f"step_{step:04d}.json", row)
    np.savez_compressed(folder / f"step_{step:04d}.npz", sample_ids=np.asarray(indices),
                        g_norm=np.asarray(ngs), h_norm=np.asarray(nhs),
                        **{f"{a}_norm": np.asarray(n) for a, n in norms.items()})


def public_checkpoint(model, raw, step, output):
    path = output / "checkpoints" / ("final.pt" if step == TOTAL_STEPS else f"step_{step:04d}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model={n: t.detach().cpu() for n, t in model.state_dict().items()},
                   config=raw, optimizer_step=step, epoch=step//88, source_pretrain_sha256=PRETRAIN_SHA,
                   private_rng_in_artifact=False)
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=True)
        assert_state_equal(old, payload, "existing_public_checkpoint")
        return path
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def train(raw, output: Path, private_state_dir: Path, *, stop_after=None, device="cuda", pilot_steps=100,
          private_root_file: Path | None = None, checkpoint_callback=None, backup_every=440):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = validate_config(raw)
    if device != "cuda" or not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("training requires one NVIDIA A100; CPU/other GPU fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU is required")
    if not 50 <= pilot_steps <= 100:
        raise ValueError("pilot must be a 50..100-update prefix")
    if stop_after is not None and not pilot_steps <= stop_after <= TOTAL_STEPS:
        raise ValueError("stop_after must include the full prefix pilot")
    interval = int(cfg.control.get("checkpoint_interval", 100))
    if interval <= 0 or interval > 100:
        raise ValueError("protected checkpoint interval must be 1..100 updates")
    if backup_every <= 0:
        raise ValueError("backup interval must be positive")
    output, private_state_dir = Path(output).resolve(), Path(private_state_dir).resolve()
    root_path = Path(private_root_file).resolve() if private_root_file else private_state_dir / "root.key"
    if private_state_dir == output or output in private_state_dir.parents or output in root_path.parents:
        raise RuntimeError("private state/random root must be outside the public output tree")
    if private_state_dir in output.parents:
        raise RuntimeError("public output cannot be nested inside private state")
    from .inputs import verify_cifar_initialization
    verify_cifar_initialization(Path(str(cfg.model.weights_path)), PRETRAIN_SHA)
    if sha(str(cfg.control.get("dataset_archive", DATA))) != DATA_SHA:
        raise RuntimeError("frozen CIFAR source archive hash mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    audit = planned_accounting(cfg)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.yaml"
    if config_path.exists() and OmegaConf.to_container(OmegaConf.load(config_path), resolve=False) != raw:
        raise RuntimeError("existing output config mismatch")
    OmegaConf.save(OmegaConf.create(raw), config_path)
    write_json(output / "planned_privacy_accounting.json", audit)
    identity = dict(config_sha256=digest(raw), source=source_identity(),
                    source_pretrain_sha256=PRETRAIN_SHA, ordered_dev_sha256=DEV_SHA,
                    gpu=torch.cuda.get_device_name(), cuda_version=torch.version.cuda)
    model = model_for(cfg, device)
    optimizer, scheduler = build_optimizer(cfg, model), None
    scheduler = build_scheduler(cfg, optimizer)
    from opacus.accountants import RDPAccountant
    accountant = None if cfg.control.method == "nonprivate" else RDPAccountant()
    augmented, dev, train_ids, dev_ids = datasets(cfg)
    train_ids = np.asarray(train_ids)
    probe_gen = torch.Generator().manual_seed(20260820)
    probe_ids = [dev_ids[k] for k in torch.randperm(len(dev_ids), generator=probe_gen)[:PROBE_SIZE].tolist()]
    step, history = 0, []
    resumed = (private_state_dir / "latest.json").exists()
    if resumed:
        saved, _ = read_private_snapshot(private_state_dir)
        private_root(root_path, expected=saved["stream_secret"])
        streams = restore_payload(saved, model, optimizer, scheduler, accountant, identity=identity, device=device)
        step, history = int(saved["step"]), saved["history"]
        if not 0 <= step <= TOTAL_STEPS:
            raise RuntimeError("invalid checkpoint step")
        restored = state_payload(model, optimizer, scheduler, accountant, streams,
                                 step=step, identity=identity, history=history)
        assert_state_equal(saved, restored)
        if step >= pilot_steps and not (output / "pilot.json").exists():
            write_json(output / "pilot.json", dict(status="PASS", prefix_updates=pilot_steps,
                       full_schedule_updates=TOTAL_STEPS, full_state_roundtrip="bitwise_equal",
                       verified_on_resume_at_step=step, continued_same_run=True,
                       training_state_in_public_artifacts=False))
        if step == pilot_steps:
            public_checkpoint(model, raw, step, output)
        del restored
        del saved
    else:
        if (output / "pilot.json").exists() or list((output / "checkpoints").glob("*.pt")):
            raise RuntimeError("public artifacts exist but protected resume state is missing; refusing restart")
        public_startup(model, dev, dev_ids, cfg, output)
        streams = Streams(private_root(root_path), device)
        public_checkpoint(model, raw, 0, output)
        public_probe(model, dev, probe_ids, cfg, 0, output)
        write_private_snapshot(private_state_dir, state_payload(model, optimizer, scheduler, accountant,
                              streams, step=0, identity=identity, history=history))
        if checkpoint_callback is not None:
            checkpoint_callback(private_state_dir, 0)
    started = time.monotonic()
    invocation_initial_step = step
    prefix_update_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    final_step = TOTAL_STEPS if stop_after is None else stop_after
    if step > final_step:
        raise RuntimeError("requested stop precedes protected resume step")
    for step in range(step+1, final_step+1):
        update_started = time.monotonic()
        selected = torch.where(torch.rand(45000, generator=streams["sampling"]) < 1/88)[0].numpy()
        if len(selected):
            with streams.augmentation():
                x, y = batch(augmented, train_ids[selected], device)
            adv, probs = attack(model, x, cfg, streams["attack"])
        else:
            x = adv = torch.empty((0, 3, 32, 32), device=device)
            y = torch.empty(0, dtype=torch.long, device=device)
            probs = torch.empty((0, 10), device=device)
        update(model, optimizer, x, y, adv, probs, cfg, streams["noise"])
        if accountant is not None:
            accountant.step(noise_multiplier=float(cfg.control.sigma), sample_rate=1/88)
        prefix_update_seconds += time.monotonic()-update_started
        if step % 20 == 0:
            print(json.dumps(dict(run=str(cfg.experiment.name), step=step, total=TOTAL_STEPS,
                                  lr=optimizer.param_groups[0]["lr"])), flush=True)
        if step % STEPS_PER_EPOCH == 0:
            with isolated_public_rng():
                clean, _ = evaluate(model, dev, dev_ids, cfg)
            history.append(dict(step=step, epoch=step//88, lr=optimizer.param_groups[0]["lr"],
                                dev_clean=100*float(np.mean(clean))))
            scheduler.step()
        if step in PROBE_STEPS:
            public_checkpoint(model, raw, step, output)
            public_probe(model, dev, probe_ids, cfg, step, output)
        if step % interval == 0 or step % backup_every == 0 or step in (pilot_steps, final_step):
            before = state_payload(model, optimizer, scheduler, accountant, streams,
                                   step=step, identity=identity, history=history)
            write_private_snapshot(private_state_dir, before)
            if step == pilot_steps:
                saved, receipt = read_private_snapshot(private_state_dir)
                streams = restore_payload(saved, model, optimizer, scheduler, accountant, identity=identity, device=device)
                after = state_payload(model, optimizer, scheduler, accountant, streams,
                                      step=step, identity=identity, history=history)
                assert_state_equal(before, after)
                torch.cuda.synchronize()
                write_json(output / "pilot.json", dict(status="PASS", prefix_updates=step,
                           full_schedule_updates=TOTAL_STEPS, checkpoint_sha256=receipt["sha256"],
                           full_state_roundtrip="bitwise_equal", continued_same_run=True,
                           checkpoint_retention="protected_external_directory",
                           training_state_in_public_artifacts=False,
                           gpu_name=torch.cuda.get_device_name(), visible_gpu_count=torch.cuda.device_count(),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                           prefix_wall_seconds=time.monotonic()-started,
                           prefix_training_seconds=prefix_update_seconds,
                           timed_updates=step-invocation_initial_step,
                           prefix_updates_per_second=(step-invocation_initial_step)/prefix_update_seconds,
                           timing_scope="real prefix updates; excludes public evaluation, probe and checkpoint time"))
                public_checkpoint(model, raw, step, output)
                del saved, after
            del before
            if checkpoint_callback is not None and (step % backup_every == 0 or step in (pilot_steps, final_step)):
                checkpoint_callback(private_state_dir, step)
        (output / "history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history))
    if final_step < TOTAL_STEPS:
        status = dict(status="PARTIAL", completed_steps=final_step, planned_steps=TOTAL_STEPS,
                      config_sha256=digest(raw), official_test_accessed=False)
        write_json(output / "partial.json", status)
        return status
    if accountant is not None:
        actual_epsilon = accountant.get_epsilon(1e-5)
        if abs(actual_epsilon-audit["achieved_epsilon_primary"]) > 1e-8:
            raise RuntimeError("actual accountant history differs from complete planned schedule")
    checkpoint = public_checkpoint(model, raw, TOTAL_STEPS, output)
    with isolated_public_rng():
        clean, pgd = evaluate(model, dev, dev_ids, cfg, pgd=True)
    np.savez_compressed(output / "dev_predictions.npz", sample_ids=np.asarray(dev_ids),
                        clean=np.asarray(clean), pgd20=np.asarray(pgd))
    write_json(output / "privacy_accounting.json", {**audit, "actual_steps": TOTAL_STEPS})
    write_json(output / "rng_audit.json", dict(commitment=streams.commitment(), final_positions=streams.positions(),
               protected_state_persisted=True, public_rng_state=False, joint_release_dp_claim=False))
    result = dict(status="TRAINED", training_status="COMPLETE", run_id=str(cfg.experiment.name), method=str(cfg.control.method),
                  steps=TOTAL_STEPS, completed_steps=TOTAL_STEPS,
                  checkpoint_sha256=sha(checkpoint), config_sha256=digest(raw),
                  ordered_dev_sha256=digest(dev_ids), probe_ids_sha256=digest(probe_ids),
                  dev_count=len(dev_ids), dev_clean=100*float(np.mean(clean)), dev_pgd20=100*float(np.mean(pgd)),
                  autoattack_status="NOT_RUN", official_test_accessed=False,
                  resumed=resumed, runtime_this_invocation_seconds=time.monotonic()-started,
                  source_identity=identity["source"])
    write_json(output / "run_summary.json", result)
    if (output / "partial.json").exists():
        write_json(output / "partial.json", dict(status="SUPERSEDED_BY_COMPLETE", completed_steps=TOTAL_STEPS))
    return result




def main():
    parser = argparse.ArgumentParser(description="Frozen CIFAR gradient-control training; one A100.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-state-dir", type=Path, required=True)
    parser.add_argument("--private-root-file", type=Path)
    parser.add_argument("--pilot-steps", type=int, default=100)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=False)
    print(json.dumps(train(raw, args.output, args.private_state_dir,
        stop_after=args.stop_after, private_root_file=args.private_root_file,
        pilot_steps=args.pilot_steps), sort_keys=True))

if __name__ == "__main__":
    main()
