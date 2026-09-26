"""SSG-initialized final-iterate private runs, with common J/U random streams.

No raw private loss, per-record gradient, batch membership or RNG seed is saved.
Diagnostic geometry and losses use only the fixed public-development records.
"""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from trustml.convnext_cifar100.preflight import write_json
from trustml.convnext_gtsrb.accounting import horizon
from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.common import code_identity, diagnostic_rng, tensor_digest
from trustml.convnext_gtsrb.data import digest
from trustml.convnext_gtsrb.evaluation import evaluate
from trustml.convnext_gtsrb.mechanism import (gradients, contribution, private_release,
                                           assign_gradient, lr_at, validate_outer_kl_target)
from trustml.convnext_gtsrb.model import build_model, sha256
from trustml.convnext_gtsrb.probe import audit_geometry
from .stages import coupling_key
from .telemetry import optimizer_step


def matched_streams(master_key, row, device="cuda"):
    anchor = coupling_key(row)
    seeds = {name: int.from_bytes(hmac.new(master_key, f"{anchor}/{name}".encode(), hashlib.sha256).digest()[:8], "big") % 2**63
             for name in ["sampler", "attack", "noise", "augmentation"]}
    torch.manual_seed(seeds["augmentation"])
    random.seed(seeds["augmentation"])
    np.random.seed(seeds["augmentation"] % 2**32)
    generators = {name: torch.Generator(device="cpu" if name == "sampler" else device).manual_seed(seeds[name])
                  for name in ["sampler", "attack", "noise"]}
    report = {"coupling_key": anchor, "stream_commitments": {
        name: hashlib.sha256(str(seed).encode()).hexdigest() for name, seed in seeds.items()},
        "raw_seeds_persisted": False, "exact_matched_J_U_coupling": True,
        "generator": "torch.Generator; not CSPRNG-certified", "public_model_seed": row["seed_label"],
        "privacy_scope": "one fixed run; NOT a joint release of the correlated J/U pair"}
    return generators, report


def stream_positions(streams):
    return {name: hashlib.sha256(gen.get_state().numpy().tobytes()).hexdigest() for name, gen in streams.items()}


def training_microbatch(protocol, row):
    """Legacy deployments keep8; larger batches require a public audit lock."""
    audit = protocol.get("execution", {}).get("microbatch_audit")
    if audit is None:
        return 8
    from .microbatch_probe import choose
    if audit.get("status") != "PASS" or audit.get("source_checkpoint_sha256") != row["source_checkpoint_sha256"]:
        raise ValueError("unverified physical batch calibration")
    selected = choose(audit["rows"])
    if selected != audit.get("selected_microbatch_by_role") or selected["Joint"] != selected["Componentwise"]:
        raise ValueError("invalid matched physical batch lock")
    policy = protocol["execution"].get("microbatch_policy", "maximum_safe")
    if policy == "fastest_common_safe":
        from .microbatch_probe import choose_fastest_common
        selected = choose_fastest_common(audit["rows"])
    elif policy != "maximum_safe":
        raise ValueError("unknown microbatch policy")
    batch = selected["CleanDP" if row["role"] == "Natural" else row["role"]]
    if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= 512:
        raise ValueError("invalid physical batch size")
    return batch


def uses_bounded_query(row):
    if row["role"] not in {"Natural", "CleanDP", "Joint", "Componentwise"}:
        raise ValueError("unrequested scientific arm")
    bounded = row["role"] != "Natural"
    zero_noise = row.get("zero_noise_control", False)
    if not isinstance(zero_noise, bool):
        raise ValueError("zero-noise control flag must be boolean")
    private = bounded and not zero_noise
    if row.get("private", private) is not private:
        raise ValueError("role/privacy-claim flag mismatch")
    if not bounded and (row["C"] is not None or row["q"] is not None or row["beta"] != 0):
        raise ValueError("Natural must have no clipping, allocation, or TRADES")
    if zero_noise and (row["role"] not in {"Joint", "Componentwise"}
                       or row.get("control_kind") != "zero_noise"
                       or row.get("clipping_map", "hard") != "hard"
                       or row.get("outer_kl_target") != "full"
                       or row["C"] != 1. or row["beta"] != 1):
        raise ValueError("invalid zero-noise bounded-query control")
    scale = row.get("uniform_joint_scale")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))
                              or not 0 < scale <= 1
                              or row["role"] != "Joint" or zero_noise
                              or row.get("control_kind") != "uniform_attenuation"
                              or row.get("clipping_map", "hard") != "hard"
                              or row.get("outer_kl_target") != "full"
                              or row["C"] != 1. or row["beta"] != 1):
        raise ValueError("invalid uniformly attenuated Joint control")
    if row.get('clipping_map', 'hard') == 'auto_s':
        if not bounded or row['C'] != 1. or row.get('gamma') not in [.01, .1, 1.]:
            raise ValueError('invalid unit-bounded AUTO-S config')
    elif row.get('clipping_map', 'hard') != 'hard':
        raise ValueError('unknown clipping map')
    if "intervention" in row:
        from .confirmation import validate_intervention
        validate_intervention(row)
    return bounded


def is_private_run(row):
    """Whether this run makes a DP claim and therefore performs Gaussian releases."""
    bounded = uses_bounded_query(row)
    return bounded and not row.get("zero_noise_control", False)


def accumulate_natural(model, x, y, denominator=512):
    """Unclipped CE gradient sum / fixed logical denominator, no noise draw."""
    loss = F.cross_entropy(model(x), y, reduction="sum") / denominator
    if not torch.isfinite(loss):
        raise RuntimeError("scientific failure: nonfinite nonprivate CE")
    loss.backward()


def load_selected(assets_root, selected_path, expected_hash):
    assets_root, selected_path = Path(assets_root), Path(selected_path)
    if sha256(selected_path) != expected_hash:
        raise ValueError("selected public checkpoint changed")
    metadata = json.loads((assets_root / "assets.json").read_text())["convnext_tiny_cvst_clean.pt"]
    model, architecture = build_model(assets_root / "convnext_tiny_cvst_clean.pt", metadata, seed=42)
    state = torch.load(selected_path, map_location="cpu", weights_only=False)
    if state["source_checkpoint_sha256"] != metadata["sha256"]:
        raise ValueError("public pretrain did not inherit the pinned clean initialization")
    model.load_state_dict(state["model"], strict=True)
    if not all(torch.equal(v, state["model"][k]) for k, v in model.state_dict().items()):
        raise ValueError("public weights, including classifier, not loaded exactly")
    architecture.update(newly_initialized=[], checkpoint_sha256=expected_hash,
                        source_kind="selected_robust_public_SSG", classifier_reinitialized=False,
                        public_pretrain_epoch=state["epoch"])
    return model.eval(), architecture


def public_trades_loss(model, data, output):
    """Full 4020-dev task/robust loss; independent of private data and randomness."""
    values = []
    for start in range(0, len(data.split["indices"]["dev"]), 64):
        ids = data.split["indices"]["dev"][start:start+64]
        x, y = data.batch(ids)
        x, y = x.cuda(), y.cuda()
        uniform = public_uniform(x, [data.split["records"][i]["image_id"] for i in ids], 20260921, "ssg_dev_trades")
        adv, _ = attack(model, x, y, objective="kl", uniform=uniform)
        with torch.no_grad():
            clean, robust = model(x), model(adv)
            task = F.cross_entropy(clean, y, reduction="none")
            kl = F.kl_div(robust.log_softmax(-1), clean.softmax(-1), reduction="none").sum(-1)
        values.extend(torch.stack([task, kl], dim=1).cpu().tolist())
    array = np.asarray(values)
    if array.shape != (4020, 2) or not np.isfinite(array).all():
        raise RuntimeError("invalid public TRADES evaluation")
    np.savez_compressed(Path(output) / "public_trades_losses.npz", losses=array,
                        dev_indices=data.split["indices"]["dev"])
    return {"dev_task_loss": float(array[:, 0].mean()), "dev_robust_loss": float(array[:, 1].mean())}


def train(row, protocol, data, calibration, privacy, assets_root, selected_path, master_key, output):
    outer_kl_target = validate_outer_kl_target(row.get("outer_kl_target", "full"))
    outer_kl_metadata = {"outer_kl_target": outer_kl_target} if "outer_kl_target" in row else {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bounded = uses_bounded_query(row)
    private = is_private_run(row)
    trades = row["role"] in {"Joint", "Componentwise"}
    if private and (privacy["status"] != "PASS" or max(privacy["rdp_epsilon_upper"], privacy["prv_epsilon_upper"]) > 4):
        raise ValueError("private accountant gate failed")
    h = horizon(len(data.split["indices"]["private"]))
    microbatch = training_microbatch(protocol, row)
    audit = protocol.get("execution", {}).get("microbatch_audit")
    if audit is not None and (audit["probe_hash"] != data.split["hashes"]["probe"] or audit["split_hash"] != data.split["split_hash"]):
        raise ValueError("physical batch audit split/probe mismatch")
    write_json(output / "training_execution.json", dict(training_microbatch=microbatch,
               logical_batch=512, microbatch_audit_hash=digest(audit) if audit else None,
               microbatch_policy=protocol.get("execution", {}).get("microbatch_policy", "legacy_8"),
               private=private, bounded_per_record_query=bounded,
               one_gaussian_release_per_logical_step=private,
               **outer_kl_metadata))
    if h["steps"] != 3436 or privacy["steps"] != h["steps"] or privacy["sample_rate"] != h["sample_rate"]:
        raise ValueError("horizon/accountant/sampler mismatch")
    model, architecture = load_selected(assets_root, selected_path, row["source_checkpoint_sha256"])
    initial_digest = tensor_digest(dict(model.named_parameters()))
    model.cuda().eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if row["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=row["lr"], momentum=.9, nesterov=True, weight_decay=5e-4)
    elif row["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(parameters, lr=row["lr"], betas=(.9, .95), eps=1e-8, weight_decay=.005)
    else:
        raise ValueError("unauthorized optimizer")
    streams, randomness = matched_streams(master_key, row)
    randomness["exact_matched_J_U_coupling"] = trades and row.get('rng_coupling') != 'independent_per_run'
    if row.get('rng_coupling') == 'independent_per_run':
        randomness['coupling_design'] = 'Independent per-run secret roots; matching refers to scientific parameters, not realized noise'
    if not private:
        randomness["privacy_scope"] = ("NONE: bounded zero-noise control"
                                       if bounded else "NONE: nonprivate CE control")
    write_json(output / "randomness_audit.json", randomness)
    write_json(output / "scientific_config.json", row)
    write_json(output / "architecture.json", architecture)
    sigma, C = (privacy["noise_multiplier"] if private else 0.), row["C"]
    accounting = ({**privacy, "private": True, "bounded_per_record_query": True,
                   "sensitivity_bound": C,
                   "absolute_noise_std_before_averaging": sigma * C} if private else
                  dict(status=("NOT_APPLICABLE_ZERO_NOISE_CONTROL" if bounded
                               else "NOT_APPLICABLE_NONPRIVATE"), private=False,
                       bounded_per_record_query=bounded, epsilon=None, delta=None,
                       noise_multiplier=0., sensitivity_bound=C if bounded else None,
                       absolute_noise_std_before_averaging=0., gaussian_release_count=0,
                       privacy_guarantee="NONE"))
    write_json(output / "gtsrb_privacy_accounting.json", accounting)
    code = code_identity()
    write_json(output / "code_identity.json", code)
    (output / "checkpoints").mkdir()
    if private:
        from opacus.accountants import RDPAccountant, PRVAccountant
        rdp, prv = RDPAccountant(), PRVAccountant()
    import wandb
    wb = wandb.init(project="ssg-gtsrb-v1", name=row["run_id"], mode="offline", dir=str(output), config=row, reinit=True)
    curves, telemetry, compute = [], [], []
    milestones = {step: int(epoch) for epoch, step in h["checkpoint_steps"].items()}
    started = time.monotonic()
    geometry_config = {"private": private, "bounded_query": bounded,
                       "role": "DPOnly" if row["role"] == "CleanDP" else row["role"],
                       "C": C, "q": row["q"], "sigma": sigma, "geometry_q_grid": protocol["geometry"]["diagnostic_q"],
                       "geometry_attack_seed": 20260921, **outer_kl_metadata}
    if row.get('clipping_map') == 'auto_s':
        geometry_config.update(clipping_map='auto_s', gamma=row['gamma'], lr=row['lr'])
    if "intervention" in row:
        geometry_config.update(intervention=row['intervention'], intervention_q=row['intervention_q'])
    if "uniform_joint_scale" in row:
        geometry_config["uniform_joint_scale"] = row["uniform_joint_scale"]

    def diagnose(step):
        epoch = milestones[step]
        path = output / "checkpoints" / ("final.pt" if epoch == 50 else f"epoch{epoch:03d}.pt")
        if path.exists():
            raise FileExistsError("immutable private checkpoint already exists")
        state = {"trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                 "optimizer_steps": step, "release_count": step if private else 0,
                 "bounded_query_steps": step if bounded else 0, "scientific_config": row,
                 "source_checkpoint_sha256": row["source_checkpoint_sha256"], "trainable_mask_hash": architecture["trainable_mask_hash"]}
        if epoch == 50:
            state["model"] = {n: p.detach().cpu() for n, p in model.state_dict().items()}
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, path)
        identity = {**code, "run_id": row["run_id"], "config_hash": digest(row), "checkpoint_sha256": sha256(path),
                    "source_checkpoint_sha256": row["source_checkpoint_sha256"], "step": step, "expected_epoch": step * h["sample_rate"],
                    "split_hash": data.split["split_hash"], "probe_hash": data.split["hashes"]["probe"],
                    "trainable_mask_hash": architecture["trainable_mask_hash"], "beta": row["beta"],
                    "private": private, "geometry_counterfactual_beta": 1, **outer_kl_metadata,
                    "C": C, "q": row["q"], "sigma": sigma, "optimizer": row["optimizer"],
                    "attack_seed": 20260921, "attack_protocol": protocol["attack"],
                    "normalization": "ImageNet mean/std once inside model wrapper"}
        with diagnostic_rng():
            t = time.monotonic()
            evaluation_output = output / f"evaluation_epoch{epoch:03d}"
            metrics = evaluate(model, data, evaluation_output, seed=42, final=epoch == 50)
            if epoch == 50 and protocol["private"].get("final_pgd_full_dev", False):
                full_data = copy.copy(data)
                full_data.split = copy.deepcopy(data.split)
                full_data.split["indices"]["trajectory"] = list(data.split["indices"]["dev"])
                full_data.split["hashes"]["trajectory"] = data.split["hashes"]["dev"]
                full = evaluate(model, full_data, evaluation_output / "full_dev_pgd20", seed=42, final=False)
                # Preserve the original 1024-record metric/NPZ agreement.
                metrics.update(pgd20_full_dev=full["pgd20"], pgd20_full_dev_examples=full["attack_examples"],
                    pgd20_full_dev_source="full_dev_pgd20/public_dev_predictions.npz")
            if trades:
                metrics.update(public_trades_loss(model, data, evaluation_output))
            if any(v for k, v in metrics.items() if k.endswith("attack_time_posthoc_mismatches")):
                raise RuntimeError("attack numerical disagreement: artifacts retained for diagnosis")
            write_json(evaluation_output / "metrics.json", metrics)
            write_json(evaluation_output / "provenance.json", identity)
            curves.append({"epoch": epoch, "step": step, "checkpoint_sha256": sha256(path), **metrics})
            write_json(output / "gtsrb_learning_curves.json", curves)
            compute.append({"phase": "public_evaluation", "epoch": epoch, "seconds": time.monotonic() - t})
            if epoch in protocol["private"]["geometry_epochs"]:
                t = time.monotonic()
                audit_geometry(model, data, architecture, calibration, geometry_config,
                               output / f"geometry_epoch{epoch:03d}", identity, microbatch=2)
                compute.append({"phase": "public_geometry", "epoch": epoch, "seconds": time.monotonic() - t})
        write_json(output / "gtsrb_compute_ledger.json", compute)
        wb.log({k: v for k, v in curves[-1].items() if isinstance(v, (float, int))}, step=step)
        print(json.dumps({"phase": "private_diagnostic", "run_id": row["run_id"], **curves[-1]}), flush=True)
        torch.cuda.empty_cache()

    ids = np.asarray(data.split["indices"]["private"])
    train_seconds = 0.
    try:
        diagnose(0)
        for step in range(1, h["steps"] + 1):
            t = time.monotonic()
            selected = ids[torch.nonzero(torch.rand(h["N_private"], generator=streams["sampler"]) < h["sample_rate"]).flatten().numpy()]
            summed = torch.zeros(architecture["trainable_parameter_count"], device="cuda") if bounded else None
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(selected), microbatch):
                x, y = data.batch(selected[start:start+microbatch], augment=True)
                x, y = x.cuda(), y.cuda()
                if not bounded:
                    accumulate_natural(model, x, y)
                    continue
                adv = None
                if trades:
                    uniform = torch.rand(x.shape, device="cuda", generator=streams["attack"])
                    adv, _ = attack(model, x, y, objective="kl", uniform=uniform)
                values = gradients(model, x, y, trades_adv=adv, beta=1.,
                                   outer_kl_target=outer_kl_target)
                if "intervention" in row:
                    from .confirmation import intervention_query
                    update = intervention_query(values, row)
                elif row.get('clipping_map') == 'auto_s':
                    from .autos_map import query
                    update = query(values, geometry_config['role'], row['gamma'], row['q'])
                else:
                    update = contribution(values, geometry_config["role"], C, row["q"])
                if "uniform_joint_scale" in row:
                    update = update * row["uniform_joint_scale"]
                if not torch.isfinite(update).all() or (update.norm(dim=-1) > C * (1 + 2e-5)).any():
                    raise RuntimeError("scientific failure: nonfinite or sensitivity violation")
                summed.add_(update.sum(0))
                del values, update
            if private:
                sanitized = private_release(summed, sigma=sigma, bound=C, generator=streams["noise"], denominator=512)
                assign_gradient(model, sanitized)
            elif bounded:
                # Explicit non-DP control: same bounded query and fixed logical
                # denominator, but no Gaussian draw and no accountant step.
                assign_gradient(model, summed / 512)
            else:
                # Empty Poisson batches still receive the scheduled SGD step.
                for parameter in parameters:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("scientific failure: nonfinite nonprivate gradient")
            lr = lr_at(step, row["lr"], total=h["steps"], warmup=h["warmup_steps"])
            optimizer.param_groups[0]["lr"] = lr
            if bounded:
                scalars = optimizer_step(optimizer, parameters)
                scalars["gradient_provenance"] = ("gaussian_sanitized"
                                                   if private else "bounded_noiseless_mean")
                if private:
                    rdp.step(noise_multiplier=sigma, sample_rate=h["sample_rate"])
                    prv.step(noise_multiplier=sigma, sample_rate=h["sample_rate"])
            else:
                optimizer.step()
                # Do not mislabel raw gradients as sanitized DP telemetry.
                scalars = {"learning_rate": lr, "private": False}
            if not all(torch.isfinite(p).all() for p in parameters):
                raise RuntimeError("scientific failure: nonfinite parameters")
            duration = time.monotonic() - t
            train_seconds += duration
            observation = {"step": step, "release_count": step if private else 0,
                           "bounded_query_steps": step if bounded else 0,
                           "step_seconds": duration, **scalars}
            telemetry.append(observation)
            with (output / "optimizer_telemetry.jsonl").open("a") as handle:
                handle.write(json.dumps(observation, allow_nan=False) + "\n")
            if step == 1 or step % 20 == 0:
                wb.log({k: v for k, v in observation.items() if isinstance(v, (float, int))}, step=step)
                print(json.dumps({"phase": "private_training", "run_id": row["run_id"], **observation}), flush=True)
            if step in milestones:
                diagnose(step)
        if private:
            if sum(v[2] for v in rdp.history) != 3436 or rdp.history != prv.history:
                raise RuntimeError("optimizer/accountant release mismatch")
            accounting.update(achieved_rdp=rdp.get_epsilon(delta=1e-5, alphas=privacy["orders"]),
                              achieved_prv=prv.get_epsilon(delta=1e-5, eps_error=.1, delta_error=1e-8))
            if max(accounting["achieved_rdp"], accounting["achieved_prv"]) > 4:
                raise RuntimeError("scientific failure: achieved epsilon exceeds target")
        write_json(output / "gtsrb_privacy_accounting.json", accounting)
        write_json(output / "accountant_history.json", {"rdp": rdp.history if private else [],
                   "prv": prv.history if private else [], "releases": 3436 if private else 0,
                   "release_count": 3436 if private else 0,
                   "bounded_query_steps": 3436 if bounded else 0, "private": private})
        randomness.update(final_stream_position_hashes=stream_positions(streams), initial_model_digest=initial_digest)
        write_json(output / "randomness_audit.json", randomness)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(telemetry), output / "gtsrb_optimizer_telemetry.parquet", compression="zstd")
        compute.append({"phase": ("private_training" if private else
                                  "bounded_zero_noise_training" if bounded else "nonprivate_training"),
                        "seconds": train_seconds})
        write_json(output / "gtsrb_compute_ledger.json", compute)
        final = {**row, "scientific_config": row, "status": "PASS", "optimizer_steps": 3436,
                 "release_count": 3436 if private else 0,
                 "bounded_query_steps": 3436 if bounded else 0,
                 "private": private, "bounded_per_record_query": bounded,
                 "dev_clean": curves[-1]["dev_clean"],
                 "pgd20": curves[-1].get("pgd20_full_dev", curves[-1]["pgd20"]),
                 "pgd20_examples": curves[-1].get("pgd20_full_dev_examples", curves[-1]["attack_examples"]),
                 "final_public_evaluation": curves[-1], "accounting": accounting, "randomness_audit": randomness,
                 "checkpoint_sha256": sha256(output / "checkpoints/final.pt"),
                 "elapsed_seconds": time.monotonic() - started, "wandb_run_id": wb.id, "official_test_loaded": False}
        write_json(output / "summary.json", final)
    finally:
        wb.finish()
    return final
