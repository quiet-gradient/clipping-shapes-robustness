"""One DataSphere/A100 process: public pretrain -> lock -> SGD -> AdamW.

Stage commits are independent. No automatic retry of scientific failure and no
regeneration of a completed scientific run. No raw private randomness persisted.
"""
import argparse
import copy
import json
from pathlib import Path
import secrets
import time
from types import SimpleNamespace

import numpy as np
import torch

from trustml.convnext_cifar100.preflight import write_json, configure_gpu
from trustml.convnext_gtsrb.accounting import calibrate as calibrate_privacy
from trustml.convnext_gtsrb.common import diagnostic_rng
from trustml.convnext_gtsrb.data import GTSRBTrainOnly
from trustml.convnext_gtsrb.evaluation import evaluate
from trustml.convnext_gtsrb.model import sha256
from trustml.convnext_gtsrb.probe import calibrate_clipping
from trustml.convnext_gtsrb.runner import public_training_benchmark
from .persistence import TransactionStore, transaction_keys
from .pretrain import run as pretrain, validate_source_gate
from .private import load_selected, train
from .stages import initial_grid, anchors, component_grid, pareto
from .reports import plot_frontier, aggregate_campaign


def verify_completed_run(output, config, summary):
    from .private import is_private_run, uses_bounded_query
    private = is_private_run(config)
    bounded = uses_bounded_query(config)
    if summary["status"] != "PASS" or summary["optimizer_steps"] != 3436 or summary["release_count"] != (3436 if private else 0):
        raise RuntimeError("cannot commit incomplete run")
    if not private and (summary.get("private") is not False or
                        summary["accounting"].get("status") not in {
                            "NOT_APPLICABLE_NONPRIVATE", "NOT_APPLICABLE_ZERO_NOISE_CONTROL"}):
        raise RuntimeError("nonprivate baseline incorrectly labelled DP")
    observed_bounded = summary.get("bounded_query_steps")
    if observed_bounded is not None and observed_bounded != (3436 if bounded else 0):
        raise RuntimeError("bounded-query step count mismatch")
    if config.get("control_kind") and observed_bounded is None:
        raise RuntimeError("control run lacks bounded-query step audit")
    if summary["scientific_config"] != config or sha256(output / "checkpoints/final.pt") != summary["checkpoint_sha256"]:
        raise RuntimeError("run/checkpoint identity mismatch")
    for epoch in [0, 10, 20, 30, 40, 50]:
        for name in ["gtsrb_geometry_per_record.parquet", "gtsrb_geometry_layers.parquet",
                     "gtsrb_geometry_manifest.json", "gtsrb_geometry_aggregate.json"]:
            if not (output / f"geometry_epoch{epoch:03d}" / name).is_file():
                raise RuntimeError("required public geometry missing")
    if not (output / "gtsrb_optimizer_telemetry.parquet").is_file():
        raise RuntimeError("optimizer telemetry missing")


def check_coupling(joint, component):
    a, b = joint["randomness_audit"], component["randomness_audit"]
    for name in ["coupling_key", "stream_commitments", "final_stream_position_hashes", "initial_model_digest"]:
        if a[name] != b[name]:
            raise RuntimeError("matched common-randomness invariant failed: " + name)
    for key in ["C", "lr", "optimizer", "source_checkpoint_sha256", "seed_label"]:
        if joint[key] != component[key]:
            raise RuntimeError("matched scientific config changed: " + key)
    if joint["accounting"]["noise_multiplier"] != component["accounting"]["noise_multiplier"]:
        raise RuntimeError("matched noise multiplier changed")


def report_optimizer(folder, optimizer, rows, chosen):
    prefix = "gtsrb_" + optimizer.lower()
    # Keep both the global observed envelope and each arm's envelope.
    write_json(folder / (prefix + "_pareto.json"), {"optimizer": optimizer,
        "all": [r["run_id"] for r in pareto(rows)],
        "by_arm": {arm: [r["run_id"] for r in pareto([r for r in rows if r["role"] == arm])]
                   for arm in ["CleanDP", "Joint", "Componentwise"]},
        "selected_joint_anchors": [r["run_id"] for r in chosen]})
    md = [f"# {optimizer}: SSG-initialized private GTSRB", "",
          "Fixed final iterate; public-dev only; no official test.", "",
          "| Run | Arm | C | LR | q | Clean % | PGD20 % | APGD union % | RDP epsilon |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['run_id']} | {r['role']} | {r['C']:.9g} | {r['lr']:.9g} | {r['q']} | "
                  f"{r['dev_clean']:.4f} | {r['pgd20']:.4f} | {r['final_public_evaluation']['apgd_union']:.4f} | "
                  f"{r['accounting']['achieved_rdp']:.6f} |")
    (folder / (prefix + "_results.md")).write_text("\n".join(md) + "\n")
    pairs = []
    for u in [r for r in rows if r["role"] == "Componentwise"]:
        j = next(r for r in rows if r["run_id"] == u["matched_joint_run_id"])
        check_coupling(j, u)
        pairs.append({"componentwise": u["run_id"], "joint": j["run_id"], "q": u["q"],
                      "clean_U_minus_J_pp": u["dev_clean"] - j["dev_clean"],
                      "PGD_U_minus_J_pp": u["pgd20"] - j["pgd20"], "common_rng_verified": True})
    write_json(folder / (prefix + "_matched.json"), pairs)
    lines = [f"# {optimizer}: exact matched U → J", "", "Differences are U minus J, in percentage points.", "",
             "| U | J | q | Delta clean | Delta PGD20 | Shared RNG |", "|---|---|---:|---:|---:|---|"]
    lines += [f"| {p['componentwise']} | {p['joint']} | {p['q']:.6g} | {p['clean_U_minus_J_pp']:+.4f} | {p['PGD_U_minus_J_pp']:+.4f} | PASS |" for p in pairs]
    (folder / (prefix + "_matched.md")).write_text("\n".join(lines) + "\n")


def run(args):
    protocol = json.loads(args.protocol.read_text())
    decision = json.loads(args.decision.read_text())
    audit = json.loads(args.audit.read_text())
    sources = json.loads(Path("configs/ssg_gtsrb_v1/sources.json").read_text())
    validate_source_gate(audit, sources, decision)
    if not protocol["execution"]["single_superjob"] or protocol["backend"] != "datasphere_only":
        raise ValueError("one DataSphere superjob required")
    if args.output.exists():
        raise FileExistsError("existing campaign output: recover artifacts, never silently retrain")
    routes = json.loads(args.routes.read_text())
    if set(routes) != set(transaction_keys()):
        raise ValueError("missing or unexpected stage persistence routes")
    configure_gpu()
    if "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one A100 required")
    args.output.mkdir(parents=True)
    store = TransactionStore(routes, args.output / "transactions")
    report_dir = args.output / "reports"
    report_dir.mkdir()
    write_json(report_dir / "protocol.json", protocol)
    write_json(report_dir / "source_decision.json", decision)
    started = time.monotonic()
    public = args.output / "public_pretrain"
    selection = pretrain(SimpleNamespace(audit=args.audit, sources=Path("configs/ssg_gtsrb_v1/sources.json"),
        decision=args.decision, assets_root=args.assets_root, cache=args.cache, output=public, microbatch=64), store=store)
    selected_path = public / "public_pretrain_selected.pt"
    selected_hash = sha256(selected_path)
    write_json(report_dir / "selected_public_checkpoint.json", {**selection, "sha256": selected_hash})
    old_protocol = json.loads(Path("configs/convnext_gtsrb/protocol.json").read_text())
    if sha256(args.assets_root / "GTSRB_Final_Training_Images.zip") != sources["real_training"]["sha256"]:
        raise ValueError("real training archive differs from audited source")
    data = GTSRBTrainOnly(args.assets_root / "GTSRB_Final_Training_Images.zip",
                          args.assets_root / "splits.json", old_protocol["augmentation"])
    split_manifest = json.loads(Path("reports/ssg_gtsrb_v1/gtsrb_private_split_manifest.json").read_text())
    if data.split["split_hash"] != split_manifest["split_hash"]:
        raise ValueError("real split changed from locked pilot partition")
    model, architecture = load_selected(args.assets_root, selected_path, selected_hash)
    model.cuda().eval()
    # Initial transfer sanity evaluates PGD on ALL 4020 dev records, not the
    # 1024 trajectory subset used inside private runs. Original split unmodified.
    transfer_data = copy.copy(data)
    transfer_data.split = copy.deepcopy(data.split)
    transfer_data.split["indices"]["trajectory"] = list(data.split["indices"]["dev"])
    transfer_data.split["hashes"]["trajectory"] = data.split["hashes"]["dev"]
    with diagnostic_rng():
        transfer = evaluate(model, transfer_data, args.output / "real_transfer", seed=42)
    transfer.update(checkpoint_sha256=selected_hash, selection_uses_real_dev=False,
                    attack_subset="full_real_public_dev_4020_not_private_curve_subset")
    write_json(args.output / "real_transfer/metrics.json", transfer)
    store.commit("real_transfer", args.output / "real_transfer")
    calibration_root = args.output / "calibration"
    with diagnostic_rng():
        calibration = calibrate_clipping(model, data, calibration_root, microbatch=2, seed=20260921)
    calibration["source_checkpoint_sha256"] = selected_hash
    calibration["C_CE_low"] = calibration["C_CE"] / 2
    norms = np.load(calibration_root / "initial_gradient_norms.npz")["norms"]
    calibration["clipping_fractions"]["C_CE_low"] = {
        name: float(np.mean(norms[:, i] > calibration["C_CE_low"])) for i, name in enumerate(["g", "h", "s", "a"])}
    write_json(calibration_root / "calibration.json", calibration)
    privacy = calibrate_privacy(35189)
    write_json(calibration_root / "gtsrb_privacy_accounting.json", privacy)
    write_json(report_dir / "gtsrb_private_split_manifest.json", split_manifest)
    # Check actual per-record batch implementation on public records before
    # creating ANY private optimizer step. Same physical batch8 as every arm.
    for role in ["DPOnly", "Joint", "Componentwise"]:
        target = calibration_root / role
        target.mkdir()
        with diagnostic_rng():
            public_training_benchmark(model, data, {"training_microbatch": 8, "private": True,
                "role": role, "C": calibration["C_mid"], "q": 1/3 if role == "Componentwise" else None,
                "steps": 3436, "run_id": "public_check_" + role}, target)
    store.commit("calibration", calibration_root)
    del model
    torch.cuda.empty_cache()
    master = secrets.token_bytes(32)  # Never serialized, printed, or uploaded.
    completed = []
    def execute(row):
        target = args.output / "private" / row["run_id"]
        result = train(row, protocol, data, calibration, privacy, args.assets_root, selected_path, master, target)
        if row["role"] == "Componentwise":
            joint = next(r for r in completed if r["run_id"] == row["matched_joint_run_id"])
            check_coupling(joint, result)
        verify_completed_run(target, row, result)
        store.commit(row["run_id"], target)
        completed.append(result)
        write_json(report_dir / "gtsrb_completed_runs.json", completed)
    for optimizer in ["SGD", "AdamW"]:
        prefix = "gtsrb_" + optimizer.lower()
        grid = initial_grid(optimizer, calibration, selected_hash)
        write_json(report_dir / (prefix + "_grid_manifest.json"), {"initial_grid": grid, "componentwise": "AWAITING_JOINT_ANCHORS"})
        for row in grid:
            execute(row)
        chosen = anchors([r for r in completed if r["optimizer"] == optimizer and r["role"] == "Joint"])
        components = component_grid(chosen)
        write_json(report_dir / (prefix + "_grid_manifest.json"), {"initial_grid": grid,
            "selected_anchors": [r["run_id"] for r in chosen], "componentwise": components, "max_runs": 20})
        for row in components:
            execute(row)
        rows = [r for r in completed if r["optimizer"] == optimizer]
        report_optimizer(report_dir, optimizer, rows, chosen)
        plot_frontier(report_dir, optimizer, rows)
        store.commit(optimizer.lower() + "_reports", report_dir)
        print(json.dumps({"phase": "optimizer_study_complete", "optimizer": optimizer, "runs": len(rows)}), flush=True)
    aggregate_campaign(args.output, completed, protocol)
    summary = {"status": "PASS", "backend": "datasphere", "single_superjob": True,
               "private_runs": len(completed), "source_checkpoint_sha256": selected_hash,
               "elapsed_seconds": time.monotonic() - started, "official_test_loaded": False,
               "generator_independence_established": False, "source_limitation_accepted": True}
    write_json(report_dir / "gtsrb_campaign_summary.json", summary)
    store.commit("campaign_reports", report_dir)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=Path("configs/ssg_gtsrb_v1/protocol.json"))
    p.add_argument("--decision", type=Path, default=Path("configs/ssg_gtsrb_v1/source_decision.json"))
    p.add_argument("--audit", type=Path, default=Path("reports/ssg_gtsrb_v1/ssg_gtsrb_overlap_audit.json"))
    p.add_argument("--assets-root", type=Path, default=Path("gtsrb_inputs"))
    p.add_argument("--cache", type=Path, default=Path("ssg_cache"))
    p.add_argument("--output", type=Path, default=Path("outputs/ssg_gtsrb_v1"))
    p.add_argument("--routes", type=Path, required=True)
    args = p.parse_args()
    try:
        print(json.dumps(run(args)), flush=True)
    except Exception as exc:
        # Sanitized failure type only: credentials and private state stay private.
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "FAILED.json", {"status": "STOPPED", "exception_type": type(exc).__name__,
            "automatic_retry": False, "completed_remote_commits_preserved": True})
        raise


if __name__ == "__main__":
    main()
