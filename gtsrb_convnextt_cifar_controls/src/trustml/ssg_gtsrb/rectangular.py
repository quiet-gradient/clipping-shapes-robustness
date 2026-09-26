"""Locked epoch-50, 54-run fine-tuning; one C/LR group per remote job."""
import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import shutil
import tarfile

from trustml.convnext_cifar100.preflight import write_json, configure_gpu
from trustml.convnext_gtsrb.data import digest
from trustml.convnext_gtsrb.model import sha256

SOURCE_SHA = "862b1bd943b069b0eb25de6f3c7526394d68c80fd0f4de2651ea5ab94fa12a60"
CS = [("c010", .1), ("c030", .3), ("c100", 1.)]
LRS = [("lr003", .03), ("lr010", .1), ("lr030", .3)]
QS = [("q025", .25), ("q1of3", 1/3), ("q050", .5), ("q075", .75)]
DS_GROUPS = ["c030_lr010", "c010_lr010"]


def manifest():
    groups = []
    for ctag, c in CS:
        for ltag, lr in LRS:
            group = f"{ctag}_{ltag}"
            joint_id = f"rect54_joint_{group}_s42"
            rows = []
            for arm, qtag, q in [("CleanDP", "", None), ("Joint", "", None)] + [("Componentwise", t, q) for t, q in QS]:
                row = dict(run_id=f"rect54_{arm.lower()}_{group}" + ("_"+qtag if qtag else "") + "_s42",
                    role=arm, optimizer="SGD", C=c, lr=lr, q=q, beta=0 if arm=="CleanDP" else 1,
                    source_checkpoint_sha256=SOURCE_SHA, seed_label=42)
                if arm == "Componentwise":
                    row["matched_joint_run_id"] = joint_id
                rows.append(row)
            if group in DS_GROUPS:
                groups.append(dict(group=group+"_clean", backend="datasphere", runs=rows[:1]))
                rows = rows[1:]
            groups.append(dict(group=group, backend="mlm", runs=rows))
    groups.sort(key=lambda g: (0, DS_GROUPS.index(g["group"].removesuffix("_clean")))
                if g["backend"] == "datasphere" else (1, g["group"]))
    return dict(campaign="ssg_gtsrb_rect54", source_epoch=50, source_sha256=SOURCE_SHA,
                total_training_runs=54, gpu_per_job=1, groups=groups, official_test_allowed=False)


def protocol():
    # Reuse the audited training algorithm, not the obsolete grid/controller.
    p = json.loads(Path("configs/ssg_gtsrb_v1/protocol.json").read_text())
    p.update(campaign="ssg_gtsrb_rect54", backend="datasphere_and_mlm", status="PROTOCOL_SPECIFICATION")
    p["execution"] = dict(one_C_LR_group_per_job=True, gpu_count=1, artificial_walltime_hours=None)
    p["stage_order"] = ["load_locked_epoch50", "private_runs_with_independent_commits"]
    p["private"]["final_pgd_full_dev"] = True
    p["private"]["final_evaluation"] = ["PGD20_on_4020", "APGD_CE_and_APGD_T_on_1024"]
    p["private"]["full_autoattack_stage"] = "SEPARATE_NOT_SUBMITTED_WITH_TRAINING"
    p["sgd"]["learning_rates"] = [v for _, v in LRS]
    p["grid_per_optimizer"] = dict(CleanDP=9, Joint=9, Componentwise=36,
                                  C=[v for _, v in CS], q=[v for _, v in QS], anchor_gate=False)
    p.pop("adamw")
    p["private"]["noise_multiplier"] = 1.22344970703125
    p["private"]["noise_multiplier_status"] = "VERIFIED_RECOVERED_GTSRB_ACCOUNTING"
    p["source_epoch"] = 50
    p["source_checkpoint_sha256"] = SOURCE_SHA
    return p


def selected_group(group, run_ids=None, backend=None):
    result = copy.deepcopy(group)
    if run_ids:
        wanted = set(run_ids)
        result["runs"] = [r for r in group["runs"] if r["run_id"] in wanted]
        if len(wanted) != len(run_ids) or len(result["runs"]) != len(wanted):
            raise ValueError("unknown or duplicate selected run ID")
        # A transport split may move CleanDP, but must keep the matched J/U
        # block intact so its private streams can share one RAM-only secret.
        original_matched = {r["run_id"] for r in group["runs"] if r["role"] in {"Joint", "Componentwise"}}
        selected_matched = {r["run_id"] for r in result["runs"] if r["role"] in {"Joint", "Componentwise"}}
        if selected_matched and selected_matched != original_matched:
            raise ValueError("matched Joint/Componentwise block cannot be split")
    if backend:
        if backend not in {"datasphere", "mlm"}:
            raise ValueError("unknown backend")
        result["backend"] = backend
    return result


def transport_group(name):
    """Repack existing rows without changing the locked scientific manifest.

    The fifth-card probe contains nine previously queued trainings and keeps
    its entire matched block inside one process. No private secret is shipped.
    """
    groups = manifest()["groups"]
    paired_chains = {
        "ds_chain_a_10": ["c030_lr010", "c010_lr010"],
        "ds_chain_b_10": ["c030_lr030", "c100_lr010"],
        "ds_chain_c_10": ["c010_lr030", "c030_lr003"],
    }
    if name in paired_chains:
        rows = [r for key in paired_chains[name]
                for g in groups if g["group"] == key
                for r in g["runs"] if r["role"] != "CleanDP"]
        return copy.deepcopy(dict(group=name, backend="datasphere", runs=rows))
    if name != "ds_chain_e_9":
        return copy.deepcopy(next(g for g in groups if g["group"] == name))
    anchor = next(g for g in groups if g["group"] == "c010_lr003")
    joint = [r for r in anchor["runs"] if r["role"] == "Joint"]
    clean = [r for r in anchor["runs"] if r["role"] == "CleanDP"]
    components = [r for r in anchor["runs"] if r["role"] == "Componentwise"]
    for group_name in ["c010_lr030", "c030_lr003", "c100_lr003"]:
        group = next(g for g in groups if g["group"] == group_name)
        clean += [r for r in group["runs"] if r["role"] == "CleanDP"]
    # Start Joint first to obtain actual TRADES timings, then finish baselines.
    return copy.deepcopy(dict(group=name, backend="datasphere", runs=joint + clean + components))


def fetch(url, destination, expected_sha):
    from .persistence import TransactionStore
    if destination.exists():
        if sha256(destination) != expected_sha:
            raise RuntimeError("existing input hash mismatch")
        return
    with TransactionStore.request("GET", url, stream=True) as response:
        if response.status_code != 200:
            raise RuntimeError(f"input download HTTP {response.status_code}")
        with destination.open("xb") as f:
            for block in response.iter_content(8*1024*1024):
                f.write(block)
    if sha256(destination) != expected_sha:
        raise RuntimeError("downloaded input SHA mismatch")


def run(args, plan=None):
    import sys
    plan = plan or sys.modules[__name__]
    import torch
    from scripts.platform.gtsrb_inputs import materialize
    from trustml.convnext_gtsrb.data import GTSRBTrainOnly
    from .private import train, load_selected, training_microbatch
    from trustml.convnext_gtsrb.common import diagnostic_rng
    from trustml.convnext_gtsrb.runner import public_training_benchmark
    from .campaign import verify_completed_run, check_coupling
    from .persistence import TransactionStore
    if args.output.exists():
        raise FileExistsError("existing output: reconcile committed children, do not retrain")
    group = plan.transport_group(args.group)
    payload = json.loads(args.routes.read_text())
    assert payload["group"] == args.group and payload["manifest_hash"] == digest(plan.manifest())
    group = selected_group(group, getattr(args, "run_id", None), payload["backend"])
    assert set(payload["routes"]) == {r["run_id"] for r in group["runs"]}
    configure_gpu()
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("exactly one A100 required")
    print(json.dumps(dict(phase="rect54_start", group=args.group, gpu=torch.cuda.get_device_name(),
                         source_epoch=50, pretraining=False, training_runs=len(group["runs"]))), flush=True)
    inputs = payload["inputs"]
    archive = args.input_archive or Path("training_inputs.tar")
    if args.input_archive is None:
        fetch(inputs["training_url"], archive, inputs["training_sha256"])
    materialize(archive, inputs["training_sha256"])
    source_archive = args.selected_archive or Path("selected_archive.tar.gz")
    if args.selected_archive is None:
        fetch(inputs["selected_url"], source_archive, inputs["selected_archive_sha256"])
    if sha256(source_archive) != inputs["selected_archive_sha256"]:
        raise RuntimeError("selected archive mismatch")
    selected = Path("gtsrb_inputs/public_pretrain_selected.pt")
    with tarfile.open(source_archive, "r:gz") as tar:
        member = tar.getmember("public_pretrain_selected.pt")
        if not member.isfile():
            raise ValueError("unsafe checkpoint member")
        with tar.extractfile(member) as src, selected.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    if sha256(selected) != SOURCE_SHA:
        raise ValueError("selected checkpoint mismatch")
    p = plan.protocol()
    data_protocol = json.loads(Path("configs/convnext_gtsrb/protocol.json").read_text())
    data = GTSRBTrainOnly(Path("gtsrb_inputs/GTSRB_Final_Training_Images.zip"),
                         Path("gtsrb_inputs/splits.json"), data_protocol["augmentation"])
    locked_split = json.loads(Path("reports/ssg_gtsrb_v1/gtsrb_private_split_manifest.json").read_text())
    if data.split["split_hash"] != locked_split["split_hash"]:
        raise ValueError("locked split mismatch")
    calibration = json.loads(Path("configs/ssg_gtsrb_rect54/calibration.json").read_text())
    privacy = json.loads(Path("configs/ssg_gtsrb_rect54/accounting.json").read_text())
    assert calibration["source_checkpoint_sha256"] == SOURCE_SHA
    assert privacy["noise_multiplier"] == 1.22344970703125 and privacy["steps"] == 3436
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", p)
    write_json(args.output / "group_manifest.json", group)
    store = TransactionStore(payload["routes"], args.output / "transactions")
    if payload.get("natural_kl_recovery"):
        from .natural_kl_recovery import preflight
        with diagnostic_rng():
            preflight(payload["natural_kl_recovery"], group, data, selected,
                      args.output / "natural_kl_recovery")
    model, _ = load_selected(Path("gtsrb_inputs"), selected, SOURCE_SHA)
    model.cuda().eval()
    for scientific_role in dict.fromkeys(r["role"] for r in group["runs"]):
        row = next(r for r in group["runs"] if r["role"] == scientific_role)
        role = "DPOnly" if scientific_role == "CleanDP" else scientific_role
        check = args.output / "public_checks" / role
        check.mkdir(parents=True)
        with diagnostic_rng():
            public_training_benchmark(model, data, dict(training_microbatch=training_microbatch(p, row), private=role != "Natural", role=role,
                C=row["C"], q=1/3 if role == "Componentwise" else None,
                steps=3436, run_id="public_check_"+role), check)
    del model
    torch.cuda.empty_cache()
    # The common secret never leaves RAM. All J/U pairs live in this group/job.
    master = secrets.token_bytes(32)
    completed = {}
    for row in group["runs"]:
        # A restarted parent must never silently reproduce a committed child.
        with store.request("GET", payload["routes"][row["run_id"]]["receipt_get"]) as response:
            if response.status_code != 404:
                raise RuntimeError("existing/uncertain child receipt; recovery required, no automatic retraining")
        target = args.output / row["run_id"]
        result = train(row, p, data, calibration, privacy, Path("gtsrb_inputs"), selected, master, target)
        verify_completed_run(target, row, result)
        if row["role"] == "Componentwise":
            check_coupling(completed[row["matched_joint_run_id"]], result)
        store.commit(row["run_id"], target)
        if os.environ.get("RECT_MLM") == "1":
            from scripts.platform.mlm.publish_job_artifacts import publish_job_artifacts
            publish_job_artifacts([store.root / (row["run_id"] + ".tar.gz"),
                                   store.root / (row["run_id"] + ".COMMITTED.json")])
        completed[row["run_id"]] = result
        write_json(args.output / "completed.json", list(completed.values()))
        torch.cuda.empty_cache()
    return dict(status="PASS", group=args.group, completed_runs=len(group["runs"]), official_test_loaded=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--input-archive", type=Path)
    p.add_argument("--selected-archive", type=Path)
    p.add_argument("--run-id", action="append")
    p.add_argument("--output", type=Path, default=Path("outputs/rect54"))
    args = p.parse_args()
    print(json.dumps(run(args)), flush=True)
