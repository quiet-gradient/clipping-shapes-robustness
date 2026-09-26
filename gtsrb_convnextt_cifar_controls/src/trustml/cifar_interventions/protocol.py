"""Immutable anchors, separate from old tuning and confirmation manifests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from .mechanism import ARMS

ROOT = Path("configs/cifar_interventions_v1")
PRETRAIN = "artifacts/mlm_final_v1_stage1/trades_b3_s42/checkpoints/checkpoint_pgd10_epoch_170.pt"
PRETRAIN_SHA = "53211c5f45c3a2a9171448fade104c0d3c8fd7649482827063240b3808796262"
DATA = "data/cifar-10-python.tar.gz"
DATA_SHA = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
SEEDS = (45, 46, 47)
DEV_SHA = "b6662bf813faa7289b68e5799d2e0e14e4473eeebfc504418eddbe4fe4690635"
ANCHORS = {"hard": ("hc_locked_s43", 1/3, .025),
           "autos": ("sc_locked_s43", .25, .01510270856068702)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def configs():
    for map_name, (source, q, lr) in ANCHORS.items():
        base = yaml.safe_load(Path(f"configs/finetune/runs/{source}.yaml").read_text())
        if base["training"]["optimizer"]["lr"] != lr:
            raise RuntimeError("locked anchor LR changed")
        for seed in SEEDS:
            for arm in ARMS:
                cfg = copy.deepcopy(base)
                run_id = f"cifar_dmr_{map_name}_{arm.lower()}_s{seed}"
                cfg["experiment"].update(name=run_id, seed=seed)
                cfg["protocol"].update(name="cifar_interventions_v1", stage="interventions", arm=arm,
                                       status="frozen_anchor", selection_rule="none", test_locked=True)
                # These YAMLs MUST go through this runner, not the legacy strategy dispatcher.
                cfg["intervention"] = dict(map_name=map_name, arm=arm, q=q, bound=1., gamma=.1,
                    beta=1., parent=f"{map_name}_s{seed}", steps=4400, batches_per_epoch=88,
                    sampling_probability=1/88, denominator=512, microbatch_size=32,
                    pretrain_sha256=PRETRAIN_SHA, dataset_sha256=DATA_SHA,
                    outer_kl_target="detached_clean", zero_direction="zero",
                    rng_protocol="ram_only_coupled_v1", resume_allowed=False,
                    joint_release_dp_claim=False)
                cfg["data"]["num_workers"] = 0
                cfg["training"]["physical_batch_size"] = 32
                cfg["training"]["checkpoint_every_epochs"] = 0
                # Explicit new identity; historical defense fields are provenance
                # for architecture/optimizer only, not runtime dispatch.
                cfg["strategy"] = f"intervention_{map_name}_{arm}"
                cfg["strategy_config"] = dict(strategy=cfg["strategy"], sensitivity_bound=1.,
                    noise_multiplier=1.123046875, absolute_noise_std=1.123046875, release_count_per_step=1)
                yield cfg


def materialize():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configs():
        path = ROOT / f"{cfg['experiment']['name']}.yaml"
        content = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"refusing to overwrite a different config: {path}")
        if not path.exists():
            path.write_text(content)
        rows.append(dict(run_id=cfg["experiment"]["name"], parent=cfg["intervention"]["parent"],
                         config=str(path), config_sha256=sha(path), status="PREPARED_NOT_SUBMITTED"))
    path = ROOT / "manifest.json"
    content = json.dumps(dict(protocol="cifar_interventions_v1", runs=rows), indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError("existing manifest differs; do not overwrite deployment state")
    if not path.exists():
        path.write_text(content)
    return rows


def load_parent(parent):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rows = [r for r in manifest["runs"] if r["parent"] == parent]
    if len(rows) != 5:
        raise ValueError("parent must contain exactly five interventions")
    out = []
    for row in rows:
        if sha(row["config"]) != row["config_sha256"]:
            raise RuntimeError("config hash mismatch")
        cfg = yaml.safe_load(Path(row["config"]).read_text())
        out.append(cfg)
    if [c["intervention"]["arm"] for c in out] != list(ARMS):
        raise ValueError("invalid parent arm order")
    expected = {c["experiment"]["name"]: c for c in configs()}
    for c in out:
        if c != expected.get(c["experiment"]["name"]):
            raise RuntimeError("configuration no longer matches frozen protocol")
    return out
