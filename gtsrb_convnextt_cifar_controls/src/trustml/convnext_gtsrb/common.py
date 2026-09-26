"""Auditable provenance and isolated diagnostics; no CIFAR protocol defaults."""
import contextlib
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import digest
from trustml.convnext_cifar100.model import sha256


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def diagnostic_rng():
    """AA may reseed global RNGs. Public diagnostics must not alter training."""
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


def code_identity():
    path = Path("deployment.json")
    if path.exists():
        deployment = json.loads(path.read_text())
        for file, expected in deployment["source_hashes"].items():
            if sha256(file) != expected:
                raise ValueError("deployment source changed: " + file)
        return {k: deployment[k] for k in ["git_commit", "source_hashes"]}
    raise RuntimeError("immutable deployment identity required on GPU")


def provenance(config, split, architecture, checkpoint, *, step, epoch, lr, code):
    attack_keys = ["geometry_attack_seed", "attack_steps", "attack_epsilon", "attack_step_size", "beta",
                   "kl_direction", "inner_clean_target_detached", "outer_kl_clean_target_detached"]
    attack_config = {k: config[k] for k in attack_keys}
    ids = [split["records"][i] for i in split["indices"]["probe"]]
    return {**code, "config_hash": digest(config), "run_id": config.get("run_id", "public_preflight"),
        "source_checkpoint_hash": config["source_metadata"]["sha256"], "model_checkpoint_hash": sha256(checkpoint),
        "dataset_hash": config["data_sha256"], "split_hash": split["split_hash"], "probe_hash": split["hashes"]["probe"],
        "ordered_probe_records": ids, "normalization": config["normalization"], "normalization_hash": digest(config["normalization"]),
        "attack_config": attack_config, "attack_hash": digest(attack_config), "attack_seed": config["geometry_attack_seed"],
        "beta": config["beta"], "C": config.get("C"), "q": config.get("q"), "sigma": config["sigma"],
        "epsilon_target": config["target_epsilon"] if config["private"] else None, "delta": config["delta"] if config["private"] else None,
        "sample_rate": config["sample_rate"], "release_count": step if config["private"] else 0,
        "optimizer_step": step, "expected_epoch": epoch, "LR": lr, "trainable_mask_hash": architecture["trainable_mask_hash"],
        "geometry_implementation_hash": sha256(Path(__file__).with_name("geometry.py")),
        "private": config["private"], "privacy_scope": "one fixed-hyperparameter run, not the selection campaign"}
