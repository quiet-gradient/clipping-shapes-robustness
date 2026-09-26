"""Sealed, transactional official GTSRB test evaluation.

The module deliberately never prints scientific metrics.  A caller publishes
one immutable archive/receipt per training seed and may reveal the metrics only
after the complete predeclared roster is committed.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from trustml.convnext_cifar100.preflight import write_json
from trustml.convnext_gtsrb.attacks import attack, public_uniform
from trustml.convnext_gtsrb.data import digest
from trustml.convnext_gtsrb.evaluation import autoattack_inputs
from trustml.convnext_gtsrb.model import sha256


EXPECTED_EXAMPLES = 12_630
IMAGE_PREFIX = "GTSRB/Final_Test/Images"
IMAGE_ARCHIVE_SHA256 = "48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa"
GT_ARCHIVE_SHA256 = "f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d"
AUTOATTACK_STAGES = ["apgd-ce", "apgd-t", "fab-t", "square"]
AUTOATTACK_REVISION = "0.1 / a39220048b3c9f2cca9a4d3a54604793c68eca7e"


class GTSRBOfficialTest:
    """Strict ordered official-test loader in raw [0,1] pixel space."""

    def __init__(self, images_zip: Path, ground_truth_zip: Path):
        images_zip, ground_truth_zip = Path(images_zip), Path(ground_truth_zip)
        if sha256(images_zip) != IMAGE_ARCHIVE_SHA256:
            raise RuntimeError("official GTSRB test image archive SHA-256 mismatch")
        if sha256(ground_truth_zip) != GT_ARCHIVE_SHA256:
            raise RuntimeError("official GTSRB ground-truth archive SHA-256 mismatch")

        with zipfile.ZipFile(ground_truth_zip) as source:
            names = source.namelist()
            if names != ["GT-final_test.csv"]:
                raise ValueError("unexpected official ground-truth archive members")
            with source.open(names[0]) as handle:
                rows = list(csv.DictReader(io.TextIOWrapper(handle), delimiter=";"))
        if len(rows) != EXPECTED_EXAMPLES:
            raise ValueError("official GTSRB test must contain exactly 12630 labels")

        records = []
        for index, row in enumerate(rows):
            filename = f"{index:05d}.ppm"
            if row.get("Filename") != filename or not re.fullmatch(r"\d{5}\.ppm", filename):
                raise ValueError("official GTSRB test ordering is not canonical")
            label = int(row["ClassId"])
            if not 0 <= label < 43:
                raise ValueError("official GTSRB label is outside 0..42")
            records.append({"sample_id": filename, "label": label})

        pixels = np.empty((EXPECTED_EXAMPLES, 96, 96, 3), dtype=np.uint8)
        with zipfile.ZipFile(images_zip) as source:
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate official-test ZIP paths")
            available = set(names)
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("unsafe official-test ZIP path")
            for index, row in enumerate(records):
                member = f"{IMAGE_PREFIX}/{row['sample_id']}"
                if member not in available:
                    raise ValueError("missing official GTSRB test image")
                with source.open(member) as handle:
                    image = Image.open(handle).convert("RGB").resize(
                        (96, 96), Image.Resampling.BILINEAR
                    )
                    pixels[index] = np.asarray(image)
        self.records = records
        self.pixels = pixels
        self.ordered_hash = digest(records)

    def batch(self, ids):
        indices = np.asarray(ids, dtype=np.int64)
        pixels = torch.from_numpy(self.pixels[indices].copy()).permute(0, 3, 1, 2).float().div_(255)
        labels = torch.tensor([self.records[int(i)]["label"] for i in indices], dtype=torch.long)
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise RuntimeError("official-test pixels are not raw finite [0,1]")
        return pixels, labels


def clean_and_pgd(model, data: GTSRBOfficialTest, output: Path, *, batch_size=64):
    """Clean plus deterministic per-record PGD-20, five-restart worst case."""
    device = next(model.parameters()).device
    clean_parts, pgd_parts = [], []
    started = time.monotonic()
    for begin in range(0, EXPECTED_EXAMPLES, batch_size):
        ids = list(range(begin, min(EXPECTED_EXAMPLES, begin + batch_size)))
        x, y = data.batch(ids)
        x, y = x.to(device).contiguous(), y.to(device)
        with torch.no_grad():
            logits = model(x)
        if logits.shape != (len(ids), 43) or not torch.isfinite(logits).all():
            raise RuntimeError("invalid official-test clean logits")
        clean = logits.argmax(1).eq(y)
        robust = clean.clone()
        sample_ids = [data.records[i]["sample_id"] for i in ids]
        for restart in range(5):
            uniform = public_uniform(x, sample_ids, 42, f"official_pgd20_restart{restart}")
            _, survived = attack(
                model, x, y, objective="ce", steps=20, epsilon=8 / 255,
                step_size=2 / 255, uniform=uniform, track_best=True,
            )
            robust.logical_and_(survived)
        if torch.any(robust & ~clean):
            raise RuntimeError("PGD correctness is not a subset of clean correctness")
        clean_parts.append(clean.cpu().numpy())
        pgd_parts.append(robust.cpu().numpy())
    arrays = {
        "sample_ids": np.arange(EXPECTED_EXAMPLES, dtype=np.int64),
        "clean_correct": np.concatenate(clean_parts).astype(bool),
        "pgd20_5restart_correct": np.concatenate(pgd_parts).astype(bool),
    }
    if any(value.shape != (EXPECTED_EXAMPLES,) for value in arrays.values()):
        raise RuntimeError("incomplete official-test clean/PGD vectors")
    np.savez_compressed(output / "clean_pgd_masks.npz", **arrays)
    return arrays, time.monotonic() - started


def _checked_mask(value, previous):
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.shape != previous.shape:
        raise RuntimeError("invalid AutoAttack state mask")
    if np.any(value & ~previous):
        raise RuntimeError("AutoAttack stage resurrected failed examples")
    return value.copy()


def full_autoattack(model, data: GTSRBOfficialTest, output: Path, *, epsilon: float,
                    clean_correct: np.ndarray, batch_size=64):
    """Pinned standard AutoAttack with stage masks and conservative posthoc check."""
    from autoattack import AutoAttack
    from autoattack.state import EvaluationState

    output.mkdir(parents=True, exist_ok=False)
    ids = list(range(EXPECTED_EXAMPLES))
    x, y = data.batch(ids)
    x, y = autoattack_inputs(x, y, next(model.parameters()).device)
    model.eval()
    if x.dtype != torch.float32 or any(module.training for module in model.modules()):
        raise RuntimeError("official AutoAttack requires FP32 model.eval()")

    arrays = {"sample_ids": np.arange(EXPECTED_EXAMPLES), "clean_correct": clean_correct.copy()}
    stage_results = []
    last = [clean_correct.copy(), time.monotonic()]

    class CapturedState(EvaluationState):
        def add_run_attack(self, attack_name):
            super().add_run_attack(attack_name)
            mask = _checked_mask(self.robust_flags.detach().cpu().numpy(), last[0])
            arrays["after_" + attack_name] = mask
            now = time.monotonic()
            stage_results.append({
                "attack": attack_name,
                "survivors": int(mask.sum()),
                "seconds": now - last[1],
            })
            last[:] = [mask, now]
            np.savez_compressed(output / "stage_masks.npz", **arrays)
            write_json(output / "stage_metrics.json", stage_results)
            self.to_disk(force=True)

    started = time.monotonic()
    adversary = AutoAttack(
        model, norm="Linf", eps=epsilon, version="standard", seed=42,
        device=str(x.device), log_path=str(output / "autoattack.log"), verbose=False,
    )
    if adversary.attacks_to_run != AUTOATTACK_STAGES:
        raise RuntimeError("unexpected pinned AutoAttack standard component order")
    with patch("autoattack.autoattack.EvaluationState", CapturedState):
        adv = adversary.run_standard_evaluation(
            x, y, bs=batch_size, state_path=output / "state.json"
        )
    max_linf = float((adv - x).abs().max().item())
    if (not torch.isfinite(adv).all() or max_linf > epsilon + 1e-6
            or float(adv.min()) < 0 or float(adv.max()) > 1):
        raise RuntimeError("AutoAttack returned invalid official-test images")

    posthoc_parts = []
    for begin in range(0, EXPECTED_EXAMPLES, batch_size):
        with torch.no_grad():
            logits = model(adv[begin:begin + batch_size])
        if not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite official-test posthoc logits")
        posthoc_parts.append(logits.argmax(1).eq(y[begin:begin + batch_size]).cpu().numpy())
    state = EvaluationState.from_disk(output / "state.json")
    attack_time = _checked_mask(state.robust_flags.detach().cpu().numpy(), clean_correct)
    if set(AUTOATTACK_STAGES) - set(state.run_attacks) and attack_time.any():
        raise RuntimeError("incomplete AutoAttack with surviving official-test examples")
    posthoc = np.concatenate(posthoc_parts).astype(bool) & clean_correct
    final = attack_time & posthoc
    arrays.update(
        aa_attack_time_correct=attack_time,
        aa_posthoc_correct=posthoc,
        aa_correct=final,
    )
    np.savez_compressed(output / "predictions.npz", **arrays)
    result = {
        "status": "PASS" if np.array_equal(attack_time, posthoc) else "NUMERICAL_WARNING",
        "examples": EXPECTED_EXAMPLES,
        "epsilon": epsilon,
        "norm": "Linf",
        "attack_seed": 42,
        "precision": "FP32",
        "model_mode": "eval",
        "autoattack_version": AUTOATTACK_REVISION,
        "stage_results": stage_results,
        "clean_survivors": int(clean_correct.sum()),
        "final_survivors": int(final.sum()),
        "attack_time_posthoc_mismatches": int(np.count_nonzero(attack_time != posthoc)),
        "max_linf": max_linf,
        "seconds": time.monotonic() - started,
    }
    write_json(output / "metrics.json", result)
    return result


def evaluate_official_test(model, data: GTSRBOfficialTest, output: Path, *, provenance):
    output.mkdir(parents=True, exist_ok=False)
    model.float().cuda().eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("official-test model is not FP32")
    arrays, pgd_seconds = clean_and_pgd(model, data, output)
    aa8 = full_autoattack(
        model, data, output / "aa_8_255", epsilon=8 / 255,
        clean_correct=arrays["clean_correct"],
    )
    aa4 = full_autoattack(
        model, data, output / "aa_4_255", epsilon=4 / 255,
        clean_correct=arrays["clean_correct"],
    )
    if np.any(arrays["pgd20_5restart_correct"] & ~arrays["clean_correct"]):
        raise RuntimeError("official-test PGD monotonicity failed")
    summary = {
        "schema_version": 1,
        "sealed": True,
        "reveal_state": "DO_NOT_REVEAL_UNTIL_COMPLETE_24_MODEL_ROSTER",
        "sample_count": EXPECTED_EXAMPLES,
        "ordered_test_hash": data.ordered_hash,
        "image_archive_sha256": IMAGE_ARCHIVE_SHA256,
        "ground_truth_archive_sha256": GT_ARCHIVE_SHA256,
        "clean_survivors": int(arrays["clean_correct"].sum()),
        "pgd20_5restart_survivors": int(arrays["pgd20_5restart_correct"].sum()),
        "pgd_seconds": pgd_seconds,
        "autoattack_8_255": aa8,
        "autoattack_4_255": aa4,
        "provenance": provenance,
    }
    write_json(output / "sealed_summary.json", summary)
    write_json(output / "files.json", {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    })
    return summary
