"""Run one three-seed sealed GTSRB headline official-test parent job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.platform.gtsrb_inputs import materialize
from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.model import sha256
from .night_aa import extract_checkpoint, load_final
from .official_test import (
    GT_ARCHIVE_SHA256,
    IMAGE_ARCHIVE_SHA256,
    GTSRBOfficialTest,
    evaluate_official_test,
)
from .persistence import TransactionStore
from .rectangular import fetch


class InputNotReady(RuntimeError):
    pass


def validated_role_rows(manifest, routes, role):
    """Return frozen rows while preserving campaign-specific cardinality."""
    roles = manifest["roles"]
    if role not in roles:
        raise RuntimeError("unknown headline role")
    rows = roles[role]
    campaign = manifest.get("campaign")
    if campaign == "gtsrb-reviewer-controls-official-test-v1":
        if len(rows) != 1 or rows[0]["seed_label"] not in (45, 46, 47):
            raise RuntimeError("reviewer role must contain exactly one frozen seed")
    else:
        if len(rows) != 3:
            raise RuntimeError("headline role must contain exactly three frozen seeds")
        if [row["seed_label"] for row in rows] != [45, 46, 47]:
            raise RuntimeError("headline role seeds are not exactly 45,46,47")
    if {row["run_id"] for row in rows} != set(routes["sources"]):
        raise RuntimeError("source route coverage mismatch")
    if {row["eval_id"] for row in rows} != set(routes["outputs"]):
        raise RuntimeError("output route coverage mismatch")
    return rows


def committed_receipt(route, run_id):
    with TransactionStore.request("GET", route["receipt_get"]) as response:
        if response.status_code == 404:
            raise InputNotReady(f"source checkpoint is not committed: {run_id}")
        if response.status_code != 200:
            raise RuntimeError(f"source receipt lookup HTTP {response.status_code}")
        receipt = response.json()
    if receipt.get("status") != "COMMITTED" or receipt.get("stage") != run_id:
        raise RuntimeError("invalid source COMMITTED receipt")
    file_hashes = receipt.get("file_hashes", {})
    checkpoint_sha = file_hashes.get("checkpoints/final.pt")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise RuntimeError("source receipt lacks final checkpoint SHA-256")
    if not isinstance(receipt.get("archive_sha256"), str):
        raise RuntimeError("source receipt lacks archive SHA-256")
    return receipt, checkpoint_sha


def source_checkpoint(row, route, cache):
    receipt, checkpoint_sha = committed_receipt(route, row["run_id"])
    archive = cache / f"{row['run_id']}.tar.gz"
    fetch(route["archive_get"], archive, receipt["archive_sha256"])
    checkpoint = extract_checkpoint(
        archive, cache / row["run_id"] / "final.pt", checkpoint_sha
    )
    return checkpoint, receipt, checkpoint_sha


def _already_committed(route, eval_id):
    with TransactionStore.request("GET", route["receipt_get"]) as response:
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise RuntimeError(f"output receipt lookup HTTP {response.status_code}")
        receipt = response.json()
    if receipt.get("status") != "COMMITTED" or receipt.get("stage") != eval_id:
        raise RuntimeError("invalid existing output receipt")
    return receipt


def run(args):
    configure_gpu()
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("headline official test requires exactly one A100")
    manifest = json.loads(args.manifest.read_text())
    routes = json.loads(args.routes.read_text())
    if manifest.get("sealed") is not True or manifest.get("official_test_authorized") is not True:
        raise RuntimeError("missing explicit sealed official-test authorization")
    if manifest.get("manifest_sha256") != routes.get("manifest_sha256"):
        raise RuntimeError("official-test manifest/route identity mismatch")
    rows = validated_role_rows(manifest, routes, args.role)

    # Fail before expensive input/model work if any source is not immutable yet.
    source_receipts = {}
    for row in rows:
        receipt, checkpoint_sha = committed_receipt(
            routes["sources"][row["run_id"]], row["run_id"]
        )
        source_receipts[row["run_id"]] = (receipt, checkpoint_sha)

    args.output.mkdir(parents=True, exist_ok=False)
    store = TransactionStore(routes["outputs"], args.output / "transactions")
    if not Path("gtsrb_inputs").exists():
        fetch(
            routes["inputs"]["training_url"], Path("gtsrb_training_inputs.tar"),
            routes["inputs"]["training_sha256"],
        )
        materialize(
            Path("gtsrb_training_inputs.tar"), routes["inputs"]["training_sha256"]
        )
    fetch(routes["inputs"]["test_images_url"], Path("GTSRB_Final_Test_Images.zip"),
          IMAGE_ARCHIVE_SHA256)
    fetch(routes["inputs"]["test_gt_url"], Path("GTSRB_Final_Test_GT.zip"),
          GT_ARCHIVE_SHA256)
    data = GTSRBOfficialTest(
        Path("GTSRB_Final_Test_Images.zip"), Path("GTSRB_Final_Test_GT.zip")
    )

    completed = []
    cache = Path("source_checkpoints")
    cache.mkdir(exist_ok=True)
    for row in rows:
        eval_id, run_id = row["eval_id"], row["run_id"]
        existing = _already_committed(routes["outputs"][eval_id], eval_id)
        if existing is not None:
            completed.append({"eval_id": eval_id, "status": "ALREADY_COMMITTED"})
            continue
        source_receipt, checkpoint_sha = source_receipts[run_id]
        archive = cache / f"{run_id}.tar.gz"
        fetch(routes["sources"][run_id]["archive_get"], archive,
              source_receipt["archive_sha256"])
        checkpoint = extract_checkpoint(
            archive, cache / run_id / "final.pt", checkpoint_sha
        )
        dynamic_row = dict(row)
        dynamic_row["checkpoint_sha256"] = checkpoint_sha
        model, architecture = load_final(checkpoint, dynamic_row)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        target = args.output / eval_id
        provenance = {
            "eval_id": eval_id,
            "role": args.role,
            "source_run_id": run_id,
            "training_seed": row["seed_label"],
            "scientific_config": row["config"],
            "checkpoint_sha256": checkpoint_sha,
            "source_archive_sha256": source_receipt["archive_sha256"],
            "source_receipt_object_key": source_receipt.get("object_key"),
            "architecture": architecture,
            "manifest_sha256": manifest["manifest_sha256"],
            "official_test_loaded": True,
        }
        evaluate_official_test(model, data, target, provenance=provenance)
        receipt = store.commit(eval_id, target)
        completed.append({
            "eval_id": eval_id,
            "status": "COMMITTED",
            "receipt_archive_sha256": receipt["archive_sha256"],
        })
        # Do not print or copy sealed metrics into parent-level logs.
        write_json(args.output / "completed.json", completed)
        del model
        torch.cuda.empty_cache()
    write_json(args.output / "parent_summary.json", {
        "status": "COMPLETE",
        "sealed": True,
        "role": args.role,
        "children": completed,
        "finished_at": time.time(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/gtsrb_headline_test"))
    run(parser.parse_args())
