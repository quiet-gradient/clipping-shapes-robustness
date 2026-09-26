"""Evaluate the frozen public-pretrain checkpoint on the official GTSRB test.

This is an evaluation-only reference row.  It performs no private fine-tuning,
does not participate in model selection, and uses the same official-test
implementation as the frozen headline models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
import torch

from scripts.platform.gtsrb_inputs import materialize
from trustml.convnext_cifar100.preflight import configure_gpu, write_json
from trustml.convnext_gtsrb.model import sha256
from .official_test import (
    GT_ARCHIVE_SHA256,
    IMAGE_ARCHIVE_SHA256,
    GTSRBOfficialTest,
    evaluate_official_test,
)
from .persistence import TransactionStore
from .private import load_selected
from .rectangular import SOURCE_SHA, fetch


EVAL_ID = "gtsrb_public_pretrain_reference"


def _committed_source(route):
    with TransactionStore.request("GET", route["receipt_get"]) as response:
        if response.status_code != 200:
            raise RuntimeError(f"public-pretrain receipt lookup HTTP {response.status_code}")
        receipt = response.json()
    if receipt.get("status") != "COMMITTED" or receipt.get("stage") != "public_selected":
        raise RuntimeError("invalid public-pretrain COMMITTED receipt")
    if receipt.get("file_hashes", {}).get("public_pretrain_selected.pt") != SOURCE_SHA:
        raise RuntimeError("public-pretrain checkpoint hash differs from frozen selection")
    archive_sha = receipt.get("archive_sha256")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise RuntimeError("public-pretrain receipt lacks archive SHA-256")
    return receipt


def _extract_selected(archive: Path, target: Path):
    with tarfile.open(archive) as source:
        member = source.getmember("public_pretrain_selected.pt")
        if not member.isfile() or Path(member.name).name != member.name:
            raise RuntimeError("unsafe public-pretrain archive member")
        with source.extractfile(member) as handle, target.open("xb") as output:
            shutil.copyfileobj(handle, output)
    if sha256(target) != SOURCE_SHA:
        raise RuntimeError("extracted public-pretrain checkpoint SHA-256 mismatch")


def _already_committed(route):
    with TransactionStore.request("GET", route["receipt_get"]) as response:
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise RuntimeError(f"output receipt lookup HTTP {response.status_code}")
        receipt = response.json()
    if receipt.get("status") != "COMMITTED" or receipt.get("stage") != EVAL_ID:
        raise RuntimeError("invalid existing public-pretrain evaluation receipt")
    return receipt


def run(args):
    configure_gpu()
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("public-pretrain official test requires exactly one A100")
    manifest = json.loads(args.manifest.read_text())
    routes = json.loads(args.routes.read_text())
    if manifest.get("eval_id") != EVAL_ID or manifest.get("evaluation_only") is not True:
        raise RuntimeError("invalid public-pretrain reference manifest")
    if manifest.get("optimizer_steps_added") != 0 or manifest.get("private_finetuning") is not False:
        raise RuntimeError("reference row must not train or privately fine-tune")
    if manifest.get("manifest_sha256") != routes.get("manifest_sha256"):
        raise RuntimeError("public-pretrain manifest/route identity mismatch")
    if set(routes) != {"manifest_sha256", "inputs", "source", "output"}:
        raise RuntimeError("unexpected protected route surface")

    source_receipt = _committed_source(routes["source"])
    existing = _already_committed(routes["output"])
    args.output.mkdir(parents=True, exist_ok=False)
    if existing is not None:
        write_json(args.output / "parent_summary.json", {
            "status": "ALREADY_COMMITTED",
            "eval_id": EVAL_ID,
            "finished_at": time.time(),
        })
        return

    if not Path("gtsrb_inputs").exists():
        fetch(
            routes["inputs"]["training_url"], Path("gtsrb_training_inputs.tar"),
            routes["inputs"]["training_sha256"],
        )
        materialize(
            Path("gtsrb_training_inputs.tar"), routes["inputs"]["training_sha256"]
        )
    source_archive = Path("public_selected.tar.gz")
    fetch(routes["source"]["archive_get"], source_archive, source_receipt["archive_sha256"])
    selected = Path("public_pretrain_selected.pt")
    _extract_selected(source_archive, selected)

    fetch(
        routes["inputs"]["test_images_url"], Path("GTSRB_Final_Test_Images.zip"),
        IMAGE_ARCHIVE_SHA256,
    )
    fetch(
        routes["inputs"]["test_gt_url"], Path("GTSRB_Final_Test_GT.zip"),
        GT_ARCHIVE_SHA256,
    )
    data = GTSRBOfficialTest(
        Path("GTSRB_Final_Test_Images.zip"), Path("GTSRB_Final_Test_GT.zip")
    )
    model, architecture = load_selected(Path("gtsrb_inputs"), selected, SOURCE_SHA)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    target = args.output / EVAL_ID
    provenance = {
        "eval_id": EVAL_ID,
        "role": "public_pretrain_reference",
        "source_stage": "public_selected",
        "source_checkpoint_sha256": SOURCE_SHA,
        "source_archive_sha256": source_receipt["archive_sha256"],
        "source_receipt_object_key": source_receipt.get("object_key"),
        "architecture": architecture,
        "manifest_sha256": manifest["manifest_sha256"],
        "official_test_loaded": True,
        "evaluation_only": True,
        "private_finetuning": False,
        "optimizer_steps_added": 0,
        "selection_role": "NONE_REFERENCE_ONLY",
    }
    evaluate_official_test(model, data, target, provenance=provenance)
    store = TransactionStore({EVAL_ID: routes["output"]}, args.output / "transactions")
    receipt = store.commit(EVAL_ID, target)
    write_json(args.output / "parent_summary.json", {
        "status": "COMPLETE",
        "eval_id": EVAL_ID,
        "receipt_archive_sha256": receipt["archive_sha256"],
        "finished_at": time.time(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gtsrb_public_pretrain_test")
    )
    run(parser.parse_args())
