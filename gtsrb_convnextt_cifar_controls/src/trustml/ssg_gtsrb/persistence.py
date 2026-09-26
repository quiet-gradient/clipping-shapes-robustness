"""Independent immutable stage transactions on existing private Object Storage.

Presigned routes are protected job inputs, never copied into output artifacts.
A COMMITTED receipt is published only after remote bytes have been hashed.
"""
import hashlib
import json
from pathlib import Path
import tarfile
import time
import xml.etree.ElementTree as ET

import requests

from trustml.convnext_gtsrb.model import sha256
from trustml.convnext_cifar100.preflight import write_json


class TransactionStore:
    def __init__(self, routes, local_root):
        self.routes = routes
        self.root = Path(local_root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def request(method, url, **kwargs):
        # Do not leak signed URLs or query credentials via a requests traceback.
        try:
            return requests.request(method, url, timeout=(30, 1800), **kwargs)
        except requests.RequestException:
            raise RuntimeError("artifact transport failed; signed URL suppressed") from None

    def commit(self, key, directory, paths=None):
        if key not in self.routes:
            raise ValueError("no reserved upload route for stage: " + key)
        directory = Path(directory)
        paths = sorted(directory.rglob("*")) if paths is None else [directory / p for p in paths]
        paths = transaction_files(directory, paths)
        hashes = {str(p.relative_to(directory)): sha256(p) for p in paths}
        receipt_path = self.root / (key + ".COMMITTED.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["file_hashes"] != hashes:
                raise ValueError("committed stage changed; never overwrite")
            return receipt
        archive = self.root / (key + ".tar.gz")
        manifest = self.root / (key + ".files.json")
        if archive.exists():
            if not manifest.exists() or json.loads(manifest.read_text()) != hashes:
                raise ValueError("uncommitted archive must be reconciled, not overwritten")
        else:
            write_json(manifest, hashes)
            with tarfile.open(archive, "x:gz") as tar:
                for path in paths:
                    tar.add(path, arcname=str(path.relative_to(directory)), recursive=False)
        archive_hash = sha256(archive)
        route = self.routes[key]
        with self.request("GET", route["receipt_get"]) as response:
            if response.status_code == 200:
                remote = response.json()
                if remote.get("file_hashes") != hashes or remote.get("archive_sha256") != archive_hash:
                    raise RuntimeError("remote committed stage exists; reconcile before any retraining")
                write_json(receipt_path, remote)
                return remote
            if response.status_code != 404:
                raise RuntimeError(f"receipt lookup HTTP {response.status_code}")
        # Never replace an existing weight archive, even if a previous receipt
        # upload failed. Reconcile its actual bytes against our immutable archive.
        exists = False
        with self.request("GET", route["archive_get"], stream=True) as response:
            if response.status_code == 200:
                existing_hash = hashlib.sha256()
                for chunk in response.iter_content(4 * 1024 * 1024):
                    existing_hash.update(chunk)
                if existing_hash.hexdigest() != archive_hash:
                    raise RuntimeError("remote uncommitted archive differs; refusing overwrite")
                exists = True
            elif response.status_code != 404:
                raise RuntimeError(f"artifact existence check HTTP {response.status_code}")
        if not exists:
            with archive.open("rb") as stream:
                with self.request("PUT", route["archive_put"], data=stream) as response:
                    if response.status_code not in [200, 201]:
                        try:
                            code = ET.fromstring(response.text).findtext("Code", "unknown")
                        except (ET.ParseError, AttributeError):
                            code = "non_xml_response"
                        # Never print the full XML: it may echo signed credentials.
                        raise RuntimeError(f"artifact upload HTTP {response.status_code}; code={code}")
        remote_hash, count = hashlib.sha256(), 0
        with self.request("GET", route["archive_get"], stream=True) as response:
            if response.status_code != 200:
                raise RuntimeError(f"artifact readback HTTP {response.status_code}")
            for chunk in response.iter_content(4 * 1024 * 1024):
                remote_hash.update(chunk)
                count += len(chunk)
        if remote_hash.hexdigest() != archive_hash or count != archive.stat().st_size:
            raise RuntimeError("remote artifact bytes failed SHA-256/size verification")
        receipt = {"status": "COMMITTED", "stage": key, "archive_sha256": archive_hash,
                   "archive_bytes": count, "file_hashes": hashes, "object_key": route["archive_key"],
                   "committed_at": time.time()}
        with self.request("PUT", route["receipt_put"], data=json.dumps(receipt).encode()) as response:
            if response.status_code not in [200, 201]:
                raise RuntimeError(f"commit publication HTTP {response.status_code}")
        with self.request("GET", route["receipt_get"]) as response:
            if response.status_code != 200 or response.json() != receipt:
                raise RuntimeError("COMMITTED receipt readback mismatch")
        write_json(receipt_path, receipt)
        print(json.dumps({"phase": "artifact_committed", "stage": key, "archive_sha256": archive_hash}), flush=True)
        return receipt


def transaction_files(directory, candidates):
    """Archive actual run files, never dereference W&B's external core log.

    W&B0.28 links offline-run-*/logs/debug-core.log to its process cache.
    That diagnostic alias is not a scientific artifact. All other external
    links (including dangling links and directory links) remain forbidden.
    """
    directory = Path(directory)
    root = directory.resolve()
    files = []
    for path in candidates:
        path = Path(path)
        relative = path.relative_to(directory)
        parts = relative.parts
        if path.is_symlink():
            core_log = (len(parts) == 4 and parts[0] == "wandb"
                        and parts[1].startswith("offline-run-")
                        and parts[2:] == ("logs", "debug-core.log"))
            internal_alias = (len(parts) == 2 and parts[0] == "wandb"
                              and parts[1] in {"debug.log", "debug-internal.log", "latest-run"}
                              and path.resolve().is_relative_to(root))
            if core_log or internal_alias:
                continue
            raise ValueError("unsafe transaction symlink: " + str(relative))
        if not path.resolve().is_relative_to(root):
            raise ValueError("unsafe transaction path: " + str(relative))
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError("empty transaction inputs")
    return files


def transaction_keys():
    """Pre-reserve finite stage routes without inventing calibrated C/anchors."""
    keys = [f"public_epoch{e:03d}" for e in range(10, 101, 10)]
    keys += [f"public_aa{e:03d}" for e in range(40, 101, 10)]
    keys += ["public_selected", "real_transfer", "calibration", "sgd_reports", "adamw_reports", "campaign_reports"]
    for optimizer in ["sgd", "adamw"]:
        for role in ["cleandp", "joint"]:
            keys += [f"{optimizer}_{role}_{c}_lr{i}_s42" for c in ["low", "mid"] for i in range(3)]
        # Reserve all possible parent anchors; only <=8 componentwise children run.
        keys += [f"{optimizer}_componentwise_{c}_lr{i}_s42_{q}"
                 for c in ["low", "mid"] for i in range(3) for q in ["q010", "q025", "q1of3", "q050"]]
    return keys
