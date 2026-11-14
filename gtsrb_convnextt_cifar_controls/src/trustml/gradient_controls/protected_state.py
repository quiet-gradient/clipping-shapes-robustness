"""Dataset-independent protected state helpers, extracted without behavior changes."""
import json
import os
from pathlib import Path
import secrets
import numpy as np
import torch
from .io import sha

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
