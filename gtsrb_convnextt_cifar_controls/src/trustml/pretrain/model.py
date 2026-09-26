from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from trustml.models.resnet18 import ResNet18Model
from trustml.models.wrappers import NormalizedModel


def build_cifar100_model(cfg, initialization_path: str | Path | None = None) -> NormalizedModel:
    # The run seed controls stochastic training streams; all Stage-1 and
    # Stage-2 models intentionally load the same initial state.
    seed = int(cfg.reproducibility.stream_seeds.initialization)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = ResNet18Model(num_classes=100)
        replace_batchnorm_with_groupnorm(backbone, int(cfg.model.group_norm_groups))
        model = NormalizedModel(backbone, mean=cfg.data.mean, std=cfg.data.std)
    assert_no_batchnorm(model)
    if initialization_path is not None:
        payload = torch.load(initialization_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"invalid initialization artifact: {initialization_path}")
        model.load_state_dict(payload["model"], strict=True)
        actual_hash = state_dict_sha256(model.state_dict())
        expected_hash = str(payload.get("initialization_hash", ""))
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"initialization content hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
    return model


def replace_batchnorm_with_groupnorm(module: nn.Module, groups: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if child.num_features % groups:
                raise ValueError(
                    f"GroupNorm groups={groups} does not divide channels={child.num_features} at {name}"
                )
            replacement = nn.GroupNorm(groups, child.num_features, affine=True)
            with torch.no_grad():
                if child.affine:
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batchnorm_with_groupnorm(child, groups)


def assert_no_batchnorm(model: nn.Module) -> None:
    batchnorm = [name for name, child in model.named_modules() if isinstance(child, nn.modules.batchnorm._BatchNorm)]
    if batchnorm:
        raise AssertionError(f"BatchNorm remains in public pretrain model: {batchnorm}")


def group_norm_inventory(model: nn.Module) -> dict[str, Any]:
    modules = [(name, child) for name, child in model.named_modules() if isinstance(child, nn.GroupNorm)]
    return {
        "module_count": len(modules),
        "groups": sorted({int(child.num_groups) for _, child in modules}),
        "channels": sorted({int(child.num_channels) for _, child in modules}),
    }


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def ensure_initialization_asset(cfg) -> tuple[Path, str]:
    path = Path(str(cfg.reproducibility.initialization_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"invalid initialization artifact: {path}")
        digest = state_dict_sha256(payload["model"])
        if digest != payload.get("initialization_hash"):
            raise RuntimeError(f"initialization artifact hash mismatch: {path}")
        if int(payload.get("seed", -1)) != int(cfg.reproducibility.stream_seeds.initialization):
            raise RuntimeError(f"initialization artifact seed mismatch: {path}")
        return path, digest

    model = build_cifar100_model(cfg)
    digest = state_dict_sha256(model.state_dict())
    payload = {
        "format_version": 1,
        "architecture": "cifar_resnet18",
        "normalization": "group_norm",
        "group_norm_groups": int(cfg.model.group_norm_groups),
        "seed": int(cfg.reproducibility.stream_seeds.initialization),
        "initialization_hash": digest,
        "model": model.state_dict(),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    return path, digest
