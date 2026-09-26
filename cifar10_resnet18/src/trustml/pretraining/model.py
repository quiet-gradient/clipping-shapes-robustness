from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from trustml.models.resnet18 import ResNet18Model


class PixelNormalizedModel(nn.Module):
    """A classifier with exactly one model-owned pixel normalization."""

    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model((pixels - self.mean) / self.std)


def build_cifar100_model(cfg, initialization_path: str | Path | None = None) -> PixelNormalizedModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg.reproducibility.initialization_seed))
        backbone = ResNet18Model(num_classes=100)
        replace_batchnorm_with_groupnorm(backbone, int(cfg.model.group_norm_groups))
        model = PixelNormalizedModel(backbone, cfg.data.mean, cfg.data.std)
    assert_model_contract(model, int(cfg.model.group_norm_groups))
    if initialization_path is not None:
        payload = torch.load(initialization_path, map_location="cpu", weights_only=False)
        if set(payload) != {"format_version", "seed", "state_sha256", "model"}:
            raise RuntimeError("shared initialization has an unexpected schema")
        model.load_state_dict(payload["model"], strict=True)
        observed = state_dict_sha256(model.state_dict())
        if observed != payload["state_sha256"]:
            raise RuntimeError("shared initialization content hash mismatch")
        if int(payload["seed"]) != int(cfg.reproducibility.initialization_seed):
            raise RuntimeError("shared initialization seed mismatch")
    return model


def ensure_initialization_asset(cfg, path: str | Path) -> tuple[Path, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        model = build_cifar100_model(cfg, initialization_path=path)
        return path, state_dict_sha256(model.state_dict())
    model = build_cifar100_model(cfg)
    digest = state_dict_sha256(model.state_dict())
    payload = {
        "format_version": 1,
        "seed": int(cfg.reproducibility.initialization_seed),
        "state_sha256": digest,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    _atomic_torch_save(payload, path)
    return path, digest


def replace_batchnorm_with_groupnorm(module: nn.Module, groups: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if int(child.num_features) % int(groups):
                raise ValueError("GroupNorm group count must divide every channel count")
            replacement = nn.GroupNorm(int(groups), int(child.num_features), affine=True)
            with torch.no_grad():
                if child.affine:
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batchnorm_with_groupnorm(child, groups)


def assert_model_contract(model: nn.Module, groups: int) -> None:
    batch_norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    group_norms = [module for module in model.modules() if isinstance(module, nn.GroupNorm)]
    normalizers = [module for module in model.modules() if isinstance(module, PixelNormalizedModel)]
    if batch_norms:
        raise AssertionError("BatchNorm remains in the public-pretraining model")
    if not group_norms or {module.num_groups for module in group_norms} != {int(groups)}:
        raise AssertionError("the GroupNorm inventory is invalid")
    if len(normalizers) != 1:
        raise AssertionError("the model must own exactly one normalization wrapper")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("public pretraining must update the complete model")


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

