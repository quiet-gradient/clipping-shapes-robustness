from __future__ import annotations

from pathlib import Path
from types import MethodType
import warnings

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.models.resnet as tv_resnet


class ResNet18Model(nn.Module):
    """CIFAR-style ResNet-18: 3x3 first conv, stride 1, no maxpool."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        weights_path: str | Path | None = None,
        strict_load: bool = False,
        skip_mismatched: bool = True,
        reinit_classifier_on_mismatch: bool = True,
    ) -> None:
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        _patch_residual_adds(base)
        self.model = base

        if weights_path:
            self.load_weights(
                weights_path,
                strict=strict_load,
                skip_mismatched=skip_mismatched,
                reinit_classifier_on_mismatch=reinit_classifier_on_mismatch,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def classifier(self) -> nn.Module:
        return self.model.fc

    @property
    def last_block(self) -> nn.Module:
        return self.model.layer4

    def load_weights(
        self,
        weights_path: str | Path,
        strict: bool = False,
        skip_mismatched: bool = True,
        reinit_classifier_on_mismatch: bool = True,
    ) -> None:
        # Project checkpoints contain provenance/RNG metadata in addition to
        # tensors, so PyTorch >=2.6 cannot use its new weights_only default.
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = align_state_dict_keys(state_dict, self.state_dict())
        if strict or not skip_mismatched:
            self.load_state_dict(state_dict, strict=strict)
            return

        model_state = self.state_dict()
        compatible_state = dict(model_state)
        loaded_keys = []
        skipped_keys = []
        classifier_mismatch = False
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                compatible_state[key] = value
                loaded_keys.append(key)
            elif key in model_state:
                skipped_keys.append(
                    f"{key} (checkpoint={tuple(value.shape)}, model={tuple(model_state[key].shape)})"
                )
                classifier_mismatch = classifier_mismatch or key.startswith("model.fc.")
            else:
                skipped_keys.append(f"{key} (not in model)")

        self.load_state_dict(compatible_state, strict=False)
        if not loaded_keys or "model.conv1.weight" not in loaded_keys:
            raise RuntimeError(
                f"Checkpoint {weights_path} did not load the ResNet backbone "
                f"(compatible keys={len(loaded_keys)})."
            )
        if classifier_mismatch and reinit_classifier_on_mismatch:
            self.reinit_classifier()
        if skipped_keys:
            warnings.warn(
                f"Loaded {len(loaded_keys)} compatible keys from {weights_path}; "
                f"skipped {len(skipped_keys)} incompatible keys."
            )

    def reinit_classifier(self) -> None:
        nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.01)
        if self.model.fc.bias is not None:
            nn.init.zeros_(self.model.fc.bias)


def align_state_dict_keys(state_dict: dict, model_state: dict) -> dict:
    """Align bare, wrapped, and normalized-wrapper ResNet checkpoints.

    ``NormalizedModel(ResNet18Model)`` saves backbone keys as
    ``model.model.*`` plus the wrapper's ``mean``/``std`` buffers, while the
    downstream ``ResNet18Model`` expects ``model.*``. Choose the candidate
    with the largest overlap instead of silently accepting a zero-key load.
    """
    candidates = [
        state_dict,
        {f"model.{key}": value for key, value in state_dict.items()},
        {
            key.removeprefix("model.") if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        },
    ]
    return max(candidates, key=lambda candidate: sum(key in model_state for key in candidate))


def _basic_block_forward_no_inplace(self, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = self.relu(out)
    return out


def _patch_residual_adds(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, tv_resnet.BasicBlock):
            child.forward = MethodType(_basic_block_forward_no_inplace, child)


def build_resnet18(cfg) -> ResNet18Model:
    return ResNet18Model(
        num_classes=int(cfg.get("num_classes", 10)),
        in_channels=int(cfg.get("in_channels", 3)),
        weights_path=cfg.get("weights_path", None),
        strict_load=bool(cfg.get("strict_load", False)),
        skip_mismatched=bool(cfg.get("skip_mismatched", True)),
        reinit_classifier_on_mismatch=bool(cfg.get("reinit_classifier_on_mismatch", True)),
    )
