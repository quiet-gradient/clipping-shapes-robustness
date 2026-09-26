from __future__ import annotations

from pathlib import Path
from types import MethodType
import warnings

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
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
        spectral_norm_enabled: bool = False,
        sn_power_iterations: int = 1,
        sn_coeff: float = 1.0,
    ) -> None:
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        _patch_residual_adds(base)
        self.model = base
        self._sn_applied = False
        self.sn_power_iterations = int(sn_power_iterations)
        self.sn_coeff = float(sn_coeff)

        if weights_path:
            self.load_weights(
                weights_path,
                strict=strict_load,
                skip_mismatched=skip_mismatched,
                reinit_classifier_on_mismatch=reinit_classifier_on_mismatch,
            )

        # Legacy order: weights first, then SN wrapping (plain checkpoints
        # then fit the pre-SN parameter names).
        if spectral_norm_enabled:
            self.apply_spectral_norm(self.sn_power_iterations, self.sn_coeff)

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
        state_dict = strip_spectral_norm_keys(state_dict)
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

    def apply_spectral_norm(self, n_power_iterations: int = 1, coeff: float = 1.0) -> None:
        """Wrap every Conv2d/Linear in spectral norm, as in the legacy models.

        coeff: target spectral norm (1.0 = standard SN, <1.0 = tighter
        Lipschitz constraint applied via a forward pre-hook).
        """
        if self._sn_applied:
            return

        count = 0
        modules = dict(self.model.named_modules())
        for name, module in list(modules.items()):
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent, attr_name = modules[parts[0]], parts[1]
            else:
                parent, attr_name = self.model, name
            try:
                sn_module = spectral_norm(module, n_power_iterations=n_power_iterations)
                if coeff != 1.0:
                    def _scale_hook(mod, inputs, _c=float(coeff)):
                        with torch.no_grad():
                            mod.weight.data.mul_(_c)

                    sn_module.register_forward_pre_hook(_scale_hook)
                setattr(parent, attr_name, sn_module)
                count += 1
            except Exception as exc:  # noqa: BLE001 - mirror legacy best-effort behavior.
                warnings.warn(f"Spectral norm failed for {name}: {exc}")

        self._sn_applied = True
        self.sn_power_iterations = int(n_power_iterations)
        self.sn_coeff = float(coeff)


def strip_spectral_norm_keys(state_dict: dict) -> dict:
    if not any(".weight_orig" in key for key in state_dict):
        return state_dict

    stripped = {}
    for key, value in state_dict.items():
        if ".weight_orig" in key:
            stripped[key.replace(".weight_orig", ".weight")] = value
        elif ".weight_u" in key or ".weight_v" in key:
            continue
        else:
            stripped[key] = value
    return stripped


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
    sn_cfg = cfg.get("spectral_norm", None) or {}
    return ResNet18Model(
        num_classes=int(cfg.get("num_classes", 10)),
        in_channels=int(cfg.get("in_channels", 3)),
        weights_path=cfg.get("weights_path", None),
        strict_load=bool(cfg.get("strict_load", False)),
        skip_mismatched=bool(cfg.get("skip_mismatched", True)),
        reinit_classifier_on_mismatch=bool(cfg.get("reinit_classifier_on_mismatch", True)),
        spectral_norm_enabled=bool(sn_cfg.get("enabled", False)),
        sn_power_iterations=int(sn_cfg.get("power_iterations", 1)),
        sn_coeff=float(sn_cfg.get("coeff", 1.0)),
    )
