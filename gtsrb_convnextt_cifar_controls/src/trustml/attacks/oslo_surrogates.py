from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "oslo_mia" / "weights"


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        sc = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + sc


class PreActResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.bn = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(self.in_ch, out_ch, stride))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def build_surrogate_model(name: str, num_classes: int = 10) -> nn.Module:
    normalized = name.lower()
    if normalized == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if normalized == "inception":
        model = models.inception_v3(weights=None, aux_logits=False, init_weights=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "vgg13":
        model = models.vgg13_bn(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if normalized == "resnext50":
        model = models.resnext50_32x4d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if normalized == "preactresnet18":
        return PreActResNet18(num_classes)
    raise ValueError(f"Unknown OSLO surrogate model: {name}")


VALIDATION_FAMILIES = ("densenet121", "vgg13", "shufflenet")


def available_validation_specs(weights_dir: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """Validation specs for the weight files actually present, plus missing names.

    The OSLO release ships three validation families x three seeds, but this
    repo has never had the vgg13 weights; a hard requirement on the full grid
    would make the attack permanently unrunnable. The transfer-confidence
    early-stopping only needs *an* ensemble, so use every file that exists and
    report the absentees for the caller's log/metrics.
    """
    present: list[tuple[str, Path]] = []
    missing: list[str] = []
    for family in VALIDATION_FAMILIES:
        for i in range(3):
            path = weights_dir / f"val_{family}_{i}.pth"
            if path.exists():
                present.append((family, path))
            else:
                missing.append(path.name)
    return present, missing


def load_source_and_validation_models(
    device: str | torch.device = "cuda",
    weights_dir: str | os.PathLike = DEFAULT_WEIGHTS_DIR,
) -> tuple[list[tuple[str, nn.Module]], list[tuple[str, nn.Module]]]:
    weights_dir = Path(weights_dir)
    if str(device) == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    source_specs = [
        ("inception", weights_dir / "source_inception.pth"),
        ("resnext50", weights_dir / "source_resnext50.pth"),
        ("preactresnet18", weights_dir / "source_preactresnet18.pth"),
    ]
    missing_sources = [str(path) for _, path in source_specs if not path.exists()]
    if missing_sources:
        raise FileNotFoundError("Missing OSLO source-model weights:\n" + "\n".join(missing_sources))

    validation_specs, missing_validation = available_validation_specs(weights_dir)
    if not validation_specs:
        raise FileNotFoundError(f"No OSLO validation-model weights found in {weights_dir}.")
    if missing_validation:
        print(
            f"[oslo] validation ensemble uses {len(validation_specs)}/9 models; "
            f"missing weights: {', '.join(missing_validation)}"
        )

    def load_one(name: str, path: Path) -> tuple[str, nn.Module]:
        model = build_surrogate_model(name, num_classes=10)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return name, model

    return [load_one(*spec) for spec in source_specs], [load_one(*spec) for spec in validation_specs]
