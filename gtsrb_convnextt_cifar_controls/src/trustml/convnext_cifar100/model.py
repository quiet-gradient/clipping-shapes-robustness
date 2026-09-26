"""Singh et al. ConvNeXt-T + ConvStem, with the published CIFAR strides."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

SOURCE_REVISION = "d7166c074223b89e3b7e1ec1489a8d069cc5afb7"
SOURCE_URL = "https://nc.mlcloud.uni-tuebingen.de/index.php/s/FiTToeo4RKY896P"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class ChannelLayerNorm(nn.Module):
    # Same channels-first arithmetic/epsilon as authors' utils_architecture.LayerNorm.
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        return (x - mean) / (variance + 1e-6).sqrt() * self.weight[:, None, None] + self.bias[:, None, None]


class ConvStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1), ChannelLayerNorm(48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), ChannelLayerNorm(96), nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class TransferModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def forward(self, pixels):
        return self.model((pixels - self.mean) / self.std)


def build_model(checkpoint, *, seed=42, expected_sha=None):
    import timm
    if timm.__version__ != "0.9.16":
        raise RuntimeError("locked architecture requires timm==0.9.16")
    digest = sha256(checkpoint)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("source checkpoint SHA-256 mismatch")
    torch.manual_seed(seed)
    backbone = timm.create_model("convnext_tiny", pretrained=False, drop_path_rate=0., drop_rate=0.)
    backbone.stem = ConvStem()
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in source:
        source = source["state_dict"]
    elif "model" in source and isinstance(source["model"], dict):
        source = source["model"]
    cleaned = {}
    normalization = {}
    for key, value in source.items():
        key = key.removeprefix("module.")
        if key.startswith("normalize."):
            normalization[key] = value
            continue
        cleaned[key.removeprefix("model.").removeprefix("base_model.")] = value
    if normalization:
        for name, expected in (("mean", [.485, .456, .406]), ("std", [.229, .224, .225])):
            if not torch.allclose(normalization["normalize." + name].flatten(), torch.tensor(expected)):
                raise ValueError("checkpoint normalization differs from published ImageNet recipe")
    backbone.load_state_dict(cleaned, strict=True)
    exact = sum(torch.equal(value, backbone.state_dict()[key]) for key, value in cleaned.items())
    if exact != len(cleaned):
        raise ValueError("pretrained tensors not loaded exactly")
    original_count = sum(p.numel() for p in backbone.parameters())
    backbone.stem.stem[0].stride = (1, 1)
    backbone.stem.stem[3].stride = (1, 1)
    backbone.stages[1].downsample[1].stride = (1, 1)
    backbone.reset_classifier(100)
    model = TransferModel(backbone)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("model.stages.3.", "model.head.", "model.norm_pre.")))
    model.eval()  # no dropout/stochastic depth/BN; mode fixed for all attack and outer passes
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    frozen = [name for name, module in model.named_modules()
              if list(module.parameters()) and not any(p.requires_grad for p in module.parameters())]
    info = dict(source_url=SOURCE_URL, source_revision=SOURCE_REVISION,
                checkpoint_sha256=digest, source_radius=8/255,
                source_total_params=original_count,
                model_total_params=sum(p.numel() for p in model.parameters()),
                model_trainable_params=sum(trainable.values()), trainable_tensors=trainable,
                frozen_modules=frozen, loaded_exactly=exact, source_tensor_count=len(cleaned),
                fraction_loaded_exactly=exact / len(cleaned),
                replaced_tensors=["head.fc.weight", "head.fc.bias"],
                stride_changes=["stem.stem.0:2->1", "stem.stem.3:2->1", "stages.1.downsample.1:2->1"],
                normalization_count=1, timm_version=timm.__version__)
    return model, info
