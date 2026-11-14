"""Load a released CIFAR model accepting unnormalized [0,1] images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from trustml.defenses.dp_sgd import _disable_inplace_relu, _replace_batchnorm_with_groupnorm
from trustml.models.resnet18 import ResNet18Model


class PixelModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, images):
        return self.model((images - self.mean) / self.std)


def load_release_weight(path: Path) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "trustml-cifar-eval-weights-v1":
        raise ValueError("unsupported release-weight format")
    weights = payload["model"]
    if payload["public_pretrain"]:
        weights = {name.removeprefix("model."): value for name, value in weights.items()
                   if name not in ("mean", "std")}
    model = ResNet18Model(num_classes=int(payload["num_classes"]))
    _disable_inplace_relu(model)
    _replace_batchnorm_with_groupnorm(model)
    model.load_state_dict(weights, strict=True)
    return PixelModel(model, payload["input_mean"], payload["input_std"]).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weight", type=Path)
    args = parser.parse_args()
    model = load_release_weight(args.weight)
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 32, 32))
    if logits.shape[0] != 1 or logits.shape[1] not in (10, 100) or not torch.isfinite(logits).all():
        raise RuntimeError("model forward validation failed")
    print(f"PASS: {args.weight.name}, {logits.shape[1]}-class FP32 logits")


if __name__ == "__main__":
    main()
