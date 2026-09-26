"""Load an exported GTSRB evaluation-only weight file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trustml.convnext_cifar100.model import ConvStem, TransferModel


def load_release_weight(path: Path) -> TransferModel:
    import timm

    if timm.__version__ != "0.9.16":
        raise RuntimeError("released architecture requires timm==0.9.16")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "trustml-gtsrb-eval-weights-v1":
        raise ValueError("unsupported release-weight format")
    backbone = timm.create_model(
        "convnext_tiny", pretrained=False, drop_path_rate=0.1, drop_rate=0.0
    )
    backbone.stem = ConvStem()
    backbone.reset_classifier(43)
    model = TransferModel(backbone)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weight", type=Path)
    args = parser.parse_args()
    model = load_release_weight(args.weight)
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 96, 96))
    if logits.shape != (1, 43) or not torch.isfinite(logits).all():
        raise RuntimeError("model forward validation failed")
    print(f"PASS: {args.weight.name}, 43-class FP32 logits")


if __name__ == "__main__":
    main()
