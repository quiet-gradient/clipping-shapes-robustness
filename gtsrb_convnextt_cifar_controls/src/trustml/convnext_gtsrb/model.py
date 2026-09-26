"""Clean ConvStem initialization, untouched ImageNet hierarchy, 43 classes."""
from pathlib import Path

import torch

from trustml.convnext_cifar100.model import ConvStem, TransferModel, sha256, SOURCE_REVISION
from .data import digest


def block_name(name):
    for prefix, label in [("model.stages.3.downsample.", "stage3_downsample"),
                          ("model.stages.3.blocks.0.", "stage3_block0"),
                          ("model.stages.3.blocks.1.", "stage3_block1"),
                          ("model.stages.3.blocks.2.", "stage3_block2"),
                          ("model.head.norm.", "head_norm"), ("model.norm_pre.", "head_norm"),
                          ("model.head.fc.", "classifier")]:
        if name.startswith(prefix):
            return label
    raise ValueError(f"unexpected trainable tensor: {name}")


def build_model(checkpoint, metadata, *, seed=42, drop_path_rate=0.):
    import timm
    if timm.__version__ != "0.9.16":
        raise RuntimeError("architecture lock requires timm==0.9.16")
    if metadata["source_kind"] != "clean_imagenet_convstem" or metadata["checkpoint_filename"] != "convnext_tiny_cvst_clean.pt":
        raise ValueError("pilot must not inherit adversarial ImageNet initialization")
    if sha256(checkpoint) != metadata["sha256"]:
        raise ValueError("source checkpoint hash mismatch")
    torch.manual_seed(seed)
    backbone = timm.create_model("convnext_tiny", pretrained=False, drop_path_rate=drop_path_rate, drop_rate=0.)
    backbone.stem = ConvStem()
    source = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if key in source and isinstance(source[key], dict):
            source = source[key]
            break
    cleaned = {}
    for name, value in source.items():
        name = name.removeprefix("module.")
        if name.startswith("normalize."):
            expected = {"normalize.mean": [.485, .456, .406], "normalize.std": [.229, .224, .225]}[name]
            torch.testing.assert_close(value.flatten(), torch.tensor(expected))
            continue
        name = name.removeprefix("model.").removeprefix("base_model.")
        if name in cleaned:
            raise ValueError("duplicate canonical source tensor")
        cleaned[name] = value
    backbone.load_state_dict(cleaned, strict=True)
    if not all(torch.equal(value, backbone.state_dict()[name]) for name, value in cleaned.items()):
        raise ValueError("source tensors not loaded exactly")
    source_count = sum(p.numel() for p in backbone.parameters())
    strides = {name: list(module.stride) for name, module in backbone.named_modules() if isinstance(module, torch.nn.Conv2d)}
    assert strides["stem.stem.0"] == strides["stem.stem.3"] == [2, 2]
    assert all(strides[f"stages.{i}.downsample.1"] == [2, 2] for i in [1, 2, 3])
    backbone.reset_classifier(43)
    for name, value in backbone.state_dict().items():
        if name not in {"head.fc.weight", "head.fc.bias"} and not torch.equal(value, cleaned[name]):
            raise ValueError("non-classifier tensor changed after source loading")
    model = TransferModel(backbone)
    trainable, blocks = {}, {}
    offset = 0
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(("model.stages.3.", "model.head.", "model.norm_pre.")))
        if p.requires_grad:
            label = block_name(name)
            trainable[name] = {"shape": list(p.shape), "numel": p.numel(), "block": label}
            blocks.setdefault(label, []).append([offset, offset + p.numel()])
            offset += p.numel()
    if set(blocks) != {"stage3_downsample", "stage3_block0", "stage3_block1", "stage3_block2", "head_norm", "classifier"}:
        raise ValueError("incomplete trainable block partition")
    model.eval()
    architecture = {"repr": str(backbone), "strides": strides, "resolution": 96, "classes": 43, "trainable": trainable}
    info = {"source_metadata": metadata, "source_architecture_revision": SOURCE_REVISION, "checkpoint_sha256": metadata["sha256"],
            "total_parameter_count": sum(p.numel() for p in model.parameters()), "source_parameter_count": source_count,
            "trainable_parameter_count": offset, "ordered_trainable_tensor_names": list(trainable), "trainable_tensors": trainable,
            "trainable_mask_hash": digest(trainable), "blocks": blocks,
            "frozen_modules": [n for n, m in model.named_modules() if list(m.parameters()) and not any(p.requires_grad for p in m.parameters())],
            "source_tensor_count": len(cleaned), "fraction_loaded_exactly": 1., "newly_initialized": ["head.fc.weight", "head.fc.bias"],
            "model_architecture_digest": digest(architecture), "strides": strides, "stride_changes": [],
            "normalization_count": 1, "timm_version": timm.__version__}
    return model, info
