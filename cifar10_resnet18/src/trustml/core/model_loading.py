"""Rebuild a model from a reproduction checkpoint for standalone evaluation.

Checkpoints embed their training config, which is used to rebuild the exact
architecture, including the DP BatchNorm-to-GroupNorm conversion.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from trustml.models.resnet18 import build_resnet18


def match_dp_architecture(model: torch.nn.Module, cfg) -> torch.nn.Module:
    """Reapply the DP module surgery from DPSGDDefense.setup so keys line up."""
    defenses = cfg.get("defenses", []) or []
    dp = next((d for d in defenses if str(d.get("name", "")).lower() == "dp_sgd" and d.get("enabled", True)), None)
    if dp is None:
        return model
    from opacus.validators import ModuleValidator

    from trustml.defenses.dp_sgd import _disable_inplace_relu, _replace_batchnorm_with_groupnorm

    _disable_inplace_relu(model)
    if bool(dp.get("convert_batchnorm", True)):
        _replace_batchnorm_with_groupnorm(model)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        _disable_inplace_relu(model)
    return model


def load_checkpoint_model(checkpoint_path: str, cfg_override=None):
    """Load (model, cfg) from a checkpoint, rebuilding the trained architecture.

    ``cfg_override`` (a DictConfig) is merged over the embedded config; it is
    also the fallback when the checkpoint has no embedded config at all.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint.get("config")
    if saved_cfg is None and cfg_override is None:
        raise ValueError("Checkpoint has no embedded config; pass --config explicitly.")
    if saved_cfg is not None and cfg_override is not None:
        saved_run_id = str(saved_cfg["experiment"]["name"])
        requested_run_id = str(cfg_override.experiment.name)
        if saved_run_id != requested_run_id:
            raise RuntimeError(
                f"checkpoint run ID {saved_run_id!r} != requested {requested_run_id!r}"
            )
        saved_seed = int(saved_cfg["experiment"]["seed"])
        requested_seed = int(cfg_override.experiment.seed)
        if saved_seed != requested_seed:
            raise RuntimeError(
                f"checkpoint seed {saved_seed} != requested seed {requested_seed}"
            )
    cfg = OmegaConf.create(saved_cfg) if saved_cfg is not None else cfg_override
    if cfg_override is not None and saved_cfg is not None:
        cfg = OmegaConf.merge(cfg, cfg_override)

    # Build the bare architecture (no weight loading here -- the training-time
    # weights_path may not exist on this machine, and we overwrite it anyway).
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.weights_path = None
    model = build_resnet18(model_cfg)
    model = match_dp_architecture(model, cfg)

    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # After DP surgery the only acceptable gaps are Opacus bookkeeping buffers;
    # a real architecture mismatch shows up as missing conv/fc weights.
    real_missing = [k for k in missing if k.endswith((".weight", ".bias")) and "num_batches_tracked" not in k]
    if real_missing:
        raise RuntimeError(f"checkpoint load left real params uninitialised: {real_missing[:6]} ...")
    return model, cfg
