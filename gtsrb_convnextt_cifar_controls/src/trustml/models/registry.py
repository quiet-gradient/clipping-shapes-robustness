from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from trustml.models.resnet18 import build_resnet18
from trustml.storage.artifacts import resolve_artifact_path


def build_model(cfg: DictConfig, root_cfg: DictConfig | None = None):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    weights_path = _model_weight_source(cfg)
    if weights_path is not None:
        resolved_path = resolve_artifact_path(weights_path, root_cfg)
        if resolved_path is not None:
            cfg.weights_path = str(resolved_path)

    name = str(cfg.name).lower()
    if name == "resnet18":
        return build_resnet18(cfg)
    raise ValueError(f"Unknown model: {cfg.name}")


def _model_weight_source(cfg: DictConfig):
    for key in ("weights_path", "checkpoint", "pretrained_checkpoint"):
        value = cfg.get(key, None)
        if value is not None:
            text = str(value).strip()
            if text and text.lower() not in {"none", "null"}:
                return value
    return None
