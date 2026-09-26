from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import warnings

import torch

from trustml.core.config import load_config
from omegaconf import OmegaConf

from trustml.core.model_loading import load_checkpoint_model, match_dp_architecture
from trustml.models.resnet18 import build_resnet18
from trustml.training.optimizers import apply_adaptation


ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_sanitized_checkpoint_and_model_contract(self) -> None:
        checkpoint = ROOT / "artifacts/public_pretrain/checkpoint_epoch_170.pt"
        self.assertEqual(
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "53f358fe8a0e7b87e0dbbf8b0814d885e6bb82ba5452baa22521a58bf2d68909",
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(set(payload), {"format_version", "model", "epoch", "source"})
        self.assertEqual(len(payload["model"]), 64)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in payload["model"].values()))

        cfg = load_config(ROOT / "configs/runs/ha_b1_c030_lr0050_s42.yaml")
        cfg.model.weights_path = str(checkpoint)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build_resnet18(cfg.model)
        model = match_dp_architecture(model, cfg)
        apply_adaptation(model, cfg)
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        self.assertEqual(trainable, 8_398_858)
        self.assertFalse(
            any(
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                for module in model.modules()
            )
        )
        self.assertTrue(any(isinstance(module, torch.nn.GroupNorm) for module in model.modules()))
        self.assertEqual(model.classifier.out_features, 10)

    def test_checkpoint_identity_is_checked_before_config_merge(self) -> None:
        cfg = load_config(ROOT / "configs/runs/ha_b1_c030_lr0050_s42.yaml")
        embedded = OmegaConf.to_container(cfg, resolve=True)
        embedded["experiment"]["name"] = "different_run"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            torch.save({"model": {}, "config": embedded}, path)
            with self.assertRaisesRegex(RuntimeError, "checkpoint run ID"):
                load_checkpoint_model(str(path), cfg_override=cfg)


if __name__ == "__main__":
    unittest.main()
