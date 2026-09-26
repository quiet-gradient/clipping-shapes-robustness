from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf

from trustml.pretraining.assets import (
    INITIALIZATION_FILENAME,
    prepare_pretraining_assets,
    write_asset_receipt,
)
from trustml.pretraining.config import load_pretraining_config, scientific_config_hash
from trustml.pretraining.data import build_pretraining_data
from trustml.pretraining.model import build_cifar100_model
from trustml.pretraining.reproducibility import configure_determinism, make_generator
from trustml.pretraining.trainer import PretrainingTrainer


def run_pretraining(
    config_path: str | Path,
    *,
    data_root: str | Path,
    asset_root: str | Path,
    output_root: str | Path,
    num_workers: int = 4,
    download: bool = True,
    device_name: str = "auto",
    resume: str | None = None,
    smoke_epochs: int | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> dict:
    cfg = load_pretraining_config(config_path)
    configure_determinism(int(cfg.experiment.seed))
    assets = prepare_pretraining_assets(
        cfg, data_root=data_root, asset_root=asset_root, download=download
    )
    write_asset_receipt(assets, asset_root)
    device = _select_device(device_name)
    dataloader_generator = make_generator(int(cfg.reproducibility.dataloader_seed), "cpu")
    trades_generator = make_generator(int(cfg.reproducibility.trades_seed), device)
    pgd_generator = make_generator(int(cfg.reproducibility.pgd_seed), device)
    data = build_pretraining_data(
        cfg,
        data_root=data_root,
        asset_dir=asset_root,
        dataloader_generator=dataloader_generator,
        num_workers=num_workers,
        download=False,
    )
    initialization_path = Path(asset_root) / INITIALIZATION_FILENAME
    model = build_cifar100_model(cfg, initialization_path=initialization_path)
    initialization_hash = str(assets["initialization"]["state_sha256"])
    output_dir = Path(output_root) / str(cfg.experiment.run_id)
    resume_path = _resolve_resume(resume, output_dir)
    if output_dir.exists() and resume_path is None and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is non-empty; choose another root or pass --resume auto: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    config_hash = scientific_config_hash(cfg)
    _write_environment(output_dir, device)
    trainer = PretrainingTrainer(
        cfg=cfg,
        model=model,
        data=data,
        device=device,
        output_dir=output_dir,
        config_hash=config_hash,
        initialization_hash=initialization_hash,
        dataloader_generator=dataloader_generator,
        trades_generator=trades_generator,
        pgd_generator=pgd_generator,
        smoke_epochs=smoke_epochs,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
    )
    if resume_path is not None:
        trainer.load_checkpoint(resume_path)
    return trainer.fit()


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run one final public CIFAR-100 pretraining configuration.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--asset-root", type=Path, default=Path("data/pretraining_assets"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pretraining"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'auto'.")
    parser.add_argument("--smoke-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    args = parser.parse_args(argv)
    if args.smoke_epochs is not None and args.smoke_epochs < 1:
        parser.error("--smoke-epochs must be positive")
    result = run_pretraining(
        args.config,
        data_root=args.data_root,
        asset_root=args.asset_root,
        output_root=args.output_root,
        num_workers=args.num_workers,
        download=args.download,
        device_name=args.device,
        resume=args.resume,
        smoke_epochs=args.smoke_epochs,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _resolve_resume(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    path = output_dir / "checkpoints/checkpoint_last.pt" if value == "auto" else Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing: {path}")
    return path


def _write_environment(output_dir: Path, device: torch.device) -> None:
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device_type": device.type,
        "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else "none",
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
