from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import open_dict

from trustml.pretrain.config import load_pretrain_config, parse_args
from trustml.pretrain.data import build_pretrain_data, ensure_split_assets
from trustml.pretrain.logging import (
    PretrainLogger,
    platform_metadata,
    save_environment,
    save_resolved_config,
)
from trustml.pretrain.model import (
    assert_no_batchnorm,
    build_cifar100_model,
    ensure_initialization_asset,
    group_norm_inventory,
)
from trustml.pretrain.reproducibility import configure_determinism, make_generator
from trustml.pretrain.trainer import PretrainTrainer


def run_pretraining(cfg, resume: str | None = None, prepare_assets_only: bool = False):
    configure_determinism(int(cfg.experiment.seed))
    initialization_path, initialization_hash = ensure_initialization_asset(cfg)
    split_npz, split_json, split_hash = ensure_split_assets(cfg)
    split_metadata = json.loads(split_json.read_text(encoding="utf-8"))
    archive_metadata = split_metadata.get("dataset_archive", {})
    with open_dict(cfg):
        cfg.provenance.initialization_hash = initialization_hash
        cfg.provenance.split_hash = split_hash
        cfg.provenance.initialization_path = initialization_path.as_posix()
        cfg.provenance.split_npz = split_npz.as_posix()
        cfg.provenance.split_metadata = split_json.as_posix()
        cfg.provenance.dataset_archive_md5 = archive_metadata.get("md5", "unknown")
        cfg.provenance.dataset_archive_sha256 = archive_metadata.get("sha256", "unknown")
    if prepare_assets_only:
        return {
            "initialization_path": str(initialization_path),
            "initialization_hash": initialization_hash,
            "split_npz": str(split_npz),
            "split_metadata": str(split_json),
            "split_hash": split_hash,
        }

    device = _select_device(cfg)
    streams = cfg.reproducibility.stream_seeds
    dataloader_generator = make_generator(int(streams.dataloader), "cpu")
    trades_generator = make_generator(int(streams.trades), device)
    pgd_generator = make_generator(int(streams.pgd), device)
    data = build_pretrain_data(cfg, dataloader_generator)
    if data["split_hash"] != split_hash:
        raise RuntimeError("datamodule split hash changed after asset verification")
    model = build_cifar100_model(cfg, initialization_path=initialization_path)
    assert_no_batchnorm(model)
    inventory = group_norm_inventory(model)
    if inventory["groups"] != [int(cfg.model.group_norm_groups)]:
        raise RuntimeError(f"unexpected GroupNorm inventory: {inventory}")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("full_model pretrain found frozen parameters")

    metadata = platform_metadata(cfg)
    with open_dict(cfg):
        cfg.provenance.git_commit = metadata["git_commit"]
        cfg.provenance.platform_job_id = metadata["platform_job_id"]
        cfg.model.group_norm_inventory = inventory
    output_dir = Path(str(cfg.runtime.output_root)) / str(cfg.experiment.run_id)
    resume_path = _resolve_resume(resume, output_dir)
    if output_dir.exists() and resume_path is None and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is non-empty; pass --resume auto instead of overwriting: {output_dir}"
        )
    logger = PretrainLogger(cfg, output_dir, resume=resume_path is not None)
    with open_dict(cfg):
        cfg.logging.wandb.run_id = logger.wandb_run_id
    save_resolved_config(cfg, output_dir)
    _, gpu_type = save_environment(cfg, output_dir, device)
    trainer = PretrainTrainer(
        cfg=cfg,
        model=model,
        data=data,
        logger=logger,
        device=device,
        initialization_hash=initialization_hash,
        dataloader_generator=dataloader_generator,
        trades_generator=trades_generator,
        pgd_generator=pgd_generator,
        platform_metadata=metadata,
    )
    try:
        if resume_path is not None:
            trainer.load_checkpoint(resume_path)
        summary = trainer.fit()
        summary["gpu_type"] = gpu_type
        logger.write_summary(summary)
        _write_artifact_manifest(output_dir)
        _persist_artifacts(cfg, output_dir)
        return summary
    finally:
        logger.finish()


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    cfg = load_pretrain_config(args.config, args.runtime_config)
    result = run_pretraining(
        cfg,
        resume=args.resume,
        prepare_assets_only=bool(args.prepare_assets_only),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _select_device(cfg) -> torch.device:
    if torch.cuda.is_available():
        # DataSphere exposes the allocated accelerator as logical device 0.
        # PyTorch 2.4 rejects an index-less device in ``set_device``.
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        return device
    if str(cfg.runtime.name) in {"datasphere", "mlm"}:
        raise RuntimeError(f"{cfg.runtime.name} pretraining requires a CUDA GPU")
    return torch.device("cpu")


def _resolve_resume(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    if value == "auto":
        path = output_dir / "checkpoints" / "checkpoint_last.pt"
    else:
        path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def _write_artifact_manifest(output_dir: Path) -> Path:
    manifest_path = output_dir / "artifacts_manifest.json"
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path or "/wandb/" in path.as_posix():
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {"format_version": 1, "files": files}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _persist_artifacts(cfg, output_dir: Path) -> None:
    mode = str(cfg.runtime.artifact_persistence)
    if mode in {"local", "datasphere_output"}:
        return
    if mode != "mlm_job_artifacts":
        raise ValueError(f"unknown artifact_persistence mode: {mode}")
    archive_base = Path.cwd() / f"pretrain_result__{cfg.experiment.run_id}"
    archive = Path(shutil.make_archive(str(archive_base), "gztar", root_dir=output_dir))
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError("MLM artifact persistence failed: result archive was not created")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
