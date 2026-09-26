from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torchvision
from omegaconf import OmegaConf


HISTORY_COLUMNS = [
    "epoch",
    "global_step",
    "optimizer/lr",
    "train/total_loss",
    "train/clean_ce",
    "train/robust_kl_unweighted",
    "train/robust_kl_weighted",
    "train/kl_to_ce_ratio",
    "train/clean_accuracy",
    "train/trades_kl",
    "train/approx_adv_accuracy",
    "attack/linf_mean",
    "attack/linf_max",
    "val/clean_loss",
    "val/clean_accuracy",
    "val/pgd10_accuracy",
    "val/pgd10_loss",
    "val/pgd20_accuracy",
    "val/pgd20_loss",
    "system/epoch_seconds",
    "system/elapsed_seconds",
    "system/max_gpu_memory_gb",
]


class PretrainLogger:
    def __init__(self, cfg, output_dir: Path, resume: bool = False) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = output_dir / "history.csv"
        self.summary_path = output_dir / "run_summary.json"
        self._wandb = None
        self.wandb_run = None
        if not resume or not self.history_path.exists():
            with self.history_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writeheader()
        self._init_wandb()

    @property
    def wandb_run_id(self) -> str:
        if self.wandb_run is not None:
            return str(self.wandb_run.id)
        return str(self.cfg.logging.wandb.run_id)

    def _init_wandb(self) -> None:
        mode = str(self.cfg.runtime.wandb_mode).lower()
        if mode == "disabled":
            return
        if mode not in {"online", "offline"}:
            raise ValueError(f"unsupported W&B mode: {mode}")
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(f"W&B mode is {mode}, but wandb is not installed") from exc
        self._wandb = wandb
        try:
            self.wandb_run = wandb.init(
                project=str(self.cfg.logging.wandb.project),
                entity=_optional_text(self.cfg.logging.wandb.get("entity", None)),
                name=str(self.cfg.logging.wandb.name),
                id=str(self.cfg.logging.wandb.run_id),
                resume="allow",
                mode=mode,
                dir=str(self.output_dir),
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=list(self.cfg.logging.wandb.tags),
            )
        except Exception as exc:  # W&B is telemetry, durable local artifacts remain authoritative.
            warnings.warn(f"wandb.init failed in {mode} mode; continuing with local artifacts: {exc}")
            self.wandb_run = None

    def log_epoch(self, metrics: dict[str, Any]) -> None:
        row = {column: metrics.get(column, "") for column in HISTORY_COLUMNS}
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writerow(row)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics)

    def write_summary(self, summary: dict[str, Any]) -> None:
        temp = self.summary_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.summary_path)
        if self.wandb_run is not None:
            self.wandb_run.summary.update(summary)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def save_resolved_config(cfg, output_dir: Path) -> Path:
    path = output_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    return path


def save_environment(cfg, output_dir: Path, device: torch.device) -> tuple[Path, str]:
    gpu_type = "none"
    if device.type == "cuda":
        gpu_type = torch.cuda.get_device_name(device)
    lines = [
        f"platform={cfg.runtime.name}",
        f"platform_node={platform.node()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"torch={torch.__version__}",
        f"torchvision={torchvision.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_runtime={torch.version.cuda}",
        f"cudnn={torch.backends.cudnn.version()}",
        f"device={device}",
        f"gpu_type={gpu_type}",
        f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32}",
        f"tf32_cudnn={torch.backends.cudnn.allow_tf32}",
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}",
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}",
    ]
    if device.type == "cuda":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            lines.append("nvidia_smi=" + result.stdout.strip().replace("\n", "; "))
        except (OSError, subprocess.SubprocessError) as exc:
            lines.append(f"nvidia_smi_error={type(exc).__name__}: {exc}")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        lines.extend(["", "[pip_freeze]", freeze])
    except (OSError, subprocess.SubprocessError) as exc:
        lines.extend(["", "[pip_freeze]", f"ERROR: {type(exc).__name__}: {exc}"])
    path = output_dir / "environment.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, gpu_type


def platform_metadata(cfg) -> dict[str, str]:
    job_env = str(cfg.runtime.get("platform_job_id_env", ""))
    git_env = str(cfg.runtime.get("git_commit_env", "TRUSTML_GIT_COMMIT"))
    job_id = os.environ.get(job_env, "") if job_env else ""
    if not job_id and str(cfg.runtime.name) == "mlm":
        job_id = os.environ.get("JOB_ID", "") or os.environ.get("MLM_EXECUTION_ID", "")
    if not job_id and str(cfg.runtime.name) == "datasphere":
        job_id = os.environ.get("JOB_ID", "") or os.environ.get("DATASPHERE_PROJECT_JOB_ID", "")
    git_commit = os.environ.get(git_env, "") or str(cfg.provenance.get("git_commit", "unknown"))
    name = str(cfg.runtime.name)
    return {
        "platform": name,
        "platform_job_id": job_id,
        "mlm_job_id": job_id if name == "mlm" else "",
        "datasphere_job_id": job_id if name == "datasphere" else "",
        "git_commit": git_commit,
    }


def _optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "null", "none"} else text
