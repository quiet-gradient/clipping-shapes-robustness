from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from trustml.pretrain.attacks import evaluate_pgd, trades_objective
from trustml.pretrain.autoattack import evaluate_autoattack_standard
from trustml.pretrain.model import state_dict_sha256
from trustml.pretrain.reproducibility import capture_rng_state, restore_rng_state


class PretrainTrainer:
    def __init__(
        self,
        *,
        cfg,
        model: torch.nn.Module,
        data: dict[str, Any],
        logger,
        device: torch.device,
        initialization_hash: str,
        dataloader_generator: torch.Generator,
        trades_generator: torch.Generator,
        pgd_generator: torch.Generator,
        platform_metadata: dict[str, str],
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.data = data
        self.logger = logger
        self.device = device
        self.initialization_hash = initialization_hash
        self.dataloader_generator = dataloader_generator
        self.trades_generator = trades_generator
        self.pgd_generator = pgd_generator
        self.platform_metadata = platform_metadata
        optimizer_cfg = cfg.training.optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(optimizer_cfg.learning_rate),
            momentum=float(optimizer_cfg.momentum),
            weight_decay=float(optimizer_cfg.weight_decay),
            nesterov=bool(optimizer_cfg.nesterov),
        )
        scheduler_cfg = cfg.training.scheduler
        if str(scheduler_cfg.name).lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(cfg.training.epochs),
                eta_min=float(optimizer_cfg.learning_rate) * float(scheduler_cfg.eta_min_factor),
            )
        elif str(scheduler_cfg.name).lower() == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[int(epoch) for epoch in scheduler_cfg.milestones],
                gamma=float(scheduler_cfg.gamma),
            )
        else:
            raise ValueError(f"unsupported scheduler {scheduler_cfg.name!r}")
        self.output_dir = Path(logger.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.start_epoch = 1
        self.best_metrics: dict[str, Any] = {
            "best_clean_accuracy": -math.inf,
            "best_clean_epoch": None,
            "best_pgd20_accuracy": -math.inf,
            "best_pgd20_epoch": None,
            "periodic_pgd_records": [],
            "resume_roundtrip_verified": False,
        }
        self.started_at = time.monotonic()

    @property
    def _final_v1(self) -> bool:
        return str(self.cfg.get("protocol", {}).get("name", "legacy_sweep")) == "final_v1"

    @property
    def _validation_attack(self):
        return self.cfg.validation.pgd10 if self._final_v1 else self.cfg.validation.pgd20

    @property
    def _validation_label(self) -> str:
        return "pgd10" if self._final_v1 else "pgd20"

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if str(checkpoint.get("run_id")) != str(self.cfg.experiment.run_id):
            raise RuntimeError("resume checkpoint run_id mismatch")
        if str(checkpoint.get("config_hash")) != str(self.cfg.provenance.config_hash):
            raise RuntimeError("resume checkpoint config hash mismatch")
        if str(checkpoint.get("split_hash")) != str(self.cfg.provenance.split_hash):
            raise RuntimeError("resume checkpoint split hash mismatch")
        if str(checkpoint.get("initialization_hash")) != self.initialization_hash:
            raise RuntimeError("resume checkpoint initialization hash mismatch")
        if str(checkpoint.get("wandb_run_id")) != str(self.logger.wandb_run_id):
            raise RuntimeError("resume checkpoint W&B run ID mismatch")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint["global_step"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metrics = dict(checkpoint["best_metrics"])
        restore_rng_state(
            checkpoint["rng_states"],
            self.dataloader_generator,
            self.trades_generator,
            self.pgd_generator,
        )
        return checkpoint

    def fit(self) -> dict[str, Any]:
        scientific_epochs = int(self.cfg.training.epochs)
        smoke = self.cfg.runtime.smoke
        final_epoch = int(smoke.epochs) if bool(smoke.enabled) else scientific_epochs
        max_train_batches = _optional_int(smoke.max_train_batches) if bool(smoke.enabled) else None
        max_val_batches = _optional_int(smoke.max_val_batches) if bool(smoke.enabled) else None
        if self.start_epoch == 1:
            actual = state_dict_sha256(self.model.state_dict())
            if actual != self.initialization_hash:
                raise RuntimeError(
                    f"model initialization changed before first optimizer step: {actual} != {self.initialization_hash}"
                )

        for epoch in range(self.start_epoch, final_epoch + 1):
            epoch_started = time.monotonic()
            self.data["train_dataset"].set_epoch(epoch)
            train_metrics = self._train_epoch(max_train_batches=max_train_batches)
            clean_metrics = self._evaluate_clean(max_batches=max_val_batches)
            robust_metrics: dict[str, float] = {}
            robust_due = str(self.cfg.objective.name) == "trades" and (
                epoch % int(self.cfg.validation.robust_every_epochs) == 0
                or (bool(smoke.enabled) and epoch == final_epoch)
            )
            if robust_due:
                robust_metrics = self._evaluate_pgd(
                    restarts=int(self._validation_attack.periodic_restarts),
                    max_batches=max_val_batches,
                )

            epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step()
            improved_clean = clean_metrics["val/clean_accuracy"] > float(
                self.best_metrics["best_clean_accuracy"]
            )
            if improved_clean:
                self.best_metrics["best_clean_accuracy"] = clean_metrics["val/clean_accuracy"]
                self.best_metrics["best_clean_epoch"] = epoch
            if robust_metrics:
                record = {
                    "epoch": epoch,
                    f"{self._validation_label}_accuracy": robust_metrics[f"val/{self._validation_label}_accuracy"],
                    f"{self._validation_label}_loss": robust_metrics[f"val/{self._validation_label}_loss"],
                    "clean_accuracy": clean_metrics["val/clean_accuracy"],
                    "checkpoint": f"checkpoint_{self._validation_label}_epoch_{epoch:03d}.pt",
                }
                records = list(self.best_metrics.get("periodic_pgd_records", []))
                records.append(record)
                self.best_metrics["periodic_pgd_records"] = records
                if robust_metrics[f"val/{self._validation_label}_accuracy"] > float(
                    self.best_metrics["best_pgd20_accuracy"]
                ):
                    self.best_metrics["best_pgd20_accuracy"] = robust_metrics[
                        f"val/{self._validation_label}_accuracy"
                    ]
                    self.best_metrics["best_pgd20_epoch"] = epoch

            all_metrics = {
                "epoch": epoch,
                "global_step": self.global_step,
                "optimizer/lr": epoch_lr,
                **train_metrics,
                **clean_metrics,
                **robust_metrics,
                "system/epoch_seconds": time.monotonic() - epoch_started,
                "system/elapsed_seconds": time.monotonic() - self.started_at,
                "system/max_gpu_memory_gb": (
                    torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    if self.device.type == "cuda"
                    else 0.0
                ),
            }
            if not all(math.isfinite(float(value)) for value in all_metrics.values() if isinstance(value, float)):
                raise FloatingPointError(f"non-finite epoch metrics: {all_metrics}")

            last_path = self.save_checkpoint("checkpoint_last.pt", epoch, all_metrics)
            if improved_clean:
                shutil.copy2(last_path, self.checkpoint_dir / "checkpoint_best_clean.pt")
            if robust_metrics:
                periodic_path = self.save_checkpoint(
                    f"checkpoint_{self._validation_label}_epoch_{epoch:03d}.pt", epoch, all_metrics
                )
                if int(self.best_metrics["best_pgd20_epoch"]) == epoch:
                    shutil.copy2(periodic_path, self.checkpoint_dir / f"checkpoint_best_{self._validation_label}.pt")
                self._retain_top_robust_checkpoints()
            self.logger.log_epoch(all_metrics)

            if bool(smoke.enabled) and bool(smoke.resume_probe) and epoch == 1:
                expected_hash = state_dict_sha256(self.model.state_dict())
                expected_step = self.global_step
                self.load_checkpoint(last_path)
                if state_dict_sha256(self.model.state_dict()) != expected_hash or self.global_step != expected_step:
                    raise RuntimeError("smoke resume round-trip did not restore model/global_step")
                self.best_metrics["resume_roundtrip_verified"] = True
                self.start_epoch = epoch + 1

        selected_path = self.checkpoint_dir / "checkpoint_best_clean.pt"
        final_robust: dict[str, Any] = {}
        if str(self.cfg.objective.name) == "trades" and self.best_metrics["periodic_pgd_records"]:
            selected_path, final_robust = self._select_final_robust(max_batches=max_val_batches)
        summary = {
            "run_id": str(self.cfg.experiment.run_id),
            "objective": str(self.cfg.objective.name),
            "best_clean_accuracy": float(self.best_metrics["best_clean_accuracy"]),
            "best_clean_epoch": self.best_metrics["best_clean_epoch"],
            "best_pgd20_accuracy": _finite_or_none(self.best_metrics["best_pgd20_accuracy"]),
            "best_pgd20_epoch": self.best_metrics["best_pgd20_epoch"],
            **final_robust,
            "selected_checkpoint": str(selected_path.relative_to(self.output_dir)),
            "completed_successfully": True,
            "gpu_type": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "none",
            "config_hash": str(self.cfg.provenance.config_hash),
            "split_hash": str(self.cfg.provenance.split_hash),
            "initialization_hash": self.initialization_hash,
            "wandb_run_id": str(self.logger.wandb_run_id),
            "wandb_mode": str(self.cfg.runtime.wandb_mode),
            "wandb_active": bool(self.logger.wandb_run is not None),
            **self.platform_metadata,
            "resume_roundtrip_verified": bool(self.best_metrics.get("resume_roundtrip_verified")),
        }
        self.logger.write_summary(summary)
        return summary

    def _train_epoch(self, max_train_batches: int | None) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "correct": 0.0,
            "examples": 0.0,
            "kl": 0.0,
            "adv_correct": 0.0,
            "linf_sum": 0.0,
            "linf_max": 0.0,
        }
        for batch_index, (pixels, targets) in enumerate(self.data["train"]):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            pixels = pixels.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
                raise AssertionError("training transform emitted pixels outside [0,1]")
            self.optimizer.zero_grad(set_to_none=True)
            if str(self.cfg.objective.name) == "trades":
                result = trades_objective(
                    self.model,
                    pixels,
                    targets,
                    beta=float(self.cfg.objective.beta),
                    epsilon=float(self.cfg.objective.epsilon),
                    step_size=float(self.cfg.objective.step_size),
                    steps=int(self.cfg.objective.steps),
                    generator=self.trades_generator,
                    random_start=bool(self.cfg.objective.random_start),
                    attack_init=str(self.cfg.objective.get("attack_init", "uniform")),
                    gaussian_std=float(self.cfg.objective.get("gaussian_std", 0.001)),
                )
                loss = result.total_loss
                ce = result.clean_ce
                kl = result.trades_kl
                clean_predictions = None
                totals["adv_correct"] += result.approx_adv_accuracy * len(targets)
                totals["correct"] += result.clean_accuracy * len(targets)
                totals["linf_sum"] += result.linf_mean * len(targets)
                totals["linf_max"] = max(totals["linf_max"], result.linf_max)
            else:
                logits = self.model(pixels)
                ce = F.cross_entropy(logits, targets, reduction="mean")
                loss = ce
                kl = torch.zeros((), device=self.device)
                clean_predictions = logits.detach().argmax(1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at global step {self.global_step}")
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            batch_size = int(targets.numel())
            totals["loss"] += float(loss.detach().item()) * batch_size
            totals["ce"] += float(ce.detach().item()) * batch_size
            totals["kl"] += float(kl.detach().item()) * batch_size
            if clean_predictions is not None:
                totals["correct"] += int(clean_predictions.eq(targets).sum().item())
            totals["examples"] += batch_size
        examples = int(totals["examples"])
        if examples == 0:
            raise RuntimeError("training processed zero examples")
        metrics = {
            "train/total_loss": totals["loss"] / examples,
            "train/clean_ce": totals["ce"] / examples,
            "train/clean_accuracy": totals["correct"] / examples,
        }
        if str(self.cfg.objective.name) == "trades":
            metrics.update(
                {
                    "train/trades_kl": totals["kl"] / examples,
                    "train/robust_kl_unweighted": totals["kl"] / examples,
                    "train/robust_kl_weighted": float(self.cfg.objective.beta) * totals["kl"] / examples,
                    "train/kl_to_ce_ratio": (totals["kl"] / examples) / max(totals["ce"] / examples, 1e-12),
                    "train/approx_adv_accuracy": totals["adv_correct"] / examples,
                    "attack/linf_mean": totals["linf_sum"] / examples,
                    "attack/linf_max": totals["linf_max"],
                }
            )
            if totals["linf_max"] > float(self.cfg.objective.epsilon) + 1e-6:
                raise AssertionError("TRADES perturbation exceeded the configured L-inf bound")
        return metrics

    @torch.no_grad()
    def _evaluate_clean(self, max_batches: int | None) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch_index, (pixels, targets) in enumerate(self.data["val"]):
            if max_batches is not None and batch_index >= max_batches:
                break
            pixels = pixels.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(pixels)
            losses = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += float(losses.item())
            total_correct += int(logits.argmax(1).eq(targets).sum().item())
            total += int(targets.numel())
        self.model.train(was_training)
        if total == 0:
            raise RuntimeError("clean validation processed zero examples")
        return {"val/clean_loss": total_loss / total, "val/clean_accuracy": total_correct / total}

    def _evaluate_pgd(self, restarts: int, max_batches: int | None) -> dict[str, float]:
        attack = self._validation_attack
        metrics = evaluate_pgd(
            self.model,
            self.data["val"],
            self.device,
            epsilon=float(attack.epsilon),
            step_size=float(attack.step_size),
            steps=int(attack.steps),
            restarts=restarts,
            generator=self.pgd_generator,
            max_batches=max_batches,
        )
        return {f"val/{self._validation_label}_{key}": value for key, value in metrics.items()}

    def _retain_top_robust_checkpoints(self) -> None:
        """Retain exactly the current three PGD leaders; history keeps all scores."""
        key = f"{self._validation_label}_accuracy"
        records = list(self.best_metrics.get("periodic_pgd_records", []))
        leaders = sorted(records, key=lambda item: (-float(item[key]), int(item["epoch"])))[: int(self.cfg.validation.top_k)]
        keep = {str(record["checkpoint"]) for record in leaders}
        for record in records:
            path = self.checkpoint_dir / str(record["checkpoint"])
            if str(record["checkpoint"]) not in keep and path.exists():
                path.unlink()

    def _select_final_robust(self, max_batches: int | None) -> tuple[Path, dict[str, Any]]:
        key = f"{self._validation_label}_accuracy"
        records = sorted(
            self.best_metrics["periodic_pgd_records"],
            key=lambda item: (-float(item[key]), int(item["epoch"])),
        )[: int(self.cfg.validation.top_k)]
        evaluated: list[dict[str, Any]] = []
        current_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        for record in records:
            path = self.checkpoint_dir / str(record["checkpoint"])
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model"], strict=True)
            if self._final_v1:
                report = evaluate_autoattack_standard(
                    self.model, self.data["val"], self.device,
                    epsilon=float(self._validation_attack.epsilon),
                    batch_size=int(self.cfg.training.batch_size),
                    seed=int(self.cfg.experiment.seed),
                    expected_examples=int(self.cfg.data.val_per_class) * int(self.cfg.data.num_classes),
                    output_dir=self.output_dir / "autoattack" / f"epoch_{int(record['epoch']):03d}",
                )
                pgd_accuracy = float(record[key])
                square_accuracy = float(report["components"]["square"])
                evaluated.append({
                    **record,
                    "aa_accuracy": float(report["overall_accuracy"]),
                    "aa_components": report["components"],
                    "gap_pgd_aa": pgd_accuracy - float(report["overall_accuracy"]),
                    "gap_pgd_square": pgd_accuracy - square_accuracy,
                    "unstable": (pgd_accuracy - square_accuracy) >= 0.08,
                    "autoattack_report": str((self.output_dir / "autoattack" / f"epoch_{int(record['epoch']):03d}" / "autoattack_report.json").relative_to(self.output_dir)),
                })
            else:
                metrics = self._evaluate_pgd(
                    restarts=int(self._validation_attack.final_restarts), max_batches=max_batches
                )
                evaluated.append({**record, "pgd20_5r_accuracy": metrics["val/pgd20_accuracy"], "pgd20_5r_loss": metrics["val/pgd20_loss"]})
        self.model.load_state_dict(current_state, strict=True)
        if self._final_v1:
            stable = [candidate for candidate in evaluated if not candidate["unstable"]]
            contenders = stable or evaluated
            selected = _select_one_standard_error(contenders, metric="aa_accuracy")
            source = self.checkpoint_dir / str(selected["checkpoint"])
            destination = self.checkpoint_dir / "checkpoint_selected_autoattack.pt"
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            checkpoint["final_robust_selection"] = {"selected": selected, "candidates": evaluated, "stability_threshold": 0.08}
            _atomic_torch_save(checkpoint, destination)
            return destination, {"selected_autoattack_accuracy": float(selected["aa_accuracy"]), "selected_autoattack_epoch": int(selected["epoch"]), "autoattack_candidates": evaluated}

        selected = evaluated[0]
        threshold = float(self.cfg.validation.clean_tie_threshold)
        for candidate in evaluated[1:]:
            delta = float(candidate["pgd20_5r_accuracy"]) - float(selected["pgd20_5r_accuracy"])
            if delta >= threshold:
                selected = candidate
            elif abs(delta) < threshold:
                clean_delta = float(candidate["clean_accuracy"]) - float(selected["clean_accuracy"])
                if clean_delta > 0 or (clean_delta == 0 and int(candidate["epoch"]) < int(selected["epoch"])):
                    selected = candidate
        source = self.checkpoint_dir / str(selected["checkpoint"])
        destination = self.checkpoint_dir / "checkpoint_best_pgd20_5r.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        checkpoint["final_robust_selection"] = {
            "selected": selected,
            "candidates": evaluated,
            "clean_tie_threshold": threshold,
        }
        _atomic_torch_save(checkpoint, destination)
        return destination, {
            "best_pgd20_5r_accuracy": float(selected["pgd20_5r_accuracy"]),
            "best_pgd20_5r_epoch": int(selected["epoch"]),
            "pgd20_5r_candidates": evaluated,
        }

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, Any]) -> Path:
        path = self.checkpoint_dir / filename
        state = {
            "format_version": 1,
            "run_id": str(self.cfg.experiment.run_id),
            "model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "metrics": dict(metrics),
            "best_metrics": self.best_metrics,
            "rng_states": capture_rng_state(
                self.dataloader_generator, self.trades_generator, self.pgd_generator
            ),
            "resolved_config": OmegaConf.to_container(self.cfg, resolve=True),
            "config_hash": str(self.cfg.provenance.config_hash),
            "split_hash": str(self.cfg.provenance.split_hash),
            "initialization_hash": self.initialization_hash,
            "git_commit": self.platform_metadata["git_commit"],
            "wandb_run_id": str(self.logger.wandb_run_id),
            **self.platform_metadata,
        }
        _atomic_torch_save(state, path)
        return path


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _optional_int(value) -> int | None:
    if value is None or str(value).lower() in {"none", "null", ""}:
        return None
    return int(value)


def _finite_or_none(value) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _select_one_standard_error(candidates: list[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    """Choose within one binomial standard error, then clean accuracy, then epoch."""
    if not candidates:
        raise RuntimeError("no robust checkpoint candidates")
    best = max(float(candidate[metric]) for candidate in candidates)
    n = 1000  # the final_v1 validation split is fixed at 1k examples
    standard_error = math.sqrt(max(0.0, best * (1.0 - best)) / n)
    within = [candidate for candidate in candidates if best - float(candidate[metric]) <= standard_error + 1e-12]
    return sorted(
        within,
        key=lambda candidate: (-float(candidate["clean_accuracy"]), int(candidate["epoch"])),
    )[0]
