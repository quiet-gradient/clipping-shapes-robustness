from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from trustml.pretraining.attacks import evaluate_pgd, trades_objective
from trustml.pretraining.autoattack import evaluate_autoattack_standard
from trustml.pretraining.model import state_dict_sha256
from trustml.pretraining.reproducibility import capture_rng_state, restore_rng_state
from trustml.pretraining.selection import mark_consistency, select_within_run


HISTORY_COLUMNS = (
    "epoch",
    "global_step",
    "learning_rate",
    "train_total_loss",
    "train_clean_ce",
    "train_robust_kl",
    "train_clean_accuracy",
    "train_adversarial_accuracy",
    "train_linf_max",
    "validation_clean_loss",
    "validation_clean_accuracy",
    "validation_pgd10_loss",
    "validation_pgd10_accuracy",
    "validation_pgd10_linf_max",
    "epoch_seconds",
)


class PretrainingTrainer:
    def __init__(
        self,
        *,
        cfg,
        model: torch.nn.Module,
        data: dict[str, Any],
        device: torch.device,
        output_dir: str | Path,
        config_hash: str,
        initialization_hash: str,
        dataloader_generator: torch.Generator,
        trades_generator: torch.Generator,
        pgd_generator: torch.Generator,
        smoke_epochs: int | None = None,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = str(config_hash)
        self.initialization_hash = str(initialization_hash)
        self.dataloader_generator = dataloader_generator
        self.trades_generator = trades_generator
        self.pgd_generator = pgd_generator
        self.smoke_epochs = smoke_epochs
        self.max_train_batches = max_train_batches
        self.max_validation_batches = max_validation_batches
        optimizer = cfg.training.optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(optimizer.learning_rate),
            momentum=float(optimizer.momentum),
            weight_decay=float(optimizer.weight_decay),
            nesterov=bool(optimizer.nesterov),
        )
        scheduler = cfg.training.scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(value) for value in scheduler.milestones],
            gamma=float(scheduler.gamma),
        )
        self.start_epoch = 1
        self.global_step = 0
        self.state: dict[str, Any] = {
            "best_clean_accuracy": -math.inf,
            "best_clean_epoch": None,
            "pgd10_records": [],
            "resume_roundtrip_verified": False,
        }
        self.history_path = self.output_dir / "history.csv"

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "run_id": str(self.cfg.experiment.run_id),
            "config_hash": self.config_hash,
            "split_hash": str(self.data["split_hash"]),
            "initialization_hash": self.initialization_hash,
        }
        for key, value in expected.items():
            if str(checkpoint.get(key)) != value:
                raise RuntimeError(f"resume checkpoint {key} mismatch")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.state = dict(checkpoint["selection_state"])
        restore_rng_state(
            checkpoint["rng_states"],
            self.dataloader_generator,
            self.trades_generator,
            self.pgd_generator,
        )

    def fit(self) -> dict[str, Any]:
        final_epoch = int(self.smoke_epochs or self.cfg.training.epochs)
        if self.start_epoch == 1:
            observed = state_dict_sha256(self.model.state_dict())
            if observed != self.initialization_hash:
                raise RuntimeError("model state changed before the first optimizer step")
            self._initialize_history()
        elif not self.history_path.is_file():
            raise RuntimeError("resume history is missing")

        for epoch in range(self.start_epoch, final_epoch + 1):
            started = time.monotonic()
            self.data["train_dataset"].set_epoch(epoch)
            train_metrics = self._train_epoch()
            clean_metrics = self._evaluate_clean()
            pgd_metrics: dict[str, float] = {}
            if str(self.cfg.objective.name) == "trades":
                pgd_metrics = self._evaluate_pgd10()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step()

            if clean_metrics["validation_clean_accuracy"] > float(self.state["best_clean_accuracy"]):
                self.state["best_clean_accuracy"] = clean_metrics["validation_clean_accuracy"]
                self.state["best_clean_epoch"] = epoch
                improved_clean = True
            else:
                improved_clean = False
            if pgd_metrics:
                record = {
                    "epoch": epoch,
                    "clean_accuracy": clean_metrics["validation_clean_accuracy"],
                    "pgd10_accuracy": pgd_metrics["validation_pgd10_accuracy"],
                    "pgd10_loss": pgd_metrics["validation_pgd10_loss"],
                    "checkpoint": f"checkpoint_pgd10_epoch_{epoch:03d}.pt",
                }
                self.state["pgd10_records"] = [*self.state["pgd10_records"], record]

            metrics = {
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate": learning_rate,
                **train_metrics,
                **clean_metrics,
                **pgd_metrics,
                "epoch_seconds": time.monotonic() - started,
            }
            if not all(
                math.isfinite(float(value))
                for value in metrics.values()
                if isinstance(value, (int, float))
            ):
                raise FloatingPointError("an epoch produced non-finite metrics")
            last_path = self.save_checkpoint("checkpoint_last.pt", epoch, metrics)
            if improved_clean:
                shutil.copy2(last_path, self.checkpoint_dir / "checkpoint_best_clean.pt")
            if pgd_metrics:
                self.save_checkpoint(f"checkpoint_pgd10_epoch_{epoch:03d}.pt", epoch, metrics)
                self._retain_top_pgd10_checkpoints()
            self._append_history(metrics)

            if self.smoke_epochs is not None and epoch == 1:
                expected_hash = state_dict_sha256(self.model.state_dict())
                expected_step = self.global_step
                self.load_checkpoint(last_path)
                if state_dict_sha256(self.model.state_dict()) != expected_hash:
                    raise RuntimeError("resume round trip changed model tensors")
                if self.global_step != expected_step:
                    raise RuntimeError("resume round trip changed the optimizer step")
                self.state["resume_roundtrip_verified"] = True

        robust_selection: dict[str, Any] = {}
        selected_checkpoint = self.checkpoint_dir / "checkpoint_best_clean.pt"
        if str(self.cfg.objective.name) == "trades" and self.smoke_epochs is None:
            selected_checkpoint, robust_selection = self._run_final_autoattack_selection()
        summary = {
            "format_version": 1,
            "status": "SMOKE_COMPLETE" if self.smoke_epochs is not None else "COMPLETE",
            "run_id": str(self.cfg.experiment.run_id),
            "objective": str(self.cfg.objective.name),
            "beta": None if self.cfg.objective.beta is None else float(self.cfg.objective.beta),
            "seed": int(self.cfg.experiment.seed),
            "best_clean_accuracy": float(self.state["best_clean_accuracy"]),
            "best_clean_epoch": int(self.state["best_clean_epoch"]),
            "selected_checkpoint": selected_checkpoint.relative_to(self.output_dir).as_posix(),
            "config_hash": self.config_hash,
            "split_hash": str(self.data["split_hash"]),
            "initialization_hash": self.initialization_hash,
            "resume_roundtrip_verified": bool(self.state["resume_roundtrip_verified"]),
            **robust_selection,
        }
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_examples = 0
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total_clean_correct = 0.0
        total_adversarial_correct = 0.0
        linf_max = 0.0
        for batch_index, (pixels, targets) in enumerate(self.data["train"]):
            if self.max_train_batches is not None and batch_index >= self.max_train_batches:
                break
            pixels = pixels.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
                raise AssertionError("training transforms must emit pixels in [0,1]")
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
                    gaussian_std=float(self.cfg.objective.gaussian_std),
                    generator=self.trades_generator,
                )
                loss = result.total_loss
                clean_ce = result.clean_ce
                robust_kl = result.robust_kl
                total_clean_correct += result.clean_accuracy * len(targets)
                total_adversarial_correct += result.adversarial_accuracy * len(targets)
                linf_max = max(linf_max, result.linf_max)
            else:
                logits = self.model(pixels)
                clean_ce = F.cross_entropy(logits, targets, reduction="mean")
                robust_kl = torch.zeros((), device=self.device)
                loss = clean_ce
                total_clean_correct += int(logits.detach().argmax(1).eq(targets).sum().item())
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            batch_size = int(targets.numel())
            total_examples += batch_size
            total_loss += float(loss.detach().item()) * batch_size
            total_ce += float(clean_ce.detach().item()) * batch_size
            total_kl += float(robust_kl.detach().item()) * batch_size
        if total_examples == 0:
            raise RuntimeError("training processed zero examples")
        if linf_max > float(self.cfg.objective.epsilon) + 1e-6:
            raise AssertionError("TRADES exceeded the configured perturbation radius")
        return {
            "train_total_loss": total_loss / total_examples,
            "train_clean_ce": total_ce / total_examples,
            "train_robust_kl": total_kl / total_examples,
            "train_clean_accuracy": total_clean_correct / total_examples,
            "train_adversarial_accuracy": total_adversarial_correct / total_examples,
            "train_linf_max": linf_max,
        }

    @torch.no_grad()
    def _evaluate_clean(self) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch_index, (pixels, targets) in enumerate(self.data["validation"]):
            if self.max_validation_batches is not None and batch_index >= self.max_validation_batches:
                break
            pixels = pixels.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(pixels)
            total_loss += float(F.cross_entropy(logits, targets, reduction="sum").item())
            total_correct += int(logits.argmax(1).eq(targets).sum().item())
            total += int(targets.numel())
        self.model.train(was_training)
        if total == 0:
            raise RuntimeError("clean validation processed zero examples")
        return {
            "validation_clean_loss": total_loss / total,
            "validation_clean_accuracy": total_correct / total,
        }

    def _evaluate_pgd10(self) -> dict[str, float]:
        attack = self.cfg.validation.pgd10
        result = evaluate_pgd(
            self.model,
            self.data["validation"],
            self.device,
            epsilon=float(attack.epsilon),
            step_size=float(attack.step_size),
            steps=int(attack.steps),
            restarts=int(attack.restarts),
            generator=self.pgd_generator,
            max_batches=self.max_validation_batches,
        )
        return {
            "validation_pgd10_loss": result["loss"],
            "validation_pgd10_accuracy": result["accuracy"],
            "validation_pgd10_linf_max": result["linf_max"],
        }

    def _retain_top_pgd10_checkpoints(self) -> None:
        records = sorted(
            self.state["pgd10_records"],
            key=lambda record: (-float(record["pgd10_accuracy"]), int(record["epoch"])),
        )
        keep = {
            record["checkpoint"] for record in records[: int(self.cfg.selection.pgd_top_k)]
        }
        for record in records:
            path = self.checkpoint_dir / str(record["checkpoint"])
            if record["checkpoint"] not in keep and path.exists():
                path.unlink()

    def _run_final_autoattack_selection(self) -> tuple[Path, dict[str, Any]]:
        records = sorted(
            self.state["pgd10_records"],
            key=lambda record: (-float(record["pgd10_accuracy"]), int(record["epoch"])),
        )[: int(self.cfg.selection.pgd_top_k)]
        if len(records) != 3:
            raise RuntimeError("final robust selection requires exactly three PGD-10 candidates")
        current = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        evaluated = []
        try:
            for record in records:
                source = self.checkpoint_dir / str(record["checkpoint"])
                checkpoint = torch.load(source, map_location="cpu", weights_only=False)
                self.model.load_state_dict(checkpoint["model"], strict=True)
                report = evaluate_autoattack_standard(
                    self.model,
                    self.data["validation"],
                    self.device,
                    epsilon=float(self.cfg.autoattack.epsilon),
                    batch_size=int(self.cfg.autoattack.batch_size),
                    seed=int(self.cfg.autoattack.seed),
                    expected_examples=int(self.cfg.autoattack.examples),
                    output_dir=self.output_dir / "autoattack" / f"epoch_{int(record['epoch']):03d}",
                )
                candidate = {
                    **record,
                    "autoattack_accuracy": float(report["overall_accuracy"]),
                    "autoattack_components": report["component_accuracy"],
                    "square_accuracy": float(report["component_accuracy"]["square"]),
                    "autoattack_report": (
                        Path("autoattack") / f"epoch_{int(record['epoch']):03d}" / "report.json"
                    ).as_posix(),
                }
                evaluated.append(
                    mark_consistency(
                        candidate,
                        threshold=float(
                            self.cfg.selection.pgd_minus_square_instability_threshold
                        ),
                    )
                )
        finally:
            self.model.load_state_dict(current, strict=True)
        selected = select_within_run(
            evaluated, examples=int(self.cfg.selection.binomial_examples)
        )
        source = self.checkpoint_dir / str(selected["checkpoint"])
        destination = self.checkpoint_dir / "checkpoint_selected_autoattack.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        checkpoint["final_selection"] = {"selected": selected, "candidates": evaluated}
        _atomic_torch_save(checkpoint, destination)
        return destination, {
            "selected_autoattack_accuracy": float(selected["autoattack_accuracy"]),
            "selected_autoattack_epoch": int(selected["epoch"]),
            "selected_clean_accuracy": float(selected["clean_accuracy"]),
            "selected_pgd10_accuracy": float(selected["pgd10_accuracy"]),
            "autoattack_candidates": evaluated,
        }

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, Any]) -> Path:
        path = self.checkpoint_dir / filename
        payload = {
            "format_version": 1,
            "run_id": str(self.cfg.experiment.run_id),
            "model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "metrics": dict(metrics),
            "selection_state": self.state,
            "rng_states": capture_rng_state(
                self.dataloader_generator, self.trades_generator, self.pgd_generator
            ),
            "resolved_config": OmegaConf.to_container(self.cfg, resolve=True),
            "config_hash": self.config_hash,
            "split_hash": str(self.data["split_hash"]),
            "initialization_hash": self.initialization_hash,
        }
        _atomic_torch_save(payload, path)
        return path

    def _initialize_history(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writeheader()

    def _append_history(self, metrics: dict[str, Any]) -> None:
        row = {column: metrics.get(column, "") for column in HISTORY_COLUMNS}
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writerow(row)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
