from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from trustml.attacks.pgd import PGDAttack
from trustml.core.checkpointing import save_checkpoint, unwrap_model
from trustml.core.seed import capture_rng_state, restore_rng_state
from trustml.defenses.combined import collect_defense_metrics
from trustml.evaluation.clean import evaluate_clean
from trustml.evaluation.robustness import evaluate_robustness
from trustml.training.dp_diagnostics import (
    auto_s_component_grad_sums_from_current_samples,
    auto_s_grad_sums_from_current_samples,
    clipped_grad_sums_from_current_samples,
    param_grad_norm,
    per_sample_clip_diagnostics,
    preclip_metric_names,
)
from trustml.training.gradient_probe import GradientProbeStateRestorationError, run_gradient_probe
from trustml.training.optimizers import (
    apply_adaptation,
    build_optimizer,
    build_scheduler,
    parameter_summary,
)


class Trainer:
    def __init__(self, cfg, model, data, defenses, logger) -> None:
        self.cfg = cfg
        self.model = model
        self.data = data
        self.defenses = defenses
        self.logger = logger
        self.device = self._resolve_device()
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.optimizer_step = 0
        self.current_epoch = 0
        self.start_epoch = 1
        self.best_metric = -float("inf")
        self.last_metrics: dict[str, float] = {}
        self.manual_d_dp_passes = 0
        self._gradient_probe_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_gradient_probe_metrics: dict[str, float] = {}
        self._gradient_probe_last_optimizer_step: int | None = None
        self._gradient_probe_target_steps: set[int] = set()

    @classmethod
    def from_config(cls, cfg, model, data, defenses, logger):
        return cls(cfg=cfg, model=model, data=data, defenses=defenses, logger=logger)

    def fit(self) -> dict[str, float]:
        self.model.to(self.device)
        apply_adaptation(self.model, self.cfg)
        self.optimizer = build_optimizer(self.cfg, self.model)

        for defense in self.defenses:
            self.model, self.optimizer, self.data["train"] = defense.setup(
                self.model, self.optimizer, self.data["train"], self.cfg
            )
            if defense.setup_metrics:
                self.logger.log_metrics(defense.setup_metrics, step=0)

        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self._configure_gradient_probe_schedule()
        self.logger.log_metrics(parameter_summary(self.model), step=0)
        self.logger.log_metrics(self._setup_metrics(), step=0)
        self._prepare_gradient_probe()
        resume_from = self.cfg.training.get("resume_from", None)
        if resume_from is not None and str(resume_from).strip().lower() not in {
            "",
            "none",
            "null",
        }:
            self.load_training_checkpoint(resume_from)
        probe_cfg = self._gradient_probe_cfg()
        if (
            self.start_epoch == 1
            and probe_cfg is not None
            and bool(probe_cfg.get("run_at_start", True))
        ):
            self._run_gradient_probe(trigger="start")

        epochs = int(self.cfg.training.get("epochs", 1))
        if epochs <= 0:
            metrics = self.evaluate_split("val", prefix="val")
            self.save_checkpoint("latest.pt", epoch=0, metrics=metrics)
            self.last_metrics = metrics
            return metrics

        final_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate_split("val", prefix="val")
            final_metrics = {**train_metrics, **val_metrics}
            final_metrics.update(self._epoch_metrics(train_metrics, val_metrics))
            self.logger.log_metrics(final_metrics, step=epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            score = val_metrics.get("val/clean_acc", train_metrics.get("train/acc", 0.0))
            if score > self.best_metric:
                self.best_metric = score
                path = self.save_checkpoint("best.pt", epoch=epoch, metrics=final_metrics)
                self.logger.log_artifact(path, artifact_path="best-checkpoint")

            every = int(self.cfg.training.get("checkpoint_every_epochs", 1))
            if every > 0 and epoch % every == 0:
                self.save_checkpoint("latest.pt", epoch=epoch, metrics=final_metrics)

        self.last_metrics = final_metrics
        return final_metrics

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        max_batches = self.cfg.training.get("max_train_batches", None)
        log_every = int(self.cfg.training.get("log_every_n_steps", 50))
        dp_defense = self._defense("dp_sgd")
        trades_defense = self._defense("trades")
        dp_strategy = getattr(dp_defense, "dp_strategy", None)
        use_auto_s = dp_defense is not None and bool(getattr(dp_defense, "auto_s", False))
        use_auto_s_componentwise_trades = (
            dp_defense is not None
            and trades_defense is not None
            and getattr(dp_defense, "regularizer_name", None) == "trades"
            and dp_strategy == "C"
            and bool(getattr(dp_defense, "componentwise_one_release", False))
            and use_auto_s
        )
        use_componentwise_trades = (
            dp_defense is not None
            and trades_defense is not None
            and getattr(dp_defense, "regularizer_name", None) == "trades"
            and dp_strategy == "C"
            and bool(getattr(dp_defense, "componentwise_one_release", False))
            and not use_auto_s
        )
        loader = self.data["train"]
        physical_batch_size = self.cfg.training.get("physical_batch_size", None)
        if physical_batch_size is not None and int(physical_batch_size) < int(
            self.cfg.data.get("batch_size", 0)
        ):
            raise ValueError(
                "The released experiment matrix requires physical_batch_size to be null "
                "or at least the logical batch size."
            )

        iterator = tqdm(
            enumerate(loader), total=len(loader), desc=f"epoch {epoch}", leave=False
        )
        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            x, y = self._move_batch(batch)

            if use_auto_s_componentwise_trades:
                logits, loss, batch_metrics = self._auto_s_componentwise_trades_step(
                    x, y, dp_defense, trades_defense
                )
            elif use_componentwise_trades:
                logits, loss, batch_metrics = self._componentwise_trades_step(
                    x, y, dp_defense, trades_defense
                )
            else:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                base_loss = F.cross_entropy(logits, y)
                context = {
                    "logits": logits,
                    "base_loss": base_loss,
                    "epoch": epoch,
                    "step": self.global_step,
                }
                loss = base_loss
                for defense in self.defenses:
                    loss = defense.loss(self.model, (x, y), loss, context)
                loss.backward()
                batch_metrics = collect_defense_metrics(self.defenses)
                if use_auto_s:
                    self._apply_auto_s_update(dp_defense)
                else:
                    self.optimizer.step()

            # The released configurations expose only public-development and
            # privacy-accounting outputs, never private-batch gradient telemetry.
            batch_metrics = self._without_private_gradient_metrics(batch_metrics)
            batch_size = int(y.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int(logits.detach().argmax(dim=1).eq(y).sum().item())
            total_seen += batch_size
            self.global_step += 1

            optimizer_step_skipped = bool(
                getattr(self.optimizer, "_is_last_step_skipped", False)
            )
            if not optimizer_step_skipped:
                self.optimizer_step += 1
                self._maybe_run_gradient_probe()

            if batch_metrics:
                self.logger.log_metrics(batch_metrics, step=self.global_step - 1)
            if log_every > 0 and self.global_step % log_every == 0:
                metrics = {
                    "train/batch_loss": float(loss.detach().item()),
                    "train/batch_acc": total_correct / max(1, total_seen),
                    "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                }
                metrics.update(batch_metrics)
                self.logger.log_metrics(metrics, step=self.global_step)

        probe_cfg = self._gradient_probe_cfg()
        if probe_cfg is not None and bool(probe_cfg.get("run_at_epoch_end", True)):
            self._run_gradient_probe(trigger="epoch_end")

        metrics = {
            "train/loss": total_loss / max(1, total_seen),
            "train/acc": total_correct / max(1, total_seen),
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        dp_defense = self._defense("dp_sgd")
        if dp_defense is not None:
            epsilon = dp_defense.get_epsilon()
            if epsilon is not None:
                metrics["dp/epsilon"] = epsilon
                metrics["DP epsilon"] = epsilon
            metrics.update(
                dp_defense.accountant_metrics(
                    manual_task_passes=0,
                    manual_regularizer_passes=0,
                    manual_d_passes=self.manual_d_dp_passes,
                )
            )
        return metrics

    def evaluate_split(self, split: str, prefix: str) -> dict[str, float]:
        dataloader = self.data.get(split)
        if dataloader is None and split == "val":
            dataloader = self.data.get("test")
        if dataloader is None:
            return {}
        max_batches = self.cfg.training.get("max_eval_batches", None)
        metrics = evaluate_clean(self.model, dataloader, self.device, max_batches=max_batches)
        return {f"{prefix}/{key}": value for key, value in metrics.items()}

    def evaluate(self) -> dict[str, float]:
        protocol = self.cfg.get("protocol", {}) or {}
        test_locked = bool(protocol.get("test_locked", False))
        evaluation_split = "val" if test_locked else "test"
        evaluation_prefix = "dev" if test_locked else "test"
        metrics = self.evaluate_split(evaluation_split, prefix=evaluation_prefix)

        attacks = []
        attacks_cfg = self.cfg.get("attacks", None)
        pgd_cfg = attacks_cfg.get("pgd", None) if attacks_cfg is not None else None
        if pgd_cfg is not None and bool(pgd_cfg.get("enabled", False)):
            attacks.append(
                PGDAttack(
                    pgd_cfg,
                    mean=tuple(self.cfg.data.mean),
                    std=tuple(self.cfg.data.std),
                )
            )
        if attacks and evaluation_split in self.data:
            metrics.update(
                evaluate_robustness(
                    self.model, self.data[evaluation_split], attacks, self.device
                )
            )

        if "test/clean_acc" in metrics:
            metrics["Test Accuracy"] = float(metrics["test/clean_acc"]) * 100.0
        if "dev/clean_acc" in metrics:
            metrics["Dev Accuracy"] = float(metrics["dev/clean_acc"]) * 100.0
        if metrics:
            self.logger.log_metrics(metrics, step=self.global_step)
        return metrics

    def save_checkpoint(
        self,
        name: str = "latest.pt",
        epoch: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_metrics = self._release_safe_checkpoint_metrics(metrics or {})
        checkpoint_dir = self.logger.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = Path("outputs") / str(self.cfg.experiment.name) / "checkpoints"
        checkpoint_path = save_checkpoint(
            Path(checkpoint_dir) / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch if epoch is not None else self.current_epoch,
            metrics=checkpoint_metrics,
            cfg=self.cfg,
            extra_state=self._training_checkpoint_extra_state(),
        )
        return checkpoint_path

    def _training_checkpoint_extra_state(self) -> dict[str, Any]:
        dp_defense = self._defense("dp_sgd")
        accountant_state = None
        if dp_defense is not None and getattr(dp_defense, "privacy_engine", None) is not None:
            accountant_state = dp_defense.privacy_engine.accountant.state_dict()
        return {
            "format_version": 2,
            "trainer_state": {
                "global_step": int(self.global_step),
                "optimizer_step": int(self.optimizer_step),
                "current_epoch": int(self.current_epoch),
                "best_metric": float(self.best_metric),
                "manual_d_dp_passes": int(self.manual_d_dp_passes),
                "gradient_probe_last_optimizer_step": self._gradient_probe_last_optimizer_step,
            },
            "rng_state": capture_rng_state(),
            "data_loader_state": self._capture_data_loader_state(),
            "privacy_accountant": accountant_state,
        }

    def _capture_data_loader_state(self) -> dict[str, Any]:
        """Describe the next epoch's sampler state for exact boundary resume.

        The campaign checkpoints only at epoch boundaries.  PyTorch and the
        Opacus Poisson sampler use either the loader's explicit generator or
        the global Torch RNG, which is already captured beside this payload.
        """
        loader = self.data.get("train")
        if loader is None:
            return {
                "format_version": 1,
                "boundary": "epoch",
                "next_epoch": int(self.current_epoch) + 1,
                "unavailable": True,
            }
        generator = getattr(loader, "generator", None)
        return {
            "format_version": 1,
            "boundary": "epoch",
            "next_epoch": int(self.current_epoch) + 1,
            "loader_class": type(loader).__name__,
            "sampler_class": type(getattr(loader, "sampler", None)).__name__,
            "batch_sampler_class": type(getattr(loader, "batch_sampler", None)).__name__,
            "num_batches": int(len(loader)),
            "uses_global_torch_rng": generator is None,
            "generator_state": generator.get_state() if generator is not None else None,
        }

    def load_training_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Resume an epoch-boundary checkpoint without resetting privacy state."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "model",
            "optimizer",
            "scheduler",
            "epoch",
            "trainer_state",
            "rng_state",
            "data_loader_state",
            "privacy_accountant",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(f"resume checkpoint is missing state: {missing}")

        unwrap_model(self.model).load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        dp_defense = self._defense("dp_sgd")
        accountant_state = checkpoint["privacy_accountant"]
        if accountant_state is not None:
            if dp_defense is None or getattr(dp_defense, "privacy_engine", None) is None:
                raise RuntimeError("resume checkpoint has privacy state but DP-SGD is absent")
            dp_defense.privacy_engine.accountant.load_state_dict(accountant_state)

        state = checkpoint["trainer_state"]
        self.global_step = int(state["global_step"])
        self.optimizer_step = int(state["optimizer_step"])
        self.current_epoch = int(state["current_epoch"])
        self.best_metric = float(state["best_metric"])
        self.manual_d_dp_passes = int(state.get("manual_d_dp_passes", 0))
        self._gradient_probe_last_optimizer_step = state.get(
            "gradient_probe_last_optimizer_step"
        )
        self.start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])
        self._restore_data_loader_state(checkpoint["data_loader_state"])
        return checkpoint

    def _restore_data_loader_state(self, state: dict[str, Any]) -> None:
        if bool(state.get("unavailable", False)):
            if self.data.get("train") is not None:
                raise RuntimeError("resume checkpoint lacks the active DataLoader state")
            return
        loader = self.data["train"]
        if state.get("boundary") != "epoch":
            raise RuntimeError("only epoch-boundary DataLoader resume is supported")
        if int(state.get("num_batches", -1)) != int(len(loader)):
            raise RuntimeError("resume DataLoader length mismatch")
        if int(state.get("next_epoch", -1)) != int(self.start_epoch):
            raise RuntimeError("resume DataLoader epoch mismatch")
        generator = getattr(loader, "generator", None)
        saved_generator = state.get("generator_state")
        if saved_generator is not None:
            if generator is None:
                raise RuntimeError("resume checkpoint has a generator state but active loader does not")
            generator.set_state(saved_generator)
        elif not bool(state.get("uses_global_torch_rng", False)):
            raise RuntimeError("resume checkpoint has no usable DataLoader RNG state")

    def _release_safe_checkpoint_metrics(
        self, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        local_cfg = (self.cfg.get("logging", {}) or {}).get("local", {}) or {}
        include_train = bool(local_cfg.get("include_private_train_metrics", True))
        include_gradients = bool(local_cfg.get("include_private_train_gradients", True))
        if include_train and include_gradients:
            return dict(metrics)

        def private_train_metric(key: str) -> bool:
            if key.startswith("train/"):
                return key != "train/lr"
            if key in {"Train Loss", "Train Accuracy"}:
                return True
            if key == "defense/trades_kl":
                return True
            return key.startswith(
                ("TRADES CE", "TRADES KL", "TRADES weighted KL", "TRADES total loss")
            )

        result = {}
        for key, value in metrics.items():
            text = str(key)
            if not include_gradients and text.startswith(
                ("A ", "B ", "C ", "D ", "H-C ", "S ", "S-C ")
            ):
                continue
            if not include_train and private_train_metric(text):
                continue
            result[key] = value
        return result

    def _move_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch[0], batch[1]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def _resolve_device(self) -> torch.device:
        requested = str(self.cfg.training.get("device", "auto")).lower()
        # On a GPU job a silent CPU fallback turns a ~4h run into a multi-day
        # one, so "auto" refuses to quietly downgrade unless explicitly allowed.
        # Set training.allow_cpu_fallback=true (or device=cpu) for CPU on purpose.
        allow_cpu_fallback = bool(self.cfg.training.get("allow_cpu_fallback", False))
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if allow_cpu_fallback:
                return torch.device("cpu")
            raise RuntimeError(
                "training.device=auto but CUDA is not available. This usually means "
                "a CPU-only torch wheel was installed on a GPU VM. Install a CUDA "
                "build of torch, or set training.allow_cpu_fallback=true / device=cpu "
                "to run on CPU on purpose."
            )
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda but torch.cuda.is_available() is False.")
        return torch.device(requested)


    def _componentwise_trades_step(self, x: torch.Tensor, y: torch.Tensor, dp_defense, trades_defense):
        """H-C: two clipping maps followed by exactly one Gaussian release."""
        if not hasattr(self.optimizer, "original_optimizer"):
            raise RuntimeError("H-C requires an Opacus DPOptimizer")

        self.optimizer.zero_grad(set_to_none=True)
        x_adv = trades_defense._generate_adversarial(self.model, x.detach())

        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()
        task_sums, task_preclip = clipped_grad_sums_from_current_samples(
            self.optimizer, dp_defense.max_grad_norm
        )

        self.optimizer.zero_grad(set_to_none=True)
        clean_probs = self._clean_probs_without_grad_sample_hooks(x)
        kl_loss = trades_defense.kl_loss(self.model, x_adv, clean_probs)
        kl_loss.backward()

        stats = self._apply_d_combined_update(dp_defense, task_sums, trades_defense.beta)
        expected = float(dp_defense.sensitivity_bound)
        if abs(float(stats["sensitivity"]) - expected) > 1e-10 * max(1.0, expected):
            raise RuntimeError(
                "H-C sensitivity mismatch: "
                f"Cg + beta*Cr={stats['sensitivity']}, configured Delta={expected}"
            )
        combined_grad_norm = param_grad_norm(list(self.optimizer.params))
        self.optimizer.original_optimizer.step()

        loss = ce_loss.detach() + trades_defense.beta * kl_loss.detach()
        weighted_kl = float(trades_defense.beta) * float(kl_loss.detach().item())
        metrics = {
            "TRADES CE": float(ce_loss.detach().item()),
            "TRADES KL": float(kl_loss.detach().item()),
            "TRADES weighted KL": float(weighted_kl),
            "TRADES total loss": float(loss.detach().item()),
            "TRADES weighted KL to total loss ratio": float(
                weighted_kl / (loss.detach().item() + 1e-12)
            ),
        }
        metrics.update(
            self._one_release_channel_metrics(
                "H-C", "TRADES", stats, task_preclip, combined_grad_norm
            )
        )
        return logits.detach(), loss, metrics

    def _auto_s_componentwise_trades_step(
        self, x: torch.Tensor, y: torch.Tensor, dp_defense, trades_defense
    ):
        """S-C: componentwise AUTO-S followed by one Gaussian release.

        The task branch and the beta-weighted TRADES branch are transformed
        independently. Their unaveraged per-example sums are combined before
        noise, averaging, accounting, momentum, and the optimizer step.
        """
        if not hasattr(self.optimizer, "original_optimizer"):
            raise RuntimeError("S-C requires an Opacus DPOptimizer")

        q_regularizer = float(dp_defense.componentwise_q_r)
        task_cap = 1.0 - q_regularizer
        robust_cap = q_regularizer
        gamma = float(dp_defense.auto_s_gamma)
        beta = float(trades_defense.beta)

        self.optimizer.zero_grad(set_to_none=True)
        x_adv = trades_defense._generate_adversarial(self.model, x.detach())

        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()
        task_sums, task_stats = auto_s_component_grad_sums_from_current_samples(
            self.optimizer,
            gamma=gamma,
            sensitivity_cap=task_cap,
            vector_scale=1.0,
        )

        self.optimizer.zero_grad(set_to_none=True)
        clean_probs = self._clean_probs_without_grad_sample_hooks(x)
        kl_loss = trades_defense.kl_loss(self.model, x_adv, clean_probs)
        kl_loss.backward()
        robust_sums, robust_stats = auto_s_component_grad_sums_from_current_samples(
            self.optimizer,
            gamma=gamma,
            sensitivity_cap=robust_cap,
            # h_i = beta * r_i is formed before its norm and denominator.
            vector_scale=beta,
            clear_state=False,
        )

        stats = self._apply_auto_s_componentwise_update(
            dp_defense,
            task_sums=task_sums,
            robust_sums=robust_sums,
            task_stats=task_stats,
            robust_stats=robust_stats,
        )
        combined_grad_norm = param_grad_norm(list(self.optimizer.params))
        # Momentum sees only the already privatized, combined update.
        self.optimizer.original_optimizer.step()

        loss = ce_loss.detach() + beta * kl_loss.detach()
        weighted_kl = beta * float(kl_loss.detach().item())
        metrics = {
            "TRADES CE": float(ce_loss.detach().item()),
            "TRADES KL": float(kl_loss.detach().item()),
            "TRADES weighted KL": weighted_kl,
            "TRADES total loss": float(loss.detach().item()),
            "TRADES weighted KL to total loss ratio": float(
                weighted_kl / (loss.detach().item() + 1e-12)
            ),
            "S-C q_regularizer": q_regularizer,
            "S-C gamma": gamma,
            "S-C task input norm mean": float(task_stats["input_norm_mean"]),
            "S-C task input norm max": float(task_stats["input_norm_max"]),
            "S-C task output norm mean": float(task_stats["output_norm_mean"]),
            "S-C task output norm max": float(task_stats["output_norm_max"]),
            "S-C robust input norm mean": float(robust_stats["input_norm_mean"]),
            "S-C robust input norm max": float(robust_stats["input_norm_max"]),
            "S-C robust output norm mean": float(robust_stats["output_norm_mean"]),
            "S-C robust output norm max": float(robust_stats["output_norm_max"]),
            "S-C task sensitivity cap": task_cap,
            "S-C robust sensitivity cap": robust_cap,
            "S-C sensitivity bound": 1.0,
            "S-C combined signal norm": float(stats["combined_signal_norm"]),
            "S-C combined DP grad norm": float(combined_grad_norm),
            "S-C noise norm": float(stats["noise_norm"]),
            "S-C expected noise norm": float(stats["expected_noise_norm"]),
            "S-C SNR": float(stats["snr"]),
            "S-C noise multiplier": float(stats["noise_multiplier"]),
            "S-C absolute noise std": float(stats["absolute_noise_std"]),
            "S-C accountant steps this batch": 1.0,
            "S-C optimizer steps this batch": 1.0,
        }
        return logits.detach(), loss, metrics

    def _apply_auto_s_componentwise_update(
        self,
        dp_defense,
        *,
        task_sums: list[torch.Tensor | None],
        robust_sums: list[torch.Tensor | None],
        task_stats: dict[str, float],
        robust_stats: dict[str, float],
    ) -> dict[str, float]:
        """Combine S-C branches and perform one sampled-Gaussian release."""
        optimizer = self.optimizer
        params = list(optimizer.params)
        q_regularizer = float(dp_defense.componentwise_q_r)
        task_cap = 1.0 - q_regularizer
        robust_cap = q_regularizer
        tolerance = 1e-6
        if float(task_stats["output_norm_max"]) > task_cap + tolerance:
            raise RuntimeError("S-C task contribution exceeded its sensitivity cap")
        if float(robust_stats["output_norm_max"]) > robust_cap + tolerance:
            raise RuntimeError("S-C robust contribution exceeded its sensitivity cap")
        if (
            float(task_stats["output_norm_max"])
            + float(robust_stats["output_norm_max"])
            > 1.0 + tolerance
        ):
            raise RuntimeError("S-C total contribution exceeded unit sensitivity")

        previous_noise_multiplier = optimizer.noise_multiplier
        previous_max_grad_norm = optimizer.max_grad_norm
        noise_multiplier = float(dp_defense.d_noise_multiplier)
        try:
            combined_signal_sq = 0.0
            parameter_count = 0
            for param, task_sum, robust_sum in zip(params, task_sums, robust_sums):
                parameter_count += param.numel()
                combined = None
                if task_sum is not None:
                    combined = task_sum.to(device=param.device, dtype=param.dtype).clone()
                if robust_sum is not None:
                    robust = robust_sum.to(device=param.device, dtype=param.dtype)
                    combined = robust.clone() if combined is None else combined.add_(robust)
                if combined is None:
                    combined = torch.zeros_like(param)
                combined_signal_sq += float(combined.detach().pow(2).sum().item())
                param.summed_grad = combined

            optimizer.max_grad_norm = 1.0
            optimizer.noise_multiplier = noise_multiplier
            optimizer.add_noise()
            optimizer.scale_grad()

            denominator = 1.0
            if getattr(optimizer, "loss_reduction", "mean") == "mean":
                # S-C forbids physical microbatching, so there is exactly one
                # logical Poisson sample and one Opacus denominator here.
                denominator = float(optimizer.expected_batch_size)
            noise_sq = 0.0
            for param in params:
                if param.grad is None or getattr(param, "summed_grad", None) is None:
                    continue
                signal_grad = param.summed_grad.detach().view_as(param.grad) / denominator
                noise_sq += float((param.grad.detach() - signal_grad).pow(2).sum().item())

            # One accountant event for the one observable Gaussian output.
            step_hook = getattr(optimizer, "step_hook", None)
            if step_hook is not None:
                step_hook(optimizer)
            self.manual_d_dp_passes += 1

            combined_signal_norm = combined_signal_sq**0.5 / denominator
            noise_norm = noise_sq**0.5
            expected_noise_norm = (
                noise_multiplier * (float(parameter_count) ** 0.5) / denominator
            )
            return {
                "combined_signal_norm": combined_signal_norm,
                "noise_norm": noise_norm,
                "expected_noise_norm": expected_noise_norm,
                "snr": combined_signal_norm / (noise_norm + 1e-12),
                "noise_multiplier": noise_multiplier,
                "absolute_noise_std": noise_multiplier,
                "sensitivity": 1.0,
                "aggregation_denominator": denominator,
            }
        finally:
            optimizer.noise_multiplier = previous_noise_multiplier
            optimizer.max_grad_norm = previous_max_grad_norm
            for param in params:
                if hasattr(param, "grad_sample"):
                    param.grad_sample = None
                if hasattr(param, "summed_grad"):
                    param.summed_grad = None


    def _apply_d_combined_update(self, dp_defense, task_sums, regularizer_weight: float) -> dict:
        """Strategy D: separate clipping per channel, one joint noise draw.

        The regularizer per-sample grads must already be in the optimizer. The
        combined per-sample contribution clip(g_i) + w*clip(r_i) has L2 norm at
        most max_grad_norm + |w|*c_regularizer_clip_norm, so a single Gaussian
        mechanism with that sensitivity privatizes the whole update in one
        accountant step per batch.
        """
        optimizer = self.optimizer
        params = list(optimizer.params)
        previous_noise_multiplier = optimizer.noise_multiplier
        previous_max_grad_norm = optimizer.max_grad_norm
        weight = float(regularizer_weight)
        sensitivity = float(dp_defense.max_grad_norm) + abs(weight) * float(dp_defense.c_regularizer_clip_norm)

        try:
            optimizer.max_grad_norm = float(dp_defense.c_regularizer_clip_norm)
            regularizer_preclip = per_sample_clip_diagnostics(optimizer)
            optimizer.clip_and_accumulate()
            accumulated_iterations = max(1, int(optimizer.accumulated_iterations))
            grad_scale = 1.0
            if getattr(optimizer, "loss_reduction", "mean") == "mean":
                grad_scale = float(optimizer.expected_batch_size) * float(accumulated_iterations)

            task_signal_sq = 0.0
            regularizer_signal_sq = 0.0
            combined_signal_sq = 0.0
            param_count = 0
            for param, task_sum in zip(params, task_sums):
                param_count += param.numel()
                regularizer_sum = (
                    param.summed_grad.detach() if getattr(param, "summed_grad", None) is not None else None
                )
                combined = None
                if task_sum is not None:
                    task_signal_sq += float(task_sum.pow(2).sum().item())
                    combined = task_sum.to(device=param.device, dtype=param.dtype).clone()
                if regularizer_sum is not None:
                    regularizer_signal_sq += float(regularizer_sum.pow(2).sum().item())
                    scaled = regularizer_sum.mul(weight)
                    combined = scaled if combined is None else combined.add_(scaled)
                if combined is None:
                    combined = torch.zeros_like(param)
                combined_signal_sq += float(combined.pow(2).sum().item())
                param.summed_grad = combined

            optimizer.max_grad_norm = sensitivity
            optimizer.noise_multiplier = float(dp_defense.d_noise_multiplier)
            optimizer.add_noise()
            optimizer.scale_grad()

            noise_sq = 0.0
            for param in params:
                if param.grad is None or getattr(param, "summed_grad", None) is None:
                    continue
                signal_grad = param.summed_grad.detach().view_as(param.grad) / grad_scale
                noise_sq += float((param.grad.detach() - signal_grad).pow(2).sum().item())

            # The accountant hook needs grad_sample still present, so fire it
            # before clearing the per-sample state. param.grad stays assigned
            # for the caller's optimizer step.
            step_hook = getattr(optimizer, "step_hook", None)
            if step_hook is not None:
                step_hook(optimizer)
            self.manual_d_dp_passes += 1

            for param in params:
                if hasattr(param, "grad_sample"):
                    param.grad_sample = None
                if hasattr(param, "summed_grad"):
                    param.summed_grad = None

            task_signal_norm = (task_signal_sq**0.5) / grad_scale
            regularizer_signal_norm = (regularizer_signal_sq**0.5) / grad_scale
            combined_signal_norm = (combined_signal_sq**0.5) / grad_scale
            noise_norm = noise_sq**0.5
            expected_noise_norm = (
                float(dp_defense.d_noise_multiplier) * sensitivity * (float(param_count) ** 0.5) / grad_scale
            )
            return {
                "task_signal_norm": task_signal_norm,
                "regularizer_signal_norm": regularizer_signal_norm,
                "regularizer_contrib_norm": abs(weight) * regularizer_signal_norm,
                "combined_signal_norm": combined_signal_norm,
                "noise_norm": noise_norm,
                "expected_noise_norm": expected_noise_norm,
                "snr": combined_signal_norm / (noise_norm + 1e-12),
                "sensitivity": sensitivity,
                "regularizer_preclip": regularizer_preclip,
            }
        finally:
            optimizer.noise_multiplier = previous_noise_multiplier
            optimizer.max_grad_norm = previous_max_grad_norm

    def _apply_auto_s_update(self, dp_defense) -> dict[str, float]:
        """Apply one AUTO-S sampled-Gaussian release to current grad samples."""
        optimizer = self.optimizer
        if not hasattr(optimizer, "original_optimizer"):
            raise RuntimeError("AUTO-S requires an Opacus DPOptimizer")
        previous_max_grad_norm = optimizer.max_grad_norm
        try:
            stats = auto_s_grad_sums_from_current_samples(
                optimizer, gamma=float(dp_defense.auto_s_gamma)
            )
            if stats["output_norm_max"] >= 1.0:
                raise RuntimeError("AUTO-S per-example contribution norm must be strictly below one")
            optimizer.max_grad_norm = 1.0
            optimizer.add_noise()
            optimizer.scale_grad()
            step_hook = getattr(optimizer, "step_hook", None)
            if step_hook is not None:
                step_hook(optimizer)
            optimizer.original_optimizer.step()
            stats["sensitivity"] = 1.0
            stats["noise_multiplier"] = float(optimizer.noise_multiplier)
            stats["absolute_noise_std"] = float(optimizer.noise_multiplier)
            return stats
        finally:
            optimizer.max_grad_norm = previous_max_grad_norm
            for param in optimizer.params:
                if hasattr(param, "grad_sample"):
                    param.grad_sample = None
                if hasattr(param, "summed_grad"):
                    param.summed_grad = None


    def _one_release_channel_metrics(
        self,
        prefix: str,
        regularizer_label: str,
        stats: dict,
        task_preclip: dict[str, float],
        combined_grad_norm: float,
    ) -> dict[str, float]:
        contrib = float(stats["regularizer_contrib_norm"])
        metrics = {
            f"{prefix} task signal norm": float(stats["task_signal_norm"]),
            f"{prefix} {regularizer_label} signal norm": float(stats["regularizer_signal_norm"]),
            f"{prefix} {regularizer_label} signal contrib norm": contrib,
            f"{prefix} combined signal norm": float(stats["combined_signal_norm"]),
            f"{prefix} noise norm": float(stats["noise_norm"]),
            f"{prefix} expected noise norm": float(stats["expected_noise_norm"]),
            f"{prefix} combined DP grad norm": float(combined_grad_norm),
            f"{prefix} SNR": float(stats["snr"]),
            f"{prefix} sensitivity": float(stats["sensitivity"]),
            f"{prefix} {regularizer_label} signal to task ratio": contrib
            / (float(stats["task_signal_norm"]) + 1e-12),
            f"{prefix} {regularizer_label} signal to combined ratio": contrib
            / (float(stats["combined_signal_norm"]) + 1e-12),
        }
        metrics.update(preclip_metric_names(f"{prefix} task", task_preclip))
        metrics.update(
            preclip_metric_names(
                f"{prefix} {regularizer_label}", stats["regularizer_preclip"]
            )
        )
        return metrics


    def _clean_probs_without_grad_sample_hooks(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        self._set_grad_sample_hooks(False)
        try:
            with torch.no_grad():
                probs = F.softmax(self.model(x), dim=1)
        finally:
            self._set_grad_sample_hooks(True)
            if was_training:
                self.model.train()
        return probs.detach()

    def _set_grad_sample_hooks(self, enabled: bool) -> None:
        method_name = "enable_hooks" if enabled else "disable_hooks"
        method = getattr(self.model, method_name, None)
        if callable(method):
            method()

    def _defense(self, name: str):
        for defense in self.defenses:
            if getattr(defense, "name", None) == name:
                return defense
        return None


    def _gradient_probe_cfg(self):
        diagnostics = self.cfg.get("diagnostics", None) or {}
        probe_cfg = diagnostics.get("gradient_probe", None) or {}
        if not bool(probe_cfg.get("enabled", False)):
            return None
        return probe_cfg

    def _prepare_gradient_probe(self) -> None:
        """Cache a deterministic held-out public subset without consuming train RNG."""
        probe_cfg = self._gradient_probe_cfg()
        if probe_cfg is None:
            return
        if self._defense("dp_sgd") is None:
            raise ValueError("diagnostics.gradient_probe requires enabled DP-SGD")
        if self._defense("trades") is None and self._trades_probe_config() is None:
            raise ValueError("gradient diagnostics require a TRADES config for the public counterfactual")
        source = str(probe_cfg.get("source", "test")).lower()
        if source not in {"val", "test"}:
            raise ValueError("gradient diagnostics source must be 'val' or 'test'")
        if source == "test" and bool((self.cfg.get("protocol", {}) or {}).get("test_locked", False)):
            raise ValueError("official test-set gradient probes are forbidden while protocol.test_locked=true")
        loader = self.data.get(source)
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            raise ValueError(f"gradient probe requires data[{source!r}] with a map-style dataset")
        requested_size = int(probe_cfg.get("size", 128))
        if requested_size <= 0:
            raise ValueError("diagnostics.gradient_probe.size must be positive")
        manifest_path = Path(__file__).resolve().parents[3] / "manifests/data_split.json"
        if not manifest_path.is_file():
            manifest_path = Path("manifests/data_split.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        size = int(manifest["probe_size"])
        seed = int(probe_cfg.get("seed", 20260814))
        if requested_size != size or seed != int(manifest["probe_seed"]):
            raise ValueError("gradient probe size and seed must match the locked manifest")
        parent_indices = getattr(dataset, "indices", None)
        if parent_indices is None:
            raise ValueError("gradient probe requires the locked public-dev Subset")
        if [int(value) for value in parent_indices] != [
            int(value) for value in manifest["public_dev_source_indices"]
        ]:
            raise ValueError("public-development order differs from the locked manifest")
        indices = [int(value) for value in manifest["probe_subset_positions"]]
        samples = [dataset[int(index)] for index in indices]
        try:
            x = torch.stack([sample[0] for sample in samples]).cpu()
            y = torch.as_tensor([int(sample[1]) for sample in samples], dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise ValueError("gradient probe dataset must yield tensor/label pairs") from exc
        self._gradient_probe_batch = (x, y)
        self.logger.log_metrics(
            {
                "probe/setup/enabled": 1.0,
                "probe/setup/source_is_test": 1.0 if source == "test" else 0.0,
                "probe/setup/size": float(size),
                "probe/setup/microbatch_size": float(probe_cfg.get("microbatch_size", 4)),
                "probe/setup/every_n_steps": float(probe_cfg.get("every_n_steps", 200)),
                # This is the public diagnostic seed, not the DP-noise seed.
                "probe/setup/seed": float(seed),
            },
            step=0,
        )

    def _configure_gradient_probe_schedule(self) -> None:
        probe_cfg = self._gradient_probe_cfg()
        self._gradient_probe_target_steps = set()
        if probe_cfg is None:
            return
        fractions = probe_cfg.get("snapshot_fractions", None)
        if fractions is None:
            return
        epochs = max(0, int(self.cfg.training.get("epochs", 1)))
        per_epoch = len(self.data["train"])
        max_batches = self.cfg.training.get("max_train_batches", None)
        if max_batches is not None:
            per_epoch = min(per_epoch, int(max_batches))
        total_steps = epochs * per_epoch
        parsed = [float(value) for value in fractions]
        if not parsed or any(value < 0.0 or value > 1.0 for value in parsed):
            raise ValueError("gradient-probe snapshot fractions must be in [0, 1]")
        self._gradient_probe_target_steps = {
            int(round(value * total_steps)) for value in parsed
        }
        if 0.0 in parsed:
            self._gradient_probe_target_steps.add(0)
        self.logger.log_metrics(
            {
                "probe/setup/total_optimizer_steps": float(total_steps),
                "probe/setup/snapshot_count": float(len(self._gradient_probe_target_steps)),
                **{
                    f"probe/setup/snapshot_step_{index}": float(step)
                    for index, step in enumerate(sorted(self._gradient_probe_target_steps))
                },
            },
            step=0,
        )

    def _maybe_run_gradient_probe(self) -> None:
        probe_cfg = self._gradient_probe_cfg()
        if probe_cfg is None:
            return
        if self._gradient_probe_target_steps:
            if self.optimizer_step in self._gradient_probe_target_steps:
                self._run_gradient_probe(trigger="snapshot")
            return
        every = int(probe_cfg.get("every_n_steps", 200))
        if every > 0 and self.optimizer_step > 0 and self.optimizer_step % every == 0:
            self._run_gradient_probe(trigger="cadence")

    def _run_gradient_probe(self, trigger: str) -> dict[str, float]:
        probe_cfg = self._gradient_probe_cfg()
        if probe_cfg is None or self._gradient_probe_batch is None:
            return {}
        # A cadence hit on the final step and the epoch-end hook refer to the
        # exact same model state and public batch; do not pay for it twice.
        if self._gradient_probe_last_optimizer_step == self.optimizer_step:
            return self._last_gradient_probe_metrics

        dp_defense = self._defense("dp_sgd")
        trades_defense = self._defense("trades")
        trades_cfg = self._trades_probe_config()
        if trades_cfg is None and trades_defense is None:
            raise ValueError("gradient probe has no TRADES configuration")
        if trades_cfg is None:
            trades_cfg = {}

        def trades_value(name: str, default):
            if trades_defense is not None:
                return getattr(trades_defense, name)
            return trades_cfg.get(name, default)
        try:
            noise = dp_defense.gradient_probe_noise_multipliers(self.cfg)
            loss_reduction = str(getattr(self.optimizer, "loss_reduction", "mean")).lower()
            if loss_reduction != "mean":
                raise ValueError(
                    "gradient-probe noise formulas require DPOptimizer loss_reduction='mean'; "
                    f"got {loss_reduction!r}"
                )
            expected_batch_size = int(
                getattr(self.optimizer, "expected_batch_size", self.cfg.data.get("batch_size", 1))
            )
            probe_x = self._gradient_probe_batch[0].to(self.device, non_blocking=True)
            probe_y = self._gradient_probe_batch[1].to(self.device, non_blocking=True)
            metrics = run_gradient_probe(
                self.model,
                probe_x,
                probe_y,
                beta=float(trades_value("beta", 3.0)),
                joint_clip_norm=float(dp_defense.max_grad_norm),
                task_clip_norm=float(dp_defense.max_grad_norm),
                robust_clip_norm=float(dp_defense.c_regularizer_clip_norm),
                noise_multiplier_a=float(noise["A"]),
                noise_multiplier_c_task=float(noise["C_task"]),
                noise_multiplier_c_robust=float(noise["C_regularizer"]),
                noise_multiplier_d=float(noise["D"]),
                expected_batch_size=expected_batch_size,
                epsilon=float(trades_value("epsilon", 8.0 / 255.0)),
                step_size=float(trades_value("step_size", 2.0 / 255.0)),
                steps=int(trades_value("steps", 10)),
                mean=tuple(trades_value("mean", tuple(self.cfg.data.mean))),
                std=tuple(trades_value("std", tuple(self.cfg.data.std))),
                microbatch_size=int(probe_cfg.get("microbatch_size", 4)),
                seed=int(probe_cfg.get("seed", 20260814)),
                auto_s_gamma=float(getattr(dp_defense, "auto_s_gamma", 0.01)),
                auto_s_componentwise_q=float(
                    getattr(dp_defense, "componentwise_q_r", 0.5)
                ),
                learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            )
            trigger_code = {
                "start": 0.0,
                "cadence": 1.0,
                "epoch_end": 2.0,
                "snapshot": 3.0,
            }.get(trigger, -1.0)
            attained_epsilon = dp_defense.get_epsilon()
            strategy = (
                "CE"
                if trades_defense is None
                else str(
                    trades_defense.cfg.get(
                        "dp_strategy", trades_defense.cfg.get("strategy", "A")
                    )
                ).upper()
            )
            metrics.update(
                {
                    "probe/meta/trigger_code": trigger_code,
                    "probe/meta/global_step": float(self.global_step),
                    "probe/meta/optimizer_step": float(self.optimizer_step),
                    "probe/meta/epoch": float(self.current_epoch),
                    "probe/meta/active_strategy_code": {
                        "A": 1.0,
                        "B": 2.0,
                        "C": 3.0,
                        "D": 4.0,
                        "CE": 5.0,
                    }.get(strategy, 0.0),
                    "probe/meta/dp_target_epsilon": float(dp_defense.epsilon),
                    "probe/meta/dp_target_delta": float(dp_defense.delta),
                    "probe/meta/dp_attained_epsilon": float(attained_epsilon or 0.0),
                    "probe/meta/dp_active_noise_multiplier": float(
                        dp_defense.active_noise_multiplier or 0.0
                    ),
                    "probe/meta/dp_loss_reduction_mean": 1.0,
                    "probe/meta/counterfactual_noise_matched_target_budget": float(
                        noise["matched_target_budget"]
                    ),
                    "probe/error": 0.0,
                }
            )
            self.logger.log_metrics(metrics, step=self.global_step)
            self._last_gradient_probe_metrics = metrics
            self._gradient_probe_last_optimizer_step = self.optimizer_step
            return metrics
        except Exception as exc:
            error_metrics = {
                "probe/error": 1.0,
                "probe/meta/global_step": float(self.global_step),
                "probe/meta/optimizer_step": float(self.optimizer_step),
                "probe/meta/epoch": float(self.current_epoch),
            }
            self.logger.log_metrics(error_metrics, step=self.global_step)
            print(f"public gradient probe failed:\n{traceback.format_exc()}", file=sys.stderr)
            if isinstance(exc, GradientProbeStateRestorationError):
                raise
            if bool(probe_cfg.get("fail_on_error", False)):
                raise
            return error_metrics

    @staticmethod
    def _without_private_gradient_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """Drop strategy telemetry derived from a private training batch."""
        return {
            key: value
            for key, value in metrics.items()
            if not str(key).startswith(
                ("A ", "B ", "C ", "D ", "H-C ", "S ", "S-C ")
            )
        }

    def _setup_metrics(self) -> dict[str, float]:
        trades = self._defense("trades")
        total = sum(param.numel() for param in self.model.parameters())
        trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        adaptation = self.cfg.get("adaptation", {})
        train_last_block = bool(adaptation.get("train_last_block", False))
        metrics = {
            "CUDA available": 1.0 if torch.cuda.is_available() else 0.0,
            "Device is CUDA": 1.0 if self.device.type == "cuda" else 0.0,
            "TRADES enabled": 1.0 if trades else 0.0,
            "Trainable parameter count": float(trainable),
            "Total parameter count": float(total),
            "Trainable parameter fraction": float(trainable / max(1, total)),
            "Train last block": 1.0 if train_last_block else 0.0,
            "Batch size": float(self.cfg.data.get("batch_size", 0)),
            "Number of epochs": float(self.cfg.training.get("epochs", 0)),
            "Public gradient probe enabled": 1.0 if self._gradient_probe_cfg() is not None else 0.0,
        }
        if trades:
            metrics.update(
                {
                    "TRADES beta": float(getattr(trades, "beta", 0.0)),
                    "TRADES eps": float(getattr(trades, "epsilon", 0.0)),
                    "TRADES step_size": float(getattr(trades, "step_size", 0.0)),
                    "TRADES num_steps": float(getattr(trades, "steps", 0)),
                }
            )
            trades_strategy = str(trades.cfg.get("dp_strategy", "A")).upper()
            metrics["TRADES DP strategy code"] = {"A": 1.0, "C": 3.0}.get(
                trades_strategy, 0.0
            )
        return metrics

    def _trades_probe_config(self):
        for item in self.cfg.get("defenses", []) or []:
            if str(item.get("name", "")).lower() == "trades":
                return item
        return None

    def _epoch_metrics(self, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> dict[str, float]:
        metrics = {
            "Train Loss": float(train_metrics.get("train/loss", 0.0)),
            "Train Accuracy": float(train_metrics.get("train/acc", 0.0) * 100.0),
            "Learning Rate": float(train_metrics.get("train/lr", 0.0)),
        }
        if "val/clean_acc" in val_metrics:
            metrics["Validation Accuracy"] = float(val_metrics["val/clean_acc"] * 100.0)
        if "test/clean_acc" in val_metrics:
            metrics["Test Accuracy"] = float(val_metrics["test/clean_acc"] * 100.0)
        return metrics
