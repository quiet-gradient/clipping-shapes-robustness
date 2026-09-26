from __future__ import annotations

import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from trustml.attacks.registry import build_attacks
from trustml.core.checkpointing import save_checkpoint, unwrap_model
from trustml.core.seed import capture_rng_state, restore_rng_state
from trustml.defenses.combined import collect_defense_metrics
from trustml.evaluation.clean import evaluate_clean
from trustml.evaluation.privacy import run_privacy_suite
from trustml.evaluation.robustness import evaluate_robustness
from trustml.storage.artifacts import ArtifactStore
from trustml.training.dp_diagnostics import (
    auto_s_component_grad_sums_from_current_samples,
    auto_s_grad_sums_from_current_samples,
    assign_combined_dp_grads,
    clipped_grad_sums_from_current_samples,
    dp_grads_from_current_samples,
    grad_norm,
    param_grad_norm,
    per_sample_clip_diagnostics,
    preclip_metric_names,
    trades_a_raw_grad_diagnostics,
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
        self.manual_task_dp_passes = 0
        self.manual_regularizer_dp_passes = 0
        self.manual_d_dp_passes = 0
        self._raw_igr_buffer: dict | None = None
        self._gradient_probe_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_gradient_probe_metrics: dict[str, float] = {}
        self._gradient_probe_last_optimizer_step: int | None = None
        self._gradient_probe_target_steps: set[int] = set()
        self.artifact_store = ArtifactStore.from_config(cfg)

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
        self.logger.log_metrics(self._legacy_setup_metrics(), step=0)
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
            final_metrics.update(self._legacy_epoch_metrics(train_metrics, val_metrics))
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
        requires_input_grad = any(getattr(defense, "requires_input_grad", False) for defense in self.defenses)
        dp_defense = self._defense("dp_sgd")
        trades_defense = self._defense("trades")
        igr_defense = self._defense("igr")
        private_gradient_metrics = self._private_train_gradient_metrics_enabled()
        igr_strategy = str(igr_defense.cfg.get("dp_strategy", igr_defense.cfg.get("strategy", "A"))).upper() if igr_defense else "A"
        trades_strategy = (
            str(trades_defense.cfg.get("dp_strategy", trades_defense.cfg.get("strategy", "C"))).upper()
            if trades_defense
            else "C"
        )
        dp_strategy = getattr(dp_defense, "dp_strategy", None)
        use_auto_s = dp_defense is not None and bool(getattr(dp_defense, "auto_s", False))
        use_legacy_b_trades = dp_defense is not None and trades_defense is not None and trades_strategy == "B"
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
        use_legacy_c_trades = (
            dp_defense is not None
            and trades_defense is not None
            and getattr(dp_defense, "regularizer_name", None) == "trades"
            and dp_strategy == "C"
            and not bool(getattr(dp_defense, "componentwise_one_release", False))
        )
        use_legacy_d_trades = (
            dp_defense is not None
            and trades_defense is not None
            and getattr(dp_defense, "regularizer_name", None) == "trades"
            and dp_strategy == "D"
        )
        use_legacy_b_igr = dp_defense is not None and igr_defense is not None and igr_strategy == "B"
        use_legacy_c_igr = (
            dp_defense is not None
            and igr_defense is not None
            and getattr(dp_defense, "regularizer_name", None) == "igr"
            and dp_strategy == "C"
        )
        use_legacy_d_igr = (
            dp_defense is not None
            and igr_defense is not None
            and getattr(dp_defense, "regularizer_name", None) == "igr"
            and dp_strategy == "D"
        )
        loader = self.data["train"]
        physical_batch_size = self.cfg.training.get("physical_batch_size", None)
        use_microbatch = (
            dp_defense is not None
            and physical_batch_size is not None
            and int(physical_batch_size) > 0
            and int(physical_batch_size) < int(self.cfg.data.get("batch_size", 0))
        )
        if use_microbatch and (
            use_auto_s_componentwise_trades
            or use_componentwise_trades
            or use_legacy_c_trades
            or use_legacy_c_igr
            or use_legacy_d_trades
            or use_legacy_d_igr
            or use_auto_s
        ):
            raise ValueError(
                "Manual componentwise/AUTO-S DP paths do not support "
                "physical_batch_size < batch_size. Use physical_batch_size >= batch_size."
            )
        if use_microbatch and trades_defense is not None and float(getattr(trades_defense, "beta", 0.0)) > 0.0:
            # Legacy A path: TRADES makes two forward passes -> 2x memory.
            physical_batch_size = max(1, int(physical_batch_size) // 2)
        if use_microbatch:
            from opacus.utils.batch_memory_manager import BatchMemoryManager

            loader_ctx = BatchMemoryManager(
                data_loader=loader,
                max_physical_batch_size=int(physical_batch_size),
                optimizer=self.optimizer,
            )
        else:
            loader_ctx = nullcontext(loader)

        with loader_ctx as loader:
            iterator = enumerate(loader)
            iterator = tqdm(iterator, total=len(loader), desc=f"epoch {epoch}", leave=False)

            for batch_idx, batch in iterator:
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                x, y = self._move_batch(batch)
                if requires_input_grad:
                    x = x.detach().requires_grad_(True)

                if use_auto_s_componentwise_trades:
                    logits, loss, batch_legacy_metrics = self._auto_s_componentwise_trades_step(
                        x, y, dp_defense, trades_defense
                    )
                elif use_componentwise_trades:
                    logits, loss, batch_legacy_metrics = self._componentwise_trades_step(
                        x, y, dp_defense, trades_defense
                    )
                elif use_legacy_c_trades:
                    logits, loss, batch_legacy_metrics = self._legacy_c_trades_step(x, y, dp_defense, trades_defense)
                elif use_legacy_d_trades:
                    logits, loss, batch_legacy_metrics = self._legacy_d_trades_step(x, y, dp_defense, trades_defense)
                elif use_legacy_b_trades:
                    logits, loss, batch_legacy_metrics = self._legacy_b_trades_step(x, y, trades_defense)
                elif use_legacy_c_igr:
                    logits, loss, batch_legacy_metrics = self._legacy_c_igr_step(x, y, dp_defense, igr_defense)
                elif use_legacy_d_igr:
                    logits, loss, batch_legacy_metrics = self._legacy_d_igr_step(x, y, dp_defense, igr_defense)
                elif use_legacy_b_igr:
                    logits, loss, batch_legacy_metrics = self._legacy_b_igr_step(x, y, igr_defense)
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

                    if private_gradient_metrics and not use_auto_s:
                        a_trades_metrics = self._legacy_a_trades_diagnostics(dp_defense, trades_defense)
                    else:
                        a_trades_metrics = {}
                        # TRADES keeps these graph-attached tensors solely for
                        # the legacy private-batch diagnostic.  Do not retain a
                        # training graph when that diagnostic is disabled.
                        if trades_defense is not None:
                            trades_defense.last_ce_loss = None
                            trades_defense.last_kl_loss = None
                    loss.backward()
                    batch_legacy_metrics = collect_defense_metrics(self.defenses)
                    batch_legacy_metrics.update(a_trades_metrics)
                    if a_trades_metrics:
                        batch_legacy_metrics.update(
                            preclip_metric_names("A combined", per_sample_clip_diagnostics(self.optimizer))
                        )
                    if use_auto_s:
                        auto_s_stats = self._apply_auto_s_update(dp_defense)
                        if private_gradient_metrics:
                            batch_legacy_metrics.update(
                                {f"S {key}": float(value) for key, value in auto_s_stats.items()}
                            )
                    elif (
                        private_gradient_metrics
                        and dp_defense is not None
                        and igr_defense is not None
                        and igr_strategy == "A"
                    ):
                        previous_hook, hook_state = self._install_dp_grad_norm_step_hook()
                        try:
                            self.optimizer.step()
                        finally:
                            self.optimizer.step_hook = previous_hook
                        if hook_state["called"]:
                            batch_legacy_metrics["A combined DP grad norm"] = float(hook_state["dp_grad_norm"])
                    else:
                        self.optimizer.step()

                if not private_gradient_metrics:
                    # Legacy A/B/C/D instrumentation was computed from the
                    # private training batch.  Such telemetry is an additional
                    # non-DP output channel, so it is local-debug opt-in only.
                    # Public ``probe/*`` diagnostics are logged separately.
                    batch_legacy_metrics = self._without_private_gradient_metrics(batch_legacy_metrics)

                batch_size = int(y.numel())
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int(logits.detach().argmax(dim=1).eq(y).sum().item())
                total_seen += batch_size
                self.global_step += 1

                optimizer_step_skipped = bool(getattr(self.optimizer, "_is_last_step_skipped", False))
                if not optimizer_step_skipped:
                    self.optimizer_step += 1
                    self._maybe_run_gradient_probe()

                if batch_legacy_metrics:
                    self.logger.log_metrics(batch_legacy_metrics, step=self.global_step - 1)

                if log_every > 0 and self.global_step % log_every == 0:
                    metrics = {
                        "train/batch_loss": float(loss.detach().item()),
                        "train/batch_acc": total_correct / max(1, total_seen),
                        "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                    }
                    metrics.update(batch_legacy_metrics)
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
                    manual_task_passes=self.manual_task_dp_passes,
                    manual_regularizer_passes=self.manual_regularizer_dp_passes,
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

        # Robustness (PGD) eval stays separate from the MIA suite: it perturbs
        # the test set rather than scoring a member/non-member split. Like the
        # privacy attacks it is an optional diagnostic -- a failure is marked
        # eval/robustness_error=1 and must not cost the run its other metrics.
        attacks = build_attacks(self.cfg.get("attacks", None), data_cfg=self.cfg.data)
        if attacks and evaluation_split in self.data:
            try:
                metrics.update(
                    evaluate_robustness(
                        self.model, self.data[evaluation_split], attacks, self.device
                    )
                )
            except Exception:
                print(f"robustness evaluation failed:\n{traceback.format_exc()}", file=sys.stderr)
                metrics["eval/robustness_error"] = 1.0

        # Adversarial-loss MIA, OSLO, and offline RMIA over the seeded split;
        # each self-guards and isolates its own failures (eval/<name>_error=1).
        if not test_locked:
            metrics.update(run_privacy_suite(self.model, self.data, self.cfg, self.device))

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
        run_dir = getattr(self.logger, "output_dir", None)
        if run_dir is not None:
            self.artifact_store.register_checkpoint(
                experiment_name=str(self.cfg.experiment.name),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                alias=Path(name).stem,
                epoch=epoch if epoch is not None else self.current_epoch,
                metrics=checkpoint_metrics,
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
                "manual_task_dp_passes": int(self.manual_task_dp_passes),
                "manual_regularizer_dp_passes": int(self.manual_regularizer_dp_passes),
                "manual_d_dp_passes": int(self.manual_d_dp_passes),
                "gradient_probe_last_optimizer_step": self._gradient_probe_last_optimizer_step,
            },
            "rng_state": capture_rng_state(),
            "data_loader_state": self._capture_data_loader_state(),
            "privacy_accountant": accountant_state,
            "wandb_run_id": getattr(self.logger, "wandb_run_id", None),
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

        saved_run_id = checkpoint.get("wandb_run_id")
        active_run_id = getattr(self.logger, "wandb_run_id", None)
        if saved_run_id and active_run_id and str(saved_run_id) != str(active_run_id):
            raise RuntimeError(
                f"resume W&B run ID mismatch: {active_run_id!r} != {saved_run_id!r}"
            )

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
        self.manual_task_dp_passes = int(state.get("manual_task_dp_passes", 0))
        self.manual_regularizer_dp_passes = int(
            state.get("manual_regularizer_dp_passes", 0)
        )
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
            if key in {"Train Loss", "Train Accuracy", "IGR penalty", "defense/igr_penalty"}:
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

    def _legacy_b_igr_step(self, x: torch.Tensor, y: torch.Tensor, igr_defense):
        if not hasattr(self.optimizer, "params"):
            raise RuntimeError("Legacy B-IGR path requires an Opacus DPOptimizer.")
        params = list(self.optimizer.params)
        if self._raw_igr_buffer is None or len(self._raw_igr_buffer["grads"]) != len(params):
            self._raw_igr_buffer = {"grads": [None for _ in params], "seen": 0}

        raw_grads, igr_pen = self._raw_igr_grads(x, y, igr_defense, params)
        # The buffer accumulates across BatchMemoryManager microbatches; the
        # step hook only fires (and the buffer is cleared) on real steps.
        self._accumulate_raw_grad_buffer(self._raw_igr_buffer, raw_grads, int(y.numel()))

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()

        previous_hook, hook_state = self._install_raw_igr_step_hook(
            params=params,
            raw_buffer=self._raw_igr_buffer,
            igr_lambda=float(igr_defense.lambda_igr),
            normalize_raw_igr=bool(igr_defense.cfg.get("b_normalize_raw_igr", False)),
            raw_igr_target_ratio=float(igr_defense.cfg.get("b_raw_igr_target_ratio", 0.1)),
        )
        try:
            self.optimizer.step()
        finally:
            self.optimizer.step_hook = previous_hook
        if hook_state["called"]:
            self._clear_raw_grad_buffer(self._raw_igr_buffer)

        loss = ce_loss.detach() + float(igr_defense.lambda_igr) * igr_pen.detach()
        metrics = {
            "IGR penalty": float(igr_pen.detach().item()),
            "B DP grad norm": float(hook_state["dp_grad_norm"]),
            "B raw IGR grad norm": float(hook_state["raw_igr_grad_norm"]),
            "B raw IGR contrib norm": float(hook_state["raw_igr_contrib_norm"]),
            "B raw IGR to task ratio": float(hook_state["raw_igr_to_task_ratio"]),
            "B raw IGR to total ratio": float(hook_state["raw_igr_to_total_ratio"]),
            "B raw IGR scale": float(hook_state["raw_igr_scale"]),
            "B total grad norm": float(hook_state["total_grad_norm"]),
        }
        return logits.detach(), loss, metrics

    def _legacy_b_trades_step(self, x: torch.Tensor, y: torch.Tensor, trades_defense):
        """Strategy B for TRADES: DP task gradient + raw (non-private) beta*KL gradient.

        Not a DP scheme for the full update -- the raw KL term skips
        clipping/noise -- but it upper-bounds what a separate robust channel can
        achieve and isolates the cost of privatizing it (gap to strategy D).
        The KL term uses no labels, so a public-data variant of this step would
        be honestly DP; this in-batch version is the diagnostic.
        """
        if not hasattr(self.optimizer, "params"):
            raise RuntimeError("Legacy B-TRADES path requires an Opacus DPOptimizer.")
        params = list(self.optimizer.params)
        if self._raw_igr_buffer is None or len(self._raw_igr_buffer["grads"]) != len(params):
            self._raw_igr_buffer = {"grads": [None for _ in params], "seen": 0}

        x_detached = x.detach()
        x_adv = trades_defense._generate_adversarial(self.model, x_detached)
        clean_probs = self._clean_probs_without_grad_sample_hooks(x_detached)

        # Raw KL gradient with grad-sample hooks disabled, as in legacy B-IGR.
        for param in params:
            param.grad = None
        self._set_grad_sample_hooks(False)
        try:
            kl_loss = trades_defense.kl_loss(self.model, x_adv, clean_probs)
            kl_loss.backward()
            raw_grads = [param.grad.detach().clone() if param.grad is not None else None for param in params]
        finally:
            self._set_grad_sample_hooks(True)
            for param in params:
                param.grad = None
        self._accumulate_raw_grad_buffer(self._raw_igr_buffer, raw_grads, int(y.numel()))

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x_detached)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()

        previous_hook, hook_state = self._install_raw_igr_step_hook(
            params=params,
            raw_buffer=self._raw_igr_buffer,
            igr_lambda=float(trades_defense.beta),
            normalize_raw_igr=bool(trades_defense.cfg.get("b_normalize_raw", False)),
            raw_igr_target_ratio=float(trades_defense.cfg.get("b_raw_target_ratio", 0.1)),
        )
        try:
            self.optimizer.step()
        finally:
            self.optimizer.step_hook = previous_hook
        if hook_state["called"]:
            self._clear_raw_grad_buffer(self._raw_igr_buffer)

        loss = ce_loss.detach() + trades_defense.beta * kl_loss.detach()
        weighted_kl = float(trades_defense.beta) * float(kl_loss.detach().item())
        metrics = {
            "TRADES CE": float(ce_loss.detach().item()),
            "TRADES KL": float(kl_loss.detach().item()),
            "TRADES weighted KL": float(weighted_kl),
            "TRADES total loss": float(loss.detach().item()),
            "TRADES weighted KL to total loss ratio": float(weighted_kl / (loss.detach().item() + 1e-12)),
            "B DP grad norm": float(hook_state["dp_grad_norm"]),
            "B raw TRADES grad norm": float(hook_state["raw_igr_grad_norm"]),
            "B raw TRADES contrib norm": float(hook_state["raw_igr_contrib_norm"]),
            "B raw TRADES to task ratio": float(hook_state["raw_igr_to_task_ratio"]),
            "B raw TRADES to total ratio": float(hook_state["raw_igr_to_total_ratio"]),
            "B raw TRADES scale": float(hook_state["raw_igr_scale"]),
            "B total grad norm": float(hook_state["total_grad_norm"]),
        }
        return logits.detach(), loss, metrics

    def _legacy_c_igr_step(self, x: torch.Tensor, y: torch.Tensor, dp_defense, igr_defense):
        if not hasattr(self.optimizer, "original_optimizer"):
            raise RuntimeError("Legacy C-IGR path requires an Opacus DPOptimizer.")
        params = list(self.optimizer.params)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()
        task_grads, task_stats = dp_grads_from_current_samples(
            self.optimizer,
            run_step_hook=True,
            noise_multiplier=dp_defense.task_noise_multiplier,
            max_grad_norm=dp_defense.max_grad_norm,
        )
        self.manual_task_dp_passes += 1

        self.optimizer.zero_grad(set_to_none=True)
        _, _, igr_pen = self._igr_forward(x, y, igr_defense)
        igr_pen.backward()
        igr_grads, igr_stats = dp_grads_from_current_samples(
            self.optimizer,
            run_step_hook=True,
            noise_multiplier=dp_defense.regularizer_noise_multiplier,
            max_grad_norm=dp_defense.c_regularizer_clip_norm,
        )
        self.manual_regularizer_dp_passes += 1

        assign_combined_dp_grads(params, task_grads, igr_grads, igr_defense.lambda_igr)
        combined_grad_norm = param_grad_norm(params)
        self.optimizer.original_optimizer.step()

        loss = ce_loss.detach() + float(igr_defense.lambda_igr) * igr_pen.detach()
        task_grad_norm = grad_norm(task_grads)
        igr_grad_norm = grad_norm(igr_grads)
        igr_contrib_norm = abs(float(igr_defense.lambda_igr)) * igr_grad_norm
        metrics = {
            "IGR penalty": float(igr_pen.detach().item()),
            "C task DP grad norm": float(task_grad_norm),
            "C task DP signal norm": float(task_stats["signal_norm"]),
            "C task DP noise norm": float(task_stats["noise_norm"]),
            "C task DP expected noise norm": float(task_stats["expected_noise_norm"]),
            "C task DP SNR": float(task_stats["snr"]),
            "C task DP signal to grad ratio": float(task_stats["signal_to_noised_ratio"]),
            "C IGR DP grad norm": float(igr_grad_norm),
            "C IGR DP signal norm": float(igr_stats["signal_norm"]),
            "C IGR DP noise norm": float(igr_stats["noise_norm"]),
            "C IGR DP expected noise norm": float(igr_stats["expected_noise_norm"]),
            "C IGR DP SNR": float(igr_stats["snr"]),
            "C IGR DP signal to grad ratio": float(igr_stats["signal_to_noised_ratio"]),
            "C IGR DP contrib norm": float(igr_contrib_norm),
            "C IGR DP signal contrib norm": float(abs(float(igr_defense.lambda_igr)) * igr_stats["signal_norm"]),
            "C IGR DP noise contrib norm": float(abs(float(igr_defense.lambda_igr)) * igr_stats["noise_norm"]),
            "C combined DP grad norm": float(combined_grad_norm),
            "C IGR DP to task ratio": float(igr_contrib_norm / (task_grad_norm + 1e-12)),
            "C IGR DP to combined ratio": float(igr_contrib_norm / (combined_grad_norm + 1e-12)),
        }
        metrics.update(preclip_metric_names("C task", task_stats))
        metrics.update(preclip_metric_names("C IGR", igr_stats))
        return logits.detach(), loss, metrics

    def _legacy_c_trades_step(self, x: torch.Tensor, y: torch.Tensor, dp_defense, trades_defense):
        if not hasattr(self.optimizer, "original_optimizer"):
            raise RuntimeError("Legacy C-TRADES path requires an Opacus DPOptimizer.")

        params = list(self.optimizer.params)
        self.optimizer.zero_grad(set_to_none=True)
        x_adv = trades_defense._generate_adversarial(self.model, x.detach())

        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()
        task_grads, task_stats = dp_grads_from_current_samples(
            self.optimizer,
            run_step_hook=True,
            noise_multiplier=dp_defense.task_noise_multiplier,
            max_grad_norm=dp_defense.max_grad_norm,
        )
        self.manual_task_dp_passes += 1

        self.optimizer.zero_grad(set_to_none=True)
        clean_probs = self._clean_probs_without_grad_sample_hooks(x)
        kl_loss = trades_defense.kl_loss(self.model, x_adv, clean_probs)
        kl_loss.backward()
        trades_grads, trades_stats = dp_grads_from_current_samples(
            self.optimizer,
            run_step_hook=True,
            noise_multiplier=dp_defense.regularizer_noise_multiplier,
            max_grad_norm=dp_defense.c_regularizer_clip_norm,
        )
        self.manual_regularizer_dp_passes += 1

        assign_combined_dp_grads(params, task_grads, trades_grads, trades_defense.beta)
        combined_grad_norm = param_grad_norm(params)
        self.optimizer.original_optimizer.step()

        loss = ce_loss.detach() + trades_defense.beta * kl_loss.detach()
        task_grad_norm = grad_norm(task_grads)
        trades_grad_norm = grad_norm(trades_grads)
        trades_contrib_norm = abs(float(trades_defense.beta)) * trades_grad_norm
        weighted_kl = float(trades_defense.beta) * float(kl_loss.detach().item())

        metrics = {
            "TRADES CE": float(ce_loss.detach().item()),
            "TRADES KL": float(kl_loss.detach().item()),
            "TRADES weighted KL": float(weighted_kl),
            "TRADES total loss": float(loss.detach().item()),
            "TRADES weighted KL to total loss ratio": float(weighted_kl / (loss.detach().item() + 1e-12)),
            "C task DP grad norm": float(task_grad_norm),
            "C task DP signal norm": float(task_stats["signal_norm"]),
            "C task DP noise norm": float(task_stats["noise_norm"]),
            "C task DP expected noise norm": float(task_stats["expected_noise_norm"]),
            "C task DP SNR": float(task_stats["snr"]),
            "C task DP signal to grad ratio": float(task_stats["signal_to_noised_ratio"]),
            "C TRADES DP grad norm": float(trades_grad_norm),
            "C TRADES DP signal norm": float(trades_stats["signal_norm"]),
            "C TRADES DP noise norm": float(trades_stats["noise_norm"]),
            "C TRADES DP expected noise norm": float(trades_stats["expected_noise_norm"]),
            "C TRADES DP SNR": float(trades_stats["snr"]),
            "C TRADES DP signal to grad ratio": float(trades_stats["signal_to_noised_ratio"]),
            "C TRADES DP contrib norm": float(trades_contrib_norm),
            "C TRADES DP signal contrib norm": float(abs(float(trades_defense.beta)) * trades_stats["signal_norm"]),
            "C TRADES DP noise contrib norm": float(abs(float(trades_defense.beta)) * trades_stats["noise_norm"]),
            "C combined DP grad norm": float(combined_grad_norm),
            "C TRADES DP to task ratio": float(trades_contrib_norm / (task_grad_norm + 1e-12)),
            "C TRADES DP to combined ratio": float(trades_contrib_norm / (combined_grad_norm + 1e-12)),
        }
        metrics.update(preclip_metric_names("C task", task_stats))
        metrics.update(preclip_metric_names("C TRADES", trades_stats))
        return logits.detach(), loss, metrics

    def _legacy_d_trades_step(self, x: torch.Tensor, y: torch.Tensor, dp_defense, trades_defense):
        if not hasattr(self.optimizer, "original_optimizer"):
            raise RuntimeError("Legacy D-TRADES path requires an Opacus DPOptimizer.")

        self.optimizer.zero_grad(set_to_none=True)
        x_adv = trades_defense._generate_adversarial(self.model, x.detach())

        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()
        task_sums, task_preclip = clipped_grad_sums_from_current_samples(self.optimizer, dp_defense.max_grad_norm)

        self.optimizer.zero_grad(set_to_none=True)
        clean_probs = self._clean_probs_without_grad_sample_hooks(x)
        kl_loss = trades_defense.kl_loss(self.model, x_adv, clean_probs)
        kl_loss.backward()

        stats = self._apply_d_combined_update(dp_defense, task_sums, trades_defense.beta)
        combined_grad_norm = param_grad_norm(list(self.optimizer.params))
        self.optimizer.original_optimizer.step()

        loss = ce_loss.detach() + trades_defense.beta * kl_loss.detach()
        weighted_kl = float(trades_defense.beta) * float(kl_loss.detach().item())
        metrics = {
            "TRADES CE": float(ce_loss.detach().item()),
            "TRADES KL": float(kl_loss.detach().item()),
            "TRADES weighted KL": float(weighted_kl),
            "TRADES total loss": float(loss.detach().item()),
            "TRADES weighted KL to total loss ratio": float(weighted_kl / (loss.detach().item() + 1e-12)),
        }
        metrics.update(self._d_channel_metrics("TRADES", stats, task_preclip, combined_grad_norm))
        return logits.detach(), loss, metrics

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

    def _legacy_d_igr_step(self, x: torch.Tensor, y: torch.Tensor, dp_defense, igr_defense):
        if not hasattr(self.optimizer, "original_optimizer"):
            raise RuntimeError("Legacy D-IGR path requires an Opacus DPOptimizer.")

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(x)
        ce_loss = F.cross_entropy(logits, y)
        ce_loss.backward()
        task_sums, task_preclip = clipped_grad_sums_from_current_samples(self.optimizer, dp_defense.max_grad_norm)

        self.optimizer.zero_grad(set_to_none=True)
        _, _, igr_pen = self._igr_forward(x, y, igr_defense)
        igr_pen.backward()

        stats = self._apply_d_combined_update(dp_defense, task_sums, igr_defense.lambda_igr)
        combined_grad_norm = param_grad_norm(list(self.optimizer.params))
        self.optimizer.original_optimizer.step()

        loss = ce_loss.detach() + float(igr_defense.lambda_igr) * igr_pen.detach()
        metrics = {"IGR penalty": float(igr_pen.detach().item())}
        metrics.update(self._d_channel_metrics("IGR", stats, task_preclip, combined_grad_norm))
        return logits.detach(), loss, metrics

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

    def _d_channel_metrics(
        self,
        regularizer_label: str,
        stats: dict,
        task_preclip: dict[str, float],
        combined_grad_norm: float,
    ) -> dict[str, float]:
        return self._one_release_channel_metrics(
            "D", regularizer_label, stats, task_preclip, combined_grad_norm
        )

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

    def _igr_forward(self, x: torch.Tensor, y: torch.Tensor, igr_defense, manage_hooks: bool = True):
        x_req = x.detach().requires_grad_(True)
        logits = self.model(x_req)
        ce_loss = F.cross_entropy(logits, y)
        if manage_hooks:
            self._set_grad_sample_hooks(False)
        try:
            grad_x = torch.autograd.grad(
                outputs=ce_loss,
                inputs=x_req,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        finally:
            if manage_hooks:
                self._set_grad_sample_hooks(True)
        batch_size = max(1, int(x_req.size(0)))
        std = x_req.new_tensor(igr_defense.std).view(1, -1, 1, 1)
        grad_x_denorm = (grad_x * batch_size) / (std + 1e-12)
        igr_pen = grad_x_denorm.view(batch_size, -1).pow(2).sum(dim=1).mean()
        return logits, ce_loss, igr_pen

    def _raw_igr_grads(self, x: torch.Tensor, y: torch.Tensor, igr_defense, params: list[torch.nn.Parameter]):
        for param in params:
            param.grad = None
        self._set_grad_sample_hooks(False)
        try:
            # manage_hooks=False keeps Opacus hooks disabled through this whole
            # raw (non-private) pass, including the backward — as in legacy B.
            _, _, igr_pen = self._igr_forward(x, y, igr_defense, manage_hooks=False)
            igr_pen.backward()
            grads = [param.grad.detach().clone() if param.grad is not None else None for param in params]
        finally:
            self._set_grad_sample_hooks(True)
            for param in params:
                param.grad = None
        return grads, igr_pen.detach()

    def _accumulate_raw_grad_buffer(self, buffer: dict, grads: list[torch.Tensor | None], batch_size: int) -> None:
        for idx, grad in enumerate(grads):
            if grad is None:
                continue
            scaled_grad = grad.detach() * float(batch_size)
            if buffer["grads"][idx] is None:
                buffer["grads"][idx] = scaled_grad.clone()
            else:
                buffer["grads"][idx].add_(scaled_grad)
        buffer["seen"] += int(batch_size)

    def _clear_raw_grad_buffer(self, buffer: dict) -> None:
        buffer["grads"] = [None for _ in buffer["grads"]]
        buffer["seen"] = 0

    def _install_raw_igr_step_hook(
        self,
        params: list[torch.nn.Parameter],
        raw_buffer: dict,
        igr_lambda: float,
        normalize_raw_igr: bool,
        raw_igr_target_ratio: float,
    ):
        previous_hook = getattr(self.optimizer, "step_hook", None)
        hook_state = {
            "called": False,
            "dp_grad_norm": 0.0,
            "raw_igr_grad_norm": 0.0,
            "raw_igr_contrib_norm": 0.0,
            "raw_igr_to_task_ratio": 0.0,
            "raw_igr_to_total_ratio": 0.0,
            "raw_igr_scale": 1.0,
            "total_grad_norm": 0.0,
        }

        def hook(dp_optimizer) -> None:
            if previous_hook is not None:
                previous_hook(dp_optimizer)
            seen = int(raw_buffer["seen"])
            if seen > 0:
                raw_avgs = [
                    None if grad is None else grad.to(device=param.device, dtype=param.dtype) / float(seen)
                    for param, grad in zip(params, raw_buffer["grads"])
                ]
                dp_norm = param_grad_norm(params)
                raw_norm = grad_norm(raw_avgs)
                contrib_norm = abs(float(igr_lambda)) * raw_norm
                raw_scale = 1.0
                if normalize_raw_igr and contrib_norm > 0.0 and dp_norm > 0.0:
                    target_norm = max(0.0, float(raw_igr_target_ratio)) * dp_norm
                    raw_scale = target_norm / (contrib_norm + 1e-12)
                    contrib_norm = target_norm

                hook_state["dp_grad_norm"] = dp_norm
                hook_state["raw_igr_grad_norm"] = raw_norm
                hook_state["raw_igr_contrib_norm"] = contrib_norm
                hook_state["raw_igr_to_task_ratio"] = contrib_norm / (dp_norm + 1e-12)
                hook_state["raw_igr_scale"] = raw_scale

                for param, raw_avg in zip(params, raw_avgs):
                    if raw_avg is None:
                        continue
                    if param.grad is None:
                        param.grad = raw_avg.mul(float(igr_lambda) * raw_scale)
                    else:
                        param.grad.add_(raw_avg, alpha=float(igr_lambda) * raw_scale)

                total_norm = param_grad_norm(params)
                hook_state["total_grad_norm"] = total_norm
                hook_state["raw_igr_to_total_ratio"] = contrib_norm / (total_norm + 1e-12)

            hook_state["called"] = True

        self.optimizer.step_hook = hook
        return previous_hook, hook_state

    def _legacy_a_trades_diagnostics(self, dp_defense, trades_defense) -> dict[str, float]:
        """Legacy DP+TRADES strategy-A raw gradient diagnostics (before backward)."""
        if trades_defense is None:
            return {}
        ce_loss = getattr(trades_defense, "last_ce_loss", None)
        kl_loss = getattr(trades_defense, "last_kl_loss", None)
        trades_defense.last_ce_loss = None
        trades_defense.last_kl_loss = None
        if dp_defense is None or ce_loss is None or kl_loss is None:
            return {}
        params = list(getattr(self.optimizer, "params", []))
        if not params:
            return {}

        self._set_grad_sample_hooks(False)
        try:
            stats = trades_a_raw_grad_diagnostics(ce_loss, kl_loss, params, float(trades_defense.beta))
        finally:
            self._set_grad_sample_hooks(True)
        return {
            "A CE raw grad norm": float(stats["ce_norm"]),
            "A TRADES raw grad norm": float(stats["kl_norm"]),
            "A TRADES raw contrib norm": float(stats["trades_contrib_norm"]),
            "A combined raw grad norm": float(stats["combined_norm"]),
            "A TRADES raw to CE ratio": float(stats["trades_to_ce_ratio"]),
            "A TRADES raw to combined ratio": float(stats["trades_to_combined_ratio"]),
            "A CE TRADES grad cosine": float(stats["cosine"]),
        }

    def _install_dp_grad_norm_step_hook(self):
        previous_hook = getattr(self.optimizer, "step_hook", None)
        params = list(getattr(self.optimizer, "params", []))
        hook_state = {"called": False, "dp_grad_norm": 0.0}

        def hook(dp_optimizer) -> None:
            if previous_hook is not None:
                previous_hook(dp_optimizer)
            hook_state["dp_grad_norm"] = param_grad_norm(params)
            hook_state["called"] = True

        self.optimizer.step_hook = hook
        return previous_hook, hook_state

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

    def _private_train_gradient_metrics_enabled(self) -> bool:
        diagnostics = self.cfg.get("diagnostics", None) or {}
        private_cfg = diagnostics.get("private_train_gradients", None) or {}
        return bool(private_cfg.get("enabled", False))

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
        size = min(requested_size, len(dataset))
        generator = torch.Generator(device="cpu").manual_seed(int(probe_cfg.get("seed", 20260814)))
        indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
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
                "probe/setup/seed": float(probe_cfg.get("seed", 20260814)),
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
        """Drop legacy strategy telemetry derived from a private train batch."""
        return {
            key: value
            for key, value in metrics.items()
            if not str(key).startswith(
                ("A ", "B ", "C ", "D ", "H-C ", "S ", "S-C ")
            )
        }

    def _legacy_setup_metrics(self) -> dict[str, float]:
        igr = self._defense("igr")
        trades = self._defense("trades")
        label_smoothing = self._defense("label_smoothing")
        total = sum(param.numel() for param in self.model.parameters())
        trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        adaptation = self.cfg.get("adaptation", {})
        train_last_block = bool(adaptation.get("train_last_block", False))
        physical_batch_size = self.cfg.training.get("physical_batch_size", None)
        if physical_batch_size is None:
            physical_batch_size = self.cfg.data.get("batch_size", 0)
        sn_cfg = self.cfg.model.get("spectral_norm", None) or {}
        sn_enabled = bool(sn_cfg.get("enabled", False))
        metrics = {
            "CUDA available": 1.0 if torch.cuda.is_available() else 0.0,
            "Device is CUDA": 1.0 if self.device.type == "cuda" else 0.0,
            "Label smoothing": float(getattr(label_smoothing, "smoothing", 0.0) if label_smoothing else 0.0),
            "Spectral Norm enabled": 1.0 if sn_enabled else 0.0,
            "IGR enabled": 1.0 if igr else 0.0,
            "TRADES enabled": 1.0 if trades else 0.0,
            "Trainable parameter count": float(trainable),
            "Total parameter count": float(total),
            "Trainable parameter fraction": float(trainable / max(1, total)),
            "Train last block": 1.0 if train_last_block else 0.0,
            "Batch size": float(self.cfg.data.get("batch_size", 0)),
            "Physical batch size": float(physical_batch_size),
            "Number of epochs": float(self.cfg.training.get("epochs", 0)),
            "Private train-gradient diagnostics enabled": (
                1.0 if self._private_train_gradient_metrics_enabled() else 0.0
            ),
            "Public gradient probe enabled": 1.0 if self._gradient_probe_cfg() is not None else 0.0,
        }
        if sn_enabled:
            metrics["SN power_iterations"] = float(sn_cfg.get("power_iterations", 1))
            metrics["SN coeff"] = float(sn_cfg.get("coeff", 1.0))
        if igr:
            metrics["IGR lambda"] = float(getattr(igr, "lambda_igr", 0.0))
            strategy = str(igr.cfg.get("dp_strategy", igr.cfg.get("strategy", "A"))).upper()
            metrics["IGR DP strategy code"] = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}.get(strategy, 0.0)
            metrics["C IGR clip norm"] = float(igr.cfg.get("c_igr_clip_norm", 1.0))
            if strategy == "B":
                metrics["B normalize raw IGR"] = 1.0 if bool(igr.cfg.get("b_normalize_raw_igr", False)) else 0.0
                metrics["B raw IGR target ratio"] = float(igr.cfg.get("b_raw_igr_target_ratio", 0.1))
        if trades:
            metrics.update(
                {
                    "TRADES beta": float(getattr(trades, "beta", 0.0)),
                    "TRADES eps": float(getattr(trades, "epsilon", 0.0)),
                    "TRADES step_size": float(getattr(trades, "step_size", 0.0)),
                    "TRADES num_steps": float(getattr(trades, "steps", 0)),
                }
            )
            trades_strategy = str(trades.cfg.get("dp_strategy", trades.cfg.get("strategy", "C"))).upper()
            metrics["TRADES DP strategy code"] = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}.get(trades_strategy, 0.0)
            if trades_strategy == "C":
                metrics["C TRADES clip norm"] = float(trades.cfg.get("c_trades_clip_norm", 1.0))
            elif trades_strategy == "D":
                metrics["D TRADES clip norm"] = float(trades.cfg.get("c_trades_clip_norm", 1.0))
            elif trades_strategy == "B":
                metrics["B normalize raw TRADES"] = 1.0 if bool(trades.cfg.get("b_normalize_raw", False)) else 0.0
                metrics["B raw TRADES target ratio"] = float(trades.cfg.get("b_raw_target_ratio", 0.1))
        return metrics

    def _trades_probe_config(self):
        for item in self.cfg.get("defenses", []) or []:
            if str(item.get("name", "")).lower() == "trades":
                return item
        return None

    def _legacy_epoch_metrics(self, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> dict[str, float]:
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
