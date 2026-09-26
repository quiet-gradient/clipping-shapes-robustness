"""AutoAttack integration for full-dataset robustness evaluation.

The upstream AutoAttack API expects images in pixel space (``[0, 1]``), while
TrustML dataloaders yield normalized tensors.  This module converts the inputs
back to pixel space and wraps the target model with the matching normalization,
so the threat radius remains the conventional CIFAR-10 ``Linf`` radius.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch

from trustml.attacks.pgd import _set_grad_sample_hooks


class _NormalizedModel(torch.nn.Module):
    """Expose a normalized TrustML model as a model over raw ``[0, 1]`` pixels."""

    def __init__(self, model: torch.nn.Module, mean, std) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))


@dataclass(frozen=True)
class _PreparedInputs:
    normalized: torch.Tensor
    pixel_legacy: torch.Tensor
    pixel: torch.Tensor
    labels: torch.Tensor
    input_diagnostics: dict[str, Any]


class PixelWrapperValidationError(RuntimeError):
    """Fail-closed pixel-wrapper error with machine-readable evidence."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class AutoAttackAttack:
    """Run the official AutoAttack ensemble through the robustness interface.

    ``version=standard`` is the publication/RobustBench protocol: APGD-CE,
    targeted APGD-DLR, targeted FAB, and Square Attack.  The dependency is
    imported lazily so training and non-AA evaluations do not require it.
    """

    def __init__(
        self,
        cfg,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ) -> None:
        self.epsilon = float(cfg.get("epsilon", 8.0 / 255.0))
        self.norm = str(cfg.get("norm", "Linf"))
        self.version = str(cfg.get("version", "standard"))
        self.batch_size = int(cfg.get("batch_size", 128))
        self.seed = int(cfg.get("seed", 42))
        self.max_batches = cfg.get("max_batches", None)
        self.expected_examples = cfg.get("expected_examples", None)
        self.require_full_test = bool(cfg.get("require_full_test", False))
        self.verbose = bool(cfg.get("verbose", True))
        self.log_path = cfg.get("log_path", None)
        self.state_path = cfg.get("state_path", None)
        requested_components = cfg.get("attacks_to_run", None)
        self.attacks_to_run = (
            None if requested_components is None else [str(value) for value in requested_components]
        )
        self.mean = tuple(float(v) for v in cfg.get("mean", mean))
        self.std = tuple(float(v) for v in cfg.get("std", std))
        # CIFAR images originate as uint8 values.  Recovering pixels with the
        # algebraic inverse of torchvision Normalize leaves a few float32 ULPs
        # of error; applying Normalize again on CUDA can then move borderline
        # predictions.  Validate that the inverse is on the uint8 lattice and
        # snap to that lattice before the attack.  A wrong mean/std does not
        # silently pass: it produces a large lattice or round-trip residual.
        self.pixel_quantization_levels = int(cfg.get("pixel_quantization_levels", 256))
        self.pixel_lattice_tolerance = float(cfg.get("pixel_lattice_tolerance", 1e-6))
        self.roundtrip_tolerance = float(cfg.get("roundtrip_tolerance", 1e-6))
        self.max_prediction_disagreements = int(cfg.get("max_prediction_disagreements", 16))
        self.capture_component_trace = bool(cfg.get("capture_component_trace", False))
        # Development reports retain the upstream attack-time labels for
        # backward compatibility.  The post-lock official-test evaluator sets
        # this flag so its headline is a fresh FP32 forward pass over the final
        # adversarial examples.
        self.headline_from_posthoc = bool(cfg.get("headline_from_posthoc", False))
        self.last_diagnostics: dict[str, Any] | None = None
        self.last_output_diagnostics: dict[str, Any] | None = None
        self.last_component_trace: dict[str, Any] | None = None

        if self.version != "standard":
            raise ValueError(
                "TrustML publication evaluation requires AutoAttack version='standard'; "
                f"got {self.version!r}."
            )
        if self.norm not in {"Linf", "L2", "L1"}:
            raise ValueError(f"Unsupported AutoAttack norm: {self.norm}")
        if self.batch_size <= 0:
            raise ValueError("AutoAttack batch_size must be positive.")
        if self.pixel_quantization_levels < 2:
            raise ValueError("pixel_quantization_levels must be at least 2.")
        if self.pixel_lattice_tolerance < 0.0 or self.roundtrip_tolerance < 0.0:
            raise ValueError("Pixel-wrapper tolerances must be non-negative.")
        if self.max_prediction_disagreements < 0:
            raise ValueError("max_prediction_disagreements must be non-negative.")
        allowed_components = {"apgd-ce", "apgd-dlr", "fab-t", "square"}
        if self.attacks_to_run is not None and (
            not self.attacks_to_run or not set(self.attacks_to_run).issubset(allowed_components)
        ):
            raise ValueError(f"invalid AutoAttack components: {self.attacks_to_run}")
        if self.require_full_test and self.max_batches is not None:
            raise ValueError("Full AutoAttack cannot set max_batches.")

    def _prepare_inputs(self, dataloader) -> _PreparedInputs:
        mean = torch.tensor(self.mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(1, -1, 1, 1)
        normalized_images: list[torch.Tensor] = []
        legacy_pixels: list[torch.Tensor] = []
        quantized_pixels: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        lattice_scale = float(self.pixel_quantization_levels - 1)
        for batch_idx, (x, y) in enumerate(dataloader):
            if self.max_batches is not None and batch_idx >= int(self.max_batches):
                break
            normalized = x.detach().cpu().float()
            raw_legacy = normalized * std + mean
            raw_quantized = (raw_legacy * lattice_scale).round() / lattice_scale
            normalized_images.append(normalized)
            legacy_pixels.append(raw_legacy)
            quantized_pixels.append(raw_quantized)
            labels.append(y.detach().cpu().long())
        if not normalized_images:
            raise ValueError("AutoAttack received an empty dataloader.")
        normalized_all = torch.cat(normalized_images, dim=0)
        pixel_legacy_all = torch.cat(legacy_pixels, dim=0)
        pixel_all = torch.cat(quantized_pixels, dim=0)
        y_all = torch.cat(labels, dim=0)
        expected = None if self.expected_examples is None else int(self.expected_examples)
        if expected is not None and len(y_all) != expected:
            raise ValueError(f"AutoAttack expected {expected} examples, got {len(y_all)}.")
        if self.require_full_test and len(y_all) != 10_000:
            raise ValueError(f"Full CIFAR-10 AutoAttack requires 10000 examples, got {len(y_all)}.")

        lattice_error = float((pixel_legacy_all - pixel_all).abs().max().item())
        pixel_min = float(pixel_legacy_all.min().item())
        pixel_max = float(pixel_legacy_all.max().item())
        input_diagnostics: dict[str, Any] = {
            "n_examples": int(y_all.numel()),
            "pixel_quantization_levels": self.pixel_quantization_levels,
            "pixel_lattice_tolerance": self.pixel_lattice_tolerance,
            "input_lattice_max_abs_error": lattice_error,
            "legacy_pixel_min": pixel_min,
            "legacy_pixel_max": pixel_max,
            "quantized_pixel_min": float(pixel_all.min().item()),
            "quantized_pixel_max": float(pixel_all.max().item()),
        }
        input_issues: list[str] = []
        if lattice_error > self.pixel_lattice_tolerance:
            input_issues.append(
                f"uint8 lattice residual {lattice_error:.9g} exceeds "
                f"{self.pixel_lattice_tolerance:.9g}"
            )
        if pixel_min < -self.pixel_lattice_tolerance or pixel_max > 1.0 + self.pixel_lattice_tolerance:
            input_issues.append(f"recovered pixel range [{pixel_min:.9g}, {pixel_max:.9g}] is invalid")
        if float(pixel_all.min().item()) < 0.0 or float(pixel_all.max().item()) > 1.0:
            input_issues.append("quantized pixels are outside [0, 1]")
        input_diagnostics["input_validation_passed"] = not input_issues
        input_diagnostics["input_validation_issues"] = input_issues
        if input_issues:
            diagnostics = {
                **input_diagnostics,
                "validation_passed": False,
                "validation_issues": input_issues,
            }
            self.last_diagnostics = diagnostics
            raise PixelWrapperValidationError(
                "AutoAttack pixel recovery failed closed; check dataset normalization.",
                diagnostics,
            )

        return _PreparedInputs(
            normalized=normalized_all,
            pixel_legacy=pixel_legacy_all,
            pixel=pixel_all,
            labels=y_all,
            input_diagnostics=input_diagnostics,
        )

    def _collect_pixel_inputs(self, dataloader) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility helper used by cheap integration tests."""

        prepared = self._prepare_inputs(dataloader)
        return prepared.pixel, prepared.labels

    @staticmethod
    @torch.no_grad()
    def _predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
        predictions: list[torch.Tensor] = []
        for start in range(0, len(x), batch_size):
            logits = model(x[start : start + batch_size].to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("AutoAttack evaluation produced non-finite model logits.")
            predictions.append(logits.argmax(dim=1).cpu())
        return torch.cat(predictions, dim=0)

    @torch.no_grad()
    def _diagnose_prepared(
        self,
        model: torch.nn.Module,
        wrapped: _NormalizedModel,
        prepared: _PreparedInputs,
        device: torch.device,
    ) -> dict[str, Any]:
        direct_predictions: list[torch.Tensor] = []
        legacy_predictions: list[torch.Tensor] = []
        wrapper_predictions: list[torch.Tensor] = []
        legacy_roundtrip_max = 0.0
        roundtrip_max = 0.0
        legacy_roundtrip_sum = 0.0
        roundtrip_sum = 0.0
        value_count = 0

        for start in range(0, len(prepared.labels), self.batch_size):
            stop = start + self.batch_size
            normalized = prepared.normalized[start:stop].to(device)
            pixel_legacy = prepared.pixel_legacy[start:stop].to(device)
            pixel = prepared.pixel[start:stop].to(device)
            legacy_roundtrip = wrapped.normalize(pixel_legacy)
            roundtrip = wrapped.normalize(pixel)

            legacy_error = (legacy_roundtrip - normalized).abs()
            error = (roundtrip - normalized).abs()
            legacy_roundtrip_max = max(legacy_roundtrip_max, float(legacy_error.max().item()))
            roundtrip_max = max(roundtrip_max, float(error.max().item()))
            legacy_roundtrip_sum += float(legacy_error.double().sum().item())
            roundtrip_sum += float(error.double().sum().item())
            value_count += int(error.numel())

            direct_logits = model(normalized)
            legacy_logits = model(legacy_roundtrip)
            wrapper_logits = model(roundtrip)
            for logits in (direct_logits, legacy_logits, wrapper_logits):
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Pixel-wrapper diagnostics produced non-finite model logits.")
            direct_predictions.append(direct_logits.argmax(dim=1).cpu())
            legacy_predictions.append(legacy_logits.argmax(dim=1).cpu())
            wrapper_predictions.append(wrapper_logits.argmax(dim=1).cpu())

        direct_pred = torch.cat(direct_predictions)
        legacy_pred = torch.cat(legacy_predictions)
        wrapper_pred = torch.cat(wrapper_predictions)
        labels = prepared.labels
        direct_correct = direct_pred.eq(labels)
        legacy_correct = legacy_pred.eq(labels)
        wrapper_correct = wrapper_pred.eq(labels)

        diagnostics: dict[str, Any] = {
            **prepared.input_diagnostics,
            "roundtrip_tolerance": self.roundtrip_tolerance,
            "max_prediction_disagreements": self.max_prediction_disagreements,
            "legacy_roundtrip_max_abs_error": legacy_roundtrip_max,
            "legacy_roundtrip_mean_abs_error": legacy_roundtrip_sum / max(1, value_count),
            "roundtrip_max_abs_error": roundtrip_max,
            "roundtrip_mean_abs_error": roundtrip_sum / max(1, value_count),
            "direct_clean_correct": int(direct_correct.sum().item()),
            "legacy_wrapper_clean_correct": int(legacy_correct.sum().item()),
            "wrapper_clean_correct": int(wrapper_correct.sum().item()),
            "direct_clean_acc": float(direct_correct.float().mean().item()),
            "legacy_wrapper_clean_acc": float(legacy_correct.float().mean().item()),
            "wrapper_clean_acc": float(wrapper_correct.float().mean().item()),
            "legacy_prediction_disagreement_count": int(legacy_pred.ne(direct_pred).sum().item()),
            "prediction_disagreement_count": int(wrapper_pred.ne(direct_pred).sum().item()),
            "legacy_correctness_disagreement_count": int(legacy_correct.ne(direct_correct).sum().item()),
            "correctness_disagreement_count": int(wrapper_correct.ne(direct_correct).sum().item()),
            "legacy_clean_correct_delta": int(legacy_correct.sum().item() - direct_correct.sum().item()),
            "clean_correct_delta": int(wrapper_correct.sum().item() - direct_correct.sum().item()),
        }
        issues: list[str] = []
        if roundtrip_max > self.roundtrip_tolerance:
            issues.append(
                f"quantized round-trip error {roundtrip_max:.9g} exceeds "
                f"{self.roundtrip_tolerance:.9g}"
            )
        if diagnostics["prediction_disagreement_count"] > self.max_prediction_disagreements:
            issues.append(
                f"direct/wrapper prediction disagreements "
                f"{diagnostics['prediction_disagreement_count']} exceed "
                f"{self.max_prediction_disagreements}"
            )
        diagnostics["validation_passed"] = not issues
        diagnostics["validation_issues"] = issues
        return diagnostics

    def diagnose(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
    ) -> dict[str, Any]:
        """Compare direct, legacy-wrapper, and quantized-wrapper clean paths."""

        was_training = model.training
        model.eval()
        _set_grad_sample_hooks(model, False)
        try:
            prepared = self._prepare_inputs(dataloader)
            wrapped = _NormalizedModel(model, self.mean, self.std).to(device).eval()
            diagnostics = self._diagnose_prepared(model, wrapped, prepared, device)
            self.last_diagnostics = diagnostics
            if not diagnostics["validation_passed"]:
                raise PixelWrapperValidationError(
                    "AutoAttack pixel-wrapper diagnostics failed closed.",
                    diagnostics,
                )
            return diagnostics
        finally:
            _set_grad_sample_hooks(model, True)
            if was_training:
                model.train()

    def run(
        self,
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
        progress: bool = True,
    ) -> dict[str, float]:
        try:
            from autoattack import AutoAttack
        except ImportError as exc:  # pragma: no cover - exercised only in a broken environment
            raise RuntimeError(
                "AutoAttack is not installed. Install the pinned official dependency "
                "from requirements.lock before running this evaluation."
            ) from exc

        was_training = model.training
        model.eval()
        _set_grad_sample_hooks(model, False)
        try:
            prepared = self._prepare_inputs(dataloader)
            wrapped = _NormalizedModel(model, self.mean, self.std).to(device).eval()
            diagnostics = self._diagnose_prepared(model, wrapped, prepared, device)
            self.last_diagnostics = diagnostics
            if not diagnostics["validation_passed"]:
                raise PixelWrapperValidationError(
                    "AutoAttack pixel-wrapper diagnostics failed closed.",
                    diagnostics,
                )
            x_orig, y_orig = prepared.pixel, prepared.labels

            log_path = None if self.log_path in {None, "", "null"} else str(self.log_path)
            if log_path is not None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            state_path = None if self.state_path in {None, "", "null"} else Path(str(self.state_path))
            if state_path is not None:
                state_path.parent.mkdir(parents=True, exist_ok=True)

            adversary = AutoAttack(
                wrapped,
                norm=self.norm,
                eps=self.epsilon,
                seed=self.seed,
                verbose=self.verbose and progress,
                version="standard",
                device=device,
                log_path=log_path,
            )
            if self.attacks_to_run is not None:
                adversary.attacks_to_run = list(self.attacks_to_run)
            clean_pred = self._predict(wrapped, x_orig, device, self.batch_size)
            component_trace: dict[str, Any] | None = None
            aa_started_at = time.perf_counter()
            if self.capture_component_trace:
                component_trace = self._run_with_component_trace(
                    adversary,
                    x_orig,
                    y_orig,
                    clean_pred,
                    state_path=state_path,
                )
                x_adv = component_trace.pop("_x_adv")
                attack_time_predictions = component_trace.pop("_attack_time_predictions")
            else:
                x_adv, attack_time_predictions = adversary.run_standard_evaluation(
                    x_orig,
                    y_orig,
                    bs=self.batch_size,
                    return_labels=True,
                    state_path=state_path,
                )
            aa_total_seconds = time.perf_counter() - aa_started_at
            if not torch.isfinite(x_adv).all():
                raise RuntimeError("AutoAttack produced non-finite adversarial examples.")
            if not isinstance(attack_time_predictions, torch.Tensor):
                raise RuntimeError("AutoAttack did not return attack-time prediction labels.")
            if attack_time_predictions.shape != y_orig.shape:
                raise RuntimeError(
                    "AutoAttack returned prediction labels with an unexpected shape: "
                    f"{tuple(attack_time_predictions.shape)} != {tuple(y_orig.shape)}."
                )
            pixel_tolerance = 1e-6
            adv_min = float(x_adv.min().item())
            adv_max = float(x_adv.max().item())
            if adv_min < -pixel_tolerance or adv_max > 1.0 + pixel_tolerance:
                raise RuntimeError(
                    "AutoAttack produced adversarial examples outside pixel bounds: "
                    f"min={adv_min}, max={adv_max}."
                )
            posthoc_adv_pred = self._predict(wrapped, x_adv.cpu(), device, self.batch_size)
        finally:
            _set_grad_sample_hooks(model, True)
            if was_training:
                model.train()

        clean_mask = clean_pred.eq(y_orig)
        attack_time_predictions = attack_time_predictions.detach().cpu().long()
        attack_time_adv_mask = attack_time_predictions.eq(y_orig)
        posthoc_adv_mask = posthoc_adv_pred.eq(y_orig)
        clean_correct = int(clean_mask.sum().item())
        attack_time_adv_correct = int(attack_time_adv_mask.sum().item())
        posthoc_adv_correct = int(posthoc_adv_mask.sum().item())
        headline_mask = posthoc_adv_mask if self.headline_from_posthoc else attack_time_adv_mask
        adv_correct = int(headline_mask.sum().item())
        total = int(y_orig.numel())
        if adv_correct > clean_correct:
            raise RuntimeError(
                "AutoAttack headline robust count exceeds clean correct count: "
                f"{adv_correct} > {clean_correct}."
            )
        attack_success = int((clean_mask & ~headline_mask).sum().item())
        self.last_output_diagnostics = {
            "headline_source": (
                "posthoc_fp32_logits"
                if self.headline_from_posthoc
                else "autoattack_return_labels"
            ),
            "n_examples": total,
            "headline_adv_correct": adv_correct,
            "attack_time_adv_correct": attack_time_adv_correct,
            "posthoc_adv_correct": posthoc_adv_correct,
            "posthoc_adv_correct_delta": posthoc_adv_correct - attack_time_adv_correct,
            "attack_time_vs_posthoc_prediction_disagreement_count": int(
                attack_time_predictions.ne(posthoc_adv_pred).sum().item()
            ),
            "attack_time_vs_posthoc_correctness_disagreement_count": int(
                attack_time_adv_mask.ne(posthoc_adv_mask).sum().item()
            ),
        }
        if component_trace is not None:
            component_trace["total_seconds"] = aa_total_seconds
            component_trace["clean_correct"] = clean_mask.tolist()
            component_trace["attack_time_final_robust_correct"] = attack_time_adv_mask.tolist()
            component_trace["posthoc_final_robust_correct"] = posthoc_adv_mask.tolist()
            component_trace["final_robust_correct"] = headline_mask.tolist()
            component_trace["final_robust_correct_source"] = self.last_output_diagnostics[
                "headline_source"
            ]
            self.last_component_trace = component_trace

        flat_delta = (x_adv.cpu() - x_orig).reshape(total, -1)
        if self.norm == "Linf":
            perturbation = flat_delta.abs().max(dim=1).values
        elif self.norm == "L2":
            perturbation = flat_delta.norm(p=2, dim=1)
        else:
            perturbation = flat_delta.norm(p=1, dim=1)
        max_perturbation = float(perturbation.max().item())
        tolerance = max(1e-6, abs(self.epsilon) * 1e-4)
        if max_perturbation > self.epsilon + tolerance:
            raise RuntimeError(
                f"AutoAttack produced an out-of-budget perturbation: {max_perturbation} > {self.epsilon}."
            )

        clean_acc = clean_correct / max(1, total)
        adv_acc = adv_correct / max(1, total)
        posthoc_adv_acc = posthoc_adv_correct / max(1, total)
        asr = attack_success / max(1, clean_correct)
        return {
            "attack/autoattack_clean_acc": clean_acc,
            "attack/autoattack_adv_acc": adv_acc,
            "attack/autoattack_asr": asr,
            "attack/autoattack_n_examples": float(total),
            "attack/autoattack_max_perturbation": max_perturbation,
            "attack/autoattack_adv_acc_posthoc": posthoc_adv_acc,
            "attack/autoattack_attack_time_adv_correct": float(attack_time_adv_correct),
            "attack/autoattack_posthoc_adv_correct": float(posthoc_adv_correct),
            "AUTOATTACK_CLEAN_ACC": clean_acc,
            "AUTOATTACK_ADV_ACC": adv_acc,
            "AUTOATTACK_ADV_ACC_POSTHOC": posthoc_adv_acc,
            "AUTOATTACK_ASR": asr,
        }

    def _run_with_component_trace(
        self,
        adversary,
        x_orig: torch.Tensor,
        y_orig: torch.Tensor,
        clean_pred: torch.Tensor,
        *,
        state_path: Path | None,
    ) -> dict[str, Any]:
        """Run upstream standard AA while observing, but not changing, its state.

        The official implementation updates ``EvaluationState.robust_flags``
        after every attack batch and calls ``add_run_attack`` at each component
        boundary.  Temporarily wrapping that boundary method gives us exact
        per-example survivor masks without re-running prefixes or forking the
        attack implementation.  Perturb methods are wrapped only to timestamp
        the first batch of each component.  All wrappers are restored even if
        AutoAttack raises.
        """

        expected_order = ("apgd-ce", "apgd-t", "fab-t", "square")
        actual_order = tuple(str(name) for name in adversary.attacks_to_run)
        if actual_order != expected_order:
            raise RuntimeError(
                "component tracing requires the exact AutoAttack standard order "
                f"{expected_order}, got {actual_order}"
            )
        if state_path is not None:
            raise RuntimeError("component tracing requires a fresh run without state restoration")

        import autoattack.autoattack as upstream_autoattack

        state_cls = upstream_autoattack.EvaluationState
        original_add_run_attack = state_cls.add_run_attack
        stage_started: dict[str, float] = {}
        masks: dict[str, list[bool]] = {}
        reached: dict[str, int] = {}
        durations: dict[str, float] = {}
        clean_mask = clean_pred.eq(y_orig)
        previous_after = int(clean_mask.sum().item())

        perturb_bindings = (
            (adversary.apgd, "perturb", "apgd-ce"),
            (adversary.apgd_targeted, "perturb", "apgd-t"),
            (adversary.fab, "perturb", "fab-t"),
            (adversary.square, "perturb", "square"),
        )
        original_perturb: list[tuple[Any, str, Any]] = []

        for owner, attribute, stage in perturb_bindings:
            original = getattr(owner, attribute)
            original_perturb.append((owner, attribute, original))

            def timed_perturb(*args, _original=original, _stage=stage, **kwargs):
                stage_started.setdefault(_stage, time.perf_counter())
                return _original(*args, **kwargs)

            setattr(owner, attribute, timed_perturb)

        def capture_boundary(state, attack: str) -> None:
            nonlocal previous_after
            finished = time.perf_counter()
            robust_flags = state.robust_flags
            if robust_flags is None:
                raise RuntimeError(f"AutoAttack did not expose robust flags after {attack}")
            current_mask = robust_flags.detach().cpu().bool()
            if current_mask.shape != y_orig.shape:
                raise RuntimeError(
                    f"AutoAttack robust flag shape after {attack} is {tuple(current_mask.shape)}, "
                    f"expected {tuple(y_orig.shape)}"
                )
            reached[str(attack)] = previous_after
            after = int(current_mask.sum().item())
            previous_after = after
            masks[str(attack)] = current_mask.tolist()
            started = stage_started.get(str(attack))
            if started is None:
                raise RuntimeError(f"AutoAttack component {attack} completed without calling perturb")
            durations[str(attack)] = finished - started
            original_add_run_attack(state, attack)

        state_cls.add_run_attack = capture_boundary
        try:
            x_adv, attack_time_predictions = adversary.run_standard_evaluation(
                x_orig,
                y_orig,
                bs=self.batch_size,
                return_labels=True,
                state_path=None,
            )
        finally:
            state_cls.add_run_attack = original_add_run_attack
            for owner, attribute, original in original_perturb:
                setattr(owner, attribute, original)

        missing = [name for name in expected_order if name not in masks]
        if missing:
            # If no clean-correct examples survive, upstream exits early.  The
            # remaining masks are deterministically all-false and the attacks
            # have zero runtime/reach; preserve a complete four-stage schema.
            if previous_after != 0:
                raise RuntimeError(f"AutoAttack component trace is incomplete: {missing}")
            all_false = [False] * int(y_orig.numel())
            for name in missing:
                reached[name] = 0
                masks[name] = list(all_false)
                durations[name] = 0.0

        return {
            "official_component_order": list(expected_order),
            "component_durations_seconds": durations,
            "component_reached_examples": reached,
            "correct_after_component": masks,
            "_x_adv": x_adv,
            "_attack_time_predictions": attack_time_predictions,
        }
