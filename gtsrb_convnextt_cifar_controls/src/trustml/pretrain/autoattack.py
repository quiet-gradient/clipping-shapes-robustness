"""Pixel-space AutoAttack evaluation for the public CIFAR-100 pretrain protocol.

Unlike the older CIFAR-10 evaluator, pretraining dataloaders already emit raw
``[0, 1]`` pixels and the model owns its single normalization wrapper.  Keeping
that contract here prevents an accidental second normalization before AA.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from trustml.models.wrappers import NormalizedModel


COMPONENTS = ("apgd-ce", "apgd-dlr", "fab-t", "square")


def evaluate_autoattack_standard(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    epsilon: float,
    batch_size: int,
    seed: int,
    expected_examples: int,
    output_dir: Path,
) -> dict[str, object]:
    """Evaluate every standard AA component and persist per-example flags."""
    try:
        from autoattack import AutoAttack
    except ImportError as exc:  # dependency is deliberately lazy for local unit tests
        raise RuntimeError("AutoAttack is required for final_v1 checkpoint selection") from exc

    normalization_wrappers = sum(isinstance(module, NormalizedModel) for module in model.modules())
    if normalization_wrappers != 1:
        raise AssertionError(f"expected exactly one model-owned normalization, found {normalization_wrappers}")
    pixels, labels = _collect_pixels(dataloader, expected_examples)
    if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
        raise AssertionError("AutoAttack inputs must be raw pixels in [0,1]")
    model_was_training = model.training
    model.eval()
    # FAB's CUDA projection invokes cumsum, which PyTorch 2.4 does not expose
    # as a deterministic algorithm. This scope is evaluation-only: training,
    # checkpoint resume, PGD-10, and all selection inputs retain strict mode.
    with _autoattack_determinism_scope():
        try:
            clean_correct = _correctness(model, pixels, labels, device, batch_size)
            component_correct: dict[str, np.ndarray] = {}
            for component in COMPONENTS:
                attack = AutoAttack(model, norm="Linf", eps=float(epsilon), version="standard", seed=int(seed), verbose=False)
                attack.attacks_to_run = [component]
                adversarial = attack.run_standard_evaluation(
                    pixels.to(device), labels.to(device), bs=int(batch_size), return_labels=False
                )
                _assert_linf(pixels.to(device), adversarial, epsilon)
                component_correct[component] = _correctness(model, adversarial.cpu(), labels, device, batch_size).numpy()

            standard = AutoAttack(model, norm="Linf", eps=float(epsilon), version="standard", seed=int(seed), verbose=False)
            adversarial = standard.run_standard_evaluation(
                pixels.to(device), labels.to(device), bs=int(batch_size), return_labels=False
            )
            _assert_linf(pixels.to(device), adversarial, epsilon)
            overall_correct = _correctness(model, adversarial.cpu(), labels, device, batch_size).numpy()
        finally:
            model.train(model_was_training)

    output_dir.mkdir(parents=True, exist_ok=True)
    flags_path = output_dir / "autoattack_per_example_correctness.npz"
    np.savez_compressed(
        flags_path,
        labels=labels.numpy(),
        clean_correct=clean_correct.numpy(),
        overall_aa_correct=overall_correct,
        **{f"{name}_correct": flags for name, flags in component_correct.items()},
    )
    report = {
        "format_version": 1,
        "protocol": "AutoAttack standard, Linf 8/255, validation 1k",
        "n_examples": int(labels.numel()),
        "epsilon": float(epsilon),
        "input_space": "pixel_[0,1]",
        "model_owned_normalization_wrappers": normalization_wrappers,
        "strict_deterministic_algorithms_disabled_during_autoattack": True,
        "clean_accuracy": float(clean_correct.float().mean().item()),
        "components": {name: float(flags.mean()) for name, flags in component_correct.items()},
        "overall_accuracy": float(overall_correct.mean()),
        "per_example_correctness": flags_path.name,
    }
    (output_dir / "autoattack_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _collect_pixels(dataloader, expected_examples: int) -> tuple[torch.Tensor, torch.Tensor]:
    pixels, labels = [], []
    for batch_pixels, batch_labels in dataloader:
        pixels.append(batch_pixels.detach().cpu().float())
        labels.append(batch_labels.detach().cpu().long())
    all_pixels, all_labels = torch.cat(pixels), torch.cat(labels)
    if len(all_labels) != int(expected_examples):
        raise RuntimeError(f"AutoAttack expected {expected_examples} validation examples, got {len(all_labels)}")
    return all_pixels, all_labels


@torch.no_grad()
def _correctness(model, pixels: torch.Tensor, labels: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(labels), int(batch_size)):
        logits = model(pixels[start : start + int(batch_size)].to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits during AutoAttack evaluation")
        chunks.append(logits.argmax(1).cpu().eq(labels[start : start + int(batch_size)]))
    return torch.cat(chunks)


def _assert_linf(original: torch.Tensor, adversarial: torch.Tensor, epsilon: float) -> None:
    if float((adversarial - original).flatten(1).abs().amax().item()) > float(epsilon) + 1e-6:
        raise AssertionError("AutoAttack perturbation exceeded epsilon")


@contextmanager
def _autoattack_determinism_scope():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        if was_enabled:
            torch.use_deterministic_algorithms(False)
        yield
    finally:
        if was_enabled:
            torch.use_deterministic_algorithms(True)
