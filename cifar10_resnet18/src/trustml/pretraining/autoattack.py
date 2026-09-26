from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from trustml.pretraining.model import PixelNormalizedModel


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
    output_dir: str | Path,
) -> dict[str, object]:
    try:
        from autoattack import AutoAttack
    except ImportError as exc:
        raise RuntimeError("AutoAttack is required for final checkpoint selection") from exc
    if sum(isinstance(module, PixelNormalizedModel) for module in model.modules()) != 1:
        raise AssertionError("AutoAttack requires exactly one model-owned normalization")
    pixels, labels = _collect_pixels(dataloader, expected_examples)
    if float(pixels.min()) < 0.0 or float(pixels.max()) > 1.0:
        raise AssertionError("AutoAttack inputs must be raw pixels in [0,1]")
    was_training = model.training
    model.eval()
    with _evaluation_determinism_scope():
        try:
            clean_correct = _correctness(model, pixels, labels, device, batch_size)
            component_correct: dict[str, np.ndarray] = {}
            for component in COMPONENTS:
                attack = AutoAttack(
                    model,
                    norm="Linf",
                    eps=float(epsilon),
                    version="standard",
                    seed=int(seed),
                    verbose=False,
                )
                attack.attacks_to_run = [component]
                adversarial = attack.run_standard_evaluation(
                    pixels.to(device), labels.to(device), bs=int(batch_size), return_labels=False
                )
                _assert_linf(pixels.to(device), adversarial, epsilon)
                component_correct[component] = _correctness(
                    model, adversarial.cpu(), labels, device, batch_size
                ).numpy()
            standard = AutoAttack(
                model,
                norm="Linf",
                eps=float(epsilon),
                version="standard",
                seed=int(seed),
                verbose=False,
            )
            adversarial = standard.run_standard_evaluation(
                pixels.to(device), labels.to(device), bs=int(batch_size), return_labels=False
            )
            _assert_linf(pixels.to(device), adversarial, epsilon)
            overall_correct = _correctness(
                model, adversarial.cpu(), labels, device, batch_size
            ).numpy()
        finally:
            model.train(was_training)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flags_path = output_dir / "per_example_correctness.npz"
    np.savez_compressed(
        flags_path,
        labels=labels.numpy(),
        clean_correct=clean_correct.numpy(),
        overall_correct=overall_correct,
        **{f"{name}_correct": values for name, values in component_correct.items()},
    )
    report = {
        "format_version": 1,
        "version": "standard",
        "threat_model": "Linf",
        "epsilon": float(epsilon),
        "examples": int(labels.numel()),
        "seed": int(seed),
        "input_space": "pixel_[0,1]",
        "clean_accuracy": float(clean_correct.float().mean().item()),
        "component_accuracy": {
            name: float(values.mean()) for name, values in component_correct.items()
        },
        "overall_accuracy": float(overall_correct.mean()),
        "per_example_correctness": flags_path.name,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _collect_pixels(dataloader, expected_examples: int) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_batches = []
    label_batches = []
    for pixels, labels in dataloader:
        pixel_batches.append(pixels.detach().cpu().float())
        label_batches.append(labels.detach().cpu().long())
    all_pixels = torch.cat(pixel_batches)
    all_labels = torch.cat(label_batches)
    if len(all_labels) != int(expected_examples):
        raise RuntimeError(
            f"AutoAttack expected {expected_examples} validation examples; received {len(all_labels)}"
        )
    return all_pixels, all_labels


@torch.no_grad()
def _correctness(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    results = []
    for start in range(0, len(labels), int(batch_size)):
        logits = model(pixels[start : start + int(batch_size)].to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits during AutoAttack")
        results.append(logits.argmax(1).cpu().eq(labels[start : start + int(batch_size)]))
    return torch.cat(results)


def _assert_linf(original: torch.Tensor, adversarial: torch.Tensor, epsilon: float) -> None:
    maximum = float((adversarial - original).flatten(1).abs().amax().item())
    if maximum > float(epsilon) + 1e-6:
        raise AssertionError("AutoAttack exceeded the configured perturbation radius")


@contextmanager
def _evaluation_determinism_scope():
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        if enabled:
            torch.use_deterministic_algorithms(False)
        yield
    finally:
        if enabled:
            torch.use_deterministic_algorithms(True)

