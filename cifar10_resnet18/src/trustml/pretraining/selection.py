from __future__ import annotations

import math
from typing import Any


def binomial_standard_error(accuracy: float, examples: int = 1000) -> float:
    probability = float(accuracy)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("accuracy must be in [0,1]")
    if int(examples) <= 0:
        raise ValueError("examples must be positive")
    return math.sqrt(probability * (1.0 - probability) / int(examples))


def select_within_run(
    candidates: list[dict[str, Any]],
    *,
    examples: int = 1000,
) -> dict[str, Any]:
    """Apply stability screening and the within-run one-standard-error rule."""
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    stable = [candidate for candidate in candidates if not bool(candidate.get("unstable", False))]
    if not stable:
        raise RuntimeError("all checkpoint candidates failed the PGD-versus-Square consistency screen")
    contenders = stable
    best_accuracy = max(float(candidate["autoattack_accuracy"]) for candidate in contenders)
    standard_error = binomial_standard_error(best_accuracy, examples)
    within = [
        candidate
        for candidate in contenders
        if best_accuracy - float(candidate["autoattack_accuracy"]) <= standard_error + 1e-12
    ]
    return sorted(
        within,
        key=lambda candidate: (-float(candidate["clean_accuracy"]), int(candidate["epoch"])),
    )[0]


def select_across_betas(
    candidates: list[dict[str, Any]],
    *,
    examples: int = 1000,
) -> dict[str, Any]:
    """Apply the one-standard-error rule across the seed-42 beta candidates."""
    if not candidates:
        raise ValueError("at least one beta candidate is required")
    best_accuracy = max(float(candidate["autoattack_accuracy"]) for candidate in candidates)
    standard_error = binomial_standard_error(best_accuracy, examples)
    within = [
        candidate
        for candidate in candidates
        if best_accuracy - float(candidate["autoattack_accuracy"]) <= standard_error + 1e-12
    ]
    return sorted(
        within,
        key=lambda candidate: (
            -float(candidate["clean_accuracy"]),
            float(candidate["beta"]),
            int(candidate["epoch"]),
        ),
    )[0]


def mark_consistency(
    candidate: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    result = dict(candidate)
    gap = float(result["pgd10_accuracy"]) - float(result["square_accuracy"])
    result["pgd_minus_square_gap"] = gap
    result["unstable"] = gap >= float(threshold)
    return result
