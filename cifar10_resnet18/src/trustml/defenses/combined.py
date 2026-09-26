from __future__ import annotations

from collections.abc import Iterable


def collect_defense_metrics(defenses: Iterable) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for defense in defenses:
        metrics.update(defense.metrics())
    return metrics

