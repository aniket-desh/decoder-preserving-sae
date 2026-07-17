"""Frozen statistical helpers shared by Exp10 aggregation and release closure."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def family_block_bootstrap(
    task_deltas: Mapping[str, float],
    family_by_task: Mapping[str, str],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if set(task_deltas) != set(family_by_task):
        raise ValueError("task deltas and family map must have identical keys")
    grouped: dict[str, list[float]] = {}
    for task, delta in task_deltas.items():
        grouped.setdefault(family_by_task[task], []).append(float(delta))
    families = sorted(grouped)
    if len(families) < 2:
        raise ValueError("family bootstrap requires at least two families")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(families, size=len(families), replace=True)
        draws[index] = np.mean([value for family in selected for value in grouped[family]])
    alpha = (1 - confidence_level) / 2
    return {
        "estimate": float(np.mean(list(task_deltas.values()))),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1 - alpha)),
        "bootstrap_samples": samples,
        "family_count": len(families),
    }
