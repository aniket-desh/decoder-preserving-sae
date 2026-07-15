#!/usr/bin/env python3
"""Sparse empirical sweep over structured task-prior strength."""

from __future__ import annotations

import argparse
import json
import platform
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

if __package__:
    from experiments.exp02_structured_prior import (
        crossover_weight,
        git_revision,
        run_sparse,
        write_csv,
    )
else:
    from exp02_structured_prior import (  # type: ignore[no-redef]
        crossover_weight,
        git_revision,
        run_sparse,
        write_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def paired_sweep_rows(
    metrics: list[dict[str, Any]], relative_weights: list[float]
) -> list[dict[str, Any]]:
    baselines = {
        int(row["seed"]): row for row in metrics if row["method"] == "mse"
    }
    candidates = {
        (int(row["seed"]), float(row["relative_weight"])): row
        for row in metrics
        if row["method"] == "task_prior"
    }
    seeds = sorted(baselines)
    expected = {(seed, float(weight)) for seed in seeds for weight in relative_weights}
    if set(candidates) != expected:
        missing = sorted(expected - set(candidates))
        extra = sorted(set(candidates) - expected)
        raise ValueError(f"incomplete paired sweep; missing={missing}, extra={extra}")

    rows = []
    for seed in seeds:
        baseline = baselines[seed]
        for relative_weight in relative_weights:
            candidate = candidates[(seed, float(relative_weight))]
            protected_baseline = float(baseline["protected_decoder_distortion"])
            nmse_baseline = float(baseline["test_nmse"])
            rows.append(
                {
                    "seed": seed,
                    "relative_weight": relative_weight,
                    "task_weight": candidate["task_weight"],
                    "protected_reduction_vs_mse": 1
                    - float(candidate["protected_decoder_distortion"])
                    / protected_baseline,
                    "nmse_reduction_vs_mse": 1
                    - float(candidate["test_nmse"]) / nmse_baseline,
                    "candidate_protected_decoder_distortion": candidate[
                        "protected_decoder_distortion"
                    ],
                    "mse_protected_decoder_distortion": protected_baseline,
                    "candidate_nmse": candidate["test_nmse"],
                    "mse_nmse": nmse_baseline,
                }
            )
    return rows


def run_sweep(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    settings = config["empirical_crossover"]
    relative_weights = [float(value) for value in settings["relative_weights"]]
    if (
        not relative_weights
        or len(set(relative_weights)) != len(relative_weights)
        or any(value <= 0 for value in relative_weights)
    ):
        raise ValueError("empirical crossover weights must be unique and positive")
    threshold = crossover_weight(config["crossover"])

    baseline_metrics, baseline_groups, baseline_curves, baseline_calibrations = run_sparse(
        config,
        config["seeds"],
        threshold,
        methods=("mse",),
        relative_weight=None,
    )
    metrics = list(baseline_metrics)
    groups = list(baseline_groups)
    curves = list(baseline_curves)
    calibrations = list(baseline_calibrations)
    for relative_weight in relative_weights:
        task_metrics, task_groups, task_curves, task_calibrations = run_sparse(
            config,
            config["seeds"],
            threshold * relative_weight,
            methods=("task_prior",),
            relative_weight=relative_weight,
        )
        metrics.extend(task_metrics)
        groups.extend(task_groups)
        curves.extend(task_curves)
        calibrations.extend(task_calibrations)
    return {
        "metrics": metrics,
        "group_metrics": groups,
        "training_curves": curves,
        "calibration": calibrations,
        "paired_metrics": paired_sweep_rows(metrics, relative_weights),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = deepcopy(config)
        config["seeds"] = [0]
        config["data"]["train_samples"] = 1024
        config["data"]["validation_samples"] = 256
        config["data"]["test_samples"] = 512
        config["sae"]["steps"] = 10
        config["sae"]["log_every"] = 5
        config["evaluation"]["geometry_groups"] = 2
        config["evaluation"]["validation_geometry_groups"] = 2
        config["evaluation"]["group_samples"] = 256
        config["empirical_crossover"]["relative_weights"] = [0.5, 1.0, 2.0]
    if args.seed is not None:
        config["seeds"] = [args.seed]
    torch.set_num_threads(config.get("threads", 8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tables = run_sweep(config)
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    metadata = {
        "complete": True,
        "experiment": "exp02_prior_weight_sweep",
        "config": config,
        "config_path": str(args.config.resolve()),
        "git_revision": git_revision(),
        "crossover_weight": crossover_weight(config["crossover"]),
        "weight_parameterization": {
            "task_weight": "coefficient on the stochastic protected-target block",
            "relative_weight": (
                "task_weight divided by the separate two-direction rank-relaxation "
                "crossover; a plotting scale, not a predicted sparse transition"
            ),
            "expected_sparse_prior": "I + (task_weight / protected_dim) L L^T",
        },
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_seconds": time.perf_counter() - started,
        "smoke": args.smoke,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"completed empirical crossover sweep -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
