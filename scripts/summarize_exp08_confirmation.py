#!/usr/bin/env python3
"""Fail-closed summary for the clean matched-quality confirmation fleet."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.exp08_language_evidence import (
    disjoint_intervals,
    input_record,
    repository_state,
    validate_natural_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize_confirmation(
    evaluations: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    selected_weight: float,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    expected_seeds = sorted(int(seed) for seed in config["frontier"]["confirmation_seeds"])
    rows = []
    for payload in evaluations:
        if not payload.get("complete") or payload.get("repository") != repository:
            raise ValueError("confirmation evaluation is incomplete or from another revision")
        seed = int(payload["protocol"]["evaluation_seed"])
        paired = payload.get("paired_frontier", [])
        selected = [
            row
            for row in paired
            if math.isclose(
                float(row["decoder_weight"]), selected_weight, rel_tol=0, abs_tol=1e-12
            )
        ]
        if len(selected) != 1:
            raise ValueError(f"seed {seed} does not have one selected-weight comparison")
        comparison = selected[0]
        baseline = payload["models"][comparison["baseline"]]
        candidate = payload["models"][comparison["candidate"]]
        if int(baseline["seed"]) != seed or int(candidate["seed"]) != seed:
            raise ValueError("confirmation model seed differs from evaluation seed")
        rows.append(
            {
                "seed": seed,
                "baseline": comparison["baseline"],
                "candidate": comparison["candidate"],
                "mse": baseline,
                "dpsae": candidate,
                "nmse_ratio_to_mse": float(comparison["nmse_ratio_to_mse"]),
                "nmse_change_percent": float(comparison["nmse_change_percent"]),
                "exact_decoder_reduction": float(comparison["exact_decoder_reduction"]),
                "exact_decoder_reduction_ci95": [
                    float(value) for value in comparison["exact_decoder_reduction_ci95"]
                ],
            }
        )
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != expected_seeds:
        raise ValueError("confirmation evaluations do not match the frozen seed set")

    gate = config["frontier"]["confirmation_gate"]
    checks = {
        "nmse_ratio_every_seed": all(
            row["nmse_ratio_to_mse"] <= float(gate["maximum_nmse_ratio_every_seed"])
            for row in rows
        ),
        "median_exact_decoder_reduction": statistics.median(
            row["exact_decoder_reduction"] for row in rows
        )
        >= float(gate["minimum_median_exact_decoder_reduction"]),
        "positive_reduction_every_seed": (
            not bool(gate["require_positive_reduction_every_seed"])
            or all(row["exact_decoder_reduction"] > 0 for row in rows)
        ),
        "ci_excludes_zero_every_seed": (
            not bool(gate["require_ci_excludes_zero_every_seed"])
            or all(row["exact_decoder_reduction_ci95"][0] > 0 for row in rows)
        ),
    }
    return {
        "selected_decoder_weight": selected_weight,
        "expected_seeds": expected_seeds,
        "rows": rows,
        "gate": gate,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, nargs="+", required=True)
    parser.add_argument("--training-done", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--selected-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = repository_state()
    if repository["dirty"]:
        raise RuntimeError(f"confirmation summary requires a clean repository: {repository['status']}")
    config = json.loads(args.config.read_text())
    source = json.loads((ROOT / config["source_config"]).read_text())
    manifest = json.loads(args.run_manifest.read_text())
    training_done = json.loads(args.training_done.read_text())
    if manifest.get("repository") != repository or training_done.get("repository") != repository:
        raise ValueError("run manifest, training, and summary revisions must match")
    if not training_done.get("complete"):
        raise ValueError("confirmation training is incomplete")
    expected_training = tuple(
        int(value) for value in source["corpus"]["ranges"]["confirmation"]
    )
    if tuple(training_done["stream"]["range"]) != expected_training:
        raise ValueError("confirmation training used the wrong source interval")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("repository") != repository:
        raise ValueError("confirmation cache was not generated by the clean revision")
    if cache.get("normalized_with_sha256") != input_record(args.calibration)["sha256"]:
        raise ValueError("confirmation cache used another activation normalization")
    evaluation_interval = validate_natural_cache(cache, config, split="selection")
    disjoint_intervals(
        {
            "confirmation_training": expected_training,
            "confirmation_evaluation": evaluation_interval,
        }
    )
    evaluations = [json.loads(path.read_text()) for path in args.confirmation]
    summary = summarize_confirmation(
        evaluations,
        config,
        selected_weight=args.selected_weight,
        repository=repository,
    )
    result = {
        "complete": True,
        "experiment": "exp08_clean_confirmation_summary",
        **summary,
        "protocol": {
            "training_interval": list(expected_training),
            "evaluation_interval": list(evaluation_interval),
            "split_label": "clean confirmation [190M,195M)",
        },
        "inputs": {
            "evaluations": [input_record(path) for path in args.confirmation],
            "training_done": input_record(args.training_done),
            "models": input_record(args.models),
            "cache": input_record(args.cache),
            "calibration": input_record(args.calibration),
            "config": input_record(args.config),
            "run_manifest": input_record(args.run_manifest),
            "summarizer": input_record(Path(__file__)),
        },
        "repository": repository,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
