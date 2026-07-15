#!/usr/bin/env python3
"""Aggregate measured single-GPU time for the Exp08 closure queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.exp08_language_evidence import input_record, repository_state


def read_complete(path: Path, repository: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        raise ValueError(f"incomplete compute input: {path}")
    if payload.get("repository") != repository:
        raise ValueError(f"compute input uses another repository state: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = repository_state()
    if repository["dirty"]:
        raise RuntimeError("compute summary requires a clean repository")
    manifest = json.loads(args.run_manifest.read_text())
    if manifest.get("repository") != repository:
        raise ValueError("run manifest differs from the compute summarizer revision")

    paths = {
        "gamma_training": args.root / "gamma_sweep/done.json",
        "gamma_evaluation": args.root / "gamma_sweep_selection.json",
        "confirmation_training": args.root / "confirmation/done.json",
        "confirmation_seed0_evaluation": args.root / "confirmation_seed0.json",
        "confirmation_seed1_evaluation": args.root / "confirmation_seed1.json",
        "confirmation_seed2_evaluation": args.root / "confirmation_seed2.json",
        "robustness": args.root / "evidence/robustness.json",
        "frozen_fidelity": args.root / "evidence/frozen_fidelity.json",
        "training_overhead": args.root / "evidence/training_overhead.json",
        "task_spectrum": args.root / "task_spectrum/advantage_spectrum_summary.json",
    }
    payloads = {name: read_complete(path, repository) for name, path in paths.items()}
    cache_seconds = float((args.root / "cache_wall_seconds.txt").read_text().strip())
    training_seconds = {
        name: float(payloads[name]["training_seconds_cumulative"])
        for name in ("gamma_training", "confirmation_training")
    }
    evaluation_seconds = {
        name: float(payload["wall_seconds"])
        for name, payload in payloads.items()
        if name not in training_seconds
    }
    segments = {"clean_cache_generation": cache_seconds, **training_seconds, **evaluation_seconds}
    training_total = sum(training_seconds.values())
    evaluation_total = cache_seconds + sum(evaluation_seconds.values())
    total = training_total + evaluation_total
    result = {
        "complete": True,
        "experiment": "exp08_compute_summary",
        "scope": (
            "measured active-stage time on one GPU; excludes queue idle time and figure rendering"
        ),
        "gpu_count": 1,
        "segments_seconds": segments,
        "training_gpu_hours": training_total / 3600,
        "evaluation_and_cache_gpu_hours": evaluation_total / 3600,
        "total_measured_gpu_hours": total / 3600,
        "training_fleets": {
            name: {
                "models": len(payloads[name]["specs"]),
                "stream_tokens": int(payloads[name]["tokens_seen"]),
                "model_tokens": int(payloads[name]["tokens_seen"])
                * len(payloads[name]["specs"]),
            }
            for name in ("gamma_training", "confirmation_training")
        },
        "inputs": {name: input_record(path) for name, path in paths.items()}
        | {
            "cache_wall_seconds": input_record(args.root / "cache_wall_seconds.txt"),
            "run_manifest": input_record(args.run_manifest),
            "summarizer": input_record(Path(__file__)),
        },
        "repository": repository,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
