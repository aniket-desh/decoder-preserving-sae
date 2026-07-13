#!/usr/bin/env python3
"""One-step full-shape memory gate for the Experiment 4 confirmation fleet."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from dpsae.language_training import SAETrainSpec, TrainingFleet


ROOT = Path(__file__).resolve().parents[1]


def gibibytes(value: int) -> float:
    return value / 2**30


def main() -> None:
    config = json.loads((ROOT / "configs" / "exp04_ioi_mechanism.json").read_text())
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    k = config["sae"]["primary_k"]
    specs = [
        spec
        for seed in config["training"]["confirmation_seeds"]
        for spec in (
            SAETrainSpec(f"mse_s{seed}", "mse", seed, k),
            SAETrainSpec(f"dpsae_s{seed}", "dpsae", seed, k, decoder_weight=0.5),
            SAETrainSpec(f"whitening_s{seed}", "whitening", seed, k),
        )
    ]
    fleet = TrainingFleet(
        specs,
        input_dim=768,
        dictionary_size=config["sae"]["dictionary_size"],
        learning_rate=config["sae"]["learning_rate"],
        device=device,
        whitening=torch.eye(768, device=device),
        dead_after_steps=config["sae"]["dead_after_steps"],
        aux_k=config["sae"]["aux_k"],
    )
    activations = torch.randn(2048, 768, device=device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = fleet.train_batch(
        activations,
        step=1,
        ridge=1.0,
        group_size=config["geometry"]["group_size"],
        probes=config["geometry"]["probes"],
        probe_seed=config["seed"],
    )
    torch.cuda.synchronize()
    print(f"models={len(specs)} dictionary={config['sae']['dictionary_size']} batch=2048")
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    print(f"peak_allocated_gib={gibibytes(torch.cuda.max_memory_allocated()):.3f}")
    print(f"peak_reserved_gib={gibibytes(torch.cuda.max_memory_reserved()):.3f}")
    print(f"finite={all(torch.isfinite(torch.tensor(row['loss'])) for row in metrics.values())}")


if __name__ == "__main__":
    main()
