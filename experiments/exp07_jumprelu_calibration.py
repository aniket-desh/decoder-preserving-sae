#!/usr/bin/env python3
"""Calibrate JumpReLU sparsity controllers without reading task outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dpsae.mech_analysis import load_sae


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp07_task_fidelity.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def repository_state() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def bootstrap_mean_interval(
    values: Tensor,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    values = values.detach().cpu().double().flatten()
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(values), (samples, len(values)), generator=generator)
    estimates = values[draws].mean(1)
    tail = (1 - confidence) / 2
    return {
        "estimate": float(values.mean()),
        "low": float(estimates.quantile(tail)),
        "high": float(estimates.quantile(1 - tail)),
    }


@torch.inference_mode()
def model_l0(
    payload: dict[str, Any],
    activations: Tensor,
    *,
    device: torch.device,
    batch_tokens: int,
) -> tuple[Tensor, dict[str, Any]]:
    model = load_sae(payload, input_dim=activations.shape[-1], device=device)
    model.eval()
    counts = []
    for batch in activations.split(batch_tokens):
        code = model.encode(batch.to(device).float(), use_threshold=True)
        counts.append((code != 0).sum(1).cpu())
    thresholds = model.jump_threshold.detach()
    finite = all(torch.isfinite(value).all() for value in model.state_dict().values())
    health = {
        "finite_state": bool(finite),
        "threshold_minimum": float(thresholds.min()),
        "threshold_mean": float(thresholds.mean()),
        "threshold_maximum": float(thresholds.max()),
        "threshold_updates": int(model.threshold_updates.item()),
    }
    del model
    return torch.cat(counts), health


def evaluate_models(
    models_path: Path,
    cache: dict[str, Any],
    settings: dict[str, Any],
    *,
    device: torch.device,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payloads = torch.load(models_path, map_location="cpu", weights_only=False)
    activations = cache["activations"].flatten(0, 1)
    sequence_tokens = int(cache["activations"].shape[1])
    rows = []
    for index, (name, payload) in enumerate(sorted(payloads.items())):
        method = str(payload["spec"]["method"])
        if method not in {"mse", "dpsae"}:
            continue
        if payload.get("sparsity_mode") != "jump_relu":
            raise ValueError(f"{name} is not a JumpReLU checkpoint")
        counts, health = model_l0(
            payload,
            activations,
            device=device,
            batch_tokens=int(settings["evaluation_batch_tokens"]),
        )
        sequence_l0 = counts.reshape(-1, sequence_tokens).double().mean(1)
        interval = bootstrap_mean_interval(
            sequence_l0,
            samples=int(settings["bootstrap_samples"]),
            seed=int(settings["bootstrap_seed"]) + seed_offset + index,
        )
        rows.append(
            {
                "model": name,
                "method": method,
                "tokens": len(counts),
                "sequence_blocks": len(sequence_l0),
                "l0": interval["estimate"],
                "l0_ci95": [interval["low"], interval["high"]],
                "l0_sequence_p10": float(torch.quantile(sequence_l0, 0.10)),
                "l0_sequence_median": float(sequence_l0.median()),
                "l0_sequence_p90": float(torch.quantile(sequence_l0, 0.90)),
                **health,
            }
        )
    if {row["method"] for row in rows} != {"mse", "dpsae"}:
        raise ValueError("calibration checkpoint must contain one MSE and one DPSAE model")
    return rows, {"models": input_record(models_path)}


def interpolate_bracket(
    rows: list[dict[str, Any]],
    *,
    method: str,
    target: float,
) -> dict[str, Any]:
    candidates = sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: float(row["multiplier"]),
    )
    for low, high in zip(candidates, candidates[1:]):
        low_l0 = float(low.get("fitted_l0", low["l0"]))
        high_l0 = float(high.get("fitted_l0", high["l0"]))
        if low_l0 <= target <= high_l0 and high_l0 > low_l0:
            multiplier = float(low["multiplier"]) + (
                target - low_l0
            ) * (float(high["multiplier"]) - float(low["multiplier"])) / (
                high_l0 - low_l0
            )
            return {
                "method": method,
                "target_l0": target,
                "bracket": [
                    {"multiplier": low["multiplier"], "l0": low_l0},
                    {"multiplier": high["multiplier"], "l0": high_l0},
                ],
                "interpolated_multiplier": round(multiplier, 6),
            }
    raise RuntimeError(f"no adjacent held-out L0 bracket for {method}")


def isotonic_non_decreasing(values: list[float]) -> list[float]:
    """Return the equal-weight least-squares nondecreasing fit."""

    blocks: list[dict[str, float | int]] = []
    for index, value in enumerate(values):
        blocks.append({"start": index, "stop": index + 1, "sum": value, "count": 1})
        while len(blocks) >= 2:
            previous, current = blocks[-2:]
            previous_mean = float(previous["sum"]) / int(previous["count"])
            current_mean = float(current["sum"]) / int(current["count"])
            if previous_mean <= current_mean:
                break
            blocks[-2:] = [
                {
                    "start": previous["start"],
                    "stop": current["stop"],
                    "sum": float(previous["sum"]) + float(current["sum"]),
                    "count": int(previous["count"]) + int(current["count"]),
                }
            ]
    fitted = [0.0] * len(values)
    for block in blocks:
        mean = float(block["sum"]) / int(block["count"])
        for index in range(int(block["start"]), int(block["stop"])):
            fitted[index] = mean
    return fitted


def evaluate_grid(
    config: dict[str, Any],
    output: Path,
    *,
    device: torch.device,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    settings = config["jump_relu_calibration"]
    output.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f}")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    cache_path = ROOT / settings["cache"]
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    rows = []
    inputs: dict[str, Any] = {"cache": input_record(cache_path)}
    for index, point in enumerate(settings["grid"]):
        models_path = ROOT / point["models"]
        point_rows, point_inputs = evaluate_models(
            models_path,
            cache,
            settings,
            device=device,
            seed_offset=100 * index,
        )
        for row in point_rows:
            rows.append({"multiplier": float(point["multiplier"]), **row})
        inputs[f"models_{point['multiplier']:g}"] = point_inputs["models"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for method in ("mse", "dpsae"):
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["multiplier"],
        )
        fitted = isotonic_non_decreasing([float(row["l0"]) for row in method_rows])
        for row, fitted_l0 in zip(method_rows, fitted):
            row["fitted_l0"] = fitted_l0
    selection = {
        method: interpolate_bracket(
            rows,
            method=method,
            target=float(settings["target_l0"]),
        )
        for method in ("mse", "dpsae")
    }
    result = {
        "complete": True,
        "experiment": "jumprelu_decoder_blind_controller_grid",
        "rows": rows,
        "selection": selection,
        "selection_rule": (
            "interpolate independently between the first adjacent controller "
            "multipliers whose isotonic held-out mean-L0 fit brackets 32; do not "
            "read NMSE or decoder outcomes"
        ),
        "inputs": {
            **inputs,
            "config": input_record(Path(config["_config_path"])),
            "evaluator": input_record(Path(__file__)),
        },
        "repository": repository_state(),
        "resources": {
            "device": str(device),
            "free_gib_at_start": free_gib,
            "gpu_memory_fraction_cap": gpu_memory_fraction if device.type == "cuda" else None,
            "peak_allocated_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
            ),
        },
    }
    atomic_json(output / "jump_relu_controller_grid.json", result)
    return result


def evaluate_pair(
    config: dict[str, Any],
    output: Path,
    models_path: Path,
    label: str,
    *,
    device: torch.device,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    settings = config["jump_relu_calibration"]
    output.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f}")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    cache_path = ROOT / settings["cache"]
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    rows, pair_inputs = evaluate_models(
        models_path,
        cache,
        settings,
        device=device,
        seed_offset=10_000,
    )
    low, high = map(float, settings["accepted_l0_interval"])
    by_method = {row["method"]: row for row in rows}
    point_estimates_pass = all(low <= row["l0"] <= high for row in rows)
    confidence_intervals_pass = all(
        low <= row["l0_ci95"][0] and row["l0_ci95"][1] <= high for row in rows
    )
    inter_method_gap = abs(by_method["mse"]["l0"] - by_method["dpsae"]["l0"])
    finite = all(row["finite_state"] for row in rows)
    result = {
        "complete": True,
        "experiment": "jumprelu_decoder_blind_pair_gate",
        "label": label,
        "rows": rows,
        "checks": {
            "point_estimates_in_band": point_estimates_pass,
            "confidence_intervals_in_band": confidence_intervals_pass,
            "inter_method_l0_gap": inter_method_gap
            <= float(settings["maximum_inter_method_l0_gap"]),
            "finite_states": finite,
        },
        "inter_method_l0_gap": inter_method_gap,
        "advance": bool(
            point_estimates_pass
            and confidence_intervals_pass
            and inter_method_gap <= float(settings["maximum_inter_method_l0_gap"])
            and finite
        ),
        "outcomes_consulted": ["held-out L0", "threshold health", "finite state"],
        "outcomes_sealed": ["NMSE", "decoder distortion", "language-model loss"],
        "inputs": {
            **pair_inputs,
            "cache": input_record(cache_path),
            "config": input_record(Path(config["_config_path"])),
            "evaluator": input_record(Path(__file__)),
        },
        "repository": repository_state(),
    }
    atomic_json(output / f"jump_relu_{label}_l0_gate.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grid", "pair"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--label", default="pair")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    config["_config_path"] = str(args.config.resolve())
    output = args.output or ROOT / config["output"]
    common = {
        "device": torch.device(args.device),
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "minimum_free_gib": args.minimum_free_gib,
    }
    if args.mode == "grid":
        result = evaluate_grid(config, output, **common)
        print(json.dumps(result["selection"], indent=2), flush=True)
    else:
        if args.models is None:
            raise ValueError("--models is required in pair mode")
        result = evaluate_pair(
            config,
            output,
            args.models,
            args.label,
            **common,
        )
        print(json.dumps({"advance": result["advance"], "checks": result["checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
