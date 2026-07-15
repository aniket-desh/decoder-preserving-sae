#!/usr/bin/env python3
"""Calibrate JumpReLU sparsity controllers without reading task outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def bootstrap_paired_gap(
    left: Tensor,
    right: Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left = left.detach().cpu().double().flatten()
    right = right.detach().cpu().double().flatten()
    if left.shape != right.shape:
        raise ValueError("paired L0 sequences must have matching shapes")
    difference = left - right
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(difference),
        (samples, len(difference)),
        generator=generator,
    )
    draws = difference[indices].mean(1)
    return {
        "signed_estimate": float(difference.mean()),
        "absolute_estimate": abs(float(difference.mean())),
        "signed_ci95": [
            float(torch.quantile(draws, 0.025)),
            float(torch.quantile(draws, 0.975)),
        ],
        "absolute_upper95": float(torch.quantile(draws.abs(), 0.95)),
        "sequence_blocks": len(difference),
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
        "dictionary_size": thresholds.numel(),
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
    sequence_l0_by_method = {}
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
        sequence_l0_by_method[method] = sequence_l0
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
    paired_gap = bootstrap_paired_gap(
        sequence_l0_by_method["mse"],
        sequence_l0_by_method["dpsae"],
        samples=int(settings["bootstrap_samples"]),
        seed=int(settings["bootstrap_seed"]) + seed_offset + 90_000,
    )
    return rows, {
        "models": input_record(models_path),
        "paired_l0_gap": paired_gap,
    }


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
    training_log_path: Path | None,
    run_done_path: Path | None,
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
    point_estimates_pass = all(low <= row["l0"] <= high for row in rows)
    confidence_intervals_pass = all(
        low <= row["l0_ci95"][0] and row["l0_ci95"][1] <= high for row in rows
    )
    paired_gap = pair_inputs["paired_l0_gap"]
    inter_method_gap = float(paired_gap["absolute_estimate"])
    finite = all(row["finite_state"] for row in rows)
    checks = {
        "point_estimates_in_band": point_estimates_pass,
        "confidence_intervals_in_band": confidence_intervals_pass,
        "inter_method_l0_gap_upper95": float(paired_gap["absolute_upper95"])
        <= float(settings["maximum_inter_method_l0_gap"]),
        "finite_states": finite,
    }
    trajectory = None
    provenance = None
    extra_inputs = {}
    if (
        label == "full_horizon_screen"
        and training_log_path is None
        and run_done_path is None
    ):
        fresh = settings["full_horizon_weight_grid"]["fresh_screen"]
        training_log_path = ROOT / fresh["training_log"]
        run_done_path = ROOT / fresh["done"]
    if (training_log_path is None) != (run_done_path is None):
        raise ValueError("training log and done metadata must be supplied together")
    if training_log_path is not None and run_done_path is not None:
        grid = settings["full_horizon_weight_grid"]
        trajectory = late_l0_trajectory(
            training_log_path,
            records=int(grid["late_window_records"]),
        )
        by_name = {row["model"]: row for row in rows}
        if set(by_name) != set(trajectory):
            raise RuntimeError("training log and evaluated models do not match")
        for name, values in trajectory.items():
            values["dead_fraction_maximum"] = (
                values["dead_maximum"] / by_name[name]["dictionary_size"]
            )
        checks["late_drift"] = all(
            abs(values["late_half_shift"])
            <= float(grid["maximum_absolute_late_half_shift"])
            and abs(values["l0_slope_per_million_tokens"])
            <= float(grid["maximum_absolute_l0_slope_per_million_tokens"])
            for values in trajectory.values()
        )
        checks["dead_fraction"] = all(
            values["dead_fraction_maximum"]
            <= float(grid["maximum_dead_fraction"])
            for values in trajectory.values()
        )
        checks["trajectory_finite_health"] = all(
            values["finite_health"] for values in trajectory.values()
        )
        selection_path = output / "jump_relu_full_horizon_weight_grid.json"
        selection = json.loads(selection_path.read_text())["selection"]
        if selection is None:
            raise RuntimeError("full-horizon screen has no selected calibration weight")
        expected = {**grid, **grid["fresh_screen"]}
        done = json.loads(run_done_path.read_text())
        provenance = run_provenance_checks(
            done,
            weight=float(selection["weight"]),
            expected=expected,
        )
        expected_models = ROOT / grid["fresh_screen"]["models"]
        provenance["paths_match_frozen_screen"] = (
            models_path.resolve() == expected_models.resolve()
            and training_log_path.resolve()
            == (ROOT / grid["fresh_screen"]["training_log"]).resolve()
            and run_done_path.resolve()
            == (ROOT / grid["fresh_screen"]["done"]).resolve()
        )
        checks["run_provenance"] = all(provenance.values())
        extra_inputs = {
            "training_log": input_record(training_log_path),
            "run_done": input_record(run_done_path),
            "weight_selection": input_record(selection_path),
        }
    result = {
        "complete": True,
        "experiment": "jumprelu_decoder_blind_pair_gate",
        "label": label,
        "rows": rows,
        "checks": checks,
        "inter_method_l0_gap": inter_method_gap,
        "paired_l0_gap": paired_gap,
        "trajectory": trajectory,
        "provenance_checks": provenance,
        "advance": all(checks.values()),
        "outcomes_consulted": [
            "held-out L0",
            "paired L0 gap",
            "late-window L0 trajectory when supplied",
            "dead-feature count when supplied",
            "threshold health",
            "finite state",
            "run provenance when supplied",
        ],
        "outcomes_sealed": ["NMSE", "decoder distortion", "language-model loss"],
        "inputs": {
            "models": pair_inputs["models"],
            **extra_inputs,
            "cache": input_record(cache_path),
            "config": input_record(Path(config["_config_path"])),
            "evaluator": input_record(Path(__file__)),
        },
        "repository": repository_state(),
    }
    atomic_json(output / f"jump_relu_{label}_l0_gate.json", result)
    return result


def late_l0_trajectory(
    log_path: Path,
    *,
    records: int,
) -> dict[str, dict[str, Any]]:
    """Read only sparsity and health fields from a training log tail."""

    if records < 4:
        raise ValueError("late-window trajectory requires at least four records")
    raw = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    if len(raw) < records:
        raise ValueError(f"{log_path} has only {len(raw)} records")
    tail = raw[-records:]
    names = sorted(tail[0]["models"])
    result = {}
    for name in names:
        values = torch.tensor(
            [float(record["models"][name]["l0"]) for record in tail],
            dtype=torch.float64,
        )
        tokens = torch.tensor(
            [float(record["tokens_seen"]) for record in tail],
            dtype=torch.float64,
        )
        centered_tokens = tokens - tokens.mean()
        slope = float(
            (centered_tokens * (values - values.mean())).sum()
            / centered_tokens.square().sum().clamp_min(1e-30)
            * 1_000_000
        )
        half = records // 2
        first_mean = float(values[:half].mean())
        second_mean = float(values[-half:].mean())
        metrics = [record["models"][name] for record in tail]
        result[name] = {
            "records": records,
            "tokens_start": int(tokens[0]),
            "tokens_stop": int(tokens[-1]),
            "l0_mean": float(values.mean()),
            "l0_minimum": float(values.min()),
            "l0_maximum": float(values.max()),
            "late_half_shift": second_mean - first_mean,
            "l0_slope_per_million_tokens": slope,
            "dead_maximum": max(int(metric["dead"]) for metric in metrics),
            "threshold_minimum": min(
                float(metric["threshold_min"]) for metric in metrics
            ),
            "threshold_maximum": max(
                float(metric["threshold_max"]) for metric in metrics
            ),
            "finite_health": all(
                all(
                    math.isfinite(float(metric[key]))
                    for key in (
                        "l0",
                        "threshold_min",
                        "threshold_mean",
                        "threshold_max",
                    )
                )
                for metric in metrics
            ),
        }
    return result


def run_provenance_checks(
    done: dict[str, Any],
    *,
    weight: float,
    expected: dict[str, Any],
) -> dict[str, bool]:
    specs = done.get("specs", [])
    by_method = {spec.get("method"): spec for spec in specs}
    sparsity = done.get("sparsity_config", {})
    stream = done.get("stream", {})
    return {
        "complete": bool(done.get("complete")),
        "jump_relu_mode": done.get("sparsity_mode") == "jump_relu",
        "shared_sparsity_weight": math.isclose(
            float(sparsity.get("target_l0_loss_weight", math.nan)),
            weight,
        ),
        "shared_threshold_lr": math.isclose(
            float(sparsity.get("threshold_lr_multiplier", math.nan)),
            float(expected["shared_threshold_lr_multiplier"]),
        )
        and "threshold_lr_multipliers_by_method" not in sparsity,
        "token_budget": int(done.get("tokens_seen", -1))
        >= int(expected["token_budget"]),
        "source_range": stream.get("range_name") == expected["source_range_name"],
        "data_seed": int(stream.get("data_seed", -1)) == int(expected["data_seed"]),
        "probe_seed": int(stream.get("probe_seed_base", -1))
        == int(expected["probe_seed_base"]),
        "model_pair": len(specs) == 2 and set(by_method) == {"mse", "dpsae"},
        "model_seed": bool(specs)
        and all(int(spec.get("seed", -1)) == int(expected["model_seed"]) for spec in specs),
        "decoder_weight": "dpsae" in by_method
        and math.isclose(
            float(by_method["dpsae"].get("decoder_weight", math.nan)),
            float(expected["decoder_weight"]),
        ),
    }


def evaluate_sparsity_weight_grid(
    config: dict[str, Any],
    output: Path,
    *,
    device: torch.device,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    """Select a shared full-horizon JumpReLU sparsity weight using L0 only."""

    settings = config["jump_relu_calibration"]
    grid = settings["full_horizon_weight_grid"]
    output.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f}"
        )
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    cache_path = ROOT / settings["cache"]
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    low, high = map(float, settings["accepted_l0_interval"])
    rows = []
    point_metadata: dict[float, dict[str, Any]] = {}
    inputs: dict[str, Any] = {"cache": input_record(cache_path)}
    for index, point in enumerate(grid["points"]):
        weight = float(point["weight"])
        models_path = ROOT / point["models"]
        log_path = ROOT / point["training_log"]
        done_path = ROOT / point["done"]
        model_rows, model_inputs = evaluate_models(
            models_path,
            cache,
            settings,
            device=device,
            seed_offset=20_000 + 100 * index,
        )
        trajectories = late_l0_trajectory(
            log_path,
            records=int(grid["late_window_records"]),
        )
        done = json.loads(done_path.read_text())
        provenance = run_provenance_checks(
            done,
            weight=weight,
            expected=grid,
        )
        provenance["paths_colocated"] = (
            models_path.resolve().parent
            == log_path.resolve().parent
            == done_path.resolve().parent
        )
        point_metadata[weight] = {
            "paired_l0_gap": model_inputs["paired_l0_gap"],
            "provenance_checks": provenance,
        }
        by_name = {row["model"]: row for row in model_rows}
        if set(by_name) != set(trajectories):
            raise RuntimeError("training log and exported models do not match")
        for name in sorted(by_name):
            row = {"weight": weight, **by_name[name], **trajectories[name]}
            row["dead_fraction_maximum"] = (
                row["dead_maximum"] / row["dictionary_size"]
            )
            row["l0_interval_in_band"] = bool(
                low <= row["l0_ci95"][0] and row["l0_ci95"][1] <= high
            )
            row["late_drift_pass"] = bool(
                abs(row["late_half_shift"])
                <= float(grid["maximum_absolute_late_half_shift"])
                and abs(row["l0_slope_per_million_tokens"])
                <= float(grid["maximum_absolute_l0_slope_per_million_tokens"])
            )
            rows.append(row)
        inputs[f"models_weight_{weight:g}"] = input_record(models_path)
        inputs[f"training_log_weight_{weight:g}"] = input_record(log_path)
        inputs[f"done_weight_{weight:g}"] = input_record(done_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    weight_gates = []
    for point in grid["points"]:
        weight = float(point["weight"])
        pair = [row for row in rows if row["weight"] == weight]
        by_method = {row["method"]: row for row in pair}
        if set(by_method) != {"mse", "dpsae"}:
            raise RuntimeError("every sparsity weight must contain one model per method")
        paired_gap = point_metadata[weight]["paired_l0_gap"]
        gap = float(paired_gap["absolute_estimate"])
        checks = {
            "both_l0_intervals_in_band": all(
                row["l0_interval_in_band"] for row in pair
            ),
            "inter_method_l0_gap_upper95": float(paired_gap["absolute_upper95"])
            <= float(settings["maximum_inter_method_l0_gap"]),
            "late_drift": all(row["late_drift_pass"] for row in pair),
            "finite_health": all(
                row["finite_state"] and row["finite_health"] for row in pair
            ),
            "dead_fraction": all(
                row["dead_fraction_maximum"]
                <= float(grid["maximum_dead_fraction"])
                for row in pair
            ),
            "run_provenance": all(
                point_metadata[weight]["provenance_checks"].values()
            ),
        }
        weight_gates.append(
            {
                "weight": weight,
                "checks": checks,
                "inter_method_l0_gap": gap,
                "paired_l0_gap": paired_gap,
                "provenance_checks": point_metadata[weight]["provenance_checks"],
                "advance": all(checks.values()),
            }
        )
    advancing = [row for row in weight_gates if row["advance"]]
    selected = min(advancing, key=lambda row: row["weight"]) if advancing else None
    result = {
        "complete": True,
        "experiment": "jumprelu_decoder_blind_full_horizon_weight_grid",
        "rows": rows,
        "weight_gates": weight_gates,
        "selection": selected,
        "selection_rule": (
            "choose the smallest shared sparsity-loss weight whose held-out L0 "
            "confidence intervals both lie in the frozen band, whose method gap "
            "passes, and whose late L0 trajectory is stable; consult no task outcome"
        ),
        "outcomes_consulted": [
            "held-out L0",
            "late-window L0 trajectory",
            "dead-feature count",
            "threshold health",
            "finite state",
        ],
        "outcomes_sealed": [
            "NMSE",
            "decoder distortion",
            "language-model loss",
        ],
        "inputs": {
            **inputs,
            "config": input_record(Path(config["_config_path"])),
            "evaluator": input_record(Path(__file__)),
        },
        "repository": repository_state(),
        "resources": {
            "device": str(device),
            "free_gib_at_start": free_gib,
            "gpu_memory_fraction_cap": (
                gpu_memory_fraction if device.type == "cuda" else None
            ),
            "peak_allocated_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else None
            ),
        },
    }
    atomic_json(output / "jump_relu_full_horizon_weight_grid.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grid", "pair", "weight-grid"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--run-done", type=Path)
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
    elif args.mode == "pair":
        if args.models is None:
            raise ValueError("--models is required in pair mode")
        result = evaluate_pair(
            config,
            output,
            args.models,
            args.label,
            training_log_path=args.training_log,
            run_done_path=args.run_done,
            **common,
        )
        print(json.dumps({"advance": result["advance"], "checks": result["checks"]}, indent=2), flush=True)
    else:
        result = evaluate_sparsity_weight_grid(config, output, **common)
        print(json.dumps({"selection": result["selection"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
