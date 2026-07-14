#!/usr/bin/env python3
"""Audit the finite-probe decoder-loss gradient at frozen SAE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dpsae.language_training import sampled_decoder_loss_from_reference
from dpsae.mech_analysis import load_sae
from dpsae.task_fidelity import (
    exact_relative_gradients,
    fixed_radius_targets,
    ridge_gradient_factors,
    sampled_relative_gradients,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp07_task_fidelity.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
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


def percentile(values: list[float], probability: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability))


def log_slope(x: list[int], y: list[float]) -> float:
    log_x = torch.tensor(x, dtype=torch.float64).log()
    log_y = torch.tensor(y, dtype=torch.float64).clamp_min(1e-30).log()
    centered = log_x - log_x.mean()
    return float((centered * (log_y - log_y.mean())).sum() / centered.square().sum())


def gradient_metrics(samples: Tensor, exact: Tensor) -> dict[str, Tensor]:
    reduce_dims = tuple(range(1, samples.ndim))
    exact_norm = exact.square().sum().sqrt().clamp_min(1e-30)
    dot = (samples * exact.unsqueeze(0)).sum(dim=reduce_dims)
    norm = samples.square().sum(dim=reduce_dims).sqrt().clamp_min(1e-30)
    relative_error = (
        (samples - exact.unsqueeze(0)).square().sum(dim=reduce_dims).sqrt() / exact_norm
    )
    return {
        "dot": dot,
        "cosine": dot / (norm * exact_norm),
        "norm_ratio": norm / exact_norm,
        "relative_error": relative_error,
    }


def mean_gradient_metrics(mean: Tensor, exact: Tensor) -> dict[str, float]:
    exact_norm = exact.square().sum().sqrt().clamp_min(1e-30)
    mean_norm = mean.square().sum().sqrt().clamp_min(1e-30)
    dot = (mean * exact).sum()
    return {
        "cosine": float(dot / (mean_norm * exact_norm)),
        "norm_ratio": float(mean_norm / exact_norm),
        "relative_error": float((mean - exact).square().sum().sqrt() / exact_norm),
    }


def block_gradient_statistics(
    sampled_blocks: Tensor,
    fixed_blocks: Tensor,
    exact: Tensor,
) -> dict[str, Tensor]:
    """Compress block-mean gradients into exact inner-product statistics."""

    if sampled_blocks.shape != fixed_blocks.shape:
        raise ValueError("sampled and fixed block means must have matching shapes")
    if sampled_blocks.ndim != exact.ndim + 1 or sampled_blocks.shape[1:] != exact.shape:
        raise ValueError("block means must add one leading dimension to the exact gradient")
    sampled = sampled_blocks.double().flatten(1)
    fixed = fixed_blocks.double().flatten(1)
    reference = exact.double().flatten()
    return {
        "sampled_gram": sampled @ sampled.mT,
        "fixed_gram": fixed @ fixed.mT,
        "sampled_fixed_cross_gram": sampled @ fixed.mT,
        "sampled_exact_dot": sampled @ reference,
        "fixed_exact_dot": fixed @ reference,
        "exact_norm_squared": reference @ reference,
    }


def expected_gradient_metrics_from_weights(
    statistics: dict[str, Tensor],
    weights: Tensor,
    *,
    estimator: str,
) -> dict[str, Tensor]:
    """Evaluate U-statistical mean-gradient metrics for bootstrap weights."""

    if estimator == "sampled":
        gram = statistics["sampled_gram"]
        exact_dot = statistics["sampled_exact_dot"]
    elif estimator == "fixed":
        gram = statistics["fixed_gram"]
        exact_dot = statistics["fixed_exact_dot"]
    elif estimator == "sampled_minus_fixed":
        cross = statistics["sampled_fixed_cross_gram"]
        gram = (
            statistics["sampled_gram"]
            + statistics["fixed_gram"]
            - cross
            - cross.mT
        )
        exact_dot = None
    else:
        raise ValueError(f"unknown estimator {estimator}")

    weights = weights.to(dtype=torch.float64, device=gram.device)
    sample_count = weights.sum(1)
    if bool((sample_count <= 1).any()):
        raise ValueError("U-statistical metrics require at least two resampled blocks")
    weighted_quadratic = ((weights @ gram) * weights).sum(1)
    diagonal = weights @ gram.diagonal()
    off_diagonal = weighted_quadratic - diagonal
    pair_count = sample_count * (sample_count - 1)
    mean_norm_squared = off_diagonal / pair_count
    exact_norm_squared = statistics["exact_norm_squared"].clamp_min(1e-300)

    if estimator == "sampled_minus_fixed":
        return {
            "relative_bias": (
                mean_norm_squared.clamp_min(0) / exact_norm_squared
            ).sqrt(),
            "relative_bias_squared_unclamped": mean_norm_squared / exact_norm_squared,
        }

    dot_sum = weights @ exact_dot
    mean_dot = dot_sum / sample_count
    bias_squared = (
        off_diagonal
        - 2 * (sample_count - 1) * dot_sum
        + pair_count * exact_norm_squared
    ) / pair_count
    inferred_norm = mean_norm_squared.clamp_min(1e-300).sqrt()
    exact_norm = exact_norm_squared.sqrt()
    return {
        "cosine": (mean_dot / (inferred_norm * exact_norm)).clamp(-1, 1),
        "norm_ratio": inferred_norm / exact_norm,
        "relative_bias": (bias_squared.clamp_min(0) / exact_norm_squared).sqrt(),
        "relative_bias_squared_unclamped": bias_squared / exact_norm_squared,
    }


def interval(values: Tensor) -> list[float]:
    values = values.double()
    quantiles = torch.quantile(
        values,
        torch.tensor([0.025, 0.975], dtype=torch.float64, device=values.device),
    )
    return [float(quantiles[0]), float(quantiles[1])]


def bootstrap_weights(samples: int, blocks: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(blocks, (samples, blocks), generator=generator)
    weights = torch.zeros(samples, blocks, dtype=torch.float64)
    weights.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))
    return weights


def block_bootstrap_gradient_summary(
    statistics: dict[str, Tensor],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap whole paired blocks using recoverable Gram statistics."""

    blocks = len(statistics["sampled_exact_dot"])
    weights = bootstrap_weights(samples, blocks, seed)
    point_weights = torch.ones(1, blocks, dtype=torch.float64)
    result: dict[str, Any] = {}
    for estimator in ("sampled", "fixed", "sampled_minus_fixed"):
        point = expected_gradient_metrics_from_weights(
            statistics,
            point_weights,
            estimator=estimator,
        )
        draws = expected_gradient_metrics_from_weights(
            statistics,
            weights,
            estimator=estimator,
        )
        result[estimator] = {
            key: {
                "point": float(point[key][0]),
                "bootstrap95": interval(draws[key]),
            }
            for key in point
        }
    return result


@torch.inference_mode()
def training_reconstruction(model, activations: Tensor) -> Tensor:
    reconstruction, _ = model(activations.float(), use_threshold=False)
    return reconstruction


def target_covariances(targets: Tensor, probe_counts: list[int]) -> dict[int, Tensor]:
    result = {}
    covariance = torch.zeros(
        *targets.shape[:3],
        targets.shape[2],
        dtype=targets.dtype,
        device=targets.device,
    )
    wanted = set(probe_counts)
    for index in range(targets.shape[-1]):
        column = targets[..., index]
        covariance.add_(column.unsqueeze(-1) * column.unsqueeze(-2))
        count = index + 1
        if count in wanted:
            result[count] = covariance.clone()
    if set(result) != wanted:
        raise ValueError("probe counts exceed the generated target bank")
    return result


def autograd_spot_check(
    original: Tensor,
    reconstructed: Tensor,
    *,
    ridge: float,
    probes: int,
    seed: int,
    clamp_min: float,
) -> dict[str, float | int]:
    generator = torch.Generator(device=original.device).manual_seed(seed)
    targets, clamp_hits = fixed_radius_targets(
        1,
        original.shape[0],
        original.shape[1],
        probes,
        generator=generator,
        device=original.device,
        dtype=torch.float32,
        clamp_min=clamp_min,
    )
    targets = targets[0]
    covariance = targets @ targets.mT
    factors = ridge_gradient_factors(original.float(), reconstructed.float(), ridge=ridge)
    _, analytic, denominator = sampled_relative_gradients(
        factors,
        covariance.unsqueeze(0),
        probes=probes,
    )
    variable = reconstructed.float().detach().requires_grad_(True)
    loss = sampled_decoder_loss_from_reference(
        original.float(),
        variable,
        targets,
        ridge=ridge,
    )
    loss.backward()
    autograd = variable.grad
    analytic = analytic[0]
    exact_norm = autograd.square().sum().sqrt().clamp_min(1e-30)
    analytic_norm = analytic.square().sum().sqrt().clamp_min(1e-30)
    return {
        "target_rms_clamp_hits": clamp_hits,
        "denominator": float(denominator[0]),
        "loss": float(loss.detach()),
        "relative_error": float((analytic - autograd).square().sum().sqrt() / exact_norm),
        "cosine": float((analytic * autograd).sum() / (analytic_norm * exact_norm)),
    }


def run_model(
    config: dict[str, Any],
    output: Path,
    model_name: str,
    *,
    device: torch.device,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    started = time.time()
    settings = config["gradient_fidelity"]
    models_path = ROOT / settings["models"]
    cache_path = ROOT / settings["cache"]
    output.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f}")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)

    payloads = torch.load(models_path, map_location="cpu", weights_only=False)
    if model_name not in payloads:
        raise KeyError(f"unknown model {model_name}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    tokens_per_batch = int(settings["tokens_per_batch"])
    group_size = int(settings["group_size"])
    groups = tokens_per_batch // group_size
    batches = int(settings["batches"])
    if tokens_per_batch % group_size:
        raise ValueError("tokens per batch must divide into complete geometry groups")
    flat = cache["activations"].flatten(0, 1)
    if batches * tokens_per_batch > len(flat):
        raise ValueError("gradient batches exceed the held-out activation cache")
    model = load_sae(payloads[model_name], input_dim=flat.shape[-1], device=device)
    model.eval()
    probe_counts = [int(value) for value in settings["probe_counts"]]
    maximum_probes = max(probe_counts)
    banks = int(settings["banks"])
    bank_chunk = int(settings["bank_chunk"])
    if banks % bank_chunk:
        raise ValueError("banks must divide evenly by bank_chunk")
    bootstrap_blocks = int(settings["bootstrap_blocks"])
    if banks % bootstrap_blocks:
        raise ValueError("banks must divide evenly into bootstrap blocks")
    block_size = banks // bootstrap_blocks
    if block_size % bank_chunk:
        raise ValueError("bootstrap block size must divide evenly by bank_chunk")
    ridge = float(settings["ridge"])
    raw: dict[str, Any] = {}
    batch_rows = []
    total_target_clamp_hits = 0
    total_denominator_clamp_hits = 0
    autograd_result = None

    for batch_index in range(batches):
        print(f"{model_name}: batch {batch_index + 1}/{batches}", flush=True)
        start = batch_index * tokens_per_batch
        activations = flat[start : start + tokens_per_batch].to(device).float()
        reconstructed = training_reconstruction(model, activations)
        original_groups = activations.reshape(groups, group_size, -1).double()
        reconstructed_groups = reconstructed.reshape_as(original_groups).double()
        factors = ridge_gradient_factors(
            original_groups,
            reconstructed_groups,
            ridge=ridge,
        )
        exact_gram, exact_reconstruction = exact_relative_gradients(factors)
        exact_by_space = {
            "row_gram": exact_gram,
            "reconstruction": exact_reconstruction,
        }
        accumulators = {
            count: {
                space: {
                    "sampled_sum": torch.zeros_like(exact),
                    "fixed_sum": torch.zeros_like(exact),
                    "sampled_block_sum": torch.zeros_like(exact),
                    "fixed_block_sum": torch.zeros_like(exact),
                    "sampled_blocks": [],
                    "fixed_blocks": [],
                    "sampled": {key: [] for key in ("dot", "cosine", "norm_ratio", "relative_error")},
                    "fixed": {key: [] for key in ("dot", "cosine", "norm_ratio", "relative_error")},
                }
                for space, exact in exact_by_space.items()
            }
            for count in probe_counts
        }
        denominators = {count: [] for count in probe_counts}
        generator = torch.Generator(device=device).manual_seed(
            int(settings["probe_seed"]) + batch_index
        )
        for bank_start in range(0, banks, bank_chunk):
            targets, clamp_hits = fixed_radius_targets(
                bank_chunk,
                groups,
                group_size,
                maximum_probes,
                generator=generator,
                device=device,
                dtype=torch.float64,
                clamp_min=float(settings["target_rms_clamp"]),
            )
            total_target_clamp_hits += clamp_hits
            covariances = target_covariances(targets, probe_counts)
            for count in probe_counts:
                sampled_gram, sampled_reconstruction, denominator = sampled_relative_gradients(
                    factors,
                    covariances[count],
                    probes=count,
                )
                fixed_scale = denominator / (
                    count * factors["source_energy"].clamp_min(1e-12)
                )
                fixed_gram = sampled_gram * fixed_scale[:, None, None, None]
                fixed_reconstruction = (
                    sampled_reconstruction * fixed_scale[:, None, None, None]
                )
                total_denominator_clamp_hits += int(
                    (denominator <= float(settings["denominator_clamp"])).sum()
                )
                denominators[count].append(denominator.detach().cpu())
                for space, sampled, fixed in (
                    ("row_gram", sampled_gram, fixed_gram),
                    ("reconstruction", sampled_reconstruction, fixed_reconstruction),
                ):
                    exact = exact_by_space[space]
                    sampled_metrics = gradient_metrics(sampled, exact)
                    fixed_metrics = gradient_metrics(fixed, exact)
                    accumulator = accumulators[count][space]
                    accumulator["sampled_sum"].add_(sampled.sum(0))
                    accumulator["fixed_sum"].add_(fixed.sum(0))
                    accumulator["sampled_block_sum"].add_(sampled.sum(0))
                    accumulator["fixed_block_sum"].add_(fixed.sum(0))
                    for key in sampled_metrics:
                        accumulator["sampled"][key].append(sampled_metrics[key].detach().cpu())
                        accumulator["fixed"][key].append(fixed_metrics[key].detach().cpu())
                    if (bank_start + bank_chunk) % block_size == 0:
                        accumulator["sampled_blocks"].append(
                            (accumulator["sampled_block_sum"] / block_size)
                            .detach()
                            .float()
                            .cpu()
                        )
                        accumulator["fixed_blocks"].append(
                            (accumulator["fixed_block_sum"] / block_size)
                            .detach()
                            .float()
                            .cpu()
                        )
                        accumulator["sampled_block_sum"].zero_()
                        accumulator["fixed_block_sum"].zero_()
                del covariances[count], sampled_gram, sampled_reconstruction
                del fixed_gram, fixed_reconstruction

        rows_by_count = {}
        raw_batch = {}
        for count in probe_counts:
            denominator = torch.cat(denominators[count]).double()
            row = {
                "model": model_name,
                "batch": batch_index,
                "probes": count,
                "denominator_mean": float(denominator.mean()),
                "denominator_cv": float(denominator.std(unbiased=True) / denominator.mean().clamp_min(1e-30)),
                "spaces": {},
            }
            raw_batch[str(count)] = {"denominator": denominator.float()}
            for space, exact in exact_by_space.items():
                accumulator = accumulators[count][space]
                sampled_mean = accumulator["sampled_sum"] / banks
                fixed_mean = accumulator["fixed_sum"] / banks
                sampled_raw = {
                    key: torch.cat(values).double()
                    for key, values in accumulator["sampled"].items()
                }
                fixed_raw = {
                    key: torch.cat(values).double()
                    for key, values in accumulator["fixed"].items()
                }
                sampled_mean_metrics = mean_gradient_metrics(sampled_mean, exact)
                fixed_mean_metrics = mean_gradient_metrics(fixed_mean, exact)
                if len(accumulator["sampled_blocks"]) != bootstrap_blocks:
                    raise RuntimeError("incomplete sampled-gradient bootstrap blocks")
                sampled_blocks = torch.stack(accumulator["sampled_blocks"]).to(device)
                fixed_blocks = torch.stack(accumulator["fixed_blocks"]).to(device)
                sufficient = {
                    key: value.detach().cpu()
                    for key, value in block_gradient_statistics(
                        sampled_blocks,
                        fixed_blocks,
                        exact,
                    ).items()
                }
                bootstrap = block_bootstrap_gradient_summary(
                    sufficient,
                    samples=int(settings["bootstrap_samples"]),
                    seed=int(settings["bootstrap_seed"]) + batch_index,
                )
                sampled_rmse = float(sampled_raw["relative_error"].square().mean().sqrt())
                fixed_rmse = float(fixed_raw["relative_error"].square().mean().sqrt())
                row["spaces"][space] = {
                    "exact_gradient_norm": float(exact.square().sum().sqrt()),
                    "mean_gradient": sampled_mean_metrics,
                    "fixed_denominator_mean_gradient": fixed_mean_metrics,
                    "expected_gradient_inference": bootstrap,
                    "individual_positive_dot_fraction": float((sampled_raw["dot"] > 0).double().mean()),
                    "individual_cosine_median": float(sampled_raw["cosine"].median()),
                    "individual_cosine_p10": float(torch.quantile(sampled_raw["cosine"], 0.10)),
                    "relative_rmse": sampled_rmse,
                    "fixed_denominator_relative_rmse": fixed_rmse,
                    "fixed_denominator_mc_resolution": fixed_rmse / math.sqrt(banks),
                }
                raw_batch[str(count)][space] = {
                    "sampled": {key: value.float() for key, value in sampled_raw.items()},
                    "fixed": {key: value.float() for key, value in fixed_raw.items()},
                    "bootstrap_sufficient_statistics": sufficient,
                }
                del sampled_blocks, fixed_blocks, sufficient
            rows_by_count[count] = row
            batch_rows.append(row)

        slope_counts = [count for count in probe_counts if 2 <= count <= 32]
        for space in exact_by_space:
            slope = log_slope(
                slope_counts,
                [rows_by_count[count]["spaces"][space]["relative_rmse"] for count in slope_counts],
            )
            for count in probe_counts:
                rows_by_count[count]["spaces"][space]["rmse_log_slope_m2_to_m32"] = slope
        raw[str(batch_index)] = raw_batch

        if batch_index == int(settings["autograd_batch"]):
            autograd_result = autograd_spot_check(
                original_groups,
                reconstructed_groups,
                ridge=ridge,
                probes=int(settings["autograd_probes"]),
                seed=int(settings["probe_seed"]) + 1_000_000,
                clamp_min=float(settings["target_rms_clamp"]),
            )
        del activations, reconstructed, original_groups, reconstructed_groups, factors
        del exact_gram, exact_reconstruction, accumulators
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_path = output / f"gradient_fidelity_{model_name}_raw.pt"
    atomic_torch(raw_path, raw)
    resources = {
        "device": str(device),
        "free_gib_at_start": free_gib,
        "minimum_free_gib_guard": minimum_free_gib,
        "gpu_memory_fraction_cap": gpu_memory_fraction if device.type == "cuda" else None,
        "peak_allocated_gpu_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    payload = {
        "complete": True,
        "experiment": "finite_probe_gradient_fidelity",
        "model": model_name,
        "spec": payloads[model_name]["spec"],
        "batch_rows": batch_rows,
        "target_rms_clamp_hits": total_target_clamp_hits,
        "denominator_clamp_hits": total_denominator_clamp_hits,
        "autograd_spot_check": autograd_result,
        "protocol": {
            **settings,
            "precision": "float64 analytic audit; float32 autograd spot check",
            "support": (
                "training BatchTopK rule over each complete 2048-token batch; "
                "FP32 rule-level replay rather than bitwise BF16 support replay"
            ),
            "probe_banks_reused_across_models": True,
            "probe_counts_are_paired_prefixes": True,
            "bootstrap_blocks_are_paired_across_models_and_probe_counts": True,
        },
        "inputs": {
            "models": input_record(models_path),
            "cache": input_record(cache_path),
            "config": input_record(Path(config["_config_path"])),
            "evaluator": input_record(Path(__file__)),
            "task_fidelity_module": input_record(ROOT / "src/dpsae/task_fidelity.py"),
            "raw_metrics": input_record(raw_path),
        },
        "repository": repository_state(),
        "resources": resources,
        "wall_seconds": time.time() - started,
    }
    atomic_json(output / f"gradient_fidelity_{model_name}.json", payload)
    return payload


def model_names(settings: dict[str, Any]) -> list[str]:
    names = []
    for seed in settings["seeds"]:
        names.append(settings["baseline_template"].format(seed=seed))
        names.append(settings["candidate_template"].format(seed=seed))
    return names


def summarized_draw(values: Tensor, probability: float) -> Tensor:
    return torch.quantile(values, probability, dim=1)


def hierarchical_expected_gradient_summary(
    raw: dict[str, Any],
    settings: dict[str, Any],
    *,
    probes: int,
    space: str,
) -> dict[str, Any]:
    """Resample paired bank blocks inside resampled held-out batches."""

    samples = int(settings["bootstrap_samples"])
    batches = int(settings["batches"])
    blocks = int(settings["bootstrap_blocks"])
    seed = int(settings["bootstrap_seed"])
    by_batch: dict[str, list[Tensor]] = {
        "sampled_cosine": [],
        "sampled_norm_ratio": [],
        "sampled_relative_bias": [],
        "fixed_relative_bias": [],
        "sampled_minus_fixed_relative_bias": [],
    }
    points = {key: [] for key in by_batch}
    point_weights = torch.ones(1, blocks, dtype=torch.float64)
    for batch_index in range(batches):
        statistics = raw[str(batch_index)][str(probes)][space][
            "bootstrap_sufficient_statistics"
        ]
        weights = bootstrap_weights(samples, blocks, seed + batch_index)
        sampled = expected_gradient_metrics_from_weights(
            statistics,
            weights,
            estimator="sampled",
        )
        fixed = expected_gradient_metrics_from_weights(
            statistics,
            weights,
            estimator="fixed",
        )
        paired = expected_gradient_metrics_from_weights(
            statistics,
            weights,
            estimator="sampled_minus_fixed",
        )
        sampled_point = expected_gradient_metrics_from_weights(
            statistics,
            point_weights,
            estimator="sampled",
        )
        fixed_point = expected_gradient_metrics_from_weights(
            statistics,
            point_weights,
            estimator="fixed",
        )
        paired_point = expected_gradient_metrics_from_weights(
            statistics,
            point_weights,
            estimator="sampled_minus_fixed",
        )
        values = {
            "sampled_cosine": sampled["cosine"],
            "sampled_norm_ratio": sampled["norm_ratio"],
            "sampled_relative_bias": sampled["relative_bias"],
            "fixed_relative_bias": fixed["relative_bias"],
            "sampled_minus_fixed_relative_bias": paired["relative_bias"],
        }
        point_values = {
            "sampled_cosine": sampled_point["cosine"],
            "sampled_norm_ratio": sampled_point["norm_ratio"],
            "sampled_relative_bias": sampled_point["relative_bias"],
            "fixed_relative_bias": fixed_point["relative_bias"],
            "sampled_minus_fixed_relative_bias": paired_point["relative_bias"],
        }
        for key in by_batch:
            by_batch[key].append(values[key])
            points[key].append(float(point_values[key][0]))

    batch_generator = torch.Generator().manual_seed(seed + 10_000_000)
    batch_indices = torch.randint(
        batches,
        (samples, batches),
        generator=batch_generator,
    )
    draw_index = torch.arange(samples)[:, None]
    definitions = {
        "median_expected_cosine": ("sampled_cosine", 0.50),
        "p10_expected_cosine": ("sampled_cosine", 0.10),
        "median_expected_norm_ratio": ("sampled_norm_ratio", 0.50),
        "median_expected_relative_bias": ("sampled_relative_bias", 0.50),
        "p90_expected_relative_bias": ("sampled_relative_bias", 0.90),
        "p90_fixed_expected_relative_bias": ("fixed_relative_bias", 0.90),
        "median_paired_self_normalization_bias": (
            "sampled_minus_fixed_relative_bias",
            0.50,
        ),
        "p90_paired_self_normalization_bias": (
            "sampled_minus_fixed_relative_bias",
            0.90,
        ),
    }
    summary = {}
    for label, (key, probability) in definitions.items():
        batch_draws = torch.stack(by_batch[key])
        selected = batch_draws[batch_indices, draw_index]
        aggregate_draws = summarized_draw(selected, probability)
        point = percentile(points[key], probability)
        summary[label] = {
            "point": point,
            "hierarchical_bootstrap95": interval(aggregate_draws),
        }
    return summary


def one_sided_gate(
    result: dict[str, Any],
    threshold: float,
    *,
    direction: str,
) -> dict[str, bool]:
    low, high = result["hierarchical_bootstrap95"]
    point = float(result["point"])
    if direction == "minimum":
        return {
            "conservative_pass": low >= threshold,
            "decisive_fail": point < threshold and high < threshold,
        }
    if direction == "maximum":
        return {
            "conservative_pass": high <= threshold,
            "decisive_fail": point > threshold and low > threshold,
        }
    raise ValueError(f"unknown gate direction {direction}")


def interval_gate(
    result: dict[str, Any],
    lower: float,
    upper: float,
) -> dict[str, bool]:
    low, high = result["hierarchical_bootstrap95"]
    point = float(result["point"])
    return {
        "conservative_pass": low >= lower and high <= upper,
        "decisive_fail": (
            (point < lower and high < lower) or (point > upper and low > upper)
        ),
    }


def checkpoint_gate(
    payload: dict[str, Any],
    raw: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    gates = settings["gates"]
    primary = int(settings["autograd_probes"])
    rows = [row for row in payload["batch_rows"] if int(row["probes"]) == primary]
    denominator_cvs = [float(row["denominator_cv"]) for row in rows]
    checks: dict[str, bool] = {
        "target_rms_clamp_never_fired": payload["target_rms_clamp_hits"] == 0,
        "denominator_clamp_never_fired": payload["denominator_clamp_hits"] == 0,
        "median_denominator_cv": percentile(denominator_cvs, 0.50) <= float(gates["maximum_median_denominator_cv"]),
        "p90_denominator_cv": percentile(denominator_cvs, 0.90) <= float(gates["maximum_batch_p90_denominator_cv"]),
        "autograd_spot_check_present": payload["autograd_spot_check"] is not None,
    }
    summaries = {}
    expectation_inference = {}
    expectation_gate_details = {}
    expectation_conservative_passes = []
    expectation_decisive_failures = []
    exact_gradient_norms = []
    for space in ("row_gram", "reconstruction"):
        cosine = [float(row["spaces"][space]["mean_gradient"]["cosine"]) for row in rows]
        norm_ratio = [float(row["spaces"][space]["mean_gradient"]["norm_ratio"]) for row in rows]
        bias = [float(row["spaces"][space]["mean_gradient"]["relative_error"]) for row in rows]
        positive_dot = [float(row["spaces"][space]["individual_positive_dot_fraction"]) for row in rows]
        fixed_error = [float(row["spaces"][space]["fixed_denominator_mean_gradient"]["relative_error"]) for row in rows]
        fixed_resolution = [float(row["spaces"][space]["fixed_denominator_mc_resolution"]) for row in rows]
        slopes = [float(row["spaces"][space]["rmse_log_slope_m2_to_m32"]) for row in rows]
        space_exact_norms = [float(row["spaces"][space]["exact_gradient_norm"]) for row in rows]
        exact_gradient_norms.extend(space_exact_norms)
        summaries[space] = {
            "median_mean_gradient_cosine": percentile(cosine, 0.50),
            "p10_mean_gradient_cosine": percentile(cosine, 0.10),
            "median_norm_ratio": percentile(norm_ratio, 0.50),
            "median_relative_mean_bias": percentile(bias, 0.50),
            "p90_relative_mean_bias": percentile(bias, 0.90),
            "pooled_individual_positive_dot_fraction": sum(positive_dot) / len(positive_dot),
            "maximum_fixed_denominator_mean_error_over_resolution": max(
                error / max(float(gates["maximum_fixed_denominator_mean_error"]), resolution)
                for error, resolution in zip(fixed_error, fixed_resolution)
            ),
            "median_rmse_log_slope": percentile(slopes, 0.50),
        }
        checks[f"{space}_median_cosine"] = summaries[space]["median_mean_gradient_cosine"] >= float(gates["minimum_median_mean_gradient_cosine"])
        checks[f"{space}_p10_cosine"] = summaries[space]["p10_mean_gradient_cosine"] >= float(gates["minimum_batch_p10_mean_gradient_cosine"])
        checks[f"{space}_median_norm_ratio"] = float(gates["minimum_median_norm_ratio"]) <= summaries[space]["median_norm_ratio"] <= float(gates["maximum_median_norm_ratio"])
        checks[f"{space}_median_bias"] = summaries[space]["median_relative_mean_bias"] <= float(gates["maximum_median_relative_mean_bias"])
        checks[f"{space}_p90_bias"] = summaries[space]["p90_relative_mean_bias"] <= float(gates["maximum_batch_p90_relative_mean_bias"])
        checks[f"{space}_positive_dot"] = summaries[space]["pooled_individual_positive_dot_fraction"] >= float(gates["minimum_positive_dot_fraction"])
        checks[f"{space}_fixed_denominator"] = summaries[space]["maximum_fixed_denominator_mean_error_over_resolution"] <= 1.0
        checks[f"{space}_rmse_slope"] = float(gates["minimum_rmse_log_slope"]) <= summaries[space]["median_rmse_log_slope"] <= float(gates["maximum_rmse_log_slope"])
        inference = hierarchical_expected_gradient_summary(
            raw,
            settings,
            probes=primary,
            space=space,
        )
        expectation_inference[space] = inference
        gate_details = {
            "median_cosine": one_sided_gate(
                inference["median_expected_cosine"],
                float(gates["minimum_median_mean_gradient_cosine"]),
                direction="minimum",
            ),
            "p10_cosine": one_sided_gate(
                inference["p10_expected_cosine"],
                float(gates["minimum_batch_p10_mean_gradient_cosine"]),
                direction="minimum",
            ),
            "median_norm_ratio": interval_gate(
                inference["median_expected_norm_ratio"],
                float(gates["minimum_median_norm_ratio"]),
                float(gates["maximum_median_norm_ratio"]),
            ),
            "median_bias": one_sided_gate(
                inference["median_expected_relative_bias"],
                float(gates["maximum_median_relative_mean_bias"]),
                direction="maximum",
            ),
            "p90_bias": one_sided_gate(
                inference["p90_expected_relative_bias"],
                float(gates["maximum_batch_p90_relative_mean_bias"]),
                direction="maximum",
            ),
            "fixed_control_p90_bias": one_sided_gate(
                inference["p90_fixed_expected_relative_bias"],
                float(gates["maximum_fixed_denominator_mean_error"]),
                direction="maximum",
            ),
        }
        expectation_gate_details[space] = gate_details
        expectation_conservative_passes.extend(
            detail["conservative_pass"] for detail in gate_details.values()
        )
        expectation_decisive_failures.extend(
            detail["decisive_fail"] for detail in gate_details.values()
        )
    if payload["autograd_spot_check"] is not None:
        spot = payload["autograd_spot_check"]
        checks["autograd_relative_error"] = float(spot["relative_error"]) <= float(gates["maximum_autograd_relative_error"])
        checks["autograd_cosine"] = float(spot["cosine"]) >= float(gates["minimum_autograd_cosine"])
    else:
        checks["autograd_relative_error"] = False
        checks["autograd_cosine"] = False

    minimum_exact_norm = float(settings["minimum_exact_gradient_norm"])
    exact_norms_valid = all(
        math.isfinite(value) and value >= minimum_exact_norm
        for value in exact_gradient_norms
    )
    checks["exact_gradient_norms_resolved"] = exact_norms_valid
    auxiliary_keys = [
        "target_rms_clamp_never_fired",
        "denominator_clamp_never_fired",
        "median_denominator_cv",
        "p90_denominator_cv",
        "autograd_spot_check_present",
        "autograd_relative_error",
        "autograd_cosine",
        "row_gram_positive_dot",
        "row_gram_rmse_slope",
        "reconstruction_positive_dot",
        "reconstruction_rmse_slope",
    ]
    auxiliary_pass = all(checks[key] for key in auxiliary_keys)
    if not exact_norms_valid:
        expectation_status = "INCONCLUSIVE"
    elif not auxiliary_pass or any(expectation_decisive_failures):
        expectation_status = "FAIL"
    elif expectation_conservative_passes and all(expectation_conservative_passes):
        expectation_status = "PASS"
    else:
        expectation_status = "INCONCLUSIVE"
    return {
        "model": payload["model"],
        "method": payload["spec"]["method"],
        "probes": primary,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "finite_bank_point_gate_pass": all(checks.values()),
        "expected_gradient_fidelity_status": expectation_status,
        "summaries": summaries,
        "expected_gradient_inference": expectation_inference,
        "expected_gradient_gate_details": expectation_gate_details,
        "minimum_exact_gradient_norm": min(exact_gradient_norms),
        "denominator_cv": {
            "median": percentile(denominator_cvs, 0.50),
            "p90": percentile(denominator_cvs, 0.90),
        },
        "autograd_spot_check": payload["autograd_spot_check"],
    }


def summarize(config: dict[str, Any], output: Path) -> dict[str, Any]:
    settings = config["gradient_fidelity"]
    names = model_names(settings)
    payloads = [
        json.loads((output / f"gradient_fidelity_{name}.json").read_text())
        for name in names
    ]
    raw_payloads = [
        torch.load(
            output / f"gradient_fidelity_{name}_raw.pt",
            map_location="cpu",
            weights_only=False,
        )
        for name in names
    ]
    checkpoints = [
        checkpoint_gate(payload, raw, settings)
        for payload, raw in zip(payloads, raw_payloads)
    ]
    dpsae = [row for row in checkpoints if row["method"] == "dpsae"]
    result = {
        "complete": True,
        "experiment": "finite_probe_gradient_fidelity_summary",
        "checkpoints": checkpoints,
        "full_dpsae_fidelity_gate": all(row["all_checks_pass"] for row in dpsae),
        "full_dpsae_expected_gradient_fidelity_status": (
            "FAIL"
            if any(row["expected_gradient_fidelity_status"] == "FAIL" for row in dpsae)
            else "PASS"
            if all(row["expected_gradient_fidelity_status"] == "PASS" for row in dpsae)
            else "INCONCLUSIVE"
        ),
        "interpretation_boundary": (
            "The implemented gradient is unbiased for the expected finite-probe "
            "self-normalized objective. Passing this audit supports numerical "
            "alignment with the identity-target gradient at m=16; it is not an "
            "exact finite-m unbiasedness theorem for the identity-target ratio. "
            "Expectation-level intervals use a conservative paired block bootstrap; "
            "squared-bias inference is descriptive near a zero-bias boundary."
        ),
        "inputs": {
            name: input_record(output / f"gradient_fidelity_{name}.json")
            for name in names
        },
        "repository": repository_state(),
    }
    atomic_json(output / "gradient_fidelity_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "summarize"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    config["_config_path"] = str(args.config.resolve())
    output = args.output or ROOT / config["output"]
    if args.mode == "run":
        if not args.model_name:
            raise ValueError("--model-name is required in run mode")
        if args.model_name not in model_names(config["gradient_fidelity"]):
            raise ValueError("model is not part of the frozen gradient audit")
        result = run_model(
            config,
            output,
            args.model_name,
            device=torch.device(args.device),
            gpu_memory_fraction=args.gpu_memory_fraction,
            minimum_free_gib=args.minimum_free_gib,
        )
        print(json.dumps({"complete": result["complete"], "model": result["model"]}, indent=2))
    else:
        result = summarize(config, output)
        print(
            json.dumps(
                {
                    "full_dpsae_fidelity_gate": result["full_dpsae_fidelity_gate"],
                    "full_dpsae_expected_gradient_fidelity_status": result[
                        "full_dpsae_expected_gradient_fidelity_status"
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
