#!/usr/bin/env python3
"""Experiment 3: estimator accuracy, ridge calibration, and group scaling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from dpsae.decoder_distance import (
    batched_ridge_predict,
    batched_sampled_decoder_loss,
    batched_sampled_decoder_statistics,
)
from dpsae.plot_style import (
    COLORS,
    NEUTRAL,
    SEQUENTIAL_CMAP,
    apply_paper_style,
    clean_axis,
    savefig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--activation-device", choices=("auto", "cpu", "mps"))
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_is_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(x: Tensor, group_size: int) -> Tensor:
    usable = (x.shape[0] // group_size) * group_size
    if usable == 0:
        raise ValueError(f"group size {group_size} exceeds {x.shape[0]} samples")
    return x[:usable].reshape(usable // group_size, group_size, x.shape[1])


def choose_activation_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return requested


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def peak_process_rss_mb() -> float:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return maximum / divisor


def benchmark(
    function: Callable[[], Tensor | None],
    *,
    device: str,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / repeats


def singular_values_from_gains(gains: Tensor, n: int, ridge: float) -> Tensor:
    if torch.any((gains <= 0) | (gains >= 1)):
        raise ValueError("ridge gains must lie strictly between zero and one")
    return torch.sqrt(n * ridge * gains / (1 - gains))


def controlled_representation_pair(
    *,
    groups: int,
    n: int,
    width: int,
    effective_rank: int,
    original_gain: float,
    gain_gap: float,
    ridge: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if not 0 < effective_rank <= n:
        raise ValueError("effective rank must lie in [1, group_size]")
    if original_gain - gain_gap <= 0:
        raise ValueError("gain gap would make the reconstructed gain nonpositive")

    generator = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(
        torch.randn(groups, n, n, generator=generator), mode="reduced"
    )
    right, _ = torch.linalg.qr(
        torch.randn(groups, width, n, generator=generator), mode="reduced"
    )
    original_gains = torch.full((n,), original_gain)
    reconstructed_gains = original_gains.clone()
    reconstructed_gains[:effective_rank] -= gain_gap
    original_singular = singular_values_from_gains(original_gains, n, ridge)
    reconstructed_singular = singular_values_from_gains(
        reconstructed_gains, n, ridge
    )
    original = (left * original_singular[None, None, :]) @ right.mT
    reconstructed = (left * reconstructed_singular[None, None, :]) @ right.mT
    return original.contiguous(), reconstructed.contiguous()


def operator_statistics(original: Tensor, reconstructed: Tensor, ridge: float) -> dict[str, Tensor]:
    groups, n, _ = original.shape
    identity = torch.eye(n, dtype=original.dtype, device=original.device).expand(
        groups, n, n
    )
    pred_original = batched_ridge_predict(original, identity, ridge)
    pred_reconstructed = batched_ridge_predict(reconstructed, identity, ridge)
    delta = pred_original - pred_reconstructed
    gram = delta.mT @ delta
    trace_a = delta.square().sum(dim=(1, 2))
    trace_a2 = gram.square().sum(dim=(1, 2))
    effective_rank = trace_a.square() / trace_a2.clamp_min(1e-30)
    reference_energy = pred_original.square().sum(dim=(1, 2))
    dof = pred_original.diagonal(dim1=-2, dim2=-1).sum(dim=1)
    identity_error = (pred_original - identity).square().sum(dim=(1, 2)).sqrt()
    return {
        "trace_a": trace_a,
        "effective_rank": effective_rank,
        "reference_energy": reference_energy,
        "dof": dof,
        "identity_error": identity_error,
    }


def run_synthetic(config: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = config["groups"]
    n = config["group_size"]
    width = config["width"]
    ridge = config["ridge"]
    trials = config["trials"]
    probe_counts = config["probe_counts"]
    max_probes = max(probe_counts)
    rows: list[dict[str, Any]] = []
    maximum_exact_residual = 0.0
    maximum_rank_residual = 0.0

    print("synthetic effective-rank sweep", flush=True)
    for rank_index, target_rank in enumerate(config["effective_ranks"]):
        original, reconstructed = controlled_representation_pair(
            groups=groups,
            n=n,
            width=width,
            effective_rank=target_rank,
            original_gain=config["original_gain"],
            gain_gap=config["gain_gap"],
            ridge=ridge,
            seed=seed + 100 * rank_index,
        )
        exact = operator_statistics(original.double(), reconstructed.double(), ridge)
        exact_trace = exact["trace_a"].cpu()
        effective_rank = exact["effective_rank"].cpu()
        theoretical_trace = target_rank * config["gain_gap"] ** 2
        maximum_exact_residual = max(
            maximum_exact_residual,
            float((exact_trace - theoretical_trace).abs().max()),
        )
        maximum_rank_residual = max(
            maximum_rank_residual,
            float((effective_rank - target_rank).abs().max()),
        )

        generator = torch.Generator().manual_seed(seed + 10_000 + rank_index)
        targets = torch.randn(
            groups, n, trials, max_probes, generator=generator, dtype=torch.float32
        ).reshape(groups, n, trials * max_probes)
        numerator, _ = batched_sampled_decoder_statistics(
            original, reconstructed, targets, ridge=ridge
        )
        probe_errors = numerator.reshape(groups, trials, max_probes).double().cpu()
        cumulative = probe_errors.cumsum(dim=2)
        exact_expanded = exact_trace[:, None]

        for probe_count in probe_counts:
            estimates = cumulative[:, :, probe_count - 1] / probe_count
            relative_error = (estimates - exact_expanded) / exact_expanded
            flattened = relative_error.flatten().numpy()
            numeric_rank = float(effective_rank.mean())
            rows.append(
                {
                    "target_effective_rank": target_rank,
                    "effective_rank": numeric_rank,
                    "probe_count": probe_count,
                    "m_times_effective_rank": probe_count * numeric_rank,
                    "relative_bias": float(flattened.mean()),
                    "relative_sd": float(flattened.std(ddof=1)),
                    "relative_rmse": float(np.sqrt(np.mean(flattened**2))),
                    "median_absolute_relative_error": float(
                        np.median(np.abs(flattened))
                    ),
                    "q90_absolute_relative_error": float(
                        np.quantile(np.abs(flattened), 0.9)
                    ),
                    "predicted_relative_sd": math.sqrt(
                        2.0 / (probe_count * numeric_rank)
                    ),
                    "samples": flattened.size,
                }
            )
        print(
            f"  rank={target_rank:3d} exact={float(exact_trace.mean()):.6g} "
            f"r_eff={float(effective_rank.mean()):.3f}",
            flush=True,
        )

    diagnostics = {
        "maximum_exact_trace_residual": maximum_exact_residual,
        "maximum_effective_rank_residual": maximum_rank_residual,
    }
    return rows, diagnostics


def activation_cache_path(output_dir: Path, config: dict[str, Any]) -> Path:
    model = config["model_name"].replace("/", "-")
    return output_dir / f"{model}_layer{config['layer']}_{config['tokens']}_activations.pt"


def collect_activations(
    config: dict[str, Any], output_dir: Path
) -> tuple[Tensor, dict[str, Any]]:
    cache_path = activation_cache_path(output_dir, config)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        activations = cached["activations"]
        if activations.shape[0] >= config["tokens"]:
            print(f"loaded cached activations: {cache_path}", flush=True)
            return activations[: config["tokens"]], cached["metadata"]

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_activation_device(config["device"])
    print(f"collecting {config['tokens']} GPT-2 activations on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], local_files_only=config["local_files_only"]
    )
    dataset = load_dataset(
        config["dataset_name"],
        config["dataset_config"],
        split=config["dataset_split"],
    )
    sequence_length = config["sequence_length"]
    sequences_needed = math.ceil(config["tokens"] / sequence_length)
    tokens: list[int] = []
    for text in dataset["text"]:
        if not text.strip():
            continue
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= sequences_needed * sequence_length:
            break
    needed = sequences_needed * sequence_length
    if len(tokens) < needed:
        raise RuntimeError(f"dataset supplied {len(tokens)} tokens, need {needed}")
    input_ids = torch.tensor(tokens[:needed], dtype=torch.long).reshape(
        sequences_needed, sequence_length
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        local_files_only=config["local_files_only"],
        dtype=torch.float32,
    ).to(device)
    model.eval()
    block_index = config["layer"] - 1
    if not 0 <= block_index < len(model.transformer.h):
        raise ValueError(f"layer {config['layer']} is outside the GPT-2 block range")
    captured: list[Tensor] = []

    def capture(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())

    handle = model.transformer.h[block_index].register_forward_hook(capture)
    activation_batches: list[Tensor] = []
    peak_mps_allocated = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, sequences_needed, config["batch_size"]):
                captured.clear()
                batch = input_ids[start : start + config["batch_size"]].to(device)
                model(input_ids=batch, use_cache=False)
                if len(captured) != 1:
                    raise RuntimeError(f"activation hook fired {len(captured)} times")
                activation_batches.append(
                    captured[0].float().cpu().reshape(-1, captured[0].shape[-1])
                )
                if device == "mps":
                    peak_mps_allocated = max(
                        peak_mps_allocated, torch.mps.current_allocated_memory()
                    )
    finally:
        handle.remove()
    synchronize(device)
    activations = torch.cat(activation_batches, dim=0)[: config["tokens"]]
    metadata = {
        "model_name": config["model_name"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dataset_name": config["dataset_name"],
        "dataset_config": config["dataset_config"],
        "dataset_split": config["dataset_split"],
        "layer": config["layer"],
        "site": f"resid_post_layer_{config['layer']}",
        "tokens": int(activations.shape[0]),
        "width": int(activations.shape[1]),
        "sequence_length": sequence_length,
        "collection_device": device,
        "peak_mps_allocated_mb": peak_mps_allocated / (1024**2),
        "process_peak_rss_mb_after_collection": peak_process_rss_mb(),
        "mps_recommended_max_memory_mb": (
            torch.mps.recommended_max_memory() / (1024**2) if device == "mps" else 0
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save({"activations": activations, "metadata": metadata}, cache_path)
    print(
        f"collected activations shape={tuple(activations.shape)} "
        f"in {metadata['elapsed_seconds']:.1f}s",
        flush=True,
    )
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return activations, metadata


def preprocess_activations(
    activations: Tensor, *, seed: int, reconstruction_nmse: float
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(activations.shape[0], generator=generator)
    shuffled = activations[permutation]
    split = shuffled.shape[0] // 2
    calibration_raw, evaluation_raw = shuffled[:split], shuffled[split:]
    mean = calibration_raw.mean(dim=0, keepdim=True)
    rms = (calibration_raw - mean).square().mean().sqrt()
    calibration = (calibration_raw - mean) / rms
    evaluation = (evaluation_raw - mean) / rms
    noise = torch.randn(evaluation.shape, generator=generator)
    scale = torch.sqrt(
        reconstruction_nmse
        * evaluation.square().sum()
        / noise.square().sum().clamp_min(1e-30)
    )
    reconstructed = evaluation + scale * noise
    measured_nmse = (
        (evaluation - reconstructed).square().sum()
        / evaluation.square().sum().clamp_min(1e-30)
    )
    return calibration, evaluation, reconstructed, {
        "train_mean_norm": float(mean.norm()),
        "train_rms": float(rms),
        "reconstruction_nmse": float(measured_nmse),
    }


def calibrate_batched_ridge(
    singular_sq: Tensor, target_fraction: float, n: int, iterations: int = 80
) -> float:
    if not 0 < target_fraction < 1:
        raise ValueError("target degrees-of-freedom fraction must lie in (0, 1)")
    scale = max(float(singular_sq.max() / n), torch.finfo(singular_sq.dtype).tiny)
    low, high = scale * 1e-12, scale * 1e12
    target = target_fraction * n
    for _ in range(iterations):
        mid = math.sqrt(low * high)
        dof = (singular_sq / (singular_sq + n * mid)).sum(dim=1).mean().item()
        if dof > target:
            low = mid
        else:
            high = mid
    return math.sqrt(low * high)


def run_activation_estimator(
    config: dict[str, Any],
    calibration: Tensor,
    evaluation: Tensor,
    reconstructed: Tensor,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, float]]:
    estimator_rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    primary_ridges: dict[int, float] = {}
    trials = config["trials"]
    probe_counts = config["probe_counts"]
    max_probes = max(probe_counts)
    target_fractions = config["target_dof_fractions"]

    print("frozen-activation ridge and estimator sweep", flush=True)
    for size_index, group_size in enumerate(config["group_sizes"]):
        calibration_groups = grouped(calibration, group_size)[: config["calibration_groups"]]
        evaluation_groups = grouped(evaluation, group_size)
        reconstructed_groups = grouped(reconstructed, group_size)
        singular_sq = torch.linalg.svdvals(calibration_groups).square()
        generator = torch.Generator().manual_seed(seed + 20_000 + size_index)
        targets = torch.randn(
            evaluation_groups.shape[0],
            group_size,
            trials,
            max_probes,
            generator=generator,
        ).reshape(evaluation_groups.shape[0], group_size, trials * max_probes)

        for target_fraction in target_fractions:
            ridge = calibrate_batched_ridge(singular_sq, target_fraction, group_size)
            if math.isclose(
                target_fraction, config["primary_dof_fraction"], rel_tol=0, abs_tol=1e-12
            ):
                primary_ridges[group_size] = ridge

            exact64 = operator_statistics(
                evaluation_groups.double(), reconstructed_groups.double(), ridge
            )
            exact32 = operator_statistics(evaluation_groups, reconstructed_groups, ridge)
            exact_relative = (
                exact64["trace_a"] / exact64["reference_energy"].clamp_min(1e-30)
            ).cpu()
            fp32_relative = (
                exact32["trace_a"] / exact32["reference_energy"].clamp_min(1e-30)
            ).double().cpu()
            fp32_error = (
                (fp32_relative - exact_relative).abs()
                / exact_relative.abs().clamp_min(1e-30)
            )
            dof_fraction = (exact64["dof"] / group_size).cpu().numpy()
            identity_error = (
                exact64["identity_error"] / math.sqrt(group_size)
            ).cpu().numpy()
            distortion = exact_relative.numpy()
            effective_rank = exact64["effective_rank"].cpu().numpy()
            ridge_rows.append(
                {
                    "group_size": group_size,
                    "target_dof_fraction": target_fraction,
                    "ridge": ridge,
                    "actual_dof_mean": float(dof_fraction.mean()),
                    "actual_dof_std": float(dof_fraction.std(ddof=1)),
                    "relative_distortion_q10": float(np.quantile(distortion, 0.1)),
                    "relative_distortion_median": float(np.median(distortion)),
                    "relative_distortion_q90": float(np.quantile(distortion, 0.9)),
                    "identity_error_mean": float(identity_error.mean()),
                    "effective_rank_mean": float(effective_rank.mean()),
                    "fp32_exact_relative_error_max": float(fp32_error.max()),
                }
            )

            numerator, denominator = batched_sampled_decoder_statistics(
                evaluation_groups, reconstructed_groups, targets, ridge=ridge
            )
            numerator = numerator.reshape(
                evaluation_groups.shape[0], trials, max_probes
            ).double().cpu()
            denominator = denominator.reshape(
                evaluation_groups.shape[0], trials, max_probes
            ).double().cpu()
            cumulative_numerator = numerator.cumsum(dim=2)
            cumulative_denominator = denominator.cumsum(dim=2)
            exact_expanded = exact_relative[:, None]

            for probe_count in probe_counts:
                estimates = (
                    cumulative_numerator[:, :, probe_count - 1]
                    / cumulative_denominator[:, :, probe_count - 1].clamp_min(1e-30)
                )
                relative_error = (estimates - exact_expanded) / exact_expanded.clamp_min(
                    1e-30
                )
                flattened = relative_error.flatten().numpy()
                absolute = np.abs(flattened)
                estimator_rows.append(
                    {
                        "group_size": group_size,
                        "target_dof_fraction": target_fraction,
                        "actual_dof_fraction": float(dof_fraction.mean()),
                        "ridge": ridge,
                        "effective_rank": float(effective_rank.mean()),
                        "probe_count": probe_count,
                        "relative_bias": float(flattened.mean()),
                        "relative_rmse": float(np.sqrt(np.mean(flattened**2))),
                        "median_absolute_relative_error": float(np.median(absolute)),
                        "q10_absolute_relative_error": float(np.quantile(absolute, 0.1)),
                        "q90_absolute_relative_error": float(np.quantile(absolute, 0.9)),
                        "samples": flattened.size,
                    }
                )
        print(
            f"  group_size={group_size} groups={evaluation_groups.shape[0]} "
            f"primary_ridge={primary_ridges[group_size]:.4g}",
            flush=True,
        )

    return estimator_rows, ridge_rows, primary_ridges


def estimated_live_memory_mb(
    *, groups: int, n: int, width: int, targets: int, bytes_per_value: int = 4
) -> float:
    representation_values = 2 * groups * n * width
    target_and_prediction_values = 3 * groups * n * targets
    factor_values = 4 * groups * n * n
    statistic_values = 2 * groups * targets
    total = (
        representation_values
        + target_and_prediction_values
        + factor_values
        + statistic_values
    )
    return total * bytes_per_value / (1024**2)


def run_systems_scaling(
    config: dict[str, Any],
    estimator_config: dict[str, Any],
    evaluation: Tensor,
    reconstructed: Tensor,
    primary_ridges: dict[int, float],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []
    batch_samples = config["batch_samples"]
    probe_count = config["probe_count"]
    generator = torch.Generator().manual_seed(seed + 30_000)
    print("systems scaling", flush=True)

    for group_size in estimator_config["group_sizes"]:
        if batch_samples % group_size:
            raise ValueError("systems batch size must be divisible by every group size")
        x = grouped(evaluation[:batch_samples], group_size)
        x_hat = grouped(reconstructed[:batch_samples], group_size)
        groups = x.shape[0]
        targets = torch.randn(groups, group_size, probe_count, generator=generator)
        identity = torch.eye(group_size).expand(groups, group_size, group_size)
        ridge = primary_ridges[group_size]

        sampled_forward_ms = benchmark(
            lambda: batched_sampled_decoder_loss(x, x_hat, targets, ridge=ridge),
            device="cpu",
            warmup=config["warmup"],
            repeats=config["repeats"],
        )
        exact_forward_ms = benchmark(
            lambda: batched_sampled_decoder_statistics(x, x_hat, identity, ridge=ridge)[0],
            device="cpu",
            warmup=config["warmup"],
            repeats=config["repeats"],
        )
        x_hat_grad = x_hat.clone().requires_grad_(True)

        def forward_backward() -> Tensor:
            x_hat_grad.grad = None
            loss = batched_sampled_decoder_loss(x, x_hat_grad, targets, ridge=ridge)
            loss.backward()
            return loss

        sampled_forward_backward_ms = benchmark(
            forward_backward,
            device="cpu",
            warmup=config["warmup"],
            repeats=config["repeats"],
        )
        rows.append(
            {
                "group_size": group_size,
                "groups": groups,
                "batch_samples": batch_samples,
                "width": x.shape[2],
                "probe_count": probe_count,
                "ridge": ridge,
                "sampled_forward_ms": sampled_forward_ms,
                "sampled_forward_backward_ms": sampled_forward_backward_ms,
                "exact_forward_ms": exact_forward_ms,
                "sampled_live_memory_mb": estimated_live_memory_mb(
                    groups=groups,
                    n=group_size,
                    width=x.shape[2],
                    targets=probe_count,
                ),
                "exact_live_memory_mb": estimated_live_memory_mb(
                    groups=groups,
                    n=group_size,
                    width=x.shape[2],
                    targets=group_size,
                ),
            }
        )

        for device in ("cpu", "mps"):
            if device == "mps" and not torch.backends.mps.is_available():
                continue
            xd = x.to(device)
            xhd = x_hat.to(device)
            yd = targets.to(device)
            milliseconds = benchmark(
                lambda: batched_sampled_decoder_loss(xd, xhd, yd, ridge=ridge),
                device=device,
                warmup=config["warmup"],
                repeats=config["backend_repeats"],
            )
            xhd_grad = xhd.clone().requires_grad_(True)

            def backend_forward_backward() -> Tensor:
                xhd_grad.grad = None
                loss = batched_sampled_decoder_loss(xd, xhd_grad, yd, ridge=ridge)
                loss.backward()
                return loss

            forward_backward_ms = benchmark(
                backend_forward_backward,
                device=device,
                warmup=config["warmup"],
                repeats=config["backend_repeats"],
            )
            gradient_finite = bool(torch.isfinite(xhd_grad.grad).all().cpu())
            backend_rows.append(
                {
                    "device": device,
                    "group_size": group_size,
                    "groups": groups,
                    "probe_count": probe_count,
                    "forward_ms": milliseconds,
                    "forward_backward_ms": forward_backward_ms,
                    "gradient_finite": gradient_finite,
                }
            )
        print(
            f"  n={group_size} sampled={sampled_forward_ms:.3f}ms "
            f"forward+backward={sampled_forward_backward_ms:.3f}ms",
            flush=True,
        )
    return rows, backend_rows


def selected(
    rows: list[dict[str, str]], **conditions: int | float | str
) -> list[dict[str, str]]:
    selected_rows = []
    for row in rows:
        keep = True
        for key, value in conditions.items():
            if isinstance(value, float):
                keep &= math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-9)
            else:
                keep &= str(row[key]) == str(value)
        if keep:
            selected_rows.append(row)
    return selected_rows


def plot_estimator_accuracy(
    synthetic: list[dict[str, str]],
    activation: list[dict[str, str]],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    ranks = sorted({int(row["target_effective_rank"]) for row in synthetic})
    colors = SEQUENTIAL_CMAP(np.linspace(0.28, 0.95, len(ranks)))
    for rank, color in zip(ranks, colors):
        rows = sorted(
            selected(synthetic, target_effective_rank=rank),
            key=lambda row: int(row["probe_count"]),
        )
        ax.plot(
            [float(row["m_times_effective_rank"]) for row in rows],
            [float(row["relative_rmse"]) for row in rows],
            marker="o",
            color=color,
            label=fr"$r_{{\rm eff}}={rank}$",
        )
    x_min = min(float(row["m_times_effective_rank"]) for row in synthetic)
    x_max = max(float(row["m_times_effective_rank"]) for row in synthetic)
    theory_x = np.geomspace(x_min, x_max, 300)
    ax.plot(
        theory_x,
        np.sqrt(2 / theory_x),
        color=COLORS["theory"],
        linestyle=":",
        linewidth=1.8,
        label=r"Theory $\sqrt{2/(m r_{\rm eff})}$",
    )
    ax.set_title("Gaussian estimator follows theory")
    ax.set_xlabel(r"Probe information $m r_{\rm eff}$")
    ax.set_ylabel("Relative RMSE")
    clean_axis(ax, xlog=True, ylog=True)
    ax.legend(frameon=False, ncol=2, fontsize=7)

    ax = axes[1]
    primary_fraction = 0.25
    group_sizes = sorted({int(row["group_size"]) for row in activation})
    colors = SEQUENTIAL_CMAP(np.linspace(0.35, 0.95, len(group_sizes)))
    for group_size, color in zip(group_sizes, colors):
        rows = sorted(
            selected(
                activation,
                group_size=group_size,
                target_dof_fraction=primary_fraction,
            ),
            key=lambda row: int(row["probe_count"]),
        )
        x = np.array([int(row["probe_count"]) for row in rows])
        median = np.array(
            [float(row["median_absolute_relative_error"]) for row in rows]
        )
        q10 = np.array([float(row["q10_absolute_relative_error"]) for row in rows])
        q90 = np.array([float(row["q90_absolute_relative_error"]) for row in rows])
        ax.plot(x, median, color=color, marker="o", label=f"Group {group_size}")
        ax.fill_between(x, q10, q90, color=color, alpha=0.12, linewidth=0)
    ax.axvspan(4, 16, color=NEUTRAL["fill"], alpha=0.8, zorder=0)
    ax.set_title("Frozen GPT-2 activation estimates")
    ax.set_xlabel("Random probes $m$")
    ax.set_ylabel("Absolute relative error")
    clean_axis(ax, xlog=True, ylog=True)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, figures_dir / "exp03_estimator_accuracy")
    plt.close(fig)


def plot_ridge_calibration(rows: list[dict[str, str]], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 2.8))
    group_sizes = sorted({int(row["group_size"]) for row in rows})
    colors = SEQUENTIAL_CMAP(np.linspace(0.35, 0.95, len(group_sizes)))
    for group_size, color in zip(group_sizes, colors):
        group_rows = sorted(
            selected(rows, group_size=group_size),
            key=lambda row: float(row["actual_dof_mean"]),
        )
        x = np.array([float(row["actual_dof_mean"]) for row in group_rows])
        median = np.array(
            [float(row["relative_distortion_median"]) for row in group_rows]
        )
        low = np.array([float(row["relative_distortion_q10"]) for row in group_rows])
        high = np.array([float(row["relative_distortion_q90"]) for row in group_rows])
        axes[0].plot(x, median, color=color, marker="o", label=f"Group {group_size}")
        axes[0].fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        axes[1].plot(
            [float(row["target_dof_fraction"]) for row in group_rows],
            [float(row["actual_dof_mean"]) for row in group_rows],
            color=color,
            marker="o",
            label=f"Group {group_size}",
        )
        axes[2].plot(
            x,
            [float(row["fp32_exact_relative_error_max"]) for row in group_rows],
            color=color,
            marker="o",
            label=f"Group {group_size}",
        )
    axes[0].axvline(0.25, color=COLORS["theory"], linestyle=":")
    axes[0].set_title(r"Small ridge makes $K\approx I$")
    axes[0].set_xlabel(r"Effective degrees of freedom $d_{\rm eff}/n$")
    axes[0].set_ylabel("Exact relative distortion")
    clean_axis(axes[0], ylog=True)
    axes[0].legend(frameon=False)
    axes[1].plot([0, 1], [0, 1], color=COLORS["theory"], linestyle=":")
    axes[1].set_title("Degrees-of-freedom calibration")
    axes[1].set_xlabel(r"Target $d_{\rm eff}/n$")
    axes[1].set_ylabel(r"Observed $d_{\rm eff}/n$")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    clean_axis(axes[1])
    axes[2].axvline(0.25, color=COLORS["theory"], linestyle=":")
    axes[2].set_title("Near-zero distances lose precision")
    axes[2].set_xlabel(r"Effective degrees of freedom $d_{\rm eff}/n$")
    axes[2].set_ylabel("Max FP32 / FP64 relative error")
    clean_axis(axes[2], ylog=True)
    fig.tight_layout()
    savefig(fig, figures_dir / "exp03_ridge_calibration")
    plt.close(fig)


def plot_systems_scaling(rows: list[dict[str, str]], figures_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: int(row["group_size"]))
    x = np.array([int(row["group_size"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    axes[0].plot(
        x,
        [float(row["sampled_forward_ms"]) for row in rows],
        color=COLORS["isotropic"],
        marker="o",
        label="Sampled forward",
    )
    axes[0].plot(
        x,
        [float(row["sampled_forward_backward_ms"]) for row in rows],
        color=COLORS["mse"],
        marker="s",
        linestyle="--",
        label="Sampled forward + backward",
    )
    axes[0].plot(
        x,
        [float(row["exact_forward_ms"]) for row in rows],
        color=COLORS["theory"],
        marker="D",
        linestyle=":",
        label="Exact forward",
    )
    axes[0].set_title("Vectorized objective cost")
    axes[0].set_xlabel("Geometry-group size")
    axes[0].set_ylabel("Wall time (ms / 1,024 samples)")
    clean_axis(axes[0], ylog=True)
    axes[0].legend(frameon=False)

    axes[1].plot(
        x,
        [float(row["sampled_live_memory_mb"]) for row in rows],
        color=COLORS["isotropic"],
        marker="o",
        label="Sampled, 16 probes",
    )
    axes[1].plot(
        x,
        [float(row["exact_live_memory_mb"]) for row in rows],
        color=COLORS["theory"],
        marker="D",
        linestyle=":",
        label="Exact, identity probes",
    )
    axes[1].set_title("Estimated live tensor memory")
    axes[1].set_xlabel("Geometry-group size")
    axes[1].set_ylabel("Memory (MiB / 1,024 samples)")
    clean_axis(axes[1])
    axes[1].legend(frameon=False)
    fig.tight_layout()
    savefig(fig, figures_dir / "exp03_systems_scaling")
    plt.close(fig)


def make_plots(output_dir: Path, figures_dir: Path) -> None:
    apply_paper_style()
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plot_estimator_accuracy(
        read_csv(output_dir / "synthetic_estimator.csv"),
        read_csv(output_dir / "activation_estimator.csv"),
        figures_dir,
    )
    plot_ridge_calibration(
        read_csv(output_dir / "ridge_calibration.csv"), figures_dir
    )
    plot_systems_scaling(read_csv(output_dir / "systems_scaling.csv"), figures_dir)


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    config["synthetic"].update(
        {
            "width": 64,
            "group_size": 32,
            "groups": 2,
            "effective_ranks": [1, 4, 16, 32],
            "probe_counts": [1, 4, 8],
            "trials": 16,
        }
    )
    config["activations"].update(
        {"tokens": 1024, "sequence_length": 64, "batch_size": 2}
    )
    config["activation_estimator"].update(
        {
            "group_sizes": [32, 64],
            "target_dof_fractions": [0.25, 0.9, 0.995],
            "probe_counts": [1, 4, 8],
            "trials": 8,
            "calibration_groups": 2,
        }
    )
    config["systems"].update(
        {"batch_samples": 256, "probe_count": 8, "warmup": 1, "repeats": 2}
    )
    return config


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = smoke_config(config)
    if args.activation_device is not None:
        config["activations"]["device"] = args.activation_device
    torch.set_num_threads(config["threads"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.plots_only:
        started = time.perf_counter()
        synthetic, synthetic_diagnostics = run_synthetic(
            config["synthetic"], config["seed"]
        )
        activations, activation_metadata = collect_activations(
            config["activations"], args.output_dir
        )
        calibration, evaluation, reconstructed, preprocessing = preprocess_activations(
            activations,
            seed=config["seed"] + 1,
            reconstruction_nmse=config["activations"]["reconstruction_nmse"],
        )
        activation_rows, ridge_rows, primary_ridges = run_activation_estimator(
            config["activation_estimator"],
            calibration,
            evaluation,
            reconstructed,
            seed=config["seed"] + 2,
        )
        systems_rows, backend_rows = run_systems_scaling(
            config["systems"],
            config["activation_estimator"],
            evaluation,
            reconstructed,
            primary_ridges,
            seed=config["seed"] + 3,
        )
        write_csv(args.output_dir / "synthetic_estimator.csv", synthetic)
        write_csv(args.output_dir / "activation_estimator.csv", activation_rows)
        write_csv(args.output_dir / "ridge_calibration.csv", ridge_rows)
        write_csv(args.output_dir / "systems_scaling.csv", systems_rows)
        write_csv(args.output_dir / "backend_benchmark.csv", backend_rows)
        source_paths = [
            Path(__file__),
            args.config,
            Path("src/dpsae/decoder_distance.py"),
        ]
        metadata = {
            "config": config,
            "config_path": str(args.config),
            "git_revision": git_revision(),
            "git_dirty": git_is_dirty(),
            "source_sha256": {
                str(path): file_sha256(path) for path in source_paths if path.exists()
            },
            "synthetic_diagnostics": synthetic_diagnostics,
            "activation_metadata": activation_metadata,
            "preprocessing": preprocessing,
            "primary_ridges": primary_ridges,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "peak_process_rss_mb": peak_process_rss_mb(),
            "elapsed_seconds": time.perf_counter() - started,
            "smoke": args.smoke,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    make_plots(args.output_dir, args.figures_dir)
    print(f"completed {config['experiment']} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
