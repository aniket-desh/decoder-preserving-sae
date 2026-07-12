#!/usr/bin/env python3
"""Experiment 1: isotropic spectral mechanics under a sparse bottleneck.

The experiment has two linked parts:

1. Numerically verify the exact rank-constrained spectral theorem.
2. Train matched tied signed-TopK dictionaries on planted sparse features and
   compare MSE, isotropic decoder preservation, whitening, and decoder-only.

All generated metrics are machine-readable. Plotting is deterministic and may
be rerun from the saved CSV files with ``--plots-only``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dpsae.decoder_distance import (
    batched_ridge_predict,
    calibrate_ridge,
    effective_degrees_of_freedom,
    ridge_hat_matrix,
)
from dpsae.plot_style import (
    COLORS,
    LABELS,
    LINESTYLES,
    MARKERS,
    METHOD_ORDER,
    apply_paper_style,
    clean_axis,
    savefig,
)
from dpsae.sae import TiedSignedTopKSAE
from dpsae.spectral import decoder_gains, optimal_decoder_tail, truncated_svd


GROUPS = ("nuisance", "moderate", "weak")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    return parser.parse_args()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def orthonormal_matrix(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(rows, cols, generator=generator, dtype=torch.float64))
    return q[:, :cols]


def run_analytic(config: dict[str, Any]) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(12345)
    n = config["n_samples"]
    d = config["n_features"]
    rank = config["rank"]
    tau = config["tau"]
    candidates = config["random_candidates"]
    u = orthonormal_matrix(n, rank, generator)
    v = orthonormal_matrix(d, rank, generator)
    variance_ratio = torch.logspace(3, -3, rank, dtype=torch.float64)
    singular_values = variance_ratio.sqrt()
    x = (u * singular_values) @ v.mT
    ridge = tau / n
    theory = optimal_decoder_tail(singular_values, tau)
    k_x = ridge_hat_matrix(x, ridge)
    rows: list[dict[str, Any]] = []

    for retained in range(rank + 1):
        x_rank = truncated_svd(x, retained)
        k_rank = ridge_hat_matrix(x_rank, ridge)
        observed = (k_x - k_rank).square().sum().item()
        random_values: list[float] = []
        if retained == 0:
            random_values = [k_x.square().sum().item()] * candidates
        else:
            retained_sigmas = singular_values[:retained]
            for _ in range(candidates):
                random_u = orthonormal_matrix(n, retained, generator)
                random_v = orthonormal_matrix(d, retained, generator)
                z = (random_u * retained_sigmas) @ random_v.mT
                random_values.append(
                    (k_x - ridge_hat_matrix(z, ridge)).square().sum().item()
                )
        rows.append(
            {
                "rank": retained,
                "theory": theory[retained].item(),
                "truncated_svd": observed,
                "random_median": float(np.median(random_values)),
                "random_q10": float(np.quantile(random_values, 0.10)),
                "random_q90": float(np.quantile(random_values, 0.90)),
                "absolute_error": abs(observed - theory[retained].item()),
            }
        )

    max_error = max(row["absolute_error"] for row in rows)
    if max_error > 1e-9:
        raise AssertionError(f"spectral theorem numerical error {max_error:.3e} exceeds tolerance")
    return rows


def feature_parameters(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    total = config["true_features"]
    nuisance = config["nuisance_count"]
    moderate = config["moderate_count"]
    if nuisance + moderate >= total:
        raise ValueError("feature groups leave no weak features")
    amplitudes = torch.empty(total)
    probabilities = torch.empty(total)
    groups: list[str] = []
    amplitudes[:nuisance] = config["nuisance_amplitude"]
    probabilities[:nuisance] = config["nuisance_probability"]
    groups.extend(["nuisance"] * nuisance)
    amplitudes[nuisance : nuisance + moderate] = config["moderate_amplitude"]
    probabilities[nuisance : nuisance + moderate] = config["moderate_probability"]
    groups.extend(["moderate"] * moderate)
    amplitudes[nuisance + moderate :] = config["weak_amplitude"]
    probabilities[nuisance + moderate :] = config["weak_probability"]
    groups.extend(["weak"] * (total - nuisance - moderate))
    return amplitudes, probabilities, groups


def make_true_dictionary(config: dict[str, Any], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    total = config["true_features"]
    dim = config["input_dim"]
    clusters = config["cluster_count"]
    correlation = config["correlation"]
    cluster_vectors = torch.nn.functional.normalize(
        torch.randn(clusters, dim, generator=generator), dim=1
    )
    independent = torch.nn.functional.normalize(
        torch.randn(total, dim, generator=generator), dim=1
    )
    assignments = torch.arange(total) % clusters
    dictionary = (
        correlation * cluster_vectors[assignments]
        + math.sqrt(1 - correlation**2) * independent
    )
    return torch.nn.functional.normalize(dictionary, dim=1)


def sample_sparse_data(
    n_samples: int,
    dictionary: torch.Tensor,
    amplitudes: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    noise_std: float,
    seed: int,
    groups: list[str],
    only_group: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    active = torch.rand(n_samples, dictionary.shape[0], generator=generator) < probabilities
    if only_group is not None:
        group_mask = torch.tensor([group == only_group for group in groups])
        active &= group_mask[None, :]
    coefficients = torch.randn(n_samples, dictionary.shape[0], generator=generator)
    codes = active * coefficients * amplitudes
    x = codes @ dictionary
    if noise_std > 0:
        x += noise_std * torch.randn(x.shape, generator=generator)
    return x, codes


def fixed_preprocess(
    train: torch.Tensor, test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mean = train.mean(dim=0, keepdim=True)
    rms = (train - mean).square().mean().sqrt().item()
    return (train - mean) / rms, (test - mean) / rms, mean, rms


def whitening_matrix(train: torch.Tensor) -> torch.Tensor:
    covariance = train.mT @ train / train.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = max(eigenvalues.max().item() * 1e-3, 1e-8)
    return (eigenvectors * eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.mT


def normalized_mse(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    return (x - reconstructed).square().sum() / x.square().sum().clamp_min(1e-12)


def normalized_whitened_mse(
    x: torch.Tensor, reconstructed: torch.Tensor, whitening: torch.Tensor
) -> torch.Tensor:
    return ((x - reconstructed) @ whitening).square().sum() / (
        (x @ whitening).square().sum().clamp_min(1e-12)
    )


def decoder_group_loss(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    ridge: float,
    group_size: int,
    task_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if x.shape[0] % group_size:
        raise ValueError("batch size must be divisible by geometry group size")
    group_count = x.shape[0] // group_size
    x_grouped = x.reshape(group_count, group_size, x.shape[1])
    reconstructed_grouped = reconstructed.reshape(
        group_count, group_size, reconstructed.shape[1]
    )
    targets = torch.randint(
        0,
        2,
        (group_count, group_size, task_count),
        generator=generator,
        device=x.device,
        dtype=torch.int64,
    ).to(x.dtype)
    targets = 2 * targets - 1
    pred_original = batched_ridge_predict(x_grouped, targets, ridge)
    pred_reconstructed = batched_ridge_predict(reconstructed_grouped, targets, ridge)
    numerator = (pred_original - pred_reconstructed).square().sum(dim=(1, 2))
    denominator = pred_original.square().sum(dim=(1, 2)).clamp_min(1e-12)
    return (numerator / denominator).mean()


def gradient_norm(loss: torch.Tensor, parameter: torch.Tensor) -> float:
    gradient = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
    return gradient.norm().item()


def calibrate_objective_weights(
    model: TiedSignedTopKSAE,
    batch: torch.Tensor,
    whitening: torch.Tensor,
    *,
    ridge: float,
    group_size: int,
    task_count: int,
    clip: tuple[float, float],
    seed: int,
) -> tuple[float, float, dict[str, float]]:
    reconstruction, _ = model(batch)
    nmse = normalized_mse(batch, reconstruction)
    white = normalized_whitened_mse(batch, reconstruction, whitening)
    generator = torch.Generator(device=batch.device).manual_seed(seed)
    decoder = decoder_group_loss(
        batch,
        reconstruction,
        ridge=ridge,
        group_size=group_size,
        task_count=task_count,
        generator=generator,
    )
    norm_nmse = gradient_norm(nmse, model.dictionary)
    norm_white = gradient_norm(white, model.dictionary)
    norm_decoder = gradient_norm(decoder, model.dictionary)
    gamma_decoder = float(np.clip(norm_nmse / max(norm_decoder, 1e-12), *clip))
    gamma_white = float(np.clip(norm_nmse / max(norm_white, 1e-12), *clip))
    return gamma_decoder, gamma_white, {
        "initial_nmse": nmse.item(),
        "initial_decoder_loss": decoder.item(),
        "initial_whitened_loss": white.item(),
        "nmse_gradient_norm": norm_nmse,
        "decoder_gradient_norm": norm_decoder,
        "whitened_gradient_norm": norm_white,
    }


def train_method(
    method: str,
    initial_state: dict[str, torch.Tensor],
    train: torch.Tensor,
    batch_indices: torch.Tensor,
    whitening: torch.Tensor,
    config: dict[str, Any],
    *,
    ridge: float,
    gamma_decoder: float,
    gamma_white: float,
    seed: int,
) -> tuple[TiedSignedTopKSAE, list[dict[str, Any]], float]:
    model = TiedSignedTopKSAE(
        train.shape[1], config["dictionary_size"], config["top_k"], seed=seed
    ).to(train.device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    task_generator = torch.Generator(device=train.device).manual_seed(100_000 + seed)
    curves: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for step, indices in enumerate(batch_indices, start=1):
        batch = train[indices]
        reconstruction, _ = model(batch)
        nmse = normalized_mse(batch, reconstruction)
        white = None
        decoder = None
        if method == "whitened":
            white = normalized_whitened_mse(batch, reconstruction, whitening)
        if method in {"isotropic", "decoder_only"}:
            decoder = decoder_group_loss(
                batch,
                reconstruction,
                ridge=ridge,
                group_size=config["geometry_group_size"],
                task_count=config["random_tasks"],
                generator=task_generator,
            )
        if method == "mse":
            objective = nmse
        elif method == "isotropic":
            assert decoder is not None
            objective = nmse + gamma_decoder * decoder
        elif method == "whitened":
            assert white is not None
            objective = nmse + gamma_white * white
        elif method == "decoder_only":
            assert decoder is not None
            objective = gamma_decoder * decoder
        else:
            raise ValueError(f"unknown method {method}")

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        model.normalize_dictionary_()

        if step == 1 or step % 100 == 0 or step == len(batch_indices):
            with torch.no_grad():
                log_generator = torch.Generator(device=train.device).manual_seed(
                    200_000 + 10_000 * seed + step
                )
                log_decoder = decoder_group_loss(
                    batch,
                    reconstruction.detach(),
                    ridge=ridge,
                    group_size=config["geometry_group_size"],
                    task_count=config["random_tasks"],
                    generator=log_generator,
                )
                log_white = normalized_whitened_mse(
                    batch, reconstruction.detach(), whitening
                )
            curves.append(
                {
                    "seed": seed,
                    "method": method,
                    "step": step,
                    "objective": objective.item(),
                    "nmse": nmse.item(),
                    "decoder_loss": log_decoder.item(),
                    "whitened_loss": log_white.item(),
                }
            )
            print(
                f"seed={seed} method={method:12s} step={step:4d}/{len(batch_indices)} "
                f"nmse={nmse.item():.4f} dec={log_decoder.item():.4f}",
                flush=True,
            )
    return model, curves, time.perf_counter() - start_time


@torch.no_grad()
def exact_relative_decoder_distortion(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    ridge: float,
    group_size: int,
    groups: int,
) -> float:
    x = x[: group_size * groups].double()
    reconstructed = reconstructed[: group_size * groups].double()
    x_grouped = x.reshape(groups, group_size, x.shape[1])
    reconstructed_grouped = reconstructed.reshape(groups, group_size, reconstructed.shape[1])
    identity = torch.eye(group_size, dtype=x.dtype).expand(groups, group_size, group_size)
    k_x = batched_ridge_predict(x_grouped, identity, ridge)
    k_hat = batched_ridge_predict(reconstructed_grouped, identity, ridge)
    values = (k_x - k_hat).square().sum(dim=(1, 2)) / k_x.square().sum(dim=(1, 2))
    return values.mean().item()


@torch.no_grad()
def feature_recovery(
    model: TiedSignedTopKSAE,
    true_dictionary: torch.Tensor,
    test: torch.Tensor,
    true_codes: torch.Tensor,
    feature_groups: list[str],
) -> list[dict[str, float | str]]:
    learned = torch.nn.functional.normalize(model.dictionary.detach().cpu(), dim=1)
    truth = torch.nn.functional.normalize(true_dictionary.cpu(), dim=1)
    similarities = (truth @ learned.mT).abs().numpy()
    truth_indices, learned_indices = linear_sum_assignment(-similarities)
    learned_codes = model.encode(test).detach().cpu()
    rows = []
    for truth_index, learned_index in zip(truth_indices, learned_indices, strict=True):
        truth_active = true_codes[:, truth_index].ne(0)
        learned_active = learned_codes[:, learned_index].ne(0)
        tp = (truth_active & learned_active).sum().item()
        precision = tp / max(learned_active.sum().item(), 1)
        recall = tp / max(truth_active.sum().item(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "group": feature_groups[truth_index],
                "cosine": float(similarities[truth_index, learned_index]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def run_sparse(config: dict[str, Any], seeds: list[int]) -> tuple[list, list, list, list]:
    data_config = config["data"]
    sae_config = config["sae"]
    evaluation_config = config["evaluation"]
    all_metrics: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"preparing paired seed {seed}", flush=True)
        dictionary = make_true_dictionary(data_config, seed=10_000 + seed)
        amplitudes, probabilities, feature_groups = feature_parameters(data_config)
        train_raw, _ = sample_sparse_data(
            data_config["train_samples"],
            dictionary,
            amplitudes,
            probabilities,
            noise_std=data_config["noise_std"],
            seed=20_000 + seed,
            groups=feature_groups,
        )
        test_raw, test_codes = sample_sparse_data(
            data_config["test_samples"],
            dictionary,
            amplitudes,
            probabilities,
            noise_std=data_config["noise_std"],
            seed=30_000 + seed,
            groups=feature_groups,
        )
        train, test, train_mean, train_rms = fixed_preprocess(train_raw, test_raw)
        whitening = whitening_matrix(train)
        calibration_group = train[: sae_config["geometry_group_size"]]
        ridge = calibrate_ridge(
            calibration_group, sae_config["target_dof_fraction"], center=False
        )
        dof_fraction = (
            effective_degrees_of_freedom(calibration_group, ridge).item()
            / calibration_group.shape[0]
        )

        base_model = TiedSignedTopKSAE(
            data_config["input_dim"],
            sae_config["dictionary_size"],
            sae_config["top_k"],
            seed=40_000 + seed,
        )
        initial_state = deepcopy(base_model.state_dict())
        calibration_batch = train[: sae_config["batch_size"]]
        clip = tuple(sae_config["gradient_balance_clip"])
        gamma_decoder, gamma_white, calibration = calibrate_objective_weights(
            base_model,
            calibration_batch,
            whitening,
            ridge=ridge,
            group_size=sae_config["geometry_group_size"],
            task_count=sae_config["random_tasks"],
            clip=clip,
            seed=50_000 + seed,
        )
        calibration_rows.append(
            {
                "seed": seed,
                "ridge": ridge,
                "dof_fraction": dof_fraction,
                "gamma_decoder": gamma_decoder,
                "gamma_white": gamma_white,
                "train_rms": train_rms,
                **calibration,
            }
        )
        print(
            f"seed={seed} ridge={ridge:.4g} dof/n={dof_fraction:.3f} "
            f"gamma_dec={gamma_decoder:.3g} gamma_white={gamma_white:.3g}",
            flush=True,
        )

        index_generator = torch.Generator().manual_seed(60_000 + seed)
        batch_indices = torch.randint(
            0,
            train.shape[0],
            (sae_config["steps"], sae_config["batch_size"]),
            generator=index_generator,
        )
        group_only: dict[str, torch.Tensor] = {}
        for group_index, group in enumerate(GROUPS):
            raw, _ = sample_sparse_data(
                evaluation_config.get("group_samples", 2048),
                dictionary,
                amplitudes,
                probabilities,
                noise_std=data_config["noise_std"],
                seed=70_000 + 100 * seed + group_index,
                groups=feature_groups,
                only_group=group,
            )
            group_only[group] = (raw - train_mean) / train_rms

        for method in METHOD_ORDER:
            model, curves, elapsed = train_method(
                method,
                initial_state,
                train,
                batch_indices,
                whitening,
                sae_config,
                ridge=ridge,
                gamma_decoder=gamma_decoder,
                gamma_white=gamma_white,
                seed=seed,
            )
            all_curves.extend(curves)
            with torch.no_grad():
                reconstruction, code = model(test)
                test_nmse = normalized_mse(test, reconstruction).item()
                white_nmse = normalized_whitened_mse(test, reconstruction, whitening).item()
                decoder_distortion = exact_relative_decoder_distortion(
                    test,
                    reconstruction,
                    ridge=ridge,
                    group_size=sae_config["geometry_group_size"],
                    groups=evaluation_config["geometry_groups"],
                )
                average_l0 = code.ne(0).sum(dim=1).float().mean().item()
            all_metrics.append(
                {
                    "seed": seed,
                    "method": method,
                    "test_nmse": test_nmse,
                    "whitened_nmse": white_nmse,
                    "decoder_distortion": decoder_distortion,
                    "average_l0": average_l0,
                    "train_seconds": elapsed,
                }
            )

            recovery = feature_recovery(
                model, dictionary, test, test_codes, feature_groups
            )
            for group in GROUPS:
                group_recovery = [row for row in recovery if row["group"] == group]
                with torch.no_grad():
                    group_reconstruction, _ = model(group_only[group])
                    group_nmse = normalized_mse(
                        group_only[group], group_reconstruction
                    ).item()
                all_groups.append(
                    {
                        "seed": seed,
                        "method": method,
                        "group": group,
                        "group_nmse": group_nmse,
                        "matched_cosine": float(
                            np.mean([row["cosine"] for row in group_recovery])
                        ),
                        "support_precision": float(
                            np.mean([row["precision"] for row in group_recovery])
                        ),
                        "support_recall": float(
                            np.mean([row["recall"] for row in group_recovery])
                        ),
                        "support_f1": float(np.mean([row["f1"] for row in group_recovery])),
                    }
                )
    return all_metrics, all_groups, all_curves, calibration_rows


def quantiles(values: list[float]) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.quantile(values, [0.10, 0.50, 0.90]))


def plot_analytic(rows: list[dict[str, str]], config: dict[str, Any], figures: Path) -> None:
    rank = np.array([int(row["rank"]) for row in rows])
    theory = np.array([float(row["theory"]) for row in rows])
    measured = np.array([float(row["truncated_svd"]) for row in rows])
    random_median = np.array([float(row["random_median"]) for row in rows])
    random_q10 = np.array([float(row["random_q10"]) for row in rows])
    random_q90 = np.array([float(row["random_q90"]) for row in rows])
    floor = 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axes[0]
    ax.plot(rank, np.maximum(theory, floor), color=COLORS["theory"], linestyle=":", label="Theory")
    ax.plot(
        rank,
        np.maximum(measured, floor),
        color=COLORS["isotropic"],
        marker="s",
        markevery=2,
        label="Truncated SVD",
    )
    ax.plot(
        rank,
        np.maximum(random_median, floor),
        color=COLORS["random"],
        linestyle="--",
        label="Random rank-matched",
    )
    ax.fill_between(
        rank,
        np.maximum(random_q10, floor),
        np.maximum(random_q90, floor),
        color=COLORS["random"],
        alpha=0.15,
        linewidth=0,
    )
    ax.set_title("Optimal rank-constrained distortion")
    ax.set_xlabel("Retained rank $r$")
    ax.set_ylabel(r"$\|K(X)-K(Z)\|_F^2$")
    clean_axis(ax, ylog=True)

    ax = axes[1]
    strength = np.logspace(-3, 3, 300)
    decoder_cost = (strength / (strength + 1)) ** 2
    ax.plot(strength, strength, color=COLORS["mse"], linestyle="--", label="MSE cost")
    ax.plot(strength, decoder_cost, color=COLORS["isotropic"], label="Decoder cost")
    ax.axhline(1.0, color=COLORS["theory"], linestyle=":", linewidth=1.0)
    ax.set_title("Ridge-saturated spectral weighting")
    ax.set_xlabel(r"Mode strength $\sigma_i^2/\tau$")
    ax.set_ylabel("Per-mode omission cost")
    clean_axis(ax, xlog=True, ylog=True)

    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    fig.legend(
        handles + handles2,
        labels + labels2,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig(fig, figures / "exp01_spectral_theorem")
    plt.close(fig)


def grouped_rows(
    rows: list[dict[str, str]], method: str, group: str | None = None
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["method"] == method and (group is None or row.get("group") == group)
    ]


def plot_tradeoffs(
    metrics: list[dict[str, str]], groups: list[dict[str, str]], figures: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axes[0]
    for method in METHOD_ORDER:
        rows = grouped_rows(metrics, method)
        xs = [float(row["test_nmse"]) for row in rows]
        ys = [float(row["decoder_distortion"]) for row in rows]
        ax.scatter(xs, ys, color=COLORS[method], marker=MARKERS[method], alpha=0.35, s=18)
        ax.scatter(
            [np.median(xs)],
            [np.median(ys)],
            color=COLORS[method],
            marker=MARKERS[method],
            s=42,
            label=LABELS[method],
            zorder=3,
        )
    ax.set_title("Reconstruction–decoder trade-off")
    ax.set_xlabel("Held-out NMSE")
    ax.set_ylabel("Relative decoder distortion")
    clean_axis(ax, xlog=True, ylog=True)

    ax = axes[1]
    x_positions = np.arange(len(GROUPS))
    for method in METHOD_ORDER:
        medians, lows, highs = [], [], []
        for group in GROUPS:
            values = [float(row["group_nmse"]) for row in grouped_rows(groups, method, group)]
            low, median, high = quantiles(values)
            lows.append(low)
            medians.append(median)
            highs.append(high)
        medians_array = np.array(medians)
        ax.plot(
            x_positions,
            medians_array,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            label=LABELS[method],
        )
        ax.fill_between(x_positions, lows, highs, color=COLORS[method], alpha=0.12, linewidth=0)
    ax.set_xticks(x_positions, ["High-var. nuisance", "Moderate", "Weak"])
    ax.set_title("Capacity allocation by feature group")
    ax.set_ylabel("Group-only NMSE")
    clean_axis(ax, ylog=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig(fig, figures / "exp01_sparse_tradeoffs")
    plt.close(fig)


def plot_recovery(groups: list[dict[str, str]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    x_positions = np.arange(len(GROUPS))
    for ax, metric, title, ylabel in (
        (axes[0], "matched_cosine", "Dictionary recovery", "Matched $|\cos|$"),
        (axes[1], "support_f1", "Support recovery", "Support F1"),
    ):
        for method in METHOD_ORDER:
            medians, lows, highs = [], [], []
            for group in GROUPS:
                values = [float(row[metric]) for row in grouped_rows(groups, method, group)]
                low, median, high = quantiles(values)
                lows.append(low)
                medians.append(median)
                highs.append(high)
            ax.plot(
                x_positions,
                medians,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                label=LABELS[method],
            )
            ax.fill_between(x_positions, lows, highs, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_xticks(x_positions, ["High-var. nuisance", "Moderate", "Weak"])
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        clean_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig(fig, figures / "exp01_feature_recovery")
    plt.close(fig)


def plot_training(curves: list[dict[str, str]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    for ax, metric, title, ylabel in (
        (axes[0], "nmse", "Training reconstruction", "Batch NMSE"),
        (axes[1], "decoder_loss", "Training decoder preservation", "Sampled relative distortion"),
    ):
        for method in METHOD_ORDER:
            method_rows = grouped_rows(curves, method)
            steps = sorted({int(row["step"]) for row in method_rows})
            medians, lows, highs = [], [], []
            for step in steps:
                values = [float(row[metric]) for row in method_rows if int(row["step"]) == step]
                low, median, high = quantiles(values)
                lows.append(low)
                medians.append(median)
                highs.append(high)
            ax.plot(
                steps,
                medians,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                label=LABELS[method],
            )
            ax.fill_between(steps, lows, highs, color=COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel(ylabel)
        clean_axis(ax, ylog=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig(fig, figures / "exp01_training_diagnostics")
    plt.close(fig)


def make_plots(output_dir: Path, figures_dir: Path, config: dict[str, Any]) -> None:
    apply_paper_style()
    plot_analytic(read_csv(output_dir / "analytic.csv"), config["analytic"], figures_dir)
    metrics = read_csv(output_dir / "metrics.csv")
    groups = read_csv(output_dir / "group_metrics.csv")
    curves = read_csv(output_dir / "training_curves.csv")
    plot_tradeoffs(metrics, groups, figures_dir)
    plot_recovery(groups, figures_dir)
    plot_training(curves, figures_dir)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = deepcopy(config)
        config["seeds"] = [0]
        config["data"]["train_samples"] = 1024
        config["data"]["test_samples"] = 512
        config["sae"]["steps"] = 10
        config["evaluation"]["geometry_groups"] = 2
        config["evaluation"]["group_samples"] = 256
        config["analytic"]["random_candidates"] = 3
    torch.set_num_threads(config.get("threads", 8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.plots_only:
        start = time.perf_counter()
        analytic_rows = run_analytic(config["analytic"])
        metrics, group_metrics, curves, calibrations = run_sparse(config, config["seeds"])
        write_csv(args.output_dir / "analytic.csv", analytic_rows)
        write_csv(args.output_dir / "metrics.csv", metrics)
        write_csv(args.output_dir / "group_metrics.csv", group_metrics)
        write_csv(args.output_dir / "training_curves.csv", curves)
        write_csv(args.output_dir / "calibration.csv", calibrations)
        metadata = {
            "config": config,
            "config_path": str(args.config),
            "git_revision": git_revision(),
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - start,
            "smoke": args.smoke,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_plots(args.output_dir, args.figures_dir, config)
    print(f"completed {config['experiment']} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
