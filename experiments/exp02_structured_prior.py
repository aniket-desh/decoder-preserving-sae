#!/usr/bin/env python3
"""Experiment 2: structured-prior selection and held-out task protection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from dpsae.decoder_distance import (
    batched_sampled_decoder_loss,
    calibrate_ridge,
    effective_degrees_of_freedom,
    ridge_hat_matrix,
)
from dpsae.plot_style import (
    COLORS,
    LABELS,
    LINESTYLES,
    MARKERS,
    apply_paper_style,
    clean_axis,
    savefig,
)
from dpsae.sae import TiedSignedTopKSAE


METHODS = ("mse", "isotropic", "task_prior", "weighted_mse", "permuted_prior")
GROUPS = ("nuisance", "protected", "background", "weak")
GROUP_LABELS = ("High-var. nuisance", "Protected", "Background", "Weak")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--seed", type=int)
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def feature_parameters(
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    counts = {
        "nuisance": config["nuisance_count"],
        "protected": config["protected_count"],
        "background": config["background_count"],
    }
    counts["weak"] = config["true_features"] - sum(counts.values())
    if counts["weak"] <= 0:
        raise ValueError("feature counts leave no weak features")
    amplitudes, probabilities, groups = [], [], []
    for group in GROUPS:
        count = counts[group]
        amplitudes.extend([config[f"{group}_amplitude"]] * count)
        probabilities.extend([config[f"{group}_probability"]] * count)
        groups.extend([group] * count)
    return torch.tensor(amplitudes), torch.tensor(probabilities), groups


def make_dictionary(config: dict[str, Any], seed: int) -> torch.Tensor:
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
    feature_groups: list[str],
    only_group: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    active = torch.rand(n_samples, dictionary.shape[0], generator=generator) < probabilities
    if only_group is not None:
        mask = torch.tensor([group == only_group for group in feature_groups])
        active &= mask[None, :]
    codes = active * torch.randn(active.shape, generator=generator) * amplitudes
    x = codes @ dictionary
    if noise_std:
        x += noise_std * torch.randn(x.shape, generator=generator)
    return x, codes


def preprocess(
    train: torch.Tensor, validation: torch.Tensor, test: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    mean = train.mean(dim=0, keepdim=True)
    rms = (train - mean).square().mean().sqrt().item()
    return (
        (train - mean) / rms,
        (validation - mean) / rms,
        (test - mean) / rms,
        mean,
        rms,
    )


def normalized_mse(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    return (x - reconstructed).square().sum() / x.square().sum().clamp_min(1e-12)


def standardized_latents(
    train_codes: torch.Tensor,
    validation_codes: torch.Tensor,
    test_codes: torch.Tensor,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = train_codes[:, indices].square().mean(dim=0).sqrt().clamp_min(1e-6)
    return (
        train_codes[:, indices] / scale,
        validation_codes[:, indices] / scale,
        test_codes[:, indices] / scale,
    )


def random_task_targets(latents: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    mixing = torch.randn(latents.shape[1], count, generator=generator)
    mixing = torch.nn.functional.normalize(mixing, dim=0)
    return latents @ mixing


def ridge_weights(x: torch.Tensor, targets: torch.Tensor, ridge: float) -> torch.Tensor:
    system = x.mT @ x + x.shape[0] * ridge * torch.eye(x.shape[1], dtype=x.dtype)
    return torch.cholesky_solve(x.mT @ targets, torch.linalg.cholesky(system))


def frozen_task_loss(
    x: torch.Tensor, reconstructed: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    original = x @ weights
    return ((x - reconstructed) @ weights).square().sum() / original.square().sum().clamp_min(
        1e-12
    )


def _grouped(x: torch.Tensor, group_size: int) -> torch.Tensor:
    if x.shape[0] % group_size:
        raise ValueError("batch size must be divisible by geometry group size")
    return x.reshape(x.shape[0] // group_size, group_size, x.shape[1])


def isotropic_loss(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    ridge: float,
    group_size: int,
    task_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    x_grouped = _grouped(x, group_size)
    reconstructed_grouped = _grouped(reconstructed, group_size)
    targets = torch.randint(
        0,
        2,
        (*x_grouped.shape[:2], task_count),
        generator=generator,
        dtype=torch.int64,
    ).to(x.dtype)
    targets = (2 * targets - 1) / math.sqrt(task_count)
    return batched_sampled_decoder_loss(
        x_grouped, reconstructed_grouped, targets, ridge=ridge
    )


def structured_loss(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    task_latents: torch.Tensor,
    *,
    ridge: float,
    group_size: int,
    isotropic_tasks: int,
    protected_tasks: int,
    task_weight: float,
    generator: torch.Generator,
) -> torch.Tensor:
    x_grouped = _grouped(x, group_size)
    reconstructed_grouped = _grouped(reconstructed, group_size)
    latent_grouped = _grouped(task_latents, group_size)
    isotropic = torch.randint(
        0,
        2,
        (*x_grouped.shape[:2], isotropic_tasks),
        generator=generator,
        dtype=torch.int64,
    ).to(x.dtype)
    isotropic = (2 * isotropic - 1) / math.sqrt(isotropic_tasks)
    mixing = torch.randn(
        task_latents.shape[1], protected_tasks, generator=generator, dtype=x.dtype
    )
    mixing = torch.nn.functional.normalize(mixing, dim=0)
    protected = latent_grouped @ mixing
    protected = protected * math.sqrt(task_weight / protected_tasks)
    targets = torch.cat((isotropic, protected), dim=2)
    return batched_sampled_decoder_loss(
        x_grouped, reconstructed_grouped, targets, ridge=ridge
    )


@torch.no_grad()
def expected_structured_prior_diagnostics(
    x: torch.Tensor,
    task_latents: torch.Tensor,
    *,
    group_size: int,
    task_weight: float,
) -> dict[str, float]:
    """Describe the task prior induced in expectation by ``structured_loss``.

    Conditional on a group's protected-latent matrix ``L``, the Rademacher and
    normalized Gaussian task draws induce

        E[T T^T | L] = I + (task_weight / p) L L^T,

    where ``p`` is the number of protected latent coordinates.  This matrix is
    generally not diagonal in the left-singular basis of ``x``, so the exact
    two-direction crossover is a scale reference rather than a predicted sparse
    transition.
    """

    x_grouped = _grouped(x.double(), group_size)
    latent_grouped = _grouped(task_latents.double(), group_size)
    protected_dim = latent_grouped.shape[-1]
    identity = torch.eye(group_size, dtype=torch.float64).expand(
        x_grouped.shape[0], group_size, group_size
    )
    increment = (
        float(task_weight)
        / protected_dim
        * (latent_grouped @ latent_grouped.transpose(-1, -2))
    )
    prior = identity + increment
    sample_gram = x_grouped @ x_grouped.transpose(-1, -2)
    commutator = sample_gram @ prior - prior @ sample_gram
    normalizer = (
        torch.linalg.matrix_norm(sample_gram, ord="fro", dim=(-2, -1))
        * torch.linalg.matrix_norm(prior, ord="fro", dim=(-2, -1))
    ).clamp_min(1e-30)
    eigenvalues = torch.linalg.eigvalsh(prior)
    extra_trace_ratio = increment.diagonal(dim1=-2, dim2=-1).sum(-1) / group_size
    return {
        "expected_prior_protected_dim": float(protected_dim),
        "expected_prior_extra_trace_ratio_mean": float(extra_trace_ratio.mean()),
        "expected_prior_min_eigenvalue_mean": float(eigenvalues[:, 0].mean()),
        "expected_prior_max_eigenvalue_mean": float(eigenvalues[:, -1].mean()),
        "expected_prior_normalized_commutator_mean": float(
            (torch.linalg.matrix_norm(commutator, ord="fro", dim=(-2, -1)) / normalizer).mean()
        ),
    }


@torch.no_grad()
def exact_decoder_distortion(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    targets: torch.Tensor | None,
    *,
    ridge: float,
    group_size: int,
    groups: int,
) -> float:
    n = group_size * groups
    x_grouped = _grouped(x[:n].double(), group_size)
    reconstructed_grouped = _grouped(reconstructed[:n].double(), group_size)
    if targets is None:
        target_grouped = torch.eye(group_size, dtype=torch.float64).expand(
            groups, group_size, group_size
        )
    else:
        target_grouped = _grouped(targets[:n].double(), group_size)
    return batched_sampled_decoder_loss(
        x_grouped, reconstructed_grouped, target_grouped, ridge=ridge
    ).item()


@torch.no_grad()
def frozen_task_distortion(
    x: torch.Tensor, reconstructed: torch.Tensor, weights: torch.Tensor
) -> float:
    return frozen_task_loss(x, reconstructed, weights).item()


def gradient_norm(loss: torch.Tensor, parameter: torch.Tensor) -> float:
    return torch.autograd.grad(loss, parameter, retain_graph=True)[0].norm().item()


def crossover_weight(config: dict[str, Any]) -> float:
    nuisance = config["nuisance_strength"]
    protected = config["protected_strength"]
    q_nuisance = nuisance / (nuisance + 1)
    q_protected = protected / (protected + 1)
    return (q_nuisance / q_protected) ** 2


def run_crossover(config: dict[str, Any]) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(12345)
    n = config["n_samples"]
    q, _ = torch.linalg.qr(torch.randn(n, 2, generator=generator, dtype=torch.float64))
    strengths = torch.tensor(
        [config["nuisance_strength"], config["protected_strength"]],
        dtype=torch.float64,
    )
    x = q * strengths.sqrt()
    keep_nuisance = x.clone()
    keep_nuisance[:, 1] = 0
    keep_protected = x.clone()
    keep_protected[:, 0] = 0
    ridge = config["tau"] / n
    k_x = ridge_hat_matrix(x, ridge)
    k_nuisance = ridge_hat_matrix(keep_nuisance, ridge)
    k_protected = ridge_hat_matrix(keep_protected, ridge)
    q_values = strengths / (strengths + config["tau"])
    threshold = crossover_weight(config)
    relative_weights = np.geomspace(
        config["relative_weight_min"],
        config["relative_weight_max"],
        config["points"],
    )
    rows = []
    for ratio in relative_weights:
        omega = threshold * ratio
        sigma = q @ torch.diag(torch.tensor([1.0, omega], dtype=torch.float64)) @ q.mT
        observed_keep_nuisance = torch.einsum(
            "ij,jk,ik->", k_x - k_nuisance, sigma, k_x - k_nuisance
        ).item()
        observed_keep_protected = torch.einsum(
            "ij,jk,ik->", k_x - k_protected, sigma, k_x - k_protected
        ).item()
        theory_keep_nuisance = omega * q_values[1].square().item()
        theory_keep_protected = q_values[0].square().item()
        rows.append(
            {
                "relative_weight": ratio,
                "protected_weight": omega,
                "crossover_weight": threshold,
                "theory_keep_nuisance": theory_keep_nuisance,
                "observed_keep_nuisance": observed_keep_nuisance,
                "theory_keep_protected": theory_keep_protected,
                "observed_keep_protected": observed_keep_protected,
                "selected": (
                    "protected"
                    if observed_keep_protected < observed_keep_nuisance
                    else "nuisance"
                ),
            }
        )
    residual = max(
        max(
            abs(row["theory_keep_nuisance"] - row["observed_keep_nuisance"]),
            abs(row["theory_keep_protected"] - row["observed_keep_protected"]),
        )
        for row in rows
    )
    if residual > 1e-9:
        raise AssertionError(f"structured crossover residual {residual:.3e}")
    return rows


def calibrate_weights(
    model: TiedSignedTopKSAE,
    x: torch.Tensor,
    protected: torch.Tensor,
    permuted: torch.Tensor,
    frozen_weights: torch.Tensor,
    config: dict[str, Any],
    *,
    ridge: float,
    task_weight: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    reconstruction, _ = model(x)
    nmse = normalized_mse(x, reconstruction)
    generator = torch.Generator().manual_seed(seed)
    iso = isotropic_loss(
        x,
        reconstruction,
        ridge=ridge,
        group_size=config["geometry_group_size"],
        task_count=config["isotropic_tasks"],
        generator=generator,
    )
    task = structured_loss(
        x,
        reconstruction,
        protected,
        ridge=ridge,
        group_size=config["geometry_group_size"],
        isotropic_tasks=config["isotropic_tasks"],
        protected_tasks=config["protected_tasks"],
        task_weight=task_weight,
        generator=generator,
    )
    perm = structured_loss(
        x,
        reconstruction,
        permuted,
        ridge=ridge,
        group_size=config["geometry_group_size"],
        isotropic_tasks=config["isotropic_tasks"],
        protected_tasks=config["protected_tasks"],
        task_weight=task_weight,
        generator=generator,
    )
    frozen = frozen_task_loss(x, reconstruction, frozen_weights)
    losses = {"isotropic": iso, "task_prior": task, "weighted_mse": frozen, "permuted_prior": perm}
    norms = {"nmse": gradient_norm(nmse, model.dictionary)}
    norms.update({name: gradient_norm(loss, model.dictionary) for name, loss in losses.items()})
    clip = tuple(config["gradient_balance_clip"])
    gammas = {
        name: float(np.clip(norms["nmse"] / max(norms[name], 1e-12), *clip))
        for name in losses
    }
    values = {"initial_nmse": nmse.item()}
    values.update({f"initial_{name}": loss.item() for name, loss in losses.items()})
    values.update({f"gradient_{name}": value for name, value in norms.items()})
    return gammas, values


def train_method(
    method: str,
    initial_state: dict[str, torch.Tensor],
    train: torch.Tensor,
    train_protected: torch.Tensor,
    train_permuted: torch.Tensor,
    validation: torch.Tensor,
    validation_targets: torch.Tensor,
    batch_indices: torch.Tensor,
    frozen_weights: torch.Tensor,
    config: dict[str, Any],
    *,
    ridge: float,
    task_weight: float,
    gammas: dict[str, float],
    seed: int,
    validation_groups: int,
) -> tuple[TiedSignedTopKSAE, list[dict[str, Any]], float]:
    model = TiedSignedTopKSAE(
        train.shape[1], config["dictionary_size"], config["top_k"], seed=seed
    )
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(batch_indices), eta_min=config["min_learning_rate"]
    )
    generator = torch.Generator().manual_seed(100_000 + seed)
    curves: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step, indices in enumerate(batch_indices, start=1):
        batch = train[indices]
        reconstruction, _ = model(batch)
        nmse = normalized_mse(batch, reconstruction)
        auxiliary = None
        if method == "isotropic":
            auxiliary = isotropic_loss(
                batch,
                reconstruction,
                ridge=ridge,
                group_size=config["geometry_group_size"],
                task_count=config["isotropic_tasks"],
                generator=generator,
            )
        elif method in {"task_prior", "permuted_prior"}:
            latent_source = train_protected if method == "task_prior" else train_permuted
            auxiliary = structured_loss(
                batch,
                reconstruction,
                latent_source[indices],
                ridge=ridge,
                group_size=config["geometry_group_size"],
                isotropic_tasks=config["isotropic_tasks"],
                protected_tasks=config["protected_tasks"],
                task_weight=task_weight,
                generator=generator,
            )
        elif method == "weighted_mse":
            auxiliary = frozen_task_loss(batch, reconstruction, frozen_weights)
        elif method != "mse":
            raise ValueError(f"unknown method {method}")
        objective = nmse if auxiliary is None else nmse + gammas[method] * auxiliary
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        model.normalize_dictionary_()
        scheduler.step()

        if step == 1 or step % config["log_every"] == 0 or step == len(batch_indices):
            with torch.no_grad():
                validation_reconstruction, _ = model(validation)
                validation_nmse = normalized_mse(validation, validation_reconstruction).item()
                validation_task = exact_decoder_distortion(
                    validation,
                    validation_reconstruction,
                    validation_targets,
                    ridge=ridge,
                    group_size=config["geometry_group_size"],
                    groups=validation_groups,
                )
            curves.append(
                {
                    "seed": seed,
                    "method": method,
                    "step": step,
                    "objective": objective.item(),
                    "nmse": nmse.item(),
                    "validation_nmse": validation_nmse,
                    "validation_protected_distortion": validation_task,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(
                f"seed={seed} method={method:14s} step={step:4d}/{len(batch_indices)} "
                f"val_nmse={validation_nmse:.4f} val_task={validation_task:.4f}",
                flush=True,
            )
    return model, curves, time.perf_counter() - start


@torch.no_grad()
def feature_recovery(
    model: TiedSignedTopKSAE,
    true_dictionary: torch.Tensor,
    test: torch.Tensor,
    true_codes: torch.Tensor,
    feature_groups: list[str],
) -> list[dict[str, float | str]]:
    learned = torch.nn.functional.normalize(model.dictionary.detach(), dim=1)
    truth = torch.nn.functional.normalize(true_dictionary, dim=1)
    similarities = (truth @ learned.mT).abs().cpu().numpy()
    truth_indices, learned_indices = linear_sum_assignment(-similarities)
    learned_codes = model.encode(test)
    rows = []
    for truth_index, learned_index in zip(truth_indices, learned_indices, strict=True):
        truth_active = true_codes[:, truth_index].ne(0)
        learned_active = learned_codes[:, learned_index].ne(0)
        true_positive = (truth_active & learned_active).sum().item()
        precision = true_positive / max(learned_active.sum().item(), 1)
        recall = true_positive / max(truth_active.sum().item(), 1)
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


def run_sparse(
    config: dict[str, Any],
    seeds: list[int],
    task_weight: float,
    *,
    methods: tuple[str, ...] = METHODS,
    relative_weight: float | None = None,
) -> tuple[list, list, list, list]:
    unknown = set(methods) - set(METHODS)
    if not methods or unknown:
        raise ValueError(f"methods must be a nonempty subset of {METHODS}; got {methods}")
    data_config = config["data"]
    sae_config = config["sae"]
    evaluation = config["evaluation"]
    metrics, groups, curves, calibrations = [], [], [], []
    for seed in seeds:
        print(f"preparing paired seed {seed}", flush=True)
        dictionary = make_dictionary(data_config, 10_000 + seed)
        amplitudes, probabilities, feature_groups = feature_parameters(data_config)
        train_raw, train_codes = sample_sparse_data(
            data_config["train_samples"], dictionary, amplitudes, probabilities,
            noise_std=data_config["noise_std"], seed=20_000 + seed,
            feature_groups=feature_groups,
        )
        validation_raw, validation_codes = sample_sparse_data(
            data_config["validation_samples"], dictionary, amplitudes, probabilities,
            noise_std=data_config["noise_std"], seed=25_000 + seed,
            feature_groups=feature_groups,
        )
        test_raw, test_codes = sample_sparse_data(
            data_config["test_samples"], dictionary, amplitudes, probabilities,
            noise_std=data_config["noise_std"], seed=30_000 + seed,
            feature_groups=feature_groups,
        )
        train, validation, test, train_mean, train_rms = preprocess(
            train_raw, validation_raw, test_raw
        )
        protected_indices = [i for i, group in enumerate(feature_groups) if group == "protected"]
        background_indices = [i for i, group in enumerate(feature_groups) if group == "background"]
        train_protected, validation_protected, test_protected = standardized_latents(
            train_codes, validation_codes, test_codes, protected_indices
        )
        train_background, validation_background, test_background = standardized_latents(
            train_codes,
            validation_codes,
            test_codes,
            background_indices[: data_config["protected_count"]],
        )
        heldout_tasks = evaluation["heldout_tasks"]
        validation_targets = random_task_targets(
            validation_protected, heldout_tasks, 31_000 + seed
        )
        test_targets = random_task_targets(test_protected, heldout_tasks, 32_000 + seed)
        unrelated_targets = random_task_targets(
            test_background, heldout_tasks, 33_000 + seed
        )
        permutation = torch.randperm(
            train.shape[0], generator=torch.Generator().manual_seed(34_000 + seed)
        )
        train_permuted = train_protected[permutation]
        ridge = calibrate_ridge(
            train[: sae_config["geometry_group_size"]],
            sae_config["target_dof_fraction"],
        )
        dof = effective_degrees_of_freedom(
            train[: sae_config["geometry_group_size"]], ridge
        ).item() / sae_config["geometry_group_size"]
        frozen_weights = ridge_weights(train, train_protected, ridge)
        base_model = TiedSignedTopKSAE(
            data_config["input_dim"],
            sae_config["dictionary_size"],
            sae_config["top_k"],
            seed=40_000 + seed,
        )
        initial_state = deepcopy(base_model.state_dict())
        calibration_indices = torch.arange(sae_config["batch_size"])
        gammas, calibration = calibrate_weights(
            base_model,
            train[calibration_indices],
            train_protected[calibration_indices],
            train_permuted[calibration_indices],
            frozen_weights,
            sae_config,
            ridge=ridge,
            task_weight=task_weight,
            seed=50_000 + seed,
        )
        prior_diagnostics = expected_structured_prior_diagnostics(
            train[calibration_indices],
            train_protected[calibration_indices],
            group_size=sae_config["geometry_group_size"],
            task_weight=task_weight,
        )
        calibrations.append(
            {
                "seed": seed,
                "relative_weight": relative_weight,
                "ridge": ridge,
                "dof_fraction": dof,
                "task_weight": task_weight,
                "train_rms": train_rms,
                **{f"gamma_{key}": value for key, value in gammas.items()},
                **calibration,
                **prior_diagnostics,
            }
        )
        print(
            f"seed={seed} ridge={ridge:.4g} dof/n={dof:.3f} "
            f"gamma_task={gammas['task_prior']:.3g}",
            flush=True,
        )
        batch_indices = torch.randint(
            0,
            train.shape[0],
            (sae_config["steps"], sae_config["batch_size"]),
            generator=torch.Generator().manual_seed(60_000 + seed),
        )
        group_only = {}
        for group_index, group in enumerate(GROUPS):
            raw, _ = sample_sparse_data(
                evaluation["group_samples"], dictionary, amplitudes, probabilities,
                noise_std=data_config["noise_std"],
                seed=70_000 + 100 * seed + group_index,
                feature_groups=feature_groups,
                only_group=group,
            )
            group_only[group] = (raw - train_mean) / train_rms

        for method in methods:
            model, method_curves, elapsed = train_method(
                method,
                initial_state,
                train,
                train_protected,
                train_permuted,
                validation,
                validation_targets,
                batch_indices,
                frozen_weights,
                sae_config,
                ridge=ridge,
                task_weight=task_weight,
                gammas=gammas,
                seed=seed,
                validation_groups=evaluation["validation_geometry_groups"],
            )
            for row in method_curves:
                row["relative_weight"] = relative_weight
                row["task_weight"] = task_weight
            curves.extend(method_curves)
            with torch.no_grad():
                reconstruction, code = model(test)
                metrics.append(
                    {
                        "seed": seed,
                        "method": method,
                        "relative_weight": relative_weight,
                        "task_weight": task_weight,
                        "test_nmse": normalized_mse(test, reconstruction).item(),
                        "protected_decoder_distortion": exact_decoder_distortion(
                            test, reconstruction, test_targets, ridge=ridge,
                            group_size=sae_config["geometry_group_size"],
                            groups=evaluation["geometry_groups"],
                        ),
                        "unrelated_decoder_distortion": exact_decoder_distortion(
                            test, reconstruction, unrelated_targets, ridge=ridge,
                            group_size=sae_config["geometry_group_size"],
                            groups=evaluation["geometry_groups"],
                        ),
                        "isotropic_decoder_distortion": exact_decoder_distortion(
                            test, reconstruction, None, ridge=ridge,
                            group_size=sae_config["geometry_group_size"],
                            groups=evaluation["geometry_groups"],
                        ),
                        "frozen_task_distortion": frozen_task_distortion(
                            test, reconstruction, frozen_weights
                        ),
                        "average_l0": code.ne(0).sum(dim=1).float().mean().item(),
                        "train_seconds": elapsed,
                    }
                )
            recovery = feature_recovery(
                model, dictionary, test, test_codes, feature_groups
            )
            for group in GROUPS:
                selected = [row for row in recovery if row["group"] == group]
                with torch.no_grad():
                    group_reconstruction, _ = model(group_only[group])
                groups.append(
                    {
                        "seed": seed,
                        "method": method,
                        "relative_weight": relative_weight,
                        "task_weight": task_weight,
                        "group": group,
                        "group_nmse": normalized_mse(
                            group_only[group], group_reconstruction
                        ).item(),
                        "matched_cosine": float(np.mean([row["cosine"] for row in selected])),
                        "support_precision": float(np.mean([row["precision"] for row in selected])),
                        "support_recall": float(np.mean([row["recall"] for row in selected])),
                        "support_f1": float(np.mean([row["f1"] for row in selected])),
                    }
                )
    return metrics, groups, curves, calibrations


def quantiles(values: list[float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.1, 0.5, 0.9]))


def selected_rows(
    rows: list[dict[str, str]], method: str, group: str | None = None
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["method"] == method and (group is None or row.get("group") == group)
    ]


def shared_legend(fig, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )


def plot_crossover(rows: list[dict[str, str]], figures: Path) -> None:
    x = np.array([float(row["relative_weight"]) for row in rows])
    nuisance = np.array([float(row["observed_keep_nuisance"]) for row in rows])
    protected = np.array([float(row["observed_keep_protected"]) for row in rows])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(x, nuisance, color=COLORS["mse"], linestyle="--", label="Retain nuisance")
    ax.plot(x, protected, color=COLORS["task_prior"], label="Retain protected")
    ax.axvline(1, color=COLORS["theory"], linestyle=":", label="Predicted crossover")
    ax.set_xlabel(r"Protected weight / predicted crossover $\omega/\omega_\star$")
    ax.set_ylabel("Structured decoder distortion")
    ax.set_title("Task prior changes the retained direction")
    clean_axis(ax, xlog=True, ylog=True)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    savefig(fig, figures / "exp02_structured_crossover")
    plt.close(fig)


def plot_task_protection(metrics: list[dict[str, str]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    for method in METHODS:
        rows = selected_rows(metrics, method)
        x = [float(row["test_nmse"]) for row in rows]
        y = [float(row["protected_decoder_distortion"]) for row in rows]
        ax.scatter(x, y, color=COLORS[method], marker=MARKERS[method], alpha=0.25, s=17)
        ax.scatter(
            [np.median(x)], [np.median(y)], color=COLORS[method],
            marker=MARKERS[method], s=42, label=LABELS[method], zorder=3,
        )
    ax.set_title("Held-out protected-task trade-off")
    ax.set_xlabel("Held-out NMSE")
    ax.set_ylabel("Protected-task distortion")
    clean_axis(ax)

    ax = axes[1]
    metric_names = (
        "protected_decoder_distortion",
        "unrelated_decoder_distortion",
        "isotropic_decoder_distortion",
    )
    x_positions = np.arange(3)
    for method in METHODS:
        median, low, high = [], [], []
        for metric in metric_names:
            values = [float(row[metric]) for row in selected_rows(metrics, method)]
            q10, q50, q90 = quantiles(values)
            low.append(q10)
            median.append(q50)
            high.append(q90)
        ax.plot(
            x_positions, median, color=COLORS[method], marker=MARKERS[method],
            linestyle=LINESTYLES[method], label=LABELS[method],
        )
        ax.fill_between(x_positions, low, high, color=COLORS[method], alpha=0.10, linewidth=0)
    ax.set_xticks(x_positions, ["Protected", "Unrelated", "Isotropic"])
    ax.set_title("Selectivity across task families")
    ax.set_ylabel("Relative decoder distortion")
    clean_axis(ax, ylog=True)
    shared_legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    savefig(fig, figures / "exp02_task_protection")
    plt.close(fig)


def plot_recovery(groups: list[dict[str, str]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    x_positions = np.arange(len(GROUPS))
    for ax, metric, title, ylabel in (
        (axes[0], "matched_cosine", "Dictionary recovery", r"Matched $|\cos|$"),
        (axes[1], "support_f1", "Support recovery", "Support F1"),
    ):
        for method in METHODS:
            median, low, high = [], [], []
            for group in GROUPS:
                values = [float(row[metric]) for row in selected_rows(groups, method, group)]
                q10, q50, q90 = quantiles(values)
                low.append(q10)
                median.append(q50)
                high.append(q90)
            ax.plot(
                x_positions, median, color=COLORS[method], marker=MARKERS[method],
                linestyle=LINESTYLES[method], label=LABELS[method],
            )
            ax.fill_between(x_positions, low, high, color=COLORS[method], alpha=0.10, linewidth=0)
        ax.set_xticks(x_positions, GROUP_LABELS, rotation=10)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        clean_axis(ax)
    shared_legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    savefig(fig, figures / "exp02_feature_recovery")
    plt.close(fig)


def plot_training(curves: list[dict[str, str]], figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for ax, metric, title, ylabel in (
        (axes[0], "validation_nmse", "Validation reconstruction", "Validation NMSE"),
        (
            axes[1],
            "validation_protected_distortion",
            "Held-out task protection",
            "Protected-task distortion",
        ),
    ):
        for method in METHODS:
            rows = selected_rows(curves, method)
            steps = sorted({int(row["step"]) for row in rows})
            seeds = sorted({int(row["seed"]) for row in rows})
            for seed in seeds:
                seed_rows = sorted(
                    (row for row in rows if int(row["seed"]) == seed),
                    key=lambda row: int(row["step"]),
                )
                ax.plot(
                    [int(row["step"]) for row in seed_rows],
                    [float(row[metric]) for row in seed_rows],
                    color=COLORS[method], linestyle=LINESTYLES[method],
                    linewidth=0.65, alpha=0.12, label="_nolegend_",
                )
            median = [
                float(np.median([float(row[metric]) for row in rows if int(row["step"]) == step]))
                for step in steps
            ]
            ax.plot(
                steps, median, color=COLORS[method], linestyle=LINESTYLES[method],
                linewidth=1.8, label=LABELS[method],
            )
        ax.set_title(title)
        ax.set_xlabel("Optimization step")
        ax.set_ylabel(ylabel)
        clean_axis(ax, ylog=True)
    shared_legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    savefig(fig, figures / "exp02_training_diagnostics")
    plt.close(fig)


def make_plots(output_dir: Path, figures_dir: Path) -> None:
    apply_paper_style()
    plot_crossover(read_csv(output_dir / "crossover.csv"), figures_dir)
    plot_task_protection(read_csv(output_dir / "metrics.csv"), figures_dir)
    plot_recovery(read_csv(output_dir / "group_metrics.csv"), figures_dir)
    plot_training(read_csv(output_dir / "training_curves.csv"), figures_dir)


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
        config["crossover"]["points"] = 9
    if args.seed is not None:
        config["seeds"] = [args.seed]
    torch.set_num_threads(config.get("threads", 8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    threshold = crossover_weight(config["crossover"])
    task_weight = threshold * config["crossover"]["sparse_weight_multiplier"]
    if not args.plots_only:
        start = time.perf_counter()
        crossover = run_crossover(config["crossover"])
        metrics, groups, curves, calibrations = run_sparse(
            config, config["seeds"], task_weight
        )
        write_csv(args.output_dir / "crossover.csv", crossover)
        write_csv(args.output_dir / "metrics.csv", metrics)
        write_csv(args.output_dir / "group_metrics.csv", groups)
        write_csv(args.output_dir / "training_curves.csv", curves)
        write_csv(args.output_dir / "calibration.csv", calibrations)
        metadata = {
            "config": config,
            "config_path": str(args.config),
            "git_revision": git_revision(),
            "crossover_weight": threshold,
            "task_weight": task_weight,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - start,
            "smoke": args.smoke,
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_plots(args.output_dir, args.figures_dir)
    print(f"completed {config['experiment']} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
