#!/usr/bin/env python3
"""Frozen-checkpoint mechanism diagnostics for Experiment 4b.

The executable stages are deliberately ordered by identifying power:

tangent
    Decompose exact ridge-hat distortion into the source-Gram Frechet tangent,
    its diagonal/off-diagonal source-eigenbasis terms, and the nonlinear
    endpoint remainder. This stage uses existing activation/reconstruction
    caches only.

nonorth
    Re-encode cached activations with frozen checkpoints and replace the
    learned decoder atoms by a valid PSD counterfactual whose residual
    components are mutually orthogonal while atom norms and atom-bias inner
    products are preserved.

The support/null stage from the theory note is not exposed here. The immutable
natural caches currently contain activations and reconstructions, not frozen
selection/test code caches. Silently treating recomputed codes as immutable
cached codes would make the proposed support-only selection protocol drift.

Both implemented stages stream geometry groups, hash every immutable input,
record repository/protocol provenance, and emit explicit pass/fail diagnostics.
They do not load the language model, train an SAE, or modify source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from dpsae.mech_analysis import load_sae


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "exp04b_confirmatory"
DEFAULT_OUTPUT = ROOT / "artifacts" / "exp04b_mechanism_attribution"
DEFAULT_NATURAL_CACHE = DEFAULT_ARTIFACT / "natural_test.pt"
DEFAULT_STATIC = DEFAULT_ARTIFACT / "static_calibration.pt"
DEFAULT_RECONSTRUCTIONS = DEFAULT_ARTIFACT / "exact_reconstructions" / "baseline"
DEFAULT_MODELS = DEFAULT_ARTIFACT / "baseline_confirm" / "models.pt"

TANGENT_RATIO_TOLERANCE = 0.20
REMAINDER_RATIO_TOLERANCE = 0.20
TANGENT_COSINE_THRESHOLD = 0.95
ORTHOGONAL_SURVIVAL_THRESHOLD = 0.80


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def repository_state(root: Path = ROOT) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    return {"revision": revision, "dirty": dirty}


def resolve_ridge(ridge: float | None, static_path: Path | None) -> tuple[float, dict | None]:
    if ridge is not None:
        if ridge <= 0:
            raise ValueError("ridge must be strictly positive")
        return float(ridge), None
    if static_path is None:
        raise ValueError("either --ridge or --static-calibration is required")
    payload = torch.load(static_path, map_location="cpu", weights_only=False)
    if "ridge" not in payload:
        raise ValueError(f"static calibration has no ridge: {static_path}")
    value = float(payload["ridge"])
    if value <= 0:
        raise ValueError("calibrated ridge must be strictly positive")
    return value, file_provenance(static_path)


def load_natural_activations(
    path: Path,
    *,
    exact_tokens: int,
    group_size: int,
    max_groups: int | None,
) -> tuple[Tensor, int]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "activations" not in payload:
        raise ValueError(f"natural cache is missing activations: {path}")
    activations = payload["activations"]
    if activations.ndim != 3 or not activations.is_floating_point():
        raise ValueError("cached activations must have shape [sequences, tokens, features]")
    flat = activations.flatten(0, 1)
    if exact_tokens <= 0 or exact_tokens > len(flat):
        raise ValueError(f"exact_tokens must lie in [1, {len(flat)}]")
    available_groups = exact_tokens // group_size
    if available_groups <= 0:
        raise ValueError("exact_tokens must contain at least one complete group")
    groups = available_groups if max_groups is None else min(available_groups, max_groups)
    if groups <= 0:
        raise ValueError("max_groups must be positive")
    used = groups * group_size
    return flat[:used], groups


def iter_groups(values: Tensor, group_size: int) -> Iterable[Tensor]:
    if values.ndim != 2 or len(values) % group_size:
        raise ValueError("values must be a complete matrix of streamed groups")
    for start in range(0, len(values), group_size):
        yield values[start : start + group_size]


def ridge_hat_from_gram(gram: Tensor, tau: float) -> Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    if tau <= 0:
        raise ValueError("tau must be positive")
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    inverse = torch.linalg.solve(gram + tau * identity, identity)
    result = identity - tau * inverse
    return (result + result.mT) / 2


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return 1.0 if left.norm() == right.norm() else 0.0
    return float((left * right).sum() / denominator)


def tangent_group_statistics(original: Tensor, reconstructed: Tensor, ridge: float) -> dict[str, float]:
    """Return the exact tangent/remainder decomposition for one group."""

    if original.shape != reconstructed.shape or original.ndim != 2:
        raise ValueError("original and reconstructed must have identical rank-2 shapes")
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    original = original.float()
    reconstructed = reconstructed.float()
    n = original.shape[0]
    tau = n * ridge
    identity = torch.eye(n, device=original.device, dtype=original.dtype)
    gram = original @ original.mT
    reconstructed_gram = reconstructed @ reconstructed.mT
    perturbation = reconstructed_gram - gram
    source_resolvent = torch.linalg.solve(gram + tau * identity, identity)
    endpoint_resolvent = torch.linalg.solve(reconstructed_gram + tau * identity, identity)
    reference = identity - tau * source_resolvent
    endpoint = identity - tau * endpoint_resolvent
    exact = endpoint - reference
    tangent = tau * source_resolvent @ perturbation @ source_resolvent
    remainder = -tau * (
        source_resolvent
        @ perturbation
        @ source_resolvent
        @ perturbation
        @ endpoint_resolvent
    )
    identity_residual = (exact - tangent - remainder).norm()

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    perturbation_in_basis = eigenvectors.mT @ perturbation @ eigenvectors
    denominators = eigenvalues + tau
    tangent_in_basis = (
        tau
        * perturbation_in_basis
        / (denominators[:, None] * denominators[None, :])
    )
    tangent_diagonal = tangent_in_basis.diagonal().square().sum()
    tangent_total = tangent_in_basis.square().sum()
    tangent_off_diagonal = (tangent_total - tangent_diagonal).clamp_min(0)

    commutator = gram @ reconstructed_gram - reconstructed_gram @ gram
    commutator_denominator = gram.norm() * reconstructed_gram.norm()
    exact_norm = exact.norm()
    gram_reference = gram.square().sum().clamp_min(1e-20)
    return {
        "exact_numerator": float(exact.square().sum()),
        "denominator": float(reference.square().sum()),
        "tangent_numerator": float(tangent_total),
        "tangent_diagonal_numerator": float(tangent_diagonal),
        "tangent_off_diagonal_numerator": float(tangent_off_diagonal),
        "remainder_numerator": float(remainder.square().sum()),
        "tangent_remainder_cross": float(2 * (tangent * remainder).sum()),
        "remainder_ratio": float(remainder.norm() / exact_norm.clamp_min(1e-20)),
        "tangent_exact_cosine": _cosine(tangent, exact),
        "decomposition_residual": float(identity_residual),
        "gram_error_numerator": float(perturbation.square().sum()),
        "gram_reference": float(gram_reference),
        "normalized_gram_error": float(perturbation.square().sum() / gram_reference),
        "commutator": float(
            commutator.norm() / commutator_denominator.clamp_min(1e-20)
        ),
    }


def orthogonal_residual_gram(code: Tensor, decoder: Tensor, bias: Tensor) -> Tensor:
    """Return the valid PSD Gram with decoder residual atoms orthogonalized.

    The construction preserves every decoder-row norm and every row/bias inner
    product. It never materializes the dictionary-size squared decoder Gram.
    """

    if code.ndim != 2 or decoder.ndim != 2 or bias.ndim != 1:
        raise ValueError("code, decoder, and bias must have ranks 2, 2, and 1")
    if code.shape[1] != decoder.shape[0] or decoder.shape[1] != len(bias):
        raise ValueError("code, decoder, and bias shapes do not compose")
    code = code.float()
    decoder = decoder.float()
    bias = bias.float()
    beta = bias.norm()
    if beta <= torch.finfo(code.dtype).eps:
        residual_code = code * decoder.square().sum(1).sqrt()
        gram = residual_code @ residual_code.mT
        return (gram + gram.mT) / 2
    atom_bias = decoder @ bias
    residual_square = (decoder.square().sum(1) - atom_bias.square() / beta.square()).clamp_min(0)
    residual_code = code * residual_square.sqrt()
    shared = code @ atom_bias / beta + beta
    gram = residual_code @ residual_code.mT + shared[:, None] * shared[None, :]
    return (gram + gram.mT) / 2


def code_weighted_coherence(code: Tensor, decoder: Tensor) -> float:
    """Compute code-mass weighted squared coherence over co-active pairs."""

    numerator = code.new_zeros(())
    denominator = code.new_zeros(())
    for row in code:
        indices = (row != 0).nonzero(as_tuple=False).flatten()
        if len(indices) < 2:
            continue
        values = row[indices]
        weights = values[:, None] * values[None, :]
        off_diagonal = ~torch.eye(len(indices), device=row.device, dtype=torch.bool)
        gram = decoder[indices] @ decoder[indices].mT
        numerator += (weights[off_diagonal] * gram[off_diagonal].square()).sum()
        denominator += weights[off_diagonal].sum()
    return float(numerator / denominator.clamp_min(1e-20))


def nonorthogonal_group_statistics(
    original: Tensor,
    code: Tensor,
    decoder: Tensor,
    bias: Tensor,
    ridge: float,
) -> dict[str, float]:
    """Return exact and valid-orthogonal-counterfactual metrics for one group."""

    original = original.float()
    code = code.float()
    decoder = decoder.float()
    bias = bias.float()
    reconstruction = code @ decoder + bias
    n = len(original)
    tau = n * ridge
    source_gram = original @ original.mT
    full_gram = reconstruction @ reconstruction.mT
    orthogonal_gram = orthogonal_residual_gram(code, decoder, bias)
    reference = ridge_hat_from_gram(source_gram, tau)
    full_hat = ridge_hat_from_gram(full_gram, tau)
    orthogonal_hat = ridge_hat_from_gram(orthogonal_gram, tau)
    full_numerator = (full_hat - reference).square().sum()
    orthogonal_numerator = (orthogonal_hat - reference).square().sum()
    cross = full_gram - orthogonal_gram
    residual = source_gram - orthogonal_gram
    alignment_denominator = cross.norm() * residual.norm()
    alignment = (
        float((cross * residual).sum() / alignment_denominator)
        if alignment_denominator > 0
        else 0.0
    )
    eigenvalue = torch.linalg.eigvalsh(orthogonal_gram).min()
    return {
        "exact_numerator": float(full_numerator),
        "orthogonal_numerator": float(orthogonal_numerator),
        "nonorthogonality_benefit": float(orthogonal_numerator - full_numerator),
        "denominator": float(reference.square().sum()),
        "nmse_numerator": float((reconstruction - original).square().sum()),
        "activation_reference": float(original.square().sum()),
        "mean_l0": float((code != 0).sum(1).float().mean()),
        "code_weighted_coherence": code_weighted_coherence(code, decoder),
        "signed_cross_alignment": alignment,
        "orthogonal_gram_min_eigenvalue": float(eigenvalue),
    }


def infer_pairs(names: Sequence[str], requested: Sequence[str] | None = None) -> list[tuple[str, str]]:
    available = set(names)
    if requested:
        pairs = []
        for value in requested:
            if ":" not in value:
                raise ValueError(f"pair must have form mse:dpsae, got {value!r}")
            baseline, candidate = value.split(":", 1)
            if baseline not in available or candidate not in available:
                raise ValueError(f"pair references unavailable model: {value}")
            pairs.append((baseline, candidate))
        return pairs

    parsed: dict[tuple[str, int], dict[str, str]] = {}
    pattern = re.compile(r"^(mse|dpsae)(.*)_s(\d+)$")
    for name in names:
        match = pattern.match(name)
        if match:
            method, variant, seed = match.groups()
            parsed.setdefault((variant, int(seed)), {})[method] = name
    pairs = [
        (methods["mse"], methods["dpsae"])
        for _key, methods in sorted(parsed.items())
        if {"mse", "dpsae"} <= methods.keys()
    ]
    if not pairs:
        raise ValueError("could not infer any paired MSE/DPSAE model names")
    return pairs


def _values(rows: Sequence[Mapping[str, float]], key: str) -> Tensor:
    return torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)


def ratio_estimate(numerator: Tensor, denominator: Tensor) -> float:
    return float(numerator.sum() / denominator.sum().clamp_min(1e-20))


def bootstrap_ratio(
    numerator: Tensor,
    denominator: Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(numerator) != len(denominator) or not len(numerator):
        raise ValueError("bootstrap arrays must be nonempty and aligned")
    estimate = ratio_estimate(numerator, denominator)
    if samples <= 0:
        return {"estimate": estimate, "ci95": None, "samples": 0}
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(numerator), (samples, len(numerator)), generator=generator)
    draws = numerator[indices].sum(1) / denominator[indices].sum(1).clamp_min(1e-20)
    interval = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=draws.dtype))
    return {
        "estimate": estimate,
        "ci95": [float(interval[0]), float(interval[1])],
        "samples": samples,
    }


def _paired_metric(
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
    key: str,
    denominator: Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    difference = _values(baseline_rows, key) - _values(candidate_rows, key)
    return bootstrap_ratio(difference, denominator, samples=samples, seed=seed)


def tangent_pair_report(
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    denominator = _values(baseline_rows, "denominator")
    candidate_denominator = _values(candidate_rows, "denominator")
    torch.testing.assert_close(denominator, candidate_denominator)
    reports = {}
    for name, key in (
        ("exact_advantage", "exact_numerator"),
        ("linear_advantage", "tangent_numerator"),
        ("diagonal_linear_advantage", "tangent_diagonal_numerator"),
        ("off_diagonal_linear_advantage", "tangent_off_diagonal_numerator"),
    ):
        reports[name] = _paired_metric(
            baseline_rows,
            candidate_rows,
            key,
            denominator,
            samples=bootstrap_samples,
            seed=seed,
        )
    endpoint_numerator = (
        _values(baseline_rows, "exact_numerator")
        - _values(candidate_rows, "exact_numerator")
        - _values(baseline_rows, "tangent_numerator")
        + _values(candidate_rows, "tangent_numerator")
    )
    reports["endpoint_contrast"] = bootstrap_ratio(
        endpoint_numerator,
        denominator,
        samples=bootstrap_samples,
        seed=seed,
    )

    gram_denominator = _values(baseline_rows, "gram_reference")
    reports["raw_gram_advantage"] = bootstrap_ratio(
        _values(baseline_rows, "gram_error_numerator")
        - _values(candidate_rows, "gram_error_numerator"),
        gram_denominator,
        samples=bootstrap_samples,
        seed=seed,
    )
    exact = reports["exact_advantage"]["estimate"]
    linear = reports["linear_advantage"]["estimate"]
    maximum_remainder = max(
        float(_values(baseline_rows, "remainder_ratio").median()),
        float(_values(candidate_rows, "remainder_ratio").median()),
    )
    minimum_cosine = min(
        float(_values(baseline_rows, "tangent_exact_cosine").median()),
        float(_values(candidate_rows, "tangent_exact_cosine").median()),
    )
    tangent_ratio = linear / exact if exact != 0 else math.nan
    source_tangent_sufficient = (
        exact > 0
        and linear > 0
        and abs(tangent_ratio - 1) <= TANGENT_RATIO_TOLERANCE
        and maximum_remainder <= REMAINDER_RATIO_TOLERANCE
        and minimum_cosine >= TANGENT_COSINE_THRESHOLD
    )
    diagonal = reports["diagonal_linear_advantage"]["estimate"]
    off_diagonal = reports["off_diagonal_linear_advantage"]["estimate"]
    if max(diagonal, off_diagonal) <= 0:
        dominant = "neither"
    else:
        dominant = "diagonal" if diagonal >= off_diagonal else "off_diagonal"
    reports["diagnostics"] = {
        "exact_favors_dpsae": exact > 0,
        "endpoint_nonlinearity_necessary": exact > 0 and linear <= 0,
        "source_tangent_sufficient": source_tangent_sufficient,
        "ridge_weighting_signature": linear > 0
        and reports["raw_gram_advantage"]["estimate"] <= 0,
        "dominant_linear_component": dominant,
        "linear_to_exact_advantage_ratio": tangent_ratio,
        "maximum_model_median_remainder_ratio": maximum_remainder,
        "minimum_model_median_tangent_exact_cosine": minimum_cosine,
        "thresholds": {
            "linear_to_exact_absolute_tolerance": TANGENT_RATIO_TOLERANCE,
            "maximum_remainder_ratio": REMAINDER_RATIO_TOLERANCE,
            "minimum_tangent_exact_cosine": TANGENT_COSINE_THRESHOLD,
        },
    }
    return reports


def nonorthogonal_pair_report(
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    denominator = _values(baseline_rows, "denominator")
    torch.testing.assert_close(denominator, _values(candidate_rows, "denominator"))
    exact = _paired_metric(
        baseline_rows,
        candidate_rows,
        "exact_numerator",
        denominator,
        samples=bootstrap_samples,
        seed=seed,
    )
    orthogonal = _paired_metric(
        baseline_rows,
        candidate_rows,
        "orthogonal_numerator",
        denominator,
        samples=bootstrap_samples,
        seed=seed,
    )
    contrast = bootstrap_ratio(
        (
            _values(baseline_rows, "exact_numerator")
            - _values(candidate_rows, "exact_numerator")
            - _values(baseline_rows, "orthogonal_numerator")
            + _values(candidate_rows, "orthogonal_numerator")
        ),
        denominator,
        samples=bootstrap_samples,
        seed=seed,
    )
    baseline_benefit = ratio_estimate(
        _values(baseline_rows, "nonorthogonality_benefit"), denominator
    )
    candidate_benefit = ratio_estimate(
        _values(candidate_rows, "nonorthogonality_benefit"), denominator
    )
    exact_value = exact["estimate"]
    orthogonal_value = orthogonal["estimate"]
    survival = orthogonal_value / exact_value if exact_value != 0 else math.nan
    return {
        "exact_advantage": exact,
        "orthogonal_advantage": orthogonal,
        "nonorthogonality_contrast": contrast,
        "mse_nonorthogonality_benefit": baseline_benefit,
        "dpsae_nonorthogonality_benefit": candidate_benefit,
        "diagnostics": {
            "exact_favors_dpsae": exact_value > 0,
            "nonorthogonality_necessary": (
                exact_value > 0
                and orthogonal_value <= 0
                and contrast["estimate"] > 0
            ),
            "nonorthogonality_primary_falsified": (
                exact_value > 0
                and orthogonal_value >= ORTHOGONAL_SURVIVAL_THRESHOLD * exact_value
                and candidate_benefit <= baseline_benefit
            ),
            "orthogonal_survival_fraction": survival,
            "orthogonal_survival_threshold": ORTHOGONAL_SURVIVAL_THRESHOLD,
        },
    }


def _model_summary(rows: Sequence[Mapping[str, float]], keys: Sequence[str]) -> dict[str, float]:
    result = {}
    for key in keys:
        values = _values(rows, key)
        result[f"mean_{key}"] = float(values.mean())
        result[f"median_{key}"] = float(values.median())
    return result


def _base_result(stage: str, protocol: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict:
    return {
        "experiment": "exp04b_mechanism_attribution",
        "stage": stage,
        "complete": False,
        "started_at": utc_now(),
        "repository": repository_state(),
        "evaluator": file_provenance(Path(__file__)),
        "protocol": dict(protocol),
        "inputs": dict(inputs),
        "support_stage": {
            "implemented": False,
            "reason": (
                "immutable natural caches do not contain frozen selection/test code caches; "
                "support/null evaluation is intentionally not approximated"
            ),
        },
    }


@torch.inference_mode()
def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Cache exact-view reconstructions one checkpoint at a time."""

    started = time.perf_counter()
    if args.batch_tokens < 1:
        raise ValueError("batch_tokens must be positive")
    device = torch.device(args.device)
    activations, groups = load_natural_activations(
        args.natural_cache,
        exact_tokens=args.exact_tokens,
        group_size=args.group_size,
        max_groups=args.max_groups,
    )
    payloads = torch.load(args.models, map_location="cpu", weights_only=False)
    if not isinstance(payloads, Mapping):
        raise ValueError("models artifact must map model names to payloads")
    pairs = infer_pairs(list(payloads), args.pair)
    selected_names = sorted({name for pair in pairs for name in pair})
    args.reconstruction_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in selected_names:
        destination = args.reconstruction_dir / f"{name}.pt"
        if destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=False)
            if not isinstance(existing, Tensor) or existing.shape != activations.shape:
                raise RuntimeError(f"incompatible reconstruction cache: {destination}")
            outputs[name] = file_provenance(destination)
            continue
        model = load_sae(payloads[name], input_dim=activations.shape[1], device=device).eval()
        chunks = []
        for batch in activations.split(args.batch_tokens):
            reconstruction, _ = model(batch.to(device).float(), use_threshold=True)
            chunks.append(reconstruction.cpu().half())
        atomic_torch(destination, torch.cat(chunks))
        outputs[name] = file_provenance(destination)
        del model, chunks
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = _base_result(
        "prepare",
        {
            "group_size": args.group_size,
            "groups": groups,
            "exact_tokens_used": groups * args.group_size,
            "pairs": pairs,
            "device": str(device),
            "batch_tokens": args.batch_tokens,
            "streaming": "one checkpoint at a time",
        },
        {
            "natural_cache": file_provenance(args.natural_cache),
            "models": file_provenance(args.models),
        },
    )
    result["reconstructions"] = outputs
    result["summary"] = {"model_count": len(outputs), "pairs": len(pairs)}
    result["wall_seconds"] = time.perf_counter() - started
    result["finished_at"] = utc_now()
    result["complete"] = True
    atomic_json(args.output, result)
    return result


def run_tangent(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    ridge, static_provenance = resolve_ridge(args.ridge, args.static_calibration)
    activations, groups = load_natural_activations(
        args.natural_cache,
        exact_tokens=args.exact_tokens,
        group_size=args.group_size,
        max_groups=args.max_groups,
    )
    reconstruction_paths = {
        path.stem: path for path in sorted(args.reconstruction_dir.glob("*.pt"))
    }
    pairs = infer_pairs(list(reconstruction_paths), args.pair)
    selected_names = sorted({name for pair in pairs for name in pair})
    inputs: dict[str, Any] = {
        "natural_cache": file_provenance(args.natural_cache),
        "reconstructions": {
            name: file_provenance(reconstruction_paths[name]) for name in selected_names
        },
    }
    if static_provenance is not None:
        inputs["static_calibration"] = static_provenance
    protocol = {
        "ridge": ridge,
        "ridge_convention": "matrix regularizer = group_size * ridge",
        "group_size": args.group_size,
        "groups": groups,
        "exact_tokens_used": groups * args.group_size,
        "encoding": "existing threshold reconstruction cache",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "pairs": pairs,
        "streaming": "one geometry group at a time within one reconstruction file",
    }
    result = _base_result("tangent", protocol, inputs)
    models = {}
    for name in selected_names:
        reconstruction = torch.load(
            reconstruction_paths[name], map_location="cpu", weights_only=False
        )
        if reconstruction.ndim == 3:
            reconstruction = reconstruction.flatten(0, 1)
        if reconstruction.ndim != 2:
            raise ValueError(
                f"reconstruction {name} must have rank 2 or 3, "
                f"got shape {tuple(reconstruction.shape)}"
            )
        if (
            reconstruction.shape[1] != activations.shape[1]
            or len(reconstruction) < len(activations)
        ):
            raise ValueError(
                f"reconstruction {name} has shape {tuple(reconstruction.shape)}, "
                f"expected at least {tuple(activations.shape)}"
            )
        reconstruction = reconstruction[: len(activations)]
        rows = [
            tangent_group_statistics(original, predicted, ridge)
            for original, predicted in zip(
                iter_groups(activations, args.group_size),
                iter_groups(reconstruction, args.group_size),
            )
        ]
        models[name] = {
            "groups": rows,
            "summary": _model_summary(
                rows,
                (
                    "exact_numerator",
                    "tangent_numerator",
                    "remainder_ratio",
                    "tangent_exact_cosine",
                    "normalized_gram_error",
                    "commutator",
                ),
            ),
        }
        del reconstruction
    paired = {}
    for index, (baseline, candidate) in enumerate(pairs):
        paired[f"{baseline}:{candidate}"] = tangent_pair_report(
            models[baseline]["groups"],
            models[candidate]["groups"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100 * index,
        )
    result["models"] = models
    result["paired"] = paired
    result["summary"] = {
        "all_pairs_exact_favor_dpsae": all(
            row["diagnostics"]["exact_favors_dpsae"] for row in paired.values()
        ),
        "all_pairs_endpoint_nonlinearity_necessary": all(
            row["diagnostics"]["endpoint_nonlinearity_necessary"]
            for row in paired.values()
        ),
        "all_pairs_source_tangent_sufficient": all(
            row["diagnostics"]["source_tangent_sufficient"] for row in paired.values()
        ),
    }
    result["wall_seconds"] = time.perf_counter() - started
    result["finished_at"] = utc_now()
    result["complete"] = True
    atomic_json(args.output, result)
    return result


def run_nonorth(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ridge, static_provenance = resolve_ridge(args.ridge, args.static_calibration)
    activations, groups = load_natural_activations(
        args.natural_cache,
        exact_tokens=args.exact_tokens,
        group_size=args.group_size,
        max_groups=args.max_groups,
    )
    payloads = torch.load(args.models, map_location="cpu", weights_only=False)
    if not isinstance(payloads, Mapping):
        raise ValueError("models artifact must map model names to payloads")
    pairs = infer_pairs(list(payloads), args.pair)
    selected_names = sorted({name for pair in pairs for name in pair})
    inputs: dict[str, Any] = {
        "natural_cache": file_provenance(args.natural_cache),
        "models": file_provenance(args.models),
    }
    if static_provenance is not None:
        inputs["static_calibration"] = static_provenance
    protocol = {
        "ridge": ridge,
        "ridge_convention": "matrix regularizer = group_size * ridge",
        "group_size": args.group_size,
        "groups": groups,
        "exact_tokens_used": groups * args.group_size,
        "encoding": "frozen checkpoint threshold",
        "counterfactual": "bias-preserving orthogonal residual decoder atoms",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "pairs": pairs,
        "device": str(device),
        "streaming": "one model and one geometry group at a time",
    }
    result = _base_result("nonorth", protocol, inputs)
    models = {}
    input_dim = activations.shape[1]
    for name in selected_names:
        model = load_sae(payloads[name], input_dim=input_dim, device=device)
        decoder = model.decoder_weight.detach()
        bias = model.decoder_bias.detach()
        rows = []
        with torch.inference_mode():
            for group in iter_groups(activations, args.group_size):
                original = group.to(device).float()
                _reconstruction, code = model(original, use_threshold=True)
                rows.append(
                    nonorthogonal_group_statistics(
                        original, code, decoder, bias, ridge
                    )
                )
        models[name] = {
            "groups": rows,
            "summary": _model_summary(
                rows,
                (
                    "exact_numerator",
                    "orthogonal_numerator",
                    "nonorthogonality_benefit",
                    "mean_l0",
                    "code_weighted_coherence",
                    "signed_cross_alignment",
                    "orthogonal_gram_min_eigenvalue",
                ),
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    paired = {}
    for index, (baseline, candidate) in enumerate(pairs):
        paired[f"{baseline}:{candidate}"] = nonorthogonal_pair_report(
            models[baseline]["groups"],
            models[candidate]["groups"],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100 * index,
        )
    result["models"] = models
    result["paired"] = paired
    result["summary"] = {
        "all_pairs_exact_favor_dpsae": all(
            row["diagnostics"]["exact_favors_dpsae"] for row in paired.values()
        ),
        "all_pairs_nonorthogonality_necessary": all(
            row["diagnostics"]["nonorthogonality_necessary"] for row in paired.values()
        ),
        "all_pairs_nonorthogonality_primary_falsified": all(
            row["diagnostics"]["nonorthogonality_primary_falsified"]
            for row in paired.values()
        ),
    }
    result["resources"] = {
        "device": str(device),
        "peak_allocated_gpu_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
    }
    result["wall_seconds"] = time.perf_counter() - started
    result["finished_at"] = utc_now()
    result["complete"] = True
    atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    def common(stage: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(stage)
        child.add_argument("--natural-cache", type=Path, default=DEFAULT_NATURAL_CACHE)
        child.add_argument("--static-calibration", type=Path, default=DEFAULT_STATIC)
        child.add_argument("--ridge", type=float)
        child.add_argument("--group-size", type=int, default=128)
        child.add_argument("--exact-tokens", type=int, default=16_384)
        child.add_argument("--max-groups", type=int, default=128)
        child.add_argument("--bootstrap-samples", type=int, default=10_000)
        child.add_argument("--seed", type=int, default=2027071423)
        child.add_argument(
            "--pair",
            action="append",
            help="explicit paired model names as mse_name:dpsae_name; repeatable",
        )
        return child

    prepare = common("prepare")
    prepare.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--batch-tokens", type=int, default=4_096)
    prepare.add_argument(
        "--reconstruction-dir", type=Path, default=DEFAULT_RECONSTRUCTIONS
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "prepare.json")
    prepare.set_defaults(run=run_prepare)

    tangent = common("tangent")
    tangent.add_argument(
        "--reconstruction-dir", type=Path, default=DEFAULT_RECONSTRUCTIONS
    )
    tangent.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "tangent.json")
    tangent.set_defaults(run=run_tangent)

    nonorth = common("nonorth")
    nonorth.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    nonorth.add_argument("--device", default="cpu")
    nonorth.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "nonorth.json")
    nonorth.set_defaults(run=run_nonorth)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.ridge is not None:
        args.static_calibration = None
    result = args.run(args)
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "complete": result["complete"],
                "output": str(args.output),
                "summary": result["summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
