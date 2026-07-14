"""Exact task spectra and finite-probe gradients for ridge decoder distortion."""

from __future__ import annotations

import torch
from torch import Tensor

from .decoder_distance import batched_ridge_predict


def ridge_hat_matrices(groups: Tensor, ridge: float) -> Tensor:
    """Return one ridge hat matrix per ``[samples, features]`` group."""

    if groups.ndim != 3:
        raise ValueError("groups must have shape [groups, samples, features]")
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    count, samples, _ = groups.shape
    identity = torch.eye(samples, dtype=groups.dtype, device=groups.device)
    return batched_ridge_predict(
        groups,
        identity.expand(count, samples, samples),
        ridge,
    )


def advantage_operators(
    original: Tensor,
    baseline: Tensor,
    candidate: Tensor,
    *,
    ridge: float,
) -> dict[str, Tensor]:
    """Return exact ridge errors, task-advantage operators, and spectra.

    Positive eigenvalues of ``advantage`` are directions on which the candidate
    has lower absolute ridge-prediction error than the baseline.
    """

    if original.shape != baseline.shape or original.shape != candidate.shape:
        raise ValueError("original, baseline, and candidate groups must match")
    source_hat = ridge_hat_matrices(original, ridge)
    baseline_error = source_hat - ridge_hat_matrices(baseline, ridge)
    candidate_error = source_hat - ridge_hat_matrices(candidate, ridge)
    advantage = baseline_error.mT @ baseline_error - candidate_error.mT @ candidate_error
    eigenvalues = torch.linalg.eigvalsh(advantage)
    return {
        "source_hat": source_hat,
        "baseline_error": baseline_error,
        "candidate_error": candidate_error,
        "advantage": advantage,
        "eigenvalues": eigenvalues,
        "source_energy": source_hat.square().sum(dim=(1, 2)),
        "baseline_numerator": baseline_error.square().sum(dim=(1, 2)),
        "candidate_numerator": candidate_error.square().sum(dim=(1, 2)),
        "trace": advantage.diagonal(dim1=-2, dim2=-1).sum(-1),
    }


def fixed_radius_targets(
    banks: int,
    groups: int,
    samples: int,
    probes: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    clamp_min: float = 1e-6,
) -> tuple[Tensor, int]:
    """Draw the column-normalized Gaussian targets used during SAE training."""

    if min(banks, groups, samples, probes) < 1:
        raise ValueError("all target dimensions must be positive")
    targets = torch.randn(
        banks,
        groups,
        samples,
        probes,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    rms = targets.square().mean(dim=2, keepdim=True).sqrt()
    clamp_hits = int((rms < clamp_min).sum().item())
    targets.div_(rms.clamp_min(clamp_min))
    return targets, clamp_hits


def ridge_gradient_factors(
    original: Tensor,
    reconstructed: Tensor,
    *,
    ridge: float,
) -> dict[str, Tensor]:
    """Precompute factors for decoder-distortion gradients.

    The returned derivatives treat the source representation as fixed and
    differentiate with respect to the reconstructed row Gram matrix or the
    reconstructed activations.
    """

    if original.shape != reconstructed.shape or original.ndim != 3:
        raise ValueError("original and reconstructed groups must match and be rank 3")
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    groups, samples, _ = original.shape
    regularizer = samples * ridge
    identity = torch.eye(samples, dtype=original.dtype, device=original.device)
    identity = identity.expand(groups, samples, samples)
    source_hat = ridge_hat_matrices(original, ridge)
    gram_system = reconstructed @ reconstructed.mT + regularizer * identity
    chol = torch.linalg.cholesky(gram_system)
    inverse = torch.cholesky_solve(identity, chol)
    reconstructed_hat = identity - regularizer * inverse
    error = source_hat - reconstructed_hat
    inverse_reconstruction = inverse @ reconstructed
    return {
        "regularizer": original.new_tensor(regularizer),
        "source_hat": source_hat,
        "error": error,
        "inverse": inverse,
        "inverse_reconstruction": inverse_reconstruction,
        "source_energy": source_hat.square().sum(),
    }


def exact_relative_gradients(
    factors: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Return row-Gram and reconstruction gradients for identity targets."""

    regularizer = factors["regularizer"]
    inverse = factors["inverse"]
    error = factors["error"]
    denominator = factors["source_energy"].clamp_min(1e-12)
    gram_gradient = -2 * regularizer * (inverse @ error @ inverse) / denominator
    reconstruction_gradient = (
        -4
        * regularizer
        * (inverse @ error @ factors["inverse_reconstruction"])
        / denominator
    )
    return gram_gradient, reconstruction_gradient


def sampled_relative_gradients(
    factors: dict[str, Tensor],
    target_covariance: Tensor,
    *,
    probes: int,
    fixed_denominator: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return finite-probe row-Gram and reconstruction gradients.

    ``target_covariance`` has shape ``[banks, groups, samples, samples]`` and is
    the sum of target outer products for ``probes`` columns. With
    ``fixed_denominator=True``, the random denominator is replaced by its exact
    expectation, yielding an unbiased control for the identity-target gradient.
    """

    if target_covariance.ndim != 4:
        raise ValueError("target covariance must have shape [banks, groups, n, n]")
    if probes < 1:
        raise ValueError("probes must be positive")
    regularizer = factors["regularizer"]
    inverse = factors["inverse"]
    error = factors["error"]
    source_hat = factors["source_hat"]
    if target_covariance.shape[1:] != error.shape:
        raise ValueError("target covariance and ridge groups do not match")

    covariance = target_covariance
    middle = error.unsqueeze(0) @ covariance + covariance @ error.unsqueeze(0)
    left = inverse.unsqueeze(0) @ middle
    gram_numerator_gradient = -regularizer * (left @ inverse.unsqueeze(0))
    if fixed_denominator:
        denominator = (probes * factors["source_energy"]).expand(len(covariance))
    else:
        source_squared = source_hat @ source_hat
        denominator = torch.einsum(
            "gij,bgji->b", source_squared, covariance
        ).clamp_min(1e-12)
    gram_gradient = gram_numerator_gradient / denominator[:, None, None, None]
    reconstruction_gradient = (
        -2
        * regularizer
        * (left @ factors["inverse_reconstruction"].unsqueeze(0))
        / denominator[:, None, None, None]
    )
    return gram_gradient, reconstruction_gradient, denominator
