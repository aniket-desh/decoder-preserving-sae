"""Regularized prediction operators and decoder-preservation losses.

Rows are observations and columns are representation coordinates. Ridge is
parameterized from the average regression objective

    ||X w - y||^2 / n + ridge * ||w||^2,

so the matrix regularizer is ``n * ridge``. Representations are not centered
inside these functions by default: experiments should apply fixed statistics
estimated from the training set instead of changing the geometry per minibatch.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_representation(x: Tensor, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [n_samples, n_features]")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def _prepare(x: Tensor, center: bool) -> Tensor:
    _validate_representation(x, "x")
    return x - x.mean(dim=0, keepdim=True) if center else x


def _cholesky_solve(rhs: Tensor, chol: Tensor) -> Tensor:
    """Solve from a Cholesky factor, including on Apple Metal.

    PyTorch 2.8 does not implement ``torch.cholesky_solve`` on MPS, while its
    two constituent triangular solves are native Metal kernels. Keep the faster
    fused helper on other devices and use the equivalent formulation on MPS.
    """

    if chol.device.type == "mps":
        intermediate = torch.linalg.solve_triangular(chol, rhs, upper=False)
        return torch.linalg.solve_triangular(chol.mT, intermediate, upper=True)
    return torch.cholesky_solve(rhs, chol)


def ridge_predict(
    x: Tensor,
    targets: Tensor,
    ridge: float,
    *,
    center: bool = False,
) -> Tensor:
    """Apply the optimal ridge prediction operator without forming it.

    The smaller of the primal and dual positive-definite systems is solved with
    Cholesky. This function is differentiable with respect to ``x``.
    """

    x = _prepare(x, center)
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2 or targets.shape[0] != x.shape[0]:
        raise ValueError("targets must have shape [n_samples, n_targets]")

    n, d = x.shape
    regularizer = n * ridge
    if d <= n:
        system = x.mT @ x + regularizer * torch.eye(d, device=x.device, dtype=x.dtype)
        chol = torch.linalg.cholesky(system)
        weights = _cholesky_solve(x.mT @ targets, chol)
        return x @ weights

    gram = x @ x.mT
    system = gram + regularizer * torch.eye(n, device=x.device, dtype=x.dtype)
    chol = torch.linalg.cholesky(system)
    alpha = _cholesky_solve(targets, chol)
    return targets - regularizer * alpha


def batched_ridge_predict(x: Tensor, targets: Tensor, ridge: float) -> Tensor:
    """Batched version of :func:`ridge_predict` for geometry groups.

    ``x`` has shape ``[groups, n_samples, n_features]`` and ``targets`` has
    shape ``[groups, n_samples, n_targets]``. All Cholesky systems are factored
    in one batched kernel, avoiding a Python loop over geometry groups.
    """

    if x.ndim != 3 or targets.ndim != 3:
        raise ValueError("x and targets must both be rank-3 batched tensors")
    if x.shape[:2] != targets.shape[:2]:
        raise ValueError("x and targets must agree on groups and samples")
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    groups, n, d = x.shape
    regularizer = n * ridge
    if d <= n:
        system = x.mT @ x + regularizer * torch.eye(
            d, device=x.device, dtype=x.dtype
        ).expand(groups, d, d)
        chol = torch.linalg.cholesky(system)
        weights = _cholesky_solve(x.mT @ targets, chol)
        return x @ weights

    gram = x @ x.mT
    system = gram + regularizer * torch.eye(
        n, device=x.device, dtype=x.dtype
    ).expand(groups, n, n)
    chol = torch.linalg.cholesky(system)
    alpha = _cholesky_solve(targets, chol)
    return targets - regularizer * alpha


def batched_sampled_decoder_loss(
    original: Tensor,
    reconstructed: Tensor,
    targets: Tensor,
    *,
    ridge: float,
    relative: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Average sampled decoder disagreement over geometry groups.

    All inputs are batched over groups. ``targets`` may contain isotropic
    probes, structured task targets, or a concatenation of both; column scales
    therefore define the task-prior second moment without materializing an
    ``n_samples x n_samples`` covariance matrix.
    """

    numerator_by_target, denominator_by_target = batched_sampled_decoder_statistics(
        original,
        reconstructed,
        targets,
        ridge=ridge,
    )
    numerator = numerator_by_target.sum(dim=1)
    if not relative:
        return numerator.mean()
    denominator = denominator_by_target.sum(dim=1).clamp_min(eps)
    return (numerator / denominator).mean()


def batched_sampled_decoder_statistics(
    original: Tensor,
    reconstructed: Tensor,
    targets: Tensor,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    """Return per-group, per-target decoder error and reference energy.

    Each representation is factored in one batched kernel over all geometry
    groups, and every target is solved as a matrix right-hand side. The returned
    tensors have shape ``[groups, targets]``. Their prefix sums evaluate
    multiple probe counts without repeating either factorization or ridge solve.
    """

    if original.ndim != 3 or reconstructed.ndim != 3 or targets.ndim != 3:
        raise ValueError("original, reconstructed, and targets must be rank-3")
    if (
        original.shape[:2] != reconstructed.shape[:2]
        or original.shape[:2] != targets.shape[:2]
    ):
        raise ValueError("all inputs must agree on groups and samples")

    pred_original = batched_ridge_predict(original, targets, ridge)
    pred_reconstructed = batched_ridge_predict(reconstructed, targets, ridge)
    numerator = (pred_original - pred_reconstructed).square().sum(dim=1)
    denominator = pred_original.square().sum(dim=1)
    return numerator, denominator


def ridge_hat_matrix(x: Tensor, ridge: float = 1.0, *, center: bool = False) -> Tensor:
    """Return ``X (X.T X + n * ridge I)^-1 X.T``."""

    x = _prepare(x, center)
    identity = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    return ridge_predict(x, identity, ridge, center=False)


def effective_degrees_of_freedom(
    x: Tensor, ridge: float, *, center: bool = False
) -> Tensor:
    """Return ``trace(K_ridge(X))`` from the singular values of ``X``."""

    x = _prepare(x, center)
    singular_sq = torch.linalg.svdvals(x).square()
    return (singular_sq / (singular_sq + x.shape[0] * ridge)).sum()


def calibrate_ridge(
    x: Tensor,
    target_fraction: float,
    *,
    center: bool = False,
    iterations: int = 80,
) -> float:
    """Choose ridge so ``trace(K) / n`` matches ``target_fraction``.

    The requested fraction must be below the maximum attainable rank fraction.
    Bisection is performed in log space and returned as a Python float.
    """

    x = _prepare(x, center)
    n = x.shape[0]
    rank_fraction = torch.linalg.matrix_rank(x).item() / n
    if not 0 < target_fraction < rank_fraction:
        raise ValueError(
            f"target_fraction must lie in (0, rank(X)/n) = (0, {rank_fraction:.4g})"
        )

    singular_sq = torch.linalg.svdvals(x).square().detach()
    scale = max((singular_sq.max() / n).item(), torch.finfo(x.dtype).tiny)
    low, high = scale * 1e-12, scale * 1e12
    target = target_fraction * n
    for _ in range(iterations):
        mid = (low * high) ** 0.5
        dof = (singular_sq / (singular_sq + n * mid)).sum().item()
        if dof > target:
            low = mid
        else:
            high = mid
    return (low * high) ** 0.5


def sampled_decoder_loss(
    original: Tensor,
    reconstructed: Tensor,
    targets: Tensor,
    *,
    ridge: float,
    center: bool = False,
    relative: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Decoder disagreement estimated from explicit random task targets."""

    _validate_representation(original, "original")
    _validate_representation(reconstructed, "reconstructed")
    if original.shape[0] != reconstructed.shape[0]:
        raise ValueError("representations must contain the same number of samples")
    pred_original = ridge_predict(original, targets, ridge, center=center)
    pred_reconstructed = ridge_predict(reconstructed, targets, ridge, center=center)
    numerator = (pred_original - pred_reconstructed).square().sum()
    if not relative:
        return numerator
    return numerator / (pred_original.square().sum() + eps)


def decoder_distance(
    original: Tensor,
    reconstructed: Tensor,
    *,
    ridge: float = 1.0,
    task_covariance: Tensor | None = None,
    center: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """Compute exact task-weighted disagreement between ridge operators."""

    _validate_representation(original, "original")
    _validate_representation(reconstructed, "reconstructed")
    if original.shape[0] != reconstructed.shape[0]:
        raise ValueError("representations must contain the same number of samples")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    delta = ridge_hat_matrix(original, ridge, center=center) - ridge_hat_matrix(
        reconstructed, ridge, center=center
    )
    if task_covariance is None:
        value = delta.square().sum()
    else:
        n = original.shape[0]
        if task_covariance.shape != (n, n):
            raise ValueError(f"task_covariance must have shape [{n}, {n}]")
        task_covariance = task_covariance.to(device=delta.device, dtype=delta.dtype)
        value = torch.einsum("ij,jk,ik->", delta, task_covariance, delta)
    return value / delta.numel() if reduction == "mean" else value
