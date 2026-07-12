"""Reference implementation of the decoder-preservation objective.

Rows are observations or token positions and columns are representation
coordinates. The implementation intentionally favors clarity over scale; later
experiments should benchmark sampled-task and covariance-based estimators before
using full batch-by-batch kernels for large batches.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_representation(x: Tensor, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [n_samples, n_features]")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def ridge_hat_matrix(x: Tensor, ridge: float = 1.0, *, center: bool = True) -> Tensor:
    """Return the ridge-regression map from targets to fitted predictions.

    This computes ``X (X.T X + ridge I)^-1 X.T`` using a linear solve. Centering
    is enabled by default because decoding targets and representation-similarity
    measures commonly remove the sample mean; the experimental config must state
    this convention explicitly.
    """

    _validate_representation(x, "x")
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")

    if center:
        x = x - x.mean(dim=0, keepdim=True)
    gram = x.mT @ x
    regularized = gram + ridge * torch.eye(gram.shape[0], device=x.device, dtype=x.dtype)
    return x @ torch.linalg.solve(regularized, x.mT)


def decoder_distance(
    original: Tensor,
    reconstructed: Tensor,
    *,
    ridge: float = 1.0,
    task_covariance: Tensor | None = None,
    center: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """Measure task-weighted disagreement between optimal ridge predictions.

    With isotropic tasks, the unreduced value is ``||K_X - K_Xhat||_F^2``.
    For task covariance ``Sigma``, it is ``tr(Delta K Sigma Delta K.T)``.
    ``reduction='mean'`` divides by the number of matrix entries so loss scale is
    less sensitive to batch size; the paper must report the chosen normalization.
    """

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
        if task_covariance.device != delta.device or task_covariance.dtype != delta.dtype:
            task_covariance = task_covariance.to(device=delta.device, dtype=delta.dtype)
        value = torch.einsum("ij,jk,ik->", delta, task_covariance, delta)

    return value / delta.numel() if reduction == "mean" else value

