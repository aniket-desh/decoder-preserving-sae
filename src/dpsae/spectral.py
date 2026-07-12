"""Closed-form helpers for the isotropic spectral theorem."""

from __future__ import annotations

import torch
from torch import Tensor


def decoder_gains(singular_values: Tensor, tau: float) -> Tensor:
    if tau <= 0:
        raise ValueError("tau must be strictly positive")
    return singular_values.square() / (singular_values.square() + tau)


def optimal_decoder_tail(singular_values: Tensor, tau: float) -> Tensor:
    """Return optimal squared distortion for retained ranks 0 through s."""

    costs = decoder_gains(singular_values, tau).square()
    tails = torch.flip(torch.cumsum(torch.flip(costs, dims=(0,)), dim=0), dims=(0,))
    return torch.cat((tails, costs.new_zeros(1)))


def truncated_svd(x: Tensor, rank: int) -> Tensor:
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    if not 0 <= rank <= s.numel():
        raise ValueError("rank is outside the compact SVD range")
    return (u[:, :rank] * s[:rank]) @ vh[:rank]
