"""Small sparse autoencoders used by controlled experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TiedSignedTopKSAE(nn.Module):
    """A tied linear dictionary with an exact signed TopK bottleneck.

    The signed bottleneck matches the zero-mean Gaussian coefficients in the
    synthetic generator. Language-model experiments should use the standard
    nonnegative TopK or BatchTopK architecture instead.
    """

    def __init__(self, input_dim: int, dictionary_size: int, k: int, *, seed: int = 0):
        super().__init__()
        if not 0 < k <= dictionary_size:
            raise ValueError("k must lie between 1 and dictionary_size")
        generator = torch.Generator().manual_seed(seed)
        dictionary = torch.randn(dictionary_size, input_dim, generator=generator)
        dictionary = torch.nn.functional.normalize(dictionary, dim=1)
        self.dictionary = nn.Parameter(dictionary)
        self.k = k

    def encode(self, x: Tensor) -> Tensor:
        scores = x @ self.dictionary.mT
        indices = scores.abs().topk(self.k, dim=1).indices
        code = torch.zeros_like(scores)
        return code.scatter(1, indices, scores.gather(1, indices))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        code = self.encode(x)
        return code @ self.dictionary, code

    @torch.no_grad()
    def normalize_dictionary_(self) -> None:
        self.dictionary.copy_(torch.nn.functional.normalize(self.dictionary, dim=1))
