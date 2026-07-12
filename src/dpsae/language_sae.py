"""Nonnegative BatchTopK sparse autoencoder for language-model activations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BatchTopKSAE(nn.Module):
    """Untied BatchTopK SAE with a calibrated inference threshold."""

    def __init__(
        self,
        input_dim: int,
        dictionary_size: int,
        k: int,
        *,
        seed: int = 0,
        threshold_ema: float = 0.99,
    ) -> None:
        super().__init__()
        if not 0 < k <= dictionary_size:
            raise ValueError("k must lie in [1, dictionary_size]")
        generator = torch.Generator().manual_seed(seed)
        decoder = torch.randn(dictionary_size, input_dim, generator=generator)
        decoder = torch.nn.functional.normalize(decoder, dim=1)
        self.decoder_weight = nn.Parameter(decoder)
        self.encoder_weight = nn.Parameter(decoder.mT.clone())
        self.encoder_bias = nn.Parameter(torch.zeros(dictionary_size))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.k = k
        self.threshold_ema = threshold_ema
        self.register_buffer("activation_threshold", torch.tensor(0.0))
        self.register_buffer("threshold_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "last_active_step", torch.zeros(dictionary_size, dtype=torch.long)
        )

    def preactivations(self, x: Tensor) -> Tensor:
        return torch.relu((x - self.decoder_bias) @ self.encoder_weight + self.encoder_bias)

    def encode(self, x: Tensor, *, use_threshold: bool | None = None) -> Tensor:
        scores = self.preactivations(x)
        if use_threshold is None:
            use_threshold = not self.training and self.threshold_updates.item() > 0
        if use_threshold:
            return torch.where(scores >= self.activation_threshold, scores, 0)

        keep = min(x.shape[0] * self.k, scores.numel())
        values, indices = scores.flatten().topk(keep, sorted=False)
        code = torch.zeros_like(scores).flatten()
        code.scatter_(0, indices, values)
        code = code.reshape_as(scores)
        if self.training and values.numel():
            with torch.no_grad():
                cutoff = values.min().float()
                if self.threshold_updates.item() == 0:
                    self.activation_threshold.copy_(cutoff)
                else:
                    self.activation_threshold.lerp_(cutoff, 1 - self.threshold_ema)
                self.threshold_updates.add_(1)
        return code

    def decode(self, code: Tensor) -> Tensor:
        return code @ self.decoder_weight + self.decoder_bias

    def forward(
        self, x: Tensor, *, use_threshold: bool | None = None
    ) -> tuple[Tensor, Tensor]:
        code = self.encode(x, use_threshold=use_threshold)
        return self.decode(code), code

    @torch.no_grad()
    def update_activity_(self, code: Tensor, step: int) -> None:
        active = (code != 0).any(dim=0)
        self.last_active_step[active] = step

    def auxiliary_loss(
        self,
        x: Tensor,
        reconstruction: Tensor,
        *,
        step: int,
        dead_after_steps: int,
        aux_k: int,
    ) -> Tensor:
        """Use dead latents to reconstruct the current residual (AuxK)."""

        dead = (step - self.last_active_step) >= dead_after_steps
        dead_indices = dead.nonzero(as_tuple=False).flatten()
        if dead_indices.numel() == 0 or aux_k <= 0:
            return x.new_zeros(())
        residual = (x - reconstruction).detach()
        scores = torch.relu(
            (x - self.decoder_bias) @ self.encoder_weight[:, dead_indices]
            + self.encoder_bias[dead_indices]
        )
        keep = min(aux_k, scores.shape[1])
        values, local_indices = scores.topk(keep, dim=1, sorted=False)
        aux_code = torch.zeros_like(scores).scatter_(1, local_indices, values)
        aux_reconstruction = aux_code @ self.decoder_weight[dead_indices]
        return (residual - aux_reconstruction).square().sum() / residual.square().sum().clamp_min(
            1e-12
        )

    @torch.no_grad()
    def project_decoder_grad_(self) -> None:
        if self.decoder_weight.grad is None:
            return
        parallel = (self.decoder_weight.grad * self.decoder_weight).sum(
            dim=1, keepdim=True
        )
        self.decoder_weight.grad.sub_(parallel * self.decoder_weight)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        self.decoder_weight.copy_(
            torch.nn.functional.normalize(self.decoder_weight, dim=1)
        )

    @torch.no_grad()
    def calibrate_threshold_(self, x: Tensor) -> float:
        scores = self.preactivations(x)
        keep = min(x.shape[0] * self.k, scores.numel())
        threshold = scores.flatten().topk(keep).values.min().float()
        self.activation_threshold.copy_(threshold)
        self.threshold_updates.clamp_min_(1)
        return float(threshold)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.decoder_bias.numel()}, "
            f"dictionary_size={self.encoder_bias.numel()}, k={self.k}"
        )
