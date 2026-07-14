"""Nonnegative sparse autoencoders for language-model activations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _rectangle(value: Tensor) -> Tensor:
    return ((value > -0.5) & (value < 0.5)).to(value.dtype)


class _JumpReLU(torch.autograd.Function):
    """Hard JumpReLU with the rectangular-KDE threshold pseudo-gradient.

    The input gradient is the ordinary almost-everywhere derivative. The
    threshold pseudo-gradient is Eq. 11 of Rajamanoharan et al. (2024):
    ``-theta / bandwidth`` inside a width-``bandwidth`` rectangle around the
    jump. Thresholds are exponentiated parameters, so they remain positive.
    """

    @staticmethod
    def forward(scores: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
        if bandwidth <= 0:
            raise ValueError("JumpReLU bandwidth must be strictly positive")
        return scores * (scores > threshold)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        scores, threshold, bandwidth = inputs
        ctx.save_for_backward(scores, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx, output_gradient: Tensor) -> tuple[Tensor, Tensor, None]:
        scores, threshold = ctx.saved_tensors
        active = scores > threshold
        kernel = _rectangle((scores - threshold) / ctx.bandwidth)
        score_gradient = active * output_gradient
        threshold_gradient = (
            -(threshold / ctx.bandwidth) * kernel * output_gradient
        ).sum_to_size(threshold.shape)
        return score_gradient, threshold_gradient, None


class _JumpStep(torch.autograd.Function):
    """Hard support indicator with the matching L0 threshold estimator."""

    @staticmethod
    def forward(scores: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
        if bandwidth <= 0:
            raise ValueError("JumpReLU bandwidth must be strictly positive")
        return (scores > threshold).to(scores.dtype)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        scores, threshold, bandwidth = inputs
        ctx.save_for_backward(scores, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx, output_gradient: Tensor) -> tuple[Tensor, Tensor, None]:
        scores, threshold = ctx.saved_tensors
        kernel = _rectangle((scores - threshold) / ctx.bandwidth)
        score_gradient = torch.zeros_like(scores)
        threshold_gradient = (
            -(kernel / ctx.bandwidth) * output_gradient
        ).sum_to_size(threshold.shape)
        return score_gradient, threshold_gradient, None


def jump_relu(scores: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
    return _JumpReLU.apply(scores, threshold, bandwidth)


def jump_relu_support(scores: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
    return _JumpStep.apply(scores, threshold, bandwidth)


def jump_relu_target_l0_loss(
    scores: Tensor,
    threshold: Tensor,
    *,
    bandwidth: float,
    target_l0: int,
) -> Tensor:
    """Per-model target-L0 controller from JumpReLU Eq. 40.

    ``scores`` may be ``[tokens, features]`` or
    ``[models, tokens, features]``. The result contains one scalar per leading
    model dimension, with the token dimension averaged.
    """

    if target_l0 <= 0:
        raise ValueError("target_l0 must be positive")
    l0 = jump_relu_support(scores, threshold, bandwidth).sum(dim=-1)
    return (l0 / target_l0 - 1).square().mean(dim=-1)


class BatchTopKSAE(nn.Module):
    """Untied SAE supporting BatchTopK, token TopK, and JumpReLU."""

    def __init__(
        self,
        input_dim: int,
        dictionary_size: int,
        k: int,
        *,
        seed: int = 0,
        threshold_ema: float = 0.99,
        sparsity_mode: str = "batch_topk",
        jump_relu_init_threshold: float = 0.001,
        jump_relu_bandwidth: float = 0.001,
    ) -> None:
        super().__init__()
        if not 0 < k <= dictionary_size:
            raise ValueError("k must lie in [1, dictionary_size]")
        if sparsity_mode not in {"batch_topk", "token_topk", "jump_relu"}:
            raise ValueError(
                "sparsity_mode must be 'batch_topk', 'token_topk', or 'jump_relu'"
            )
        if jump_relu_init_threshold <= 0 or jump_relu_bandwidth <= 0:
            raise ValueError("JumpReLU threshold and bandwidth must be positive")
        generator = torch.Generator().manual_seed(seed)
        decoder = torch.randn(dictionary_size, input_dim, generator=generator)
        decoder = torch.nn.functional.normalize(decoder, dim=1)
        self.decoder_weight = nn.Parameter(decoder)
        self.encoder_weight = nn.Parameter(decoder.mT.clone())
        self.encoder_bias = nn.Parameter(torch.zeros(dictionary_size))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.k = k
        self.sparsity_mode = sparsity_mode
        self.threshold_ema = threshold_ema
        self.jump_relu_bandwidth = jump_relu_bandwidth
        if sparsity_mode == "jump_relu":
            self.log_threshold = nn.Parameter(
                torch.full((dictionary_size,), math.log(jump_relu_init_threshold))
            )
        self.register_buffer("activation_threshold", torch.tensor(0.0))
        self.register_buffer("threshold_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "last_active_step", torch.zeros(dictionary_size, dtype=torch.long)
        )

    def preactivations(self, x: Tensor) -> Tensor:
        return torch.relu((x - self.decoder_bias) @ self.encoder_weight + self.encoder_bias)

    def encode(self, x: Tensor, *, use_threshold: bool | None = None) -> Tensor:
        scores = self.preactivations(x)
        if self.sparsity_mode == "jump_relu":
            return jump_relu(scores, self.jump_threshold, self.jump_relu_bandwidth)
        if self.sparsity_mode == "token_topk":
            values, indices = scores.topk(self.k, dim=-1, sorted=False)
            code = torch.zeros_like(scores).scatter_(-1, indices, values)
            return code
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

    @property
    def jump_threshold(self) -> Tensor:
        if self.sparsity_mode != "jump_relu":
            raise AttributeError("only JumpReLU models have learned thresholds")
        return self.log_threshold.exp()

    def jump_sparsity_loss(self, scores: Tensor) -> Tensor:
        if self.sparsity_mode != "jump_relu":
            return scores.new_zeros(())
        return jump_relu_target_l0_loss(
            scores,
            self.jump_threshold,
            bandwidth=self.jump_relu_bandwidth,
            target_l0=self.k,
        )

    @torch.no_grad()
    def initialize_jump_threshold_(self, scores: Tensor) -> float:
        """Set all JumpReLU thresholds from one training-batch global cutoff.

        This is a one-time initialization, not a per-batch support rule. It
        starts the learned hard thresholds at the target average L0, after
        which Eq. 40 and the KDE pseudo-gradient control sparsity.
        """

        if self.sparsity_mode != "jump_relu":
            raise RuntimeError("only JumpReLU models have learned thresholds")
        if scores.ndim != 2 or scores.shape[1] != self.encoder_bias.numel():
            raise ValueError("scores must have shape [tokens, dictionary_size]")
        keep = min(scores.shape[0] * self.k, scores.numel())
        cutoff = scores.float().flatten().topk(keep, sorted=False).values.min()
        if not torch.isfinite(cutoff) or cutoff <= 0:
            raise RuntimeError("cannot initialize a positive JumpReLU threshold")
        lower = torch.nextafter(cutoff, torch.full_like(cutoff, -torch.inf))
        self.log_threshold.fill_(lower.log())
        self.activation_threshold.copy_(lower)
        self.threshold_updates.add_(1)
        return float(lower)

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
        result = (
            f"input_dim={self.decoder_bias.numel()}, "
            f"dictionary_size={self.encoder_bias.numel()}, k={self.k}, "
            f"sparsity_mode={self.sparsity_mode!r}"
        )
        if self.sparsity_mode == "jump_relu":
            result += f", jump_relu_bandwidth={self.jump_relu_bandwidth:g}"
        return result
