"""Matched-objective training utilities for language-model SAEs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor

from .decoder_distance import batched_ridge_predict
from .language_sae import (
    BatchTopKSAE,
    jump_relu,
    jump_relu_target_l0_loss,
)


@dataclass(frozen=True)
class SAETrainSpec:
    name: str
    method: str
    seed: int
    k: int
    decoder_weight: float = 0.0
    loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.method not in {"mse", "dpsae", "whitening", "spectral"}:
            raise ValueError(f"unknown training method: {self.method}")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be nonnegative")


def whitening_operator(activations: Tensor, *, floor_fraction: float = 1e-3) -> Tensor:
    """Return a stable inverse covariance square root."""

    covariance = activations.mT @ activations / activations.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    floor = eigenvalues.mean() * floor_fraction
    inverse_sqrt = eigenvalues.clamp_min(floor).rsqrt()
    return (eigenvectors * inverse_sqrt) @ eigenvectors.mT


def spectral_surrogate_operator(activations: Tensor, *, ridge: float) -> Tensor:
    """Return the square root of ``C (C + ridge I)^-2``.

    Applying this operator to a reconstruction residual yields the static
    ridge-saturated spectral loss implied by the isotropic decoder theorem.
    """

    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    covariance = activations.mT @ activations / activations.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    weights = eigenvalues.clamp_min(0).sqrt() / (eigenvalues + ridge)
    return (eigenvectors * weights) @ eigenvectors.mT


def sampled_decoder_loss_from_reference(
    original: Tensor,
    reconstructed: Tensor,
    targets: Tensor,
    *,
    ridge: float,
) -> Tensor:
    """Decoder distortion with the constant reference solve outside autograd."""

    with torch.no_grad():
        reference = batched_ridge_predict(original, targets, ridge)
        denominator = reference.square().sum().clamp_min(1e-12)
    prediction = batched_ridge_predict(reconstructed, targets, ridge)
    return (reference - prediction).square().sum() / denominator


class TrainingFleet:
    """SAEs trained on exactly the same activation batches."""

    def __init__(
        self,
        specs: list[SAETrainSpec],
        *,
        input_dim: int,
        dictionary_size: int,
        learning_rate: float,
        device: torch.device,
        whitening: Tensor | None = None,
        spectral: Tensor | None = None,
        aux_weight: float = 1 / 32,
        dead_after_steps: int = 2_000,
        aux_k: int = 512,
        sparsity_mode: str = "batch_topk",
        jump_relu_init_threshold: float = 0.001,
        jump_relu_init_mode: str = "fixed",
        jump_relu_bandwidth: float = 0.001,
        jump_relu_sparsity_weight: float = 1.0,
        jump_relu_threshold_lr_multiplier: float = 1.0,
        jump_relu_threshold_lr_multipliers_by_method: Mapping[str, float] | None = None,
    ) -> None:
        self.specs = specs
        self.device = device
        self.static_operators = {
            method: operator.to(device)
            for method, operator in {
                "whitening": whitening,
                "spectral": spectral,
            }.items()
            if operator is not None
        }
        self.aux_weight = aux_weight
        self.dead_after_steps = dead_after_steps
        self.aux_k = aux_k
        if sparsity_mode not in {"batch_topk", "token_topk", "jump_relu"}:
            raise ValueError(
                "sparsity_mode must be 'batch_topk', 'token_topk', or 'jump_relu'"
            )
        if jump_relu_init_threshold <= 0 or jump_relu_bandwidth <= 0:
            raise ValueError("JumpReLU threshold and bandwidth must be positive")
        if jump_relu_init_mode not in {"fixed", "topk_quantile"}:
            raise ValueError("JumpReLU init mode must be 'fixed' or 'topk_quantile'")
        if jump_relu_sparsity_weight < 0:
            raise ValueError("JumpReLU sparsity weight must be nonnegative")
        if (
            not math.isfinite(jump_relu_threshold_lr_multiplier)
            or jump_relu_threshold_lr_multiplier <= 0
        ):
            raise ValueError("JumpReLU threshold LR multiplier must be finite and positive")
        method_multipliers = dict(jump_relu_threshold_lr_multipliers_by_method or {})
        unknown_methods = set(method_multipliers) - {spec.method for spec in specs}
        if unknown_methods:
            raise ValueError(
                f"threshold LR multipliers name absent methods: {sorted(unknown_methods)}"
            )
        if any(
            not math.isfinite(multiplier) or multiplier <= 0
            for multiplier in method_multipliers.values()
        ):
            raise ValueError("method-specific threshold LR multipliers must be finite and positive")
        self.sparsity_mode = sparsity_mode
        self.jump_relu_init_threshold = jump_relu_init_threshold
        self.jump_relu_init_mode = jump_relu_init_mode
        self.jump_relu_bandwidth = jump_relu_bandwidth
        self.jump_relu_sparsity_weight = jump_relu_sparsity_weight
        self.jump_relu_threshold_lr_multiplier = jump_relu_threshold_lr_multiplier
        self.jump_relu_threshold_lr_multipliers_by_method = method_multipliers
        self.models: dict[str, BatchTopKSAE] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        for spec in specs:
            model = BatchTopKSAE(
                input_dim,
                dictionary_size,
                spec.k,
                seed=spec.seed,
                sparsity_mode=sparsity_mode,
                jump_relu_init_threshold=jump_relu_init_threshold,
                jump_relu_bandwidth=jump_relu_bandwidth,
            ).to(device)
            self.models[spec.name] = model
            if sparsity_mode == "jump_relu":
                base_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter is not model.log_threshold
                ]
                parameters = [
                    {"params": base_parameters, "lr_multiplier": 1.0},
                    {
                        "params": [model.log_threshold],
                        "lr_multiplier": method_multipliers.get(
                            spec.method, jump_relu_threshold_lr_multiplier
                        ),
                    },
                ]
            else:
                parameters = model.parameters()
            optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, betas=(0.9, 0.999)
            )
            self.optimizers[spec.name] = optimizer
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, learning_rate: float) -> None:
        """Apply one base schedule, scaling only JumpReLU threshold groups."""

        if learning_rate < 0 or not math.isfinite(learning_rate):
            raise ValueError("learning rate must be finite and nonnegative")
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * float(group.get("lr_multiplier", 1.0))

    def train_batch(
        self,
        activations: Tensor,
        *,
        step: int,
        ridge: float,
        group_size: int,
        probes: int,
        probe_seed: int,
    ) -> dict[str, dict[str, float]]:
        if activations.shape[0] % group_size:
            raise ValueError("activation batch must divide evenly into geometry groups")
        if len({spec.k for spec in self.specs}) != 1:
            raise ValueError("a vectorized training fleet must share one k")
        groups = activations.shape[0] // group_size
        dpsae_indices = [
            index for index, spec in enumerate(self.specs) if spec.method == "dpsae"
        ]
        targets = None
        grouped_original = None
        if dpsae_indices:
            generator = torch.Generator(device=self.device).manual_seed(probe_seed)
            targets = torch.randn(
                groups,
                group_size,
                probes,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
            targets.div_(
                targets.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            )
            grouped_original = activations.reshape(groups, group_size, -1)
        models = [self.models[spec.name] for spec in self.specs]
        for model, optimizer in zip(models, self.optimizers.values()):
            model.train()
            optimizer.zero_grad(set_to_none=True)

        encoder_weight = torch.stack([model.encoder_weight for model in models])
        encoder_bias = torch.stack([model.encoder_bias for model in models])
        decoder_weight = torch.stack([model.decoder_weight for model in models])
        decoder_bias = torch.stack([model.decoder_bias for model in models])
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        jump_initialized = [False] * len(models)
        with autocast:
            if self.sparsity_mode == "jump_relu":
                # The 0.001 KDE window is narrower than BF16 spacing at the
                # target-L0 cutoff, so JumpReLU scores must be formed in FP32.
                with torch.autocast(device_type=self.device.type, enabled=False):
                    centered = activations.float().unsqueeze(0) - decoder_bias[:, None, :]
                    scores = torch.relu(
                        torch.matmul(centered, encoder_weight) + encoder_bias[:, None, :]
                    )
            else:
                centered = activations.unsqueeze(0) - decoder_bias[:, None, :]
                scores = torch.relu(
                    torch.matmul(centered, encoder_weight) + encoder_bias[:, None, :]
                )
            jump_sparsity = torch.zeros(len(models), device=self.device)
            if self.sparsity_mode == "batch_topk":
                keep = activations.shape[0] * self.specs[0].k
                values, indices = scores.flatten(1).topk(keep, dim=1, sorted=False)
                flat_code = torch.zeros_like(scores).flatten(1)
                flat_code.scatter_(1, indices, values)
                code = flat_code.reshape_as(scores)
            elif self.sparsity_mode == "token_topk":
                values, indices = scores.topk(self.specs[0].k, dim=2, sorted=False)
                code = torch.zeros_like(scores).scatter_(2, indices, values)
            else:
                for index, model in enumerate(models):
                    if (
                        self.jump_relu_init_mode == "topk_quantile"
                        and model.threshold_updates.item() == 0
                    ):
                        model.initialize_jump_threshold_(scores[index])
                        jump_initialized[index] = True
                threshold = torch.stack([model.jump_threshold for model in models])[
                    :, None, :
                ]
                code = jump_relu(
                    scores,
                    threshold,
                    self.jump_relu_bandwidth,
                )
                jump_sparsity = jump_relu_target_l0_loss(
                    scores,
                    threshold,
                    bandwidth=self.jump_relu_bandwidth,
                    target_l0=self.specs[0].k,
                )
            reconstruction = torch.bmm(code, decoder_weight) + decoder_bias[:, None, :]

        reconstruction = reconstruction.float()
        residual = reconstruction - activations.unsqueeze(0)
        mse = residual.square().sum(dim=(1, 2)) / activations.square().sum().clamp_min(1e-12)
        decoder = torch.zeros(len(models), device=self.device)
        if dpsae_indices:
            assert grouped_original is not None and targets is not None
            with torch.no_grad():
                reference = batched_ridge_predict(grouped_original, targets, ridge)
                denominator = reference.square().sum().clamp_min(1e-12)
            dpsae_reconstruction = reconstruction[dpsae_indices].reshape(
                len(dpsae_indices) * groups, group_size, -1
            )
            expanded_targets = targets.repeat(len(dpsae_indices), 1, 1)
            prediction = batched_ridge_predict(dpsae_reconstruction, expanded_targets, ridge)
            prediction = prediction.reshape(len(dpsae_indices), groups, group_size, probes)
            decoder_values = (prediction - reference.unsqueeze(0)).square().sum(
                dim=(1, 2, 3)
            ) / denominator
            decoder[dpsae_indices] = decoder_values

        static_metric = torch.zeros(len(models), device=self.device)
        for method, operator in self.static_operators.items():
            indices = [
                index for index, spec in enumerate(self.specs) if spec.method == method
            ]
            if not indices:
                continue
            weighted_residual = torch.matmul(residual[indices], operator)
            weighted_original = activations @ operator
            static_metric[indices] = weighted_residual.square().sum(dim=(1, 2)) / (
                weighted_original.square().sum().clamp_min(1e-12)
            )

        primary = []
        auxiliary = []
        for index, (spec, model) in enumerate(zip(self.specs, models)):
            if spec.method == "dpsae":
                primary.append(mse[index] + spec.decoder_weight * decoder[index])
            elif spec.method in {"whitening", "spectral"}:
                if spec.method not in self.static_operators:
                    raise RuntimeError(f"{spec.method} method requires an operator")
                primary.append(mse[index] + spec.loss_weight * static_metric[index])
            else:
                primary.append(mse[index])

            if self.sparsity_mode == "jump_relu":
                primary[-1] = (
                    primary[-1]
                    + self.jump_relu_sparsity_weight * jump_sparsity[index]
                )

            dead = (step - model.last_active_step) >= self.dead_after_steps
            dead_indices = dead.nonzero(as_tuple=False).flatten()
            if dead_indices.numel() == 0 or self.aux_k <= 0:
                auxiliary.append(activations.new_zeros(()))
            else:
                dead_scores = scores[index, :, dead_indices]
                aux_keep = min(self.aux_k, dead_scores.shape[1])
                aux_values, aux_indices = dead_scores.topk(aux_keep, dim=1, sorted=False)
                aux_code = torch.zeros_like(dead_scores).scatter_(1, aux_indices, aux_values)
                aux_reconstruction = aux_code @ model.decoder_weight[dead_indices]
                detached_residual = (-residual[index]).detach()
                auxiliary.append(
                    (detached_residual - aux_reconstruction).square().sum()
                    / detached_residual.square().sum().clamp_min(1e-12)
                )

        primary_tensor = torch.stack(primary)
        auxiliary_tensor = torch.stack(auxiliary)
        losses = primary_tensor + self.aux_weight * auxiliary_tensor
        losses.sum().backward()

        metrics = {}
        for index, (spec, model, optimizer) in enumerate(
            zip(self.specs, models, self.optimizers.values())
        ):
            model.project_decoder_grad_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.normalize_decoder_()
            model.update_activity_(code[index], step)
            if self.sparsity_mode == "batch_topk":
                with torch.no_grad():
                    cutoff = values[index].min().float()
                    if model.threshold_updates.item() == 0:
                        model.activation_threshold.copy_(cutoff)
                    else:
                        model.activation_threshold.lerp_(cutoff, 1 - model.threshold_ema)
                    model.threshold_updates.add_(1)
            model_metrics = {
                "loss": float(losses[index].detach()),
                "nmse": float(mse[index].detach()),
                "decoder": float(decoder[index].detach()),
                "static": float(static_metric[index].detach()),
                "aux": float(auxiliary_tensor[index].detach()),
                "l0": float((code[index] != 0).sum(dim=1).float().mean()),
                "dead": int((step - model.last_active_step >= self.dead_after_steps).sum()),
            }
            if self.sparsity_mode == "jump_relu":
                model_metrics["sparsity"] = float(jump_sparsity[index].detach())
                model_metrics["threshold_min"] = float(
                    model.jump_threshold.detach().min()
                )
                model_metrics["threshold_mean"] = float(
                    model.jump_threshold.detach().mean()
                )
                model_metrics["threshold_max"] = float(
                    model.jump_threshold.detach().max()
                )
                model_metrics["initialization_cutoff"] = float(
                    model.activation_threshold
                    if model.threshold_updates.item()
                    else self.jump_relu_init_threshold
                )
                model_metrics["threshold_initialized"] = jump_initialized[index]
                model_metrics["threshold_learning_rate"] = float(
                    optimizer.param_groups[1]["lr"]
                )
            metrics[spec.name] = model_metrics
        return metrics

    def state_dict(self, *, step: int, tokens_seen: int) -> dict:
        state = {
            "step": step,
            "tokens_seen": tokens_seen,
            "specs": [asdict(spec) for spec in self.specs],
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
            "sparsity_mode": self.sparsity_mode,
        }
        if self.sparsity_mode == "jump_relu":
            state["sparsity_config"] = self.sparsity_config()
        return state

    def load_state_dict(self, state: dict) -> tuple[int, int]:
        if state.get("sparsity_mode", "batch_topk") != self.sparsity_mode:
            raise ValueError("checkpoint sparsity mode does not match the training fleet")
        if state.get("sparsity_config", {}) != self.sparsity_config():
            raise ValueError("checkpoint sparsity config does not match the training fleet")
        for name, model_state in state["models"].items():
            self.models[name].load_state_dict(model_state)
        for name, optimizer_state in state["optimizers"].items():
            self.optimizers[name].load_state_dict(optimizer_state)
        return int(state["step"]), int(state["tokens_seen"])

    def export_models(self) -> dict[str, dict]:
        result = {}
        for spec in self.specs:
            payload = {
                "spec": asdict(spec),
                "sparsity_mode": self.sparsity_mode,
                "state_dict": self.models[spec.name].state_dict(),
            }
            if self.sparsity_mode == "jump_relu":
                payload["sparsity_config"] = self.sparsity_config()
            result[spec.name] = payload
        return result

    def sparsity_config(self) -> dict[str, float | str | dict[str, float]]:
        if self.sparsity_mode != "jump_relu":
            return {}
        result: dict[str, float | str | dict[str, float]] = {
            "init_threshold": self.jump_relu_init_threshold,
            "initialization": self.jump_relu_init_mode,
            "bandwidth": self.jump_relu_bandwidth,
            "target_l0_loss_weight": self.jump_relu_sparsity_weight,
            "threshold_lr_multiplier": self.jump_relu_threshold_lr_multiplier,
        }
        if self.jump_relu_threshold_lr_multipliers_by_method:
            result["threshold_lr_multipliers_by_method"] = dict(
                sorted(self.jump_relu_threshold_lr_multipliers_by_method.items())
            )
        return result

    def jump_threshold_summary(self) -> dict[str, dict[str, float | int]]:
        if self.sparsity_mode != "jump_relu":
            return {}
        return {
            name: {
                "minimum": float(model.jump_threshold.detach().min()),
                "mean": float(model.jump_threshold.detach().mean()),
                "maximum": float(model.jump_threshold.detach().max()),
                "initialization_cutoff": float(
                    model.activation_threshold
                    if model.threshold_updates.item()
                    else self.jump_relu_init_threshold
                ),
                "initialization_updates": int(model.threshold_updates.item()),
            }
            for name, model in self.models.items()
        }
