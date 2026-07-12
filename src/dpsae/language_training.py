"""Matched-objective training utilities for language-model SAEs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .decoder_distance import batched_ridge_predict
from .language_sae import BatchTopKSAE


@dataclass(frozen=True)
class SAETrainSpec:
    name: str
    method: str
    seed: int
    k: int
    decoder_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in {"mse", "dpsae", "whitening"}:
            raise ValueError(f"unknown training method: {self.method}")


def whitening_operator(activations: Tensor, *, floor_fraction: float = 1e-3) -> Tensor:
    """Return a stable inverse covariance square root."""

    covariance = activations.mT @ activations / activations.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    floor = eigenvalues.mean() * floor_fraction
    inverse_sqrt = eigenvalues.clamp_min(floor).rsqrt()
    return (eigenvectors * inverse_sqrt) @ eigenvectors.mT


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
        aux_weight: float = 1 / 32,
        dead_after_steps: int = 2_000,
        aux_k: int = 512,
    ) -> None:
        self.specs = specs
        self.device = device
        self.whitening = None if whitening is None else whitening.to(device)
        self.aux_weight = aux_weight
        self.dead_after_steps = dead_after_steps
        self.aux_k = aux_k
        self.models: dict[str, BatchTopKSAE] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        for spec in specs:
            model = BatchTopKSAE(
                input_dim,
                dictionary_size,
                spec.k,
                seed=spec.seed,
            ).to(device)
            self.models[spec.name] = model
            self.optimizers[spec.name] = torch.optim.Adam(
                model.parameters(), lr=learning_rate, betas=(0.9, 0.999)
            )

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
        generator = torch.Generator(device=self.device).manual_seed(probe_seed)
        targets = torch.randn(
            groups,
            group_size,
            probes,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        targets.div_(targets.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6))
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
        with autocast:
            centered = activations.unsqueeze(0) - decoder_bias[:, None, :]
            scores = torch.relu(torch.matmul(centered, encoder_weight) + encoder_bias[:, None, :])
            keep = activations.shape[0] * self.specs[0].k
            values, indices = scores.flatten(1).topk(keep, dim=1, sorted=False)
            flat_code = torch.zeros_like(scores).flatten(1)
            flat_code.scatter_(1, indices, values)
            code = flat_code.reshape_as(scores)
            reconstruction = torch.bmm(code, decoder_weight) + decoder_bias[:, None, :]

        reconstruction = reconstruction.float()
        residual = reconstruction - activations.unsqueeze(0)
        mse = residual.square().sum(dim=(1, 2)) / activations.square().sum().clamp_min(1e-12)
        decoder = torch.zeros(len(models), device=self.device)
        dpsae_indices = [index for index, spec in enumerate(self.specs) if spec.method == "dpsae"]
        if dpsae_indices:
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

        primary = []
        auxiliary = []
        for index, (spec, model) in enumerate(zip(self.specs, models)):
            if spec.method == "dpsae":
                primary.append(mse[index] + spec.decoder_weight * decoder[index])
            elif spec.method == "whitening":
                if self.whitening is None:
                    raise RuntimeError("whitening method requires an operator")
                weighted_residual = residual[index] @ self.whitening
                weighted_original = activations @ self.whitening
                primary.append(
                    weighted_residual.square().sum() / weighted_original.square().sum()
                )
            else:
                primary.append(mse[index])

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
            with torch.no_grad():
                cutoff = values[index].min().float()
                if model.threshold_updates.item() == 0:
                    model.activation_threshold.copy_(cutoff)
                else:
                    model.activation_threshold.lerp_(cutoff, 1 - model.threshold_ema)
                model.threshold_updates.add_(1)
            metrics[spec.name] = {
                "loss": float(losses[index].detach()),
                "nmse": float(mse[index].detach()),
                "decoder": float(decoder[index].detach()),
                "aux": float(auxiliary_tensor[index].detach()),
                "l0": float((code[index] != 0).sum(dim=1).float().mean()),
                "dead": int((step - model.last_active_step >= self.dead_after_steps).sum()),
            }
        return metrics

    def state_dict(self, *, step: int, tokens_seen: int) -> dict:
        return {
            "step": step,
            "tokens_seen": tokens_seen,
            "specs": [asdict(spec) for spec in self.specs],
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state: dict) -> tuple[int, int]:
        for name, model_state in state["models"].items():
            self.models[name].load_state_dict(model_state)
        for name, optimizer_state in state["optimizers"].items():
            self.optimizers[name].load_state_dict(optimizer_state)
        return int(state["step"]), int(state["tokens_seen"])

    def export_models(self) -> dict[str, dict]:
        return {
            spec.name: {
                "spec": asdict(spec),
                "state_dict": self.models[spec.name].state_dict(),
            }
            for spec in self.specs
        }
