"""Evaluation-only natural-text diagnostics for Experiment 4b.

Integration: retain LM activations and SAE reconstructions as
``[sequences, tokens, width]`` tensors beside the matching ``[sequences,
tokens]`` token IDs. Concatenate evaluation batches only along the sequence
axis. Group indices then remain paired across every SAE, and EOS tokens let
``document_ids_from_tokens`` recover document spans without another dataset
pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor

from .corpus import TokenRange
from .decoder_distance import batched_ridge_predict


Grouping = Literal["contiguous", "shuffled", "document_balanced"]


def split_selection_test_range(
    token_range: TokenRange,
    *,
    selection_tokens: int | None = None,
    minimum_size: int = 1,
) -> dict[str, TokenRange]:
    """Split one held-out range into adjacent, disjoint selection and test ranges."""

    if minimum_size < 1:
        raise ValueError("minimum_size must be positive")
    if selection_tokens is None:
        selection_tokens = token_range.size // 2
    if not minimum_size <= selection_tokens <= token_range.size - minimum_size:
        raise ValueError("selection and test ranges must both satisfy minimum_size")
    boundary = token_range.start + selection_tokens
    return {
        "selection": TokenRange(token_range.start, boundary),
        "test": TokenRange(boundary, token_range.stop),
    }


def document_ids_from_tokens(token_ids: Tensor, *, eos_token_id: int | None) -> Tensor:
    """Infer document IDs, treating every sequence and post-EOS span as distinct."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    tokens = token_ids.detach().cpu()
    batch, sequence = tokens.shape
    if sequence == 0:
        raise ValueError("token sequences must be nonempty")
    if eos_token_id is None:
        return torch.arange(batch)[:, None].expand(batch, sequence)

    boundaries = torch.zeros_like(tokens, dtype=torch.bool)
    boundaries[:, 0] = True
    boundaries[:, 1:] = tokens[:, :-1] == eos_token_id
    local_ids = boundaries.cumsum(dim=1) - 1
    document_counts = local_ids[:, -1] + 1
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), document_counts.cumsum(0)[:-1]])
    return local_ids + offsets[:, None]


def geometry_group_indices(
    token_ids: Tensor,
    group_size: int,
    grouping: Grouping,
    *,
    seed: int = 0,
    eos_token_id: int | None = None,
) -> Tensor:
    """Return paired token indices for a natural-text geometry construction."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if group_size < 1 or token_ids.numel() < group_size:
        raise ValueError("group_size must fit within the token batch")
    if grouping not in {"contiguous", "shuffled", "document_balanced"}:
        raise ValueError(f"unknown grouping: {grouping}")

    generator = torch.Generator().manual_seed(seed)
    token_count = token_ids.numel()
    if grouping == "contiguous":
        order = torch.arange(token_count)
    elif grouping == "shuffled":
        order = torch.randperm(token_count, generator=generator)
    else:
        document_ids = document_ids_from_tokens(
            token_ids, eos_token_id=eos_token_id
        ).flatten()
        documents = []
        for document_id in document_ids.unique(sorted=True):
            indices = (document_ids == document_id).nonzero(as_tuple=False).flatten()
            documents.append(indices[torch.randperm(len(indices), generator=generator)])
        document_order = torch.randperm(len(documents), generator=generator).tolist()
        positions = [0] * len(documents)
        balanced = []
        while len(balanced) < token_count:
            for document in document_order:
                position = positions[document]
                if position < len(documents[document]):
                    balanced.append(int(documents[document][position]))
                    positions[document] += 1
        order = torch.tensor(balanced)

    usable = token_count - token_count % group_size
    return order[:usable].reshape(-1, group_size)


def apply_geometry_groups(values: Tensor, indices: Tensor) -> Tensor:
    """Gather tensors whose first two axes match the source token batch."""

    if values.ndim < 2 or indices.ndim != 2:
        raise ValueError("values and indices must have at least two and exactly two axes")
    flat = values.flatten(0, 1)
    if indices.numel() and int(indices.max()) >= len(flat):
        raise ValueError("group index exceeds the available tokens")
    return flat[indices.to(values.device)]


def _identity_prediction(groups: Tensor, ridge: float) -> Tensor:
    if groups.ndim != 3:
        raise ValueError("grouped activations must have shape [groups, samples, features]")
    group_count, samples, _ = groups.shape
    identity = torch.eye(samples, device=groups.device, dtype=torch.float32).expand(
        group_count, samples, samples
    )
    return batched_ridge_predict(groups.float(), identity, ridge)


def exact_identity_decoder_statistics(
    original: Tensor,
    reconstructed: Tensor,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    """Return exact per-group decoder error and reference energy using identity targets."""

    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed groups must have the same shape")
    reference = _identity_prediction(original, ridge)
    prediction = _identity_prediction(reconstructed, ridge)
    numerator = (prediction - reference).square().sum(dim=(1, 2))
    denominator = reference.square().sum(dim=(1, 2))
    return numerator, denominator


def bootstrap_ratio_interval(
    numerator: Tensor,
    denominator: Tensor,
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap a ratio of sums over geometry groups."""

    numerator = numerator.detach().cpu().double().flatten()
    denominator = denominator.detach().cpu().double().flatten()
    if numerator.shape != denominator.shape or not len(numerator):
        raise ValueError("numerator and denominator must be equally sized and nonempty")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must lie in (0, 1)")
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
        raise ValueError("bootstrap statistics must be finite")
    if denominator.sum() <= 0:
        raise ValueError("reference energy must be positive")

    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(numerator), (samples, len(numerator)), generator=generator)
    estimates = numerator[draws].sum(1) / denominator[draws].sum(1).clamp_min(1e-30)
    tail = (1 - confidence) / 2
    return {
        "estimate": float(numerator.sum() / denominator.sum()),
        "low": float(estimates.quantile(tail)),
        "high": float(estimates.quantile(1 - tail)),
    }


def bootstrap_paired_reduction_interval(
    baseline_numerator: Tensor,
    candidate_numerator: Tensor,
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap ``1 - candidate / baseline`` using paired geometry groups."""

    baseline = baseline_numerator.detach().cpu().double().flatten()
    candidate = candidate_numerator.detach().cpu().double().flatten()
    if baseline.shape != candidate.shape or not len(baseline):
        raise ValueError("paired statistics must be equally sized and nonempty")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must lie in (0, 1)")
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        raise ValueError("bootstrap statistics must be finite")
    if baseline.sum() <= 0:
        raise ValueError("baseline error must be positive")

    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(baseline), (samples, len(baseline)), generator=generator)
    estimates = 1 - candidate[draws].sum(1) / baseline[draws].sum(1).clamp_min(1e-30)
    tail = (1 - confidence) / 2
    return {
        "estimate": float(1 - candidate.sum() / baseline.sum()),
        "low": float(estimates.quantile(tail)),
        "high": float(estimates.quantile(1 - tail)),
    }


def exact_decoder_sweep(
    original: Tensor,
    reconstructions: Mapping[str, Tensor],
    token_ids: Tensor,
    *,
    ridges: Sequence[float],
    group_sizes: Sequence[int],
    groupings: Sequence[Grouping] = ("contiguous", "shuffled", "document_balanced"),
    eos_token_id: int | None = None,
    max_groups: int | None = None,
    bootstrap_samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Evaluate exact decoder distortion across ridge and grouping choices.

    The exact identity target materializes one ``[groups, group_size,
    group_size]`` ridge operator for the reference and one candidate at a time.
    There is deliberately no hidden chunking: set ``max_groups`` so both
    operators fit comfortably on the evaluation device. To combine explicit
    caller-side chunks, concatenate the returned per-group numerators and
    denominators before calling :func:`bootstrap_ratio_interval`.
    """

    if original.ndim != 3 or original.shape[:2] != token_ids.shape:
        raise ValueError("original must have shape [batch, sequence, features]")
    if not reconstructions or any(
        value.shape != original.shape for value in reconstructions.values()
    ):
        raise ValueError("every reconstruction must match original")
    if not ridges or any(ridge <= 0 for ridge in ridges):
        raise ValueError("ridges must be nonempty and strictly positive")
    if not group_sizes:
        raise ValueError("group_sizes must be nonempty")
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be positive")

    rows = []
    combination = 0
    for grouping in groupings:
        for group_size in group_sizes:
            indices = geometry_group_indices(
                token_ids,
                group_size,
                grouping,
                seed=seed + combination,
                eos_token_id=eos_token_id,
            )
            if max_groups is not None and len(indices) > max_groups:
                generator = torch.Generator().manual_seed(seed + 10_000 + combination)
                indices = indices[torch.randperm(len(indices), generator=generator)[:max_groups]]
            grouped_original = apply_geometry_groups(original, indices)
            for ridge in ridges:
                reference = _identity_prediction(grouped_original, float(ridge))
                denominator = reference.square().sum(dim=(1, 2))
                for name, reconstruction in reconstructions.items():
                    grouped_reconstruction = apply_geometry_groups(reconstruction, indices)
                    prediction = _identity_prediction(grouped_reconstruction, float(ridge))
                    numerator = (prediction - reference).square().sum(dim=(1, 2))
                    interval = bootstrap_ratio_interval(
                        numerator,
                        denominator,
                        samples=bootstrap_samples,
                        confidence=confidence,
                        seed=seed + combination,
                    )
                    rows.append(
                        {
                            "model": name,
                            "grouping": grouping,
                            "group_size": group_size,
                            "ridge": float(ridge),
                            "groups": len(indices),
                            "decoder_distortion": interval["estimate"],
                            "ci_low": interval["low"],
                            "ci_high": interval["high"],
                            "numerator_by_group": numerator.detach().cpu().tolist(),
                            "denominator_by_group": denominator.detach().cpu().tolist(),
                        }
                    )
            combination += 1
    return rows


def selected_feature_exposure(
    codes: Tensor,
    decoder_weight: Tensor,
    features: Tensor | Sequence[int],
    *,
    reference_activations: Tensor | None = None,
    collateral_kl: float | None = None,
    eps: float = 1e-12,
) -> dict:
    """Measure ordinary-text exposure and ablation size for one feature set."""

    if codes.ndim < 2 or decoder_weight.ndim != 2:
        raise ValueError("codes and decoder_weight must each have a feature axis")
    if codes.shape[-1] != decoder_weight.shape[0]:
        raise ValueError("code and decoder feature dimensions do not match")
    feature_ids = torch.as_tensor(features, dtype=torch.long, device=codes.device).flatten()
    if not len(feature_ids) or feature_ids.unique().numel() != len(feature_ids):
        raise ValueError("features must be nonempty and unique")
    if int(feature_ids.min()) < 0 or int(feature_ids.max()) >= codes.shape[-1]:
        raise ValueError("feature index out of bounds")

    selected_codes = codes.reshape(-1, codes.shape[-1]).float()[:, feature_ids]
    selected_decoder = decoder_weight.float()[feature_ids.to(decoder_weight.device)]
    if selected_codes.device != selected_decoder.device:
        selected_codes = selected_codes.to(selected_decoder.device)
    frequency = (selected_codes > 0).float().mean(0)
    mass = selected_codes.abs().mean(0)
    energy = selected_codes.square().mean(0) * selected_decoder.square().sum(1)
    change = selected_codes @ selected_decoder
    change_energy = change.square().sum(1).mean()
    change_rms = change_energy.sqrt()

    result = {
        "features": feature_ids.detach().cpu().tolist(),
        "feature_count": len(feature_ids),
        "active_frequency_by_feature": frequency.detach().cpu().tolist(),
        "activation_mass_by_feature": mass.detach().cpu().tolist(),
        "decoded_energy_by_feature": energy.detach().cpu().tolist(),
        "summed_active_frequency": float(frequency.sum()),
        "summed_activation_mass": float(mass.sum()),
        "summed_decoded_energy": float(energy.sum()),
        "ablation_change_energy": float(change_energy),
        "ablation_change_rms": float(change_rms),
    }
    normalizer = change_rms
    if reference_activations is not None:
        reference = reference_activations.reshape(-1, reference_activations.shape[-1]).float()
        if len(reference) != len(selected_codes) or reference.shape[1] != decoder_weight.shape[1]:
            raise ValueError("reference activations must align with codes and decoder output")
        reference_energy = reference.square().sum(1).mean().to(change_energy.device)
        relative_change = (change_energy / reference_energy.clamp_min(eps)).sqrt()
        result["ablation_relative_activation_change"] = float(relative_change)
        normalizer = relative_change
    if collateral_kl is not None:
        result["collateral_kl"] = float(collateral_kl)
        result["collateral_kl_per_activation_change"] = float(
            collateral_kl / max(float(normalizer), eps)
        )
    return result
