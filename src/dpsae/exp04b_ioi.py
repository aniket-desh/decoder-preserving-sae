"""Pure protocol and reporting helpers for confirmatory IOI analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import median
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .language_model import answer_logit_difference


@dataclass(frozen=True)
class FrozenFeatureSelection:
    """A feature count chosen once on validation and reused unchanged on test."""

    feature_count: int
    kl_budget: float
    validation_effect: float
    validation_kl: float
    rule: str = "max_effect_under_kl_budget"
    selected_on: str = "validation"
    models: int = 1
    effect_aggregation: str = "single"
    kl_aggregation: str = "single"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _matrix_and_target(codes: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if codes.ndim != 2:
        raise ValueError("codes must have shape [examples, features]")
    target = target.float().flatten().to(codes.device)
    if len(codes) != len(target):
        raise ValueError("codes and target must contain the same number of examples")
    if len(target) < 2:
        raise ValueError("at least two examples are required")
    if not torch.isfinite(codes).all() or not torch.isfinite(target).all():
        raise ValueError("codes and target must be finite")
    return codes.float(), target


def feature_target_correlation(codes: Tensor, target: Tensor) -> Tensor:
    """Score discovery features by signed univariate target correlation."""

    codes, target = _matrix_and_target(codes, target)
    centered_codes = codes - codes.mean(0)
    centered_target = target - target.mean()
    numerator = (centered_codes * centered_target[:, None]).mean(0)
    denominator = centered_codes.square().mean(0).sqrt() * centered_target.square().mean().sqrt()
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def rank_discovery_features(codes: Tensor, target: Tensor) -> Tensor:
    """Return a deterministic absolute-correlation ranking fit on discovery only."""

    scores = feature_target_correlation(codes, target)
    return scores.abs().argsort(descending=True, stable=True)


def select_feature_count(
    validation_rows: Sequence[Mapping[str, float | int | str]],
    *,
    kl_budget: float,
    effect_key: str = "ioi_effect",
    kl_key: str = "collateral_kl",
) -> FrozenFeatureSelection:
    """Freeze the best validation effect satisfying the preregistered KL budget."""

    if not isfinite(kl_budget) or kl_budget < 0:
        raise ValueError("kl_budget must be finite and nonnegative")
    candidates: list[tuple[float, float, int]] = []
    seen: set[int] = set()
    for row in validation_rows:
        if row.get("split", "validation") != "validation":
            raise ValueError("feature-count selection accepts validation rows only")
        count = int(row["features"])
        effect, kl = float(row[effect_key]), float(row[kl_key])
        if count <= 0 or count in seen:
            raise ValueError("validation feature counts must be positive and unique")
        if not isfinite(effect) or not isfinite(kl) or kl < 0:
            raise ValueError(
                "validation effects and KL values must be finite; KL must be nonnegative"
            )
        seen.add(count)
        if kl <= kl_budget:
            candidates.append((effect, kl, count))
    if not candidates:
        raise ValueError("no validation feature count satisfies the KL budget")
    effect, kl, count = max(candidates, key=lambda row: (row[0], -row[2]))
    return FrozenFeatureSelection(count, kl_budget, effect, kl)


def select_global_feature_count(
    validation_rows: Sequence[Mapping[str, float | int | str]],
    *,
    kl_budget: float,
    included_methods: Sequence[str] = ("mse", "dpsae"),
    expected_models_per_method: int | None = 3,
) -> FrozenFeatureSelection:
    """Select one global count by median effect with a worst-model KL constraint."""

    included = set(included_methods)
    if not included:
        raise ValueError("at least one selection method is required")
    if not isfinite(kl_budget) or kl_budget < 0:
        raise ValueError("kl_budget must be finite and nonnegative")
    grouped: dict[int, dict[str, tuple[float, float]]] = {}
    model_methods: dict[str, str] = {}
    for row in validation_rows:
        if row.get("split", "validation") != "validation":
            raise ValueError("global feature-count selection accepts validation rows only")
        method = str(row["method"])
        if method not in included:
            continue
        count, model = int(row["features"]), str(row["model"])
        effect, kl = float(row["ioi_effect"]), float(row["collateral_kl"])
        if count <= 0 or not isfinite(effect) or not isfinite(kl) or kl < 0:
            raise ValueError("feature counts, effects, and KL values are invalid")
        if model in grouped.setdefault(count, {}):
            raise ValueError("each model must occur once per validation feature count")
        if model in model_methods and model_methods[model] != method:
            raise ValueError("a model cannot change method across feature counts")
        model_methods[model] = method
        grouped[count][model] = (effect, kl)
    if not grouped:
        raise ValueError("no included validation models were supplied")
    model_sets = [set(rows) for rows in grouped.values()]
    if any(models != model_sets[0] for models in model_sets[1:]):
        raise ValueError("every feature count must contain the same selection models")
    model_count = len(model_sets[0])
    if expected_models_per_method is not None:
        method_counts = {
            method: sum(value == method for value in model_methods.values())
            for method in included
        }
        if any(count != expected_models_per_method for count in method_counts.values()):
            raise ValueError(
                f"expected {expected_models_per_method} models per method, found {method_counts}"
            )
    candidates = []
    for count, rows in grouped.items():
        median_effect = float(median(effect for effect, _ in rows.values()))
        maximum_kl = max(kl for _, kl in rows.values())
        if maximum_kl <= kl_budget:
            candidates.append((median_effect, maximum_kl, count))
    if not candidates:
        raise ValueError("no global feature count satisfies the worst-model KL budget")
    effect, kl, count = max(candidates, key=lambda row: (row[0], -row[2]))
    return FrozenFeatureSelection(
        count,
        kl_budget,
        effect,
        kl,
        rule="max_median_effect_with_max_model_kl_budget",
        models=model_count,
        effect_aggregation="median_across_models",
        kl_aggregation="maximum_across_models",
    )


def frozen_test_report(
    selection: FrozenFeatureSelection,
    test_rows: Sequence[Mapping[str, float | int | str]],
) -> dict:
    """Report only the test row fixed by validation, without test-set reselection."""

    if selection.selected_on != "validation":
        raise ValueError("test reporting requires a validation-frozen selection")
    matches = []
    for row in test_rows:
        if row.get("split", "test") != "test":
            raise ValueError("frozen test reporting accepts test rows only")
        if int(row["features"]) == selection.feature_count:
            matches.append(dict(row))
    if len(matches) != 1:
        raise ValueError("test rows must contain the frozen feature count exactly once")
    return {"selection": selection.to_dict(), "test": matches[0]}


def frozen_test_report_by_model(
    selection: FrozenFeatureSelection,
    test_rows: Sequence[Mapping[str, float | int | str]],
) -> dict:
    """Apply one validation-frozen count to every test model, including controls."""

    selected = []
    seen: set[str] = set()
    for row in test_rows:
        if row.get("split", "test") != "test":
            raise ValueError("frozen test reporting accepts test rows only")
        if int(row["features"]) != selection.feature_count:
            continue
        model = str(row["model"])
        if model in seen:
            raise ValueError("each model must occur once at the frozen feature count")
        seen.add(model)
        selected.append(dict(row))
    if not selected:
        raise ValueError("no test models contain the frozen feature count")
    return {
        "selection": selection.to_dict(),
        "test_by_model": sorted(selected, key=lambda row: str(row["model"])),
    }


def sample_natural_positions(
    attention_mask: Tensor,
    *,
    seed: int,
    lag_distribution: Tensor | None = None,
) -> Tensor:
    """Sample one intervention position per sequence, optionally from IOI lags."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    eligible = attention_mask.to(dtype=torch.bool).clone()
    lengths = eligible.sum(1)
    if (lengths < 2).any():
        raise ValueError(
            "each sequence needs two valid tokens so its final position can be excluded"
        )
    indices = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    final = indices.expand_as(eligible).masked_fill(~eligible, -1).max(1).values
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if lag_distribution is not None:
        lags = lag_distribution.detach().cpu().long().flatten()
        if not len(lags) or (lags <= 0).any():
            raise ValueError("lag_distribution must contain positive IOI lags")
        sampled = lags[
            torch.randint(len(lags), (len(final),), generator=generator)
        ].to(final.device)
        return (final - sampled).clamp_min(0)
    eligible[torch.arange(len(eligible), device=eligible.device), final] = False
    scores = torch.rand(attention_mask.shape, generator=generator).to(attention_mask.device)
    return scores.masked_fill(~eligible, -1).argmax(1)


def positional_kl(full_logits: Tensor, intervened_logits: Tensor, positions: Tensor) -> Tensor:
    """Compute full-SAE || intervened KL at one specified position per sequence."""

    if full_logits.shape != intervened_logits.shape or full_logits.ndim != 3:
        raise ValueError("logits must have matching [batch, sequence, vocabulary] shapes")
    if positions.shape != (full_logits.shape[0],):
        raise ValueError("positions must contain one index per sequence")
    if ((positions < 0) | (positions >= full_logits.shape[1])).any():
        raise ValueError("positions lie outside the sequence")
    rows = torch.arange(full_logits.shape[0], device=full_logits.device)
    full_log_prob = full_logits[rows, positions.to(full_logits.device)].log_softmax(-1)
    intervened_log_prob = intervened_logits[
        rows, positions.to(intervened_logits.device)
    ].log_softmax(-1)
    return (full_log_prob.exp() * (full_log_prob - intervened_log_prob)).sum(-1)


def logit_difference_report(
    original: Tensor, full_sae: Tensor, intervened: Tensor
) -> dict[str, float | int]:
    """Use one stable schema for original, reconstructed, and intervened IOI behavior."""

    original, full_sae, intervened = (
        value.float().flatten() for value in (original, full_sae, intervened)
    )
    if original.shape != full_sae.shape or original.shape != intervened.shape or not len(original):
        raise ValueError("logit-difference vectors must have the same nonempty shape")
    if not all(torch.isfinite(value).all() for value in (original, full_sae, intervened)):
        raise ValueError("logit differences must be finite")
    return {
        "examples": len(original),
        "original_model_logit_difference": float(original.mean()),
        "full_sae_logit_difference": float(full_sae.mean()),
        "intervened_logit_difference": float(intervened.mean()),
        "sae_reconstruction_delta": float((full_sae - original).mean()),
        "intervention_effect": float((full_sae - intervened).mean()),
        "total_delta_from_original": float((intervened - original).mean()),
    }


def matched_zero_ablation_report(
    *,
    original_ioi_logit_difference: Tensor,
    full_ioi_logit_difference: Tensor,
    ablated_ioi_logit_difference: Tensor,
    full_natural_logits: Tensor,
    ablated_natural_logits: Tensor,
    natural_intervention_positions: Tensor,
    natural_readout_positions: Tensor,
    natural_relative_activation_change: float | None = None,
) -> dict:
    """Summarize the same zero operator at IOI S2 and one natural-text position."""

    if natural_intervention_positions.shape != natural_readout_positions.shape:
        raise ValueError("natural intervention and readout positions must align")
    kl = positional_kl(
        full_natural_logits, ablated_natural_logits, natural_readout_positions
    )
    natural = {
        "sequences": len(natural_intervention_positions),
        "collateral_kl": float(kl.mean()),
        "intervention_positions": natural_intervention_positions.cpu().tolist(),
        "readout_positions": natural_readout_positions.cpu().tolist(),
    }
    if natural_relative_activation_change is not None:
        if (
            not isfinite(natural_relative_activation_change)
            or natural_relative_activation_change < 0
        ):
            raise ValueError("natural relative activation change must be finite and nonnegative")
        natural["relative_activation_change"] = natural_relative_activation_change
    return {
        "operator": {
            "operation": "zero_selected_sae_features",
            "ioi_position": "S2",
            "natural_interventions_per_sequence": 1,
            "natural_readout": "final_nonpadding_token",
        },
        "ioi": logit_difference_report(
            original_ioi_logit_difference,
            full_ioi_logit_difference,
            ablated_ioi_logit_difference,
        ),
        "natural_text": natural,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if isfinite(denominator) and denominator > 0 else None


def exposure_normalized_summary(
    *,
    ioi_effect: float,
    collateral_kl: float,
    exposure: Mapping[str, float | int | list],
    natural_relative_activation_change: float,
) -> dict[str, float | int | None]:
    """Normalize collateral damage by natural exposure and realized perturbation size."""

    if not isfinite(ioi_effect) or not isfinite(collateral_kl) or collateral_kl < 0:
        raise ValueError("IOI effect and nonnegative collateral KL must be finite")
    return {
        "feature_count": int(exposure["feature_count"]),
        "ioi_effect": ioi_effect,
        "collateral_kl": collateral_kl,
        "ioi_effect_per_collateral_kl": _ratio(ioi_effect, collateral_kl),
        "kl_per_total_active_frequency": _ratio(
            collateral_kl, float(exposure["summed_active_frequency"])
        ),
        "kl_per_activation_mass": _ratio(
            collateral_kl, float(exposure["summed_activation_mass"])
        ),
        "kl_per_decoded_contribution_energy": _ratio(
            collateral_kl, float(exposure["summed_decoded_energy"])
        ),
        "kl_per_natural_relative_activation_change": _ratio(
            collateral_kl, natural_relative_activation_change
        ),
    }


def exposure_matched_comparison(
    reference: Mapping[str, float | int | list],
    comparator_curve: Sequence[Mapping[str, float | int | list]],
    *,
    exposure_key: str,
    outcome_key: str = "collateral_kl",
) -> dict[str, float | str] | None:
    """Linearly interpolate a comparator outcome at the reference exposure."""

    target = float(reference[exposure_key])
    reference_outcome = float(reference[outcome_key])
    points = sorted(
        (float(row[exposure_key]), float(row[outcome_key]))
        for row in comparator_curve
    )
    if not points or target < points[0][0] or target > points[-1][0]:
        return None
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if target == left_x or right_x == left_x:
            interpolated = left_y
            break
        if left_x <= target <= right_x:
            weight = (target - left_x) / (right_x - left_x)
            interpolated = left_y + weight * (right_y - left_y)
            break
    else:
        interpolated = points[-1][1]
    return {
        "exposure": exposure_key,
        "target_exposure": target,
        "reference_outcome": reference_outcome,
        "comparator_interpolated_outcome": interpolated,
        "reference_minus_comparator": reference_outcome - interpolated,
    }


def paired_bootstrap_summary(
    baseline: Tensor,
    treatment: Tensor,
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap paired initialization seeds and report treatment minus baseline."""

    baseline, treatment = baseline.float().flatten().cpu(), treatment.float().flatten().cpu()
    if baseline.shape != treatment.shape or not len(baseline):
        raise ValueError("paired metrics must have the same nonempty shape")
    if not torch.isfinite(baseline).all() or not torch.isfinite(treatment).all():
        raise ValueError("paired metrics must be finite")
    if bootstrap_samples <= 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap_samples and confidence are invalid")
    difference = treatment - baseline
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        len(difference),
        (bootstrap_samples, len(difference)),
        generator=generator,
    )
    bootstrap = difference[indices].mean(1)
    tail = (1 - confidence) / 2
    interval = torch.quantile(bootstrap, torch.tensor([tail, 1 - tail]))
    return {
        "pairs": len(difference),
        "estimand": "mean_treatment_minus_baseline",
        "baseline_mean": float(baseline.mean()),
        "treatment_mean": float(treatment.mean()),
        "paired_difference": float(difference.mean()),
        "confidence": confidence,
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def dense_abc_patch_ceiling_report(
    *,
    original_ioi_logit_difference: Tensor,
    full_ioi_logit_difference: Tensor,
    selected_patch_logit_difference: Tensor,
    dense_patch_logit_difference: Tensor,
) -> dict:
    """Place a selected-feature ABC patch beside the original-model dense patch ceiling."""

    selected = logit_difference_report(
        original_ioi_logit_difference,
        full_ioi_logit_difference,
        selected_patch_logit_difference,
    )
    original = original_ioi_logit_difference.float().flatten()
    dense = dense_patch_logit_difference.float().flatten()
    if original.shape != dense.shape or not len(original) or not torch.isfinite(dense).all():
        raise ValueError("dense-patch logit differences must match the original-model vector")
    dense_effect = float((original - dense).mean())
    return {
        "selected_feature_patch": selected,
        "dense_activation_abc_patch": {
            "reference": "original_model",
            "original_model_logit_difference": float(original.mean()),
            "patched_logit_difference": float(dense.mean()),
            "patch_effect": dense_effect,
        },
        "selected_effect_fraction_of_dense_ceiling": _ratio(
            float(selected["intervention_effect"]), dense_effect
        ),
    }


def correct_minus_subject_target(
    logits: Tensor,
    attention_mask: Tensor,
    correct_token_id: Tensor,
    subject_token_id: Tensor,
) -> Tensor:
    """Frozen continuous IOI target: original-model correct minus subject logit."""

    return answer_logit_difference(
        logits, attention_mask, correct_token_id, subject_token_id
    ).detach()


def _fit_ridge(x: Tensor, target: Tensor, ridge: float) -> tuple[Tensor, Tensor, Tensor]:
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    standardized = (x - mean) / scale
    design = torch.cat([standardized, torch.ones(len(x), 1, device=x.device)], 1)
    penalty = torch.eye(design.shape[1], device=x.device)
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(
        design.mT @ design + len(x) * ridge * penalty,
        design.mT @ target,
    )
    return mean, scale, weights


def continuous_target_curve(
    discovery_codes: Tensor,
    discovery_target: Tensor,
    evaluation_codes: Tensor,
    evaluation_target: Tensor,
    ranking: Tensor,
    feature_counts: Sequence[int],
    *,
    ridge: float = 1e-2,
) -> list[dict[str, float | int]]:
    """Fit on discovery and score a frozen correct-minus-subject target elsewhere."""

    discovery_codes, discovery_target = _matrix_and_target(discovery_codes, discovery_target)
    evaluation_codes, evaluation_target = _matrix_and_target(evaluation_codes, evaluation_target)
    if discovery_codes.shape[1] != evaluation_codes.shape[1]:
        raise ValueError("discovery and evaluation dictionaries differ")
    if not isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    denominator = (evaluation_target - evaluation_target.mean()).square().sum()
    if denominator <= 0:
        raise ValueError("evaluation target must vary")
    rows = []
    for count in feature_counts:
        if count <= 0 or count > discovery_codes.shape[1]:
            raise ValueError("feature counts must lie within the dictionary")
        features = ranking[:count].to(discovery_codes.device)
        mean, scale, weights = _fit_ridge(
            discovery_codes[:, features], discovery_target, ridge
        )
        evaluation_x = evaluation_codes[:, features.to(evaluation_codes.device)]
        standardized = (evaluation_x - mean.to(evaluation_x.device)) / scale.to(
            evaluation_x.device
        )
        prediction = standardized @ weights[:-1].to(evaluation_x.device) + weights[-1].to(
            evaluation_x.device
        )
        residual = prediction - evaluation_target
        correlation = feature_target_correlation(prediction[:, None], evaluation_target)[0]
        rows.append(
            {
                "features": int(count),
                "r2": float(1 - residual.square().sum() / denominator),
                "mae": float(residual.abs().mean()),
                "pearson": float(correlation),
            }
        )
    return rows


def continuous_target_protocol(
    *,
    discovery_codes: Tensor,
    discovery_target: Tensor,
    validation_codes: Tensor,
    validation_target: Tensor,
    test_codes: Tensor,
    test_target: Tensor,
    feature_counts: Sequence[int],
    selection: FrozenFeatureSelection,
    ridge: float = 1e-2,
) -> dict:
    """Score the harder target using the same global count frozen by matched ablation."""

    ranking = rank_discovery_features(discovery_codes, discovery_target)
    validation = continuous_target_curve(
        discovery_codes,
        discovery_target,
        validation_codes,
        validation_target,
        ranking,
        feature_counts,
        ridge=ridge,
    )
    validation_rows = [{**row, "split": "validation"} for row in validation]
    if selection.feature_count not in set(feature_counts):
        raise ValueError("the global frozen feature count is absent from the target curve")
    test = continuous_target_curve(
        discovery_codes,
        discovery_target,
        test_codes,
        test_target,
        ranking,
        [selection.feature_count],
        ridge=ridge,
    )[0]
    test["split"] = "test"
    return {
        "target": "correct_minus_subject_logit_difference",
        "ranking": ranking.cpu().tolist(),
        "validation": validation_rows,
        **frozen_test_report(selection, [test]),
    }
