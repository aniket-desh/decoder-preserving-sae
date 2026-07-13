"""GPU execution primitives for the frozen Experiment 4b IOI protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .exp04b_ioi import positional_kl, sample_natural_positions
from .ioi import IOIExample, tokenize_ioi_examples
from .language_model import (
    ActivationStats,
    GPT2ActivationModel,
    answer_logit_difference,
)
from .language_sae import BatchTopKSAE
from .mech_analysis import (
    encode_state_pair,
    make_replacement,
    matched_random_features,
    standardized_mean_difference,
)


def confirmatory_example_splits(
    examples: Mapping[str, Sequence[IOIExample]],
    *,
    ranking_examples: int,
    selection_examples: int,
) -> dict[str, list[IOIExample]]:
    """Freeze ranking/selection from discovery and final from validation."""

    discovery = list(examples["discovery"])
    if ranking_examples <= 0 or selection_examples <= 0:
        raise ValueError("ranking and selection sizes must be positive")
    if ranking_examples + selection_examples != len(discovery):
        raise ValueError("ranking and selection must exhaust discovery exactly")
    final = list(examples["validation"])
    if not final:
        raise ValueError("the final validation split must be nonempty")
    return {
        "ranking": discovery[:ranking_examples],
        "selection": discovery[ranking_examples:],
        "test": final,
    }


@torch.inference_mode()
def collect_ioi_cache(
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    examples: Mapping[str, Sequence[IOIExample]],
    *,
    batch_size: int,
) -> dict[str, dict[str, Tensor]]:
    """Cache normalized clean/ABC S2 states and original-model behavior."""

    result = {}
    for split, split_examples in examples.items():
        clean_states, abc_states, logit_differences, lags = [], [], [], []
        for start in range(0, len(split_examples), batch_size):
            batch = split_examples[start : start + batch_size]
            clean = tokenize_ioi_examples(batch, lm.tokenizer, variant="prompt")
            abc = tokenize_ioi_examples(batch, lm.tokenizer, variant="abc_prompt")
            clean_activation = lm.activations(clean["input_ids"], clean["attention_mask"])
            abc_activation = lm.activations(abc["input_ids"], abc["attention_mask"])
            rows = torch.arange(len(batch), device=lm.device)
            clean_s2 = clean_activation[rows, clean["s2_position"].to(lm.device)]
            abc_s2 = abc_activation[rows, abc["s2_position"].to(lm.device)]
            clean_states.append(stats.normalize(clean_s2).cpu().half())
            abc_states.append(stats.normalize(abc_s2).cpu().half())
            logits = lm.logits(clean["input_ids"], clean["attention_mask"])
            logit_differences.append(
                answer_logit_difference(
                    logits,
                    clean["attention_mask"],
                    clean["io_token_id"],
                    clean["subject_token_id"],
                ).cpu()
            )
            lags.append((clean["end_position"] - clean["s2_position"]).cpu())
        result[split] = {
            "positive": torch.cat(clean_states),
            "negative": torch.cat(abc_states),
            "original_logit_difference": torch.cat(logit_differences),
            "end_s2_lag": torch.cat(lags),
        }
    return result


@torch.inference_mode()
def encode_confirmatory_states(
    model: BatchTopKSAE,
    cache: Mapping[str, Mapping[str, Tensor]],
) -> dict[str, dict[str, tuple[Tensor, Tensor]]]:
    """Encode all frozen splits once for ranking and harder-target probes."""

    result = {}
    for split, pair in cache.items():
        positive, negative, recon_positive, recon_negative = encode_state_pair(
            model, {"positive": pair["positive"], "negative": pair["negative"]}
        )
        result[split] = {
            "codes": (positive, negative),
            "reconstructions": (recon_positive, recon_negative),
        }
    return result


def duplicate_state_ranking(
    encoded_ranking: tuple[Tensor, Tensor],
    *,
    maximum: int,
) -> tuple[Tensor, Tensor]:
    """Return discovery ranking and firing-rate-matched random controls."""

    effect = standardized_mean_difference(*encoded_ranking)
    ranking = effect.abs().argsort(descending=True)
    firing_rate = (torch.cat(encoded_ranking) != 0).float().mean(0)
    random = matched_random_features(ranking[:maximum], firing_rate)
    return ranking, random


def _logit_difference(lm: GPT2ActivationModel, logits: Tensor, tokenized: dict) -> Tensor:
    return answer_logit_difference(
        logits,
        tokenized["attention_mask"],
        tokenized["io_token_id"],
        tokenized["subject_token_id"],
    ).cpu()


@torch.inference_mode()
def zero_ablation_curve(
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    model: BatchTopKSAE,
    examples: Sequence[IOIExample],
    ranking: Tensor,
    feature_counts: Sequence[int],
    *,
    original_logit_difference: Tensor,
    batch_size: int,
) -> list[dict]:
    """Zero discovery-ranked features only at S2 and retain per-example effects."""

    full_chunks = []
    ablated = {count: [] for count in feature_counts}
    changes = {count: [] for count in feature_counts}
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokenized = tokenize_ioi_examples(batch, lm.tokenizer, variant="prompt")
        full_logits = lm.logits(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            replacement=make_replacement(model, stats),
        )
        full_chunks.append(_logit_difference(lm, full_logits, tokenized))
        for count in feature_counts:
            diagnostics: dict[str, list[float]] = {}
            replacement = make_replacement(
                model,
                stats,
                positions=tokenized["s2_position"],
                features=ranking[:count],
                diagnostics=diagnostics,
            )
            logits = lm.logits(
                tokenized["input_ids"],
                tokenized["attention_mask"],
                replacement=replacement,
            )
            ablated[count].append(_logit_difference(lm, logits, tokenized))
            changes[count].extend(diagnostics["relative_activation_change"])
    full = torch.cat(full_chunks)
    original = original_logit_difference.float().cpu()
    if len(full) != len(original):
        raise ValueError("cached original IOI behavior does not align with examples")
    rows = []
    for count in feature_counts:
        intervened = torch.cat(ablated[count])
        effect = full - intervened
        rows.append(
            {
                "features": int(count),
                "examples": len(effect),
                "original_logit_difference": float(original.mean()),
                "full_sae_logit_difference": float(full.mean()),
                "ablated_logit_difference": float(intervened.mean()),
                "ioi_effect": float(effect.mean()),
                "relative_activation_change": float(
                    torch.tensor(changes[count]).mean()
                ),
                "effect_by_example": effect.tolist(),
            }
        )
    return rows


@torch.inference_mode()
def natural_zero_ablation_curve(
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    model: BatchTopKSAE,
    input_ids: Tensor,
    ranking: Tensor,
    feature_counts: Sequence[int],
    *,
    lag_distribution: Tensor,
    batch_size: int,
    seed: int,
) -> list[dict]:
    """Apply the matched zero operator once per natural sequence and read at END."""

    attention_mask = torch.ones_like(input_ids)
    intervention_positions = sample_natural_positions(
        attention_mask,
        seed=seed,
        lag_distribution=lag_distribution,
    )
    kl = {count: [] for count in feature_counts}
    changes = {count: [] for count in feature_counts}
    for start in range(0, len(input_ids), batch_size):
        ids = input_ids[start : start + batch_size]
        mask = attention_mask[start : start + batch_size]
        positions = intervention_positions[start : start + batch_size]
        full = lm.logits(ids, mask, replacement=make_replacement(model, stats))
        readout = mask.sum(1) - 1
        for count in feature_counts:
            diagnostics: dict[str, list[float]] = {}
            replacement = make_replacement(
                model,
                stats,
                positions=positions,
                features=ranking[:count],
                diagnostics=diagnostics,
            )
            intervened = lm.logits(ids, mask, replacement=replacement)
            kl[count].append(positional_kl(full, intervened, readout).cpu())
            changes[count].extend(diagnostics["relative_activation_change"])
    rows = []
    for count in feature_counts:
        values = torch.cat(kl[count])
        rows.append(
            {
                "features": int(count),
                "sequences": len(values),
                "collateral_kl": float(values.mean()),
                "natural_relative_activation_change": float(
                    torch.tensor(changes[count]).mean()
                ),
                "kl_by_sequence": values.tolist(),
                "intervention_positions": intervention_positions.tolist(),
            }
        )
    return rows


def _dense_s2_replacement(donor: Tensor, positions: Tensor):
    def replacement(hidden: Tensor) -> Tensor:
        result = hidden.clone()
        rows = torch.arange(len(hidden), device=hidden.device)
        result[rows, positions.to(hidden.device)] = donor.to(hidden.device)
        return result

    return replacement


@torch.inference_mode()
def abc_patch_ceiling(
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    model: BatchTopKSAE,
    examples: Sequence[IOIExample],
    abc_state: Tensor,
    selected_features: Tensor,
    random_features: Tensor,
    *,
    original_logit_difference: Tensor,
    batch_size: int,
) -> dict[str, list[float] | float | int]:
    """Compare selected/random SAE patches with full-code and dense ABC ceilings."""

    values = {name: [] for name in ("full", "selected", "random", "full_code", "dense")}
    all_features = torch.arange(model.encoder_bias.numel(), device=model.decoder_weight.device)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokenized = tokenize_ioi_examples(batch, lm.tokenizer, variant="prompt")
        donor_normalized = abc_state[start : start + len(batch)].to(lm.device).float()
        donor_code = model.encode(donor_normalized, use_threshold=True)
        full_logits = lm.logits(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            replacement=make_replacement(model, stats),
        )
        values["full"].append(_logit_difference(lm, full_logits, tokenized))
        for name, features in (
            ("selected", selected_features),
            ("random", random_features),
            ("full_code", all_features),
        ):
            feature_ids = features.to(lm.device)
            patch = donor_code[:, feature_ids]
            logits = lm.logits(
                tokenized["input_ids"],
                tokenized["attention_mask"],
                replacement=make_replacement(
                    model,
                    stats,
                    positions=tokenized["s2_position"],
                    features=feature_ids,
                    patch_values=patch,
                ),
            )
            values[name].append(_logit_difference(lm, logits, tokenized))
        dense_logits = lm.logits(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            replacement=_dense_s2_replacement(
                stats.denormalize(donor_normalized), tokenized["s2_position"]
            ),
        )
        values["dense"].append(_logit_difference(lm, dense_logits, tokenized))
    concatenated = {name: torch.cat(chunks) for name, chunks in values.items()}
    original = original_logit_difference.float().cpu()
    full = concatenated["full"]
    if original.shape != full.shape:
        raise ValueError("cached original IOI behavior does not align with patch examples")
    result: dict[str, list[float] | float | int] = {
        "examples": len(full),
        "original_model_logit_difference": float(original.mean()),
        "full_sae_logit_difference": float(full.mean()),
    }
    for name in ("selected", "random", "full_code"):
        effect = full - concatenated[name]
        result[f"{name}_patch_effect"] = float(effect.mean())
        result[f"{name}_patch_effect_by_example"] = effect.tolist()
    dense_effect = original - concatenated["dense"]
    result["dense_patch_effect"] = float(dense_effect.mean())
    result["dense_patch_effect_by_example"] = dense_effect.tolist()
    return result


@torch.inference_mode()
def selected_exposure_codes(
    model: BatchTopKSAE,
    normalized_activations: Tensor,
    ranking: Tensor,
    *,
    maximum: int,
    batch_size: int = 2_048,
) -> Tensor:
    """Encode only the ranked prefix needed for natural exposure diagnostics."""

    selected = ranking[:maximum].to(model.decoder_weight.device)
    chunks = []
    flat = normalized_activations.flatten(0, 1)
    for batch in flat.split(batch_size):
        code = model.encode(
            batch.to(model.decoder_weight.device).float(), use_threshold=True
        )
        chunks.append(code[:, selected].cpu().half())
    return torch.cat(chunks)
