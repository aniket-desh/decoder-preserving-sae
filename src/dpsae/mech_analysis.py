"""Held-out sparse-feature and causal analysis for the IOI experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .corpus import MemmapTokenBatcher, TokenRange
from .ioi import (
    TEMPLATE_SPLITS,
    IOIExample,
    canonical_name_splits,
    generate_ioi_examples,
    tokenize_ioi_examples,
)
from .language_model import ActivationStats, GPT2ActivationModel, answer_logit_difference
from .language_sae import BatchTopKSAE


def standardized_mean_difference(positive: Tensor, negative: Tensor) -> Tensor:
    mean_difference = positive.float().mean(0) - negative.float().mean(0)
    pooled_variance = 0.5 * (
        positive.float().var(0, unbiased=False) + negative.float().var(0, unbiased=False)
    )
    return mean_difference / pooled_variance.sqrt().clamp_min(1e-6)


def fit_ridge_binary(x: Tensor, labels: Tensor, *, ridge: float = 1e-2) -> dict[str, Tensor]:
    x = x.float()
    labels = labels.float()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    standardized = (x - mean) / scale
    design = torch.cat([standardized, torch.ones(len(x), 1)], dim=1)
    identity = torch.eye(design.shape[1], dtype=design.dtype)
    identity[-1, -1] = 0
    weights = torch.linalg.solve(
        design.mT @ design + len(x) * ridge * identity,
        design.mT @ labels,
    )
    return {"mean": mean, "scale": scale, "weights": weights}


def score_ridge_binary(probe: dict[str, Tensor], x: Tensor) -> Tensor:
    standardized = (x.float() - probe["mean"]) / probe["scale"]
    return standardized @ probe["weights"][:-1] + probe["weights"][-1]


def binary_auc(scores: Tensor, labels: Tensor) -> float:
    sorted_scores, order = scores.sort()
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    average_ranks = (starts + ends + 1).float() / 2
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.repeat_interleave(average_ranks, counts)
    positive = labels > 0
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("AUC requires both classes")
    value = (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )
    return float(value)


def binary_metrics(scores: Tensor, labels: Tensor) -> dict[str, float]:
    return {
        "accuracy": float(((scores >= 0) == (labels > 0)).float().mean()),
        "auc": binary_auc(scores, labels),
    }


def matched_random_features(
    selected: Tensor, firing_rate: Tensor, *, excluded: Tensor | None = None
) -> Tensor:
    unavailable = torch.zeros_like(firing_rate, dtype=torch.bool)
    unavailable[selected] = True
    if excluded is not None:
        unavailable[excluded] = True
    log_rate = firing_rate.clamp_min(1e-8).log()
    result = []
    for feature in selected.tolist():
        distance = (log_rate - log_rate[feature]).abs()
        distance[unavailable] = torch.inf
        match = int(distance.argmin())
        result.append(match)
        unavailable[match] = True
    return torch.tensor(result, dtype=torch.long)


def build_examples(config: dict, tokenizer) -> dict[str, list[IOIExample]]:
    names = canonical_name_splits(tokenizer, seed=config["seed"])
    counts = {
        "discovery": config["ioi"]["discovery_examples"],
        "validation": config["ioi"]["validation_examples"],
        "test": config["ioi"]["test_examples"],
    }
    return {
        split: generate_ioi_examples(
            count=counts[split],
            names=names[split],
            template_families=TEMPLATE_SPLITS[split],
            seed=config["seed"] + index,
        )
        for index, split in enumerate(("discovery", "validation", "test"))
    }


@torch.inference_mode()
def collect_state_activations(
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    examples_by_split: dict[str, list[IOIExample]],
    *,
    batch_size: int,
) -> dict[str, dict[str, Tensor]]:
    result = {}
    for split, examples in examples_by_split.items():
        prompt_chunks, abc_chunks = [], []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompt = tokenize_ioi_examples(batch, lm.tokenizer, variant="prompt")
            abc = tokenize_ioi_examples(batch, lm.tokenizer, variant="abc_prompt")
            prompt_activation = lm.activations(prompt["input_ids"], prompt["attention_mask"])
            abc_activation = lm.activations(abc["input_ids"], abc["attention_mask"])
            rows = torch.arange(len(batch), device=lm.device)
            prompt_s2 = prompt_activation[rows, prompt["s2_position"].to(lm.device)]
            abc_s2 = abc_activation[rows, abc["s2_position"].to(lm.device)]
            prompt_chunks.append(stats.normalize(prompt_s2).cpu().half())
            abc_chunks.append(stats.normalize(abc_s2).cpu().half())
        result[split] = {
            "positive": torch.cat(prompt_chunks),
            "negative": torch.cat(abc_chunks),
        }
    return result


def load_sae(payload: dict, *, input_dim: int, device: torch.device) -> BatchTopKSAE:
    spec = payload["spec"]
    sparsity_config = payload.get("sparsity_config", {})
    dictionary_size = payload["state_dict"]["encoder_bias"].numel()
    model = BatchTopKSAE(
        input_dim,
        dictionary_size,
        spec["k"],
        seed=spec["seed"],
        sparsity_mode=payload.get("sparsity_mode", "batch_topk"),
        jump_relu_init_threshold=float(sparsity_config.get("init_threshold", 0.001)),
        jump_relu_bandwidth=float(sparsity_config.get("bandwidth", 0.001)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def encode_state_pair(
    model: BatchTopKSAE, pair: dict[str, Tensor], *, batch_size: int = 512
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    codes, reconstructions = [], []
    combined = torch.cat([pair["positive"], pair["negative"]])
    for batch in combined.split(batch_size):
        reconstruction, code = model(
            batch.to(model.decoder_weight.device).float(), use_threshold=True
        )
        codes.append(code.cpu().half())
        reconstructions.append(reconstruction.cpu().half())
    code = torch.cat(codes)
    reconstruction = torch.cat(reconstructions)
    count = len(pair["positive"])
    return code[:count], code[count:], reconstruction[:count], reconstruction[count:]


def probe_curve(
    discovery: tuple[Tensor, Tensor],
    test: tuple[Tensor, Tensor],
    ranking: Tensor,
    feature_counts: Sequence[int],
) -> list[dict[str, float]]:
    train_x = torch.cat(discovery)
    train_y = torch.cat([torch.ones(len(discovery[0])), -torch.ones(len(discovery[1]))])
    test_x = torch.cat(test)
    test_y = torch.cat([torch.ones(len(test[0])), -torch.ones(len(test[1]))])
    rows = []
    for count in feature_counts:
        features = ranking[:count]
        probe = fit_ridge_binary(train_x[:, features], train_y)
        metrics = binary_metrics(score_ridge_binary(probe, test_x[:, features]), test_y)
        rows.append({"features": count, **metrics})
    return rows


def dense_probe_metrics(
    discovery: tuple[Tensor, Tensor], test: tuple[Tensor, Tensor]
) -> dict[str, float]:
    train_x = torch.cat(discovery).float()
    train_y = torch.cat([torch.ones(len(discovery[0])), -torch.ones(len(discovery[1]))])
    test_x = torch.cat(test).float()
    test_y = torch.cat([torch.ones(len(test[0])), -torch.ones(len(test[1]))])
    probe = fit_ridge_binary(train_x, train_y)
    return binary_metrics(score_ridge_binary(probe, test_x), test_y)


def make_replacement(
    model: BatchTopKSAE,
    stats: ActivationStats,
    *,
    positions: Tensor | None = None,
    features: Tensor | None = None,
    patch_values: Tensor | None = None,
    ablate_everywhere: bool = False,
    diagnostics: dict[str, list[float]] | None = None,
):
    def replacement(hidden: Tensor) -> Tensor:
        shape = hidden.shape
        normalized = stats.normalize(hidden).reshape(-1, shape[-1])
        _, code = model(normalized, use_threshold=True)
        code = code.reshape(shape[0], shape[1], -1)
        baseline = None
        if diagnostics is not None and positions is not None:
            diagnostic_rows = torch.arange(shape[0], device=code.device)
            diagnostic_positions = positions.to(code.device)
            baseline = model.decode(code[diagnostic_rows, diagnostic_positions])
        if features is not None:
            feature_ids = features.to(code.device)
            if ablate_everywhere:
                code[:, :, feature_ids] = 0
            elif positions is not None:
                rows = torch.arange(shape[0], device=code.device)[:, None]
                token_positions = positions.to(code.device)[:, None]
                columns = feature_ids[None, :]
                values = 0 if patch_values is None else patch_values.to(code.device)
                code[rows, token_positions, columns] = values
        if diagnostics is not None and positions is not None and baseline is not None:
            intervened = model.decode(code[diagnostic_rows, diagnostic_positions])
            change = intervened - baseline
            reference = normalized.reshape(shape)[diagnostic_rows, diagnostic_positions]
            relative_change = (
                change.float().square().sum() / reference.float().square().sum().clamp_min(1e-12)
            ).sqrt()
            diagnostics.setdefault("relative_activation_change", []).append(
                float(relative_change.detach())
            )
        reconstruction = model.decode(code.reshape(-1, code.shape[-1])).reshape(shape)
        return stats.denormalize(reconstruction)

    return replacement


@torch.inference_mode()
def feature_trace(
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    model: BatchTopKSAE,
    example: IOIExample,
    features: Tensor,
) -> dict:
    tokenized = tokenize_ioi_examples([example], lm.tokenizer, variant="prompt")
    activation = lm.activations(tokenized["input_ids"], tokenized["attention_mask"])
    normalized = stats.normalize(activation).reshape(-1, activation.shape[-1])
    code = model.encode(normalized, use_threshold=True).reshape(1, activation.shape[1], -1)
    length = int(tokenized["attention_mask"].sum())
    ids = tokenized["input_ids"][0, :length]
    return {
        "prompt": example.prompt,
        "tokens": [lm.tokenizer.decode([int(token)]) for token in ids],
        "features": features.tolist(),
        "activations": code[0, :length, features].float().cpu().mT.tolist(),
        "io_position": int(tokenized["io_position"][0]),
        "s1_position": int(tokenized["s1_position"][0]),
        "s2_position": int(tokenized["s2_position"][0]),
    }


@torch.inference_mode()
def causal_frontier(
    config: dict,
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    model: BatchTopKSAE,
    examples: list[IOIExample],
    abc_state: Tensor,
    ranking: Tensor,
    random_ranking: Tensor,
) -> list[dict[str, float]]:
    examples = examples[: config["ioi"]["causal_examples"]]
    abc_state = abc_state[: len(examples)]
    feature_counts = config["ioi"]["feature_counts"]
    keys = (
        "full",
        "ablated",
        "patched",
        "random_patched",
        "ablated_relative_change",
        "patched_relative_change",
        "random_patched_relative_change",
    )
    accumulators = {count: {key: [] for key in keys} for count in feature_counts}
    batch_size = config["ioi"]["batch_size"]
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokenized = tokenize_ioi_examples(batch, lm.tokenizer, variant="prompt")
        s2_positions = tokenized["s2_position"]
        full_logits = lm.logits(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            replacement=make_replacement(model, stats),
        )
        full_ld = answer_logit_difference(
            full_logits,
            tokenized["attention_mask"],
            tokenized["io_token_id"],
            tokenized["subject_token_id"],
        ).cpu()
        abc_code = model.encode(
            abc_state[start : start + len(batch)].to(lm.device).float(),
            use_threshold=True,
        )
        for count in feature_counts:
            selected = ranking[:count]
            random_features = random_ranking[:count]
            interventions = {}
            for key, features, values in (
                ("ablated", selected, None),
                ("patched", selected, abc_code[:, selected]),
                ("random_patched", random_features, abc_code[:, random_features]),
            ):
                diagnostics: dict[str, list[float]] = {}
                replacement = make_replacement(
                    model,
                    stats,
                    positions=s2_positions,
                    features=features,
                    patch_values=values,
                    diagnostics=diagnostics,
                )
                interventions[key] = (replacement, diagnostics)
            for key, (replacement, diagnostics) in interventions.items():
                logits = lm.logits(
                    tokenized["input_ids"],
                    tokenized["attention_mask"],
                    replacement=replacement,
                )
                ld = answer_logit_difference(
                    logits,
                    tokenized["attention_mask"],
                    tokenized["io_token_id"],
                    tokenized["subject_token_id"],
                ).cpu()
                accumulators[count][key].append(ld)
                relative_change = diagnostics.get("relative_activation_change")
                if not relative_change:
                    raise RuntimeError(f"missing activation-change diagnostic for {key}")
                accumulators[count][f"{key}_relative_change"].append(
                    torch.tensor(relative_change)
                )
            accumulators[count]["full"].append(full_ld)
    rows = []
    for count in feature_counts:
        values = {key: torch.cat(chunks) for key, chunks in accumulators[count].items()}
        rows.append(
            {
                "features": count,
                "full_logit_difference": float(values["full"].mean()),
                "ablation_effect": float((values["full"] - values["ablated"]).mean()),
                "abc_patch_effect": float((values["full"] - values["patched"]).mean()),
                "random_patch_effect": float(
                    (values["full"] - values["random_patched"]).mean()
                ),
                "ablation_relative_activation_change": float(
                    values["ablated_relative_change"].mean()
                ),
                "abc_patch_relative_activation_change": float(
                    values["patched_relative_change"].mean()
                ),
                "random_patch_relative_activation_change": float(
                    values["random_patched_relative_change"].mean()
                ),
            }
        )
    return rows


@torch.inference_mode()
def collateral_metrics(
    config: dict,
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    model: BatchTopKSAE,
    ranking: Tensor,
    *,
    token_path: Path,
) -> list[dict[str, float]]:
    start, stop = config["corpus"]["ranges"]["validation"]
    batcher = MemmapTokenBatcher(
        token_path,
        token_count=config["corpus"]["token_count"],
        token_range=TokenRange(start, stop),
        sequence_length=config["training"]["sequence_length"],
        batch_size=config["training"]["sequences_per_batch"],
        seed=config["seed"] + 404,
    )
    sequences_per_batch = config["training"]["sequences_per_batch"]
    batches = max(1, config["ioi"]["collateral_sequences"] // sequences_per_batch)
    kl = {count: [] for count in config["ioi"]["feature_counts"]}
    original_losses, reconstruction_losses = [], []
    for _ in range(batches):
        ids = batcher.batch()
        original_logits = lm.logits(ids)
        full_logits = lm.logits(ids, replacement=make_replacement(model, stats))
        targets = ids[:, 1:].to(lm.device)
        original_losses.append(
            F.cross_entropy(original_logits[:, :-1].flatten(0, 1), targets.flatten())
        )
        reconstruction_losses.append(
            F.cross_entropy(full_logits[:, :-1].flatten(0, 1), targets.flatten())
        )
        full_log_prob = full_logits.log_softmax(dim=-1)
        full_prob = full_log_prob.exp()
        for count in config["ioi"]["feature_counts"]:
            ablated_logits = lm.logits(
                ids,
                replacement=make_replacement(
                    model,
                    stats,
                    features=ranking[:count],
                    ablate_everywhere=True,
                ),
            )
            ablated_log_prob = ablated_logits.log_softmax(dim=-1)
            kl[count].append(
                (full_prob * (full_log_prob - ablated_log_prob)).sum(dim=-1).mean().cpu()
            )
    original_loss = float(torch.stack(original_losses).mean())
    reconstruction_loss = float(torch.stack(reconstruction_losses).mean())
    return [
        {
            "features": count,
            "collateral_kl": float(torch.stack(kl[count]).mean()),
            "original_cross_entropy": original_loss,
            "reconstruction_cross_entropy": reconstruction_loss,
        }
        for count in config["ioi"]["feature_counts"]
    ]


def analyze_model(
    config: dict,
    lm: GPT2ActivationModel,
    stats: ActivationStats,
    model: BatchTopKSAE,
    state_data: dict[str, dict[str, Tensor]],
    examples: dict[str, list[IOIExample]],
    *,
    token_path: Path,
    causal: bool,
) -> dict:
    encoded, reconstructed = {}, {}
    for split, pair in state_data.items():
        positive, negative, recon_positive, recon_negative = encode_state_pair(model, pair)
        encoded[split] = (positive, negative)
        reconstructed[split] = (recon_positive, recon_negative)
    effect = standardized_mean_difference(*encoded["discovery"])
    ranking = effect.abs().argsort(descending=True)
    firing_rate = (torch.cat(encoded["discovery"]) != 0).float().mean(0)
    maximum = max(config["ioi"]["feature_counts"])
    random_ranking = matched_random_features(ranking[:maximum], firing_rate)
    sparse_curve = probe_curve(
        encoded["discovery"], encoded["test"], ranking, config["ioi"]["feature_counts"]
    )
    original_dense = dense_probe_metrics(
        (state_data["discovery"]["positive"], state_data["discovery"]["negative"]),
        (state_data["test"]["positive"], state_data["test"]["negative"]),
    )
    reconstruction_dense = dense_probe_metrics(
        reconstructed["discovery"], reconstructed["test"]
    )
    target = 0.5 + 0.8 * (original_dense["accuracy"] - 0.5)
    threshold_count = next(
        (row["features"] for row in sparse_curve if row["accuracy"] >= target), None
    )
    result = {
        "sparse_probe_curve": sparse_curve,
        "original_dense_probe": original_dense,
        "reconstruction_dense_probe": reconstruction_dense,
        "features_to_80pct_dense": threshold_count,
        "ranked_features": ranking[:maximum].tolist(),
        "matched_random_features": random_ranking.tolist(),
        "top_effect_sizes": effect[ranking[:maximum]].tolist(),
        "top_firing_rates": firing_rate[ranking[:maximum]].tolist(),
        "feature_trace": feature_trace(
            lm, stats, model, examples["test"][0], ranking[: min(8, maximum)]
        ),
    }
    if causal:
        result["causal_frontier"] = causal_frontier(
            config,
            lm,
            stats,
            model,
            examples["test"],
            state_data["test"]["negative"],
            ranking,
            random_ranking,
        )
        result["collateral_frontier"] = collateral_metrics(
            config,
            lm,
            stats,
            model,
            ranking,
            token_path=token_path,
        )
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
