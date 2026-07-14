#!/usr/bin/env python3
"""Frozen-split, matched-operator IOI confirmation for Experiment 4b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dpsae.exp04b_execution import (
    abc_patch_ceiling,
    collect_ioi_cache,
    confirmatory_example_splits,
    duplicate_state_ranking,
    encode_confirmatory_states,
    natural_zero_ablation_curve,
    selected_exposure_codes,
    zero_ablation_curve,
)
from dpsae.exp04b_ioi import (
    FrozenFeatureSelection,
    continuous_target_curve,
    continuous_target_protocol,
    exposure_matched_comparison,
    exposure_normalized_summary,
    paired_bootstrap_summary,
    select_global_feature_count,
)
from dpsae.exp04b_natural_text import selected_feature_exposure
from dpsae.ioi import IOIExample
from dpsae.language_model import ActivationStats, GPT2ActivationModel
from dpsae.mech_analysis import build_examples, load_sae, matched_random_features


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_configs(config_path: Path) -> tuple[dict, dict]:
    config = read_json(config_path)
    source = read_json(ROOT / config["source_config"])
    return config, source


def output_path(config: dict) -> Path:
    return ROOT / "artifacts" / config["experiment"]


def source_path(config: dict) -> Path:
    return ROOT / config["source_artifact"]


def load_lm(source: dict, device: torch.device) -> GPT2ActivationModel:
    return GPT2ActivationModel.from_pretrained(
        source["model_name"], layer=source["layer"], device=device
    )


def build_confirmatory_examples(
    config: dict, source: dict, tokenizer
) -> dict[str, list[IOIExample]]:
    generated = build_examples(source, tokenizer)
    return confirmatory_example_splits(
        generated,
        ranking_examples=config["ioi"]["ranking_examples"],
        selection_examples=config["ioi"]["selection_examples"],
    )


def prepare_cache(config: dict, source: dict, device: torch.device) -> None:
    output = output_path(config)
    cache_path = output / "ioi_confirmatory_cache.pt"
    if cache_path.exists():
        print(f"IOI cache already exists: {cache_path}", flush=True)
        return
    calibration = torch.load(
        source_path(config) / "calibration.pt", map_location="cpu"
    )
    lm = load_lm(source, device)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    examples = build_confirmatory_examples(config, source, lm.tokenizer)
    cache = collect_ioi_cache(
        lm,
        stats,
        examples,
        batch_size=source["ioi"]["batch_size"],
    )
    cache["protocol"] = {
        "ranking_examples": config["ioi"]["ranking_examples"],
        "selection_examples": config["ioi"]["selection_examples"],
        "test_examples": len(examples["test"]),
        "test_source": "original_validation_split",
    }
    save_torch(cache_path, cache)


def method_name(model_name: str) -> str:
    if model_name.startswith("dpsae"):
        return "dpsae"
    if model_name.startswith("whitening"):
        return "whitening"
    if model_name.startswith("spectral"):
        return "spectral"
    return "mse"


def _model_payloads(config: dict) -> dict:
    baseline_path = output_path(config) / "baseline_confirm" / "models.pt"
    if not baseline_path.exists():
        raise FileNotFoundError(
            "baseline confirmation must finish before confirmatory IOI analysis"
        )
    return torch.load(baseline_path, map_location="cpu")


def _natural_cache(config: dict, split: str) -> dict:
    return torch.load(
        output_path(config) / f"natural_{split}.pt", map_location="cpu"
    )


def _partial_results(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _exposure_curve(
    model,
    activations: torch.Tensor,
    ranking: torch.Tensor,
    natural_curve: list[dict],
    feature_counts: list[int],
    effect_by_count: dict[int, float],
) -> list[dict]:
    maximum = max(feature_counts)
    codes = selected_exposure_codes(
        model, activations, ranking, maximum=maximum
    )
    decoder = model.decoder_weight.detach().cpu()[ranking[:maximum]].float()
    collateral = {row["features"]: row for row in natural_curve}
    rows = []
    for count in feature_counts:
        natural = collateral[count]
        exposure = selected_feature_exposure(
            codes,
            decoder,
            torch.arange(count),
            reference_activations=activations,
            collateral_kl=natural["collateral_kl"],
        )
        rows.append(
            {
                **exposure,
                **exposure_normalized_summary(
                    ioi_effect=effect_by_count[count],
                    collateral_kl=natural["collateral_kl"],
                    exposure=exposure,
                    natural_relative_activation_change=natural[
                        "natural_relative_activation_change"
                    ],
                ),
                "natural_relative_activation_change": natural[
                    "natural_relative_activation_change"
                ],
            }
        )
    return rows


def analyze_selection(config: dict, source: dict, device: torch.device) -> None:
    output = output_path(config)
    result_path = output / "ioi_selection_models.json"
    results = _partial_results(result_path)
    calibration = torch.load(source_path(config) / "calibration.pt", map_location="cpu")
    cache = torch.load(output / "ioi_confirmatory_cache.pt", map_location="cpu")
    natural = _natural_cache(config, "selection")
    lm = load_lm(source, device)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    examples = build_confirmatory_examples(config, source, lm.tokenizer)
    counts = config["ioi"]["feature_counts"]
    maximum = max(counts)
    payloads = _model_payloads(config)
    for name, payload in payloads.items():
        if name in results:
            print(f"selection already complete: {name}", flush=True)
            continue
        print(f"selection analysis: {name}", flush=True)
        input_dim = calibration["activation_stats"]["mean"].numel()
        model = load_sae(payload, input_dim=input_dim, device=device)
        encoded = encode_confirmatory_states(model, cache)
        ranking, random = duplicate_state_ranking(
            encoded["ranking"]["codes"], maximum=maximum
        )
        ioi_curve = zero_ablation_curve(
            lm,
            stats,
            model,
            examples["selection"],
            ranking,
            counts,
            original_logit_difference=cache["selection"]["original_logit_difference"],
            batch_size=source["ioi"]["batch_size"],
        )
        natural_curve = natural_zero_ablation_curve(
            lm,
            stats,
            model,
            natural["input_ids"][: config["natural_text"]["matched_sequences"]],
            ranking,
            counts,
            lag_distribution=cache["ranking"]["end_s2_lag"],
            batch_size=source["training"]["sequences_per_batch"],
            seed=config["ioi"]["seed"],
        )
        effect_by_count = {row["features"]: row["ioi_effect"] for row in ioi_curve}
        exposure = _exposure_curve(
            model,
            natural["activations"],
            ranking,
            natural_curve,
            counts,
            effect_by_count,
        )
        results[name] = {
            "method": method_name(name),
            "spec": payload["spec"],
            "ranking": ranking[:maximum].tolist(),
            "matched_random_features": random.tolist(),
            "ioi_zero_curve": ioi_curve,
            "natural_zero_curve": natural_curve,
            "exposure_curve": exposure,
        }
        save_json(result_path, results)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    select_count(config, results)


def _fallback_selection(config: dict, rows: list[dict]) -> FrozenFeatureSelection:
    counts = config["ioi"]["feature_counts"]
    core = [row for row in rows if row["method"] in {"mse", "dpsae"}]
    scored = []
    for count in counts:
        selected = [row for row in core if row["features"] == count]
        worst_kl = max(row["collateral_kl"] for row in selected)
        effects = torch.tensor([row["ioi_effect"] for row in selected])
        scored.append((worst_kl, count, float(effects.median())))
    kl, count, effect = min(scored, key=lambda value: (value[0], value[1]))
    return FrozenFeatureSelection(
        count,
        config["ioi"]["kl_budget"],
        effect,
        kl,
        rule="diagnostic_minimum_worst_kl_no_feasible_count",
        selected_on="validation",
        models=6,
        effect_aggregation="median_across_models",
        kl_aggregation="maximum_across_models",
    )


def select_count(config: dict, results: dict) -> FrozenFeatureSelection:
    rows = []
    for name, result in results.items():
        natural = {row["features"]: row for row in result["natural_zero_curve"]}
        for row in result["ioi_zero_curve"]:
            rows.append(
                {
                    "split": "validation",
                    "model": name,
                    "method": result["method"],
                    "features": row["features"],
                    "ioi_effect": row["ioi_effect"],
                    "collateral_kl": natural[row["features"]]["collateral_kl"],
                }
            )
    try:
        selection = select_global_feature_count(
            rows,
            kl_budget=config["ioi"]["kl_budget"],
        )
        feasible = True
    except ValueError as error:
        if "no global feature count" not in str(error):
            raise
        selection = _fallback_selection(config, rows)
        feasible = False
    save_json(
        output_path(config) / "ioi_feature_count_selection.json",
        {"feasible": feasible, "selection": selection.to_dict(), "validation_rows": rows},
    )
    return selection


def _selection(config: dict) -> FrozenFeatureSelection:
    value = read_json(output_path(config) / "ioi_feature_count_selection.json")
    return FrozenFeatureSelection(**value["selection"])


def _dense_target_metric(
    discovery: torch.Tensor,
    discovery_target: torch.Tensor,
    test: torch.Tensor,
    test_target: torch.Tensor,
) -> dict:
    return continuous_target_curve(
        discovery.float(),
        discovery_target,
        test.float(),
        test_target,
        torch.arange(discovery.shape[1]),
        [discovery.shape[1]],
    )[0]


def _selected_row(rows: list[dict], count: int) -> dict:
    matches = [
        row
        for row in rows
        if row.get("feature_count", row.get("features")) == count
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for frozen feature count {count}")
    return matches[0]


def paired_test_summary(
    results: dict,
    selection: FrozenFeatureSelection,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    """Pair every confirmed method to the MSE model with the same seed."""

    mse_by_seed = {
        int(result["spec"]["seed"]): result
        for result in results.values()
        if result["method"] == "mse"
    }
    rows = []
    for name, result in results.items():
        if result["method"] == "mse":
            continue
        model_seed = int(result["spec"]["seed"])
        baseline = mse_by_seed[model_seed]
        baseline_ioi = _selected_row(
            baseline["duplicate_state"]["ioi_zero_curve"], selection.feature_count
        )
        candidate_ioi = _selected_row(
            result["duplicate_state"]["ioi_zero_curve"], selection.feature_count
        )
        baseline_natural = _selected_row(
            baseline["duplicate_state"]["natural_zero_curve"], selection.feature_count
        )
        candidate_natural = _selected_row(
            result["duplicate_state"]["natural_zero_curve"], selection.feature_count
        )
        candidate_exposure = _selected_row(
            result["duplicate_state"]["exposure_curve"], selection.feature_count
        )
        baseline_exposure = baseline["duplicate_state"]["exposure_curve"]
        rows.append(
            {
                "baseline": f"mse_s{model_seed}",
                "candidate": name,
                "method": result["method"],
                "seed": model_seed,
                "ioi_effect_difference": paired_bootstrap_summary(
                    torch.tensor(baseline_ioi["effect_by_example"]),
                    torch.tensor(candidate_ioi["effect_by_example"]),
                    seed=seed + model_seed,
                    bootstrap_samples=bootstrap_samples,
                ),
                "natural_kl_difference": paired_bootstrap_summary(
                    torch.tensor(baseline_natural["kl_by_sequence"]),
                    torch.tensor(candidate_natural["kl_by_sequence"]),
                    seed=seed + 100 + model_seed,
                    bootstrap_samples=bootstrap_samples,
                ),
                "continuous_target_r2_difference": (
                    result["continuous_target"]["test"]["r2"]
                    - baseline["continuous_target"]["test"]["r2"]
                ),
                "exposure_matched_kl": {
                    key: exposure_matched_comparison(
                        candidate_exposure,
                        baseline_exposure,
                        exposure_key=key,
                    )
                    for key in (
                        "summed_active_frequency",
                        "summed_activation_mass",
                        "summed_decoded_energy",
                    )
                },
            }
        )
    return rows


def analyze_test(config: dict, source: dict, device: torch.device) -> None:
    output = output_path(config)
    selection_results = read_json(output / "ioi_selection_models.json")
    result_path = output / "ioi_test_models.json"
    results = _partial_results(result_path)
    selection = _selection(config)
    calibration = torch.load(source_path(config) / "calibration.pt", map_location="cpu")
    cache = torch.load(output / "ioi_confirmatory_cache.pt", map_location="cpu")
    natural = _natural_cache(config, "test")
    lm = load_lm(source, device)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    examples = build_confirmatory_examples(config, source, lm.tokenizer)
    counts = config["ioi"]["feature_counts"]
    payloads = _model_payloads(config)
    for name, payload in payloads.items():
        if name in results:
            print(f"test already complete: {name}", flush=True)
            continue
        print(f"confirmatory test: {name}", flush=True)
        input_dim = calibration["activation_stats"]["mean"].numel()
        model = load_sae(payload, input_dim=input_dim, device=device)
        encoded = encode_confirmatory_states(model, cache)
        duplicate_ranking = torch.tensor(selection_results[name]["ranking"])
        duplicate_random = torch.tensor(
            selection_results[name]["matched_random_features"]
        )
        test_ioi_curve = zero_ablation_curve(
            lm,
            stats,
            model,
            examples["test"],
            duplicate_ranking,
            counts,
            original_logit_difference=cache["test"]["original_logit_difference"],
            batch_size=source["ioi"]["batch_size"],
        )
        test_natural_curve = natural_zero_ablation_curve(
            lm,
            stats,
            model,
            natural["input_ids"][: config["natural_text"]["matched_sequences"]],
            duplicate_ranking,
            counts,
            lag_distribution=cache["ranking"]["end_s2_lag"],
            batch_size=source["training"]["sequences_per_batch"],
            seed=config["ioi"]["seed"] + 1,
        )
        duplicate_patch = abc_patch_ceiling(
            lm,
            stats,
            model,
            examples["test"],
            cache["test"]["negative"],
            duplicate_ranking[: selection.feature_count],
            duplicate_random[: selection.feature_count],
            original_logit_difference=cache["test"]["original_logit_difference"],
            batch_size=source["ioi"]["batch_size"],
        )
        target = {
            split: cache[split]["original_logit_difference"].float()
            for split in ("ranking", "selection", "test")
        }
        continuous = continuous_target_protocol(
            discovery_codes=encoded["ranking"]["codes"][0],
            discovery_target=target["ranking"],
            validation_codes=encoded["selection"]["codes"][0],
            validation_target=target["selection"],
            test_codes=encoded["test"]["codes"][0],
            test_target=target["test"],
            feature_counts=counts,
            selection=selection,
        )
        target_ranking = torch.tensor(continuous["ranking"])
        firing = (torch.cat(encoded["ranking"]["codes"]) != 0).float().mean(0)
        target_random = matched_random_features(
            target_ranking[: max(counts)], firing
        )
        continuous_patch = abc_patch_ceiling(
            lm,
            stats,
            model,
            examples["test"],
            cache["test"]["negative"],
            target_ranking[: selection.feature_count],
            target_random[: selection.feature_count],
            original_logit_difference=target["test"],
            batch_size=source["ioi"]["batch_size"],
        )
        continuous_natural = natural_zero_ablation_curve(
            lm,
            stats,
            model,
            natural["input_ids"][: config["natural_text"]["matched_sequences"]],
            target_ranking,
            [selection.feature_count],
            lag_distribution=cache["ranking"]["end_s2_lag"],
            batch_size=source["training"]["sequences_per_batch"],
            seed=config["ioi"]["seed"] + 2,
        )[0]
        effect = {
            row["features"]: row["ioi_effect"] for row in test_ioi_curve
        }
        exposure = _exposure_curve(
            model,
            natural["activations"],
            duplicate_ranking,
            test_natural_curve,
            counts,
            effect,
        )
        results[name] = {
            "method": method_name(name),
            "spec": payload["spec"],
            "selection": selection.to_dict(),
            "duplicate_state": {
                "ioi_zero_curve": test_ioi_curve,
                "natural_zero_curve": test_natural_curve,
                "exposure_curve": exposure,
                "patch_ceiling": duplicate_patch,
            },
            "continuous_target": {
                **continuous,
                "original_dense_test": _dense_target_metric(
                    cache["ranking"]["positive"],
                    target["ranking"],
                    cache["test"]["positive"],
                    target["test"],
                ),
                "reconstruction_dense_test": _dense_target_metric(
                    encoded["ranking"]["reconstructions"][0],
                    target["ranking"],
                    encoded["test"]["reconstructions"][0],
                    target["test"],
                ),
                "patch_ceiling": continuous_patch,
                "matched_natural": continuous_natural,
            },
        }
        save_json(result_path, results)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(
        output / "ioi_confirmatory.json",
        {
            "protocol": torch.load(
                output / "ioi_confirmatory_cache.pt", map_location="cpu"
            )["protocol"],
            "feature_count_selection": read_json(
                output / "ioi_feature_count_selection.json"
            ),
            "selection_models": selection_results,
            "test_models": results,
            "paired_test_summary": paired_test_summary(
                results,
                selection,
                bootstrap_samples=config["ioi"]["bootstrap_samples"],
                seed=config["ioi"]["seed"],
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "selection", "test", "all")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "exp04b_confirmatory.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config, source = load_configs(args.config)
    output_path(config).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    stages = ("prepare", "selection", "test") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"=== IOI {stage} ===", flush=True)
        if stage == "prepare":
            prepare_cache(config, source, device)
        elif stage == "selection":
            analyze_selection(config, source, device)
        else:
            analyze_test(config, source, device)


if __name__ == "__main__":
    main()
