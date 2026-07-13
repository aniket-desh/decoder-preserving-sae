#!/usr/bin/env python3
"""Full GPT-2 small IOI mechanism experiment for decoder-preserving SAEs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from dpsae.corpus import MemmapTokenBatcher, TokenRange, prepare_token_memmap
from dpsae.decoder_distance import batched_ridge_predict, calibrate_ridge
from dpsae.language_model import (
    ActivationStats,
    GPT2ActivationModel,
    estimate_activation_stats,
)
from dpsae.language_training import SAETrainSpec, TrainingFleet, whitening_operator
from dpsae.mech_analysis import (
    analyze_model,
    build_examples,
    collect_state_activations,
    load_sae,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path, *, smoke: bool) -> dict:
    config = json.loads(path.read_text())
    if smoke:
        config["corpus"]["token_count"] = 200_000
        config["corpus"]["ranges"] = {
            "calibration": [0, 40_000],
            "screen": [40_000, 80_000],
            "confirmation": [80_000, 130_000],
            "robustness": [130_000, 170_000],
            "validation": [170_000, 200_000],
        }
        config["sae"]["dictionary_size"] = 256
        config["sae"]["primary_k"] = 8
        config["sae"]["aux_k"] = 32
        config["training"].update(
            screen_tokens=4_096,
            confirmation_tokens=4_096,
            robustness_tokens=4_096,
            checkpoint_tokens=2_048,
            confirmation_seeds=[0],
            robustness_seeds=[0],
            decoder_weight_multipliers=[0.25, 1.0],
            log_every_steps=1,
        )
        config["geometry"].update(calibration_tokens=2_048, validation_tokens=2_048)
        config["ioi"].update(
            discovery_examples=32,
            validation_examples=16,
            test_examples=32,
            causal_examples=16,
            batch_size=8,
            feature_counts=[1, 2, 4, 8],
            collateral_sequences=8,
        )
    return config


def paths(config: dict, *, smoke: bool) -> dict[str, Path]:
    suffix = "_smoke" if smoke else ""
    output = ROOT / "artifacts" / f"{config['experiment']}{suffix}"
    return {
        "output": output,
        "tokens": output / "fineweb_gpt2_tokens.bin",
        "calibration": output / "calibration.pt",
        "selection": output / "screening_selection.json",
    }


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


def load_partial_analysis(path: Path, expected_models: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"partial analysis must be an object: {path}")
    unexpected = set(value) - expected_models
    malformed = [name for name, result in value.items() if not isinstance(result, dict)]
    if unexpected or malformed:
        raise ValueError(
            f"invalid partial analysis {path}: unexpected={sorted(unexpected)}, "
            f"malformed={sorted(malformed)}"
        )
    return value


def make_batcher(
    config: dict,
    experiment_paths: dict[str, Path],
    range_name: str,
    *,
    seed: int,
) -> MemmapTokenBatcher:
    start, stop = config["corpus"]["ranges"][range_name]
    return MemmapTokenBatcher(
        experiment_paths["tokens"],
        token_count=config["corpus"]["token_count"],
        token_range=TokenRange(start, stop),
        sequence_length=config["training"]["sequence_length"],
        batch_size=config["training"]["sequences_per_batch"],
        seed=seed,
    )


def load_lm(config: dict, device: torch.device) -> GPT2ActivationModel:
    return GPT2ActivationModel.from_pretrained(
        config["model_name"], layer=config["layer"], device=device
    )


def prepare(config: dict, experiment_paths: dict[str, Path], device: torch.device) -> None:
    lm = load_lm(config, device)
    metadata = prepare_token_memmap(
        experiment_paths["tokens"],
        tokenizer=lm.tokenizer,
        token_count=config["corpus"]["token_count"],
        dataset_name=config["corpus"]["dataset_name"],
        dataset_config=config["corpus"]["dataset_config"],
        split=config["corpus"]["split"],
    )
    save_json(experiment_paths["output"] / "corpus.json", metadata)


@torch.inference_mode()
def calibrate(config: dict, experiment_paths: dict[str, Path], device: torch.device) -> None:
    if experiment_paths["calibration"].exists():
        print(f"calibration already exists: {experiment_paths['calibration']}", flush=True)
        return
    lm = load_lm(config, device)
    batcher = make_batcher(config, experiment_paths, "calibration", seed=config["seed"])
    target = config["geometry"]["calibration_tokens"]
    batches = math.ceil(
        target
        / (
            config["training"]["sequence_length"]
            * config["training"]["sequences_per_batch"]
        )
    )
    chunks = []
    for index in range(batches):
        activation = lm.activations(batcher.batch()).reshape(-1, lm.model.config.n_embd)
        chunks.append(activation.cpu())
        print(f"calibration activation batch {index + 1}/{batches}", flush=True)
    activations = torch.cat(chunks)[:target].to(device)
    stats = estimate_activation_stats(activations)
    normalized = stats.normalize(activations)
    whitening = whitening_operator(normalized)
    group_size = config["geometry"]["group_size"]
    ridge_values = []
    ridge_group_tokens = min(
        len(normalized),
        config["geometry"]["ridge_calibration_groups"] * group_size,
    )
    for group in normalized[:ridge_group_tokens].split(group_size):
        ridge_values.append(
            calibrate_ridge(group, config["geometry"]["ridge_dof_fraction"])
        )
    ridge = float(np.median(ridge_values))
    save_torch(
        experiment_paths["calibration"],
        {
            "activation_stats": stats.state_dict(),
            "whitening": whitening.cpu(),
            "ridge": ridge,
            "ridge_values": ridge_values,
            "model_name": config["model_name"],
            "layer": config["layer"],
        },
    )
    print(f"calibrated ridge={ridge:.8g}, scale={float(stats.scale):.6g}", flush=True)


def stage_specs(config: dict, stage: str) -> list[SAETrainSpec]:
    primary_k = config["sae"]["primary_k"]
    if stage == "screen":
        specs = [
            SAETrainSpec("mse_s0", "mse", 0, primary_k),
            SAETrainSpec("whitening_s0", "whitening", 0, primary_k),
        ]
        specs += [
            SAETrainSpec(
                f"dpsae_w{weight:g}_s0", "dpsae", 0, primary_k, decoder_weight=weight
            )
            for weight in config["training"]["decoder_weight_multipliers"]
        ]
        return specs
    if stage.startswith("robustness"):
        k = int(stage.removeprefix("robustness"))
        seeds = config["training"]["robustness_seeds"]
        return [
            spec
            for seed in seeds
            for spec in (
                SAETrainSpec(f"mse_k{k}_s{seed}", "mse", seed, k),
                SAETrainSpec(f"dpsae_k{k}_s{seed}", "dpsae", seed, k),
            )
        ]
    if stage == "confirmation":
        seeds = config["training"]["confirmation_seeds"]
        return [
            spec
            for seed in seeds
            for spec in (
                SAETrainSpec(f"mse_s{seed}", "mse", seed, primary_k),
                SAETrainSpec(f"dpsae_s{seed}", "dpsae", seed, primary_k),
                SAETrainSpec(f"whitening_s{seed}", "whitening", seed, primary_k),
            )
        ]
    raise ValueError(f"unknown stage: {stage}")


def _with_selected_decoder_weight(
    specs: list[SAETrainSpec], selection_path: Path
) -> list[SAETrainSpec]:
    selection = json.loads(selection_path.read_text())
    weight = float(selection["selected_decoder_weight"])
    return [
        SAETrainSpec(spec.name, spec.method, spec.seed, spec.k, weight)
        if spec.method == "dpsae" and spec.decoder_weight == 0
        else spec
        for spec in specs
    ]


@torch.inference_mode()
def validation_activations(
    config: dict,
    experiment_paths: dict[str, Path],
    lm: GPT2ActivationModel,
    stats: ActivationStats,
) -> Tensor:
    batcher = make_batcher(config, experiment_paths, "validation", seed=config["seed"] + 99)
    target = config["geometry"]["validation_tokens"]
    chunks = []
    while sum(len(chunk) for chunk in chunks) < target:
        activation = lm.activations(batcher.batch()).reshape(-1, lm.model.config.n_embd)
        chunks.append(stats.normalize(activation).cpu())
    return torch.cat(chunks)[:target]


@torch.inference_mode()
def evaluate_fleet(
    fleet: TrainingFleet,
    activations: Tensor,
    *,
    ridge: float,
    group_size: int,
    probes: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    device = fleet.device
    x = activations.to(device)
    x = x[: (len(x) // group_size) * group_size]
    groups = len(x) // group_size
    generator = torch.Generator(device=device).manual_seed(seed)
    targets = torch.randn(groups, group_size, probes, generator=generator, device=device)
    reference = batched_ridge_predict(x.reshape(groups, group_size, -1), targets, ridge)
    denominator = reference.square().sum().clamp_min(1e-12)
    result = {}
    for spec in fleet.specs:
        model = fleet.models[spec.name].eval()
        reconstruction, code = model(x, use_threshold=True)
        prediction = batched_ridge_predict(
            reconstruction.float().reshape(groups, group_size, -1), targets, ridge
        )
        result[spec.name] = {
            "nmse": float((reconstruction.float() - x).square().sum() / x.square().sum()),
            "decoder": float((prediction - reference).square().sum() / denominator),
            "l0": float((code != 0).sum(dim=1).float().mean()),
            "dead": int((model.last_active_step == 0).sum()),
        }
    return result


def train_stage(
    config: dict,
    experiment_paths: dict[str, Path],
    device: torch.device,
    stage: str,
) -> None:
    output = experiment_paths["output"] / stage
    output.mkdir(parents=True, exist_ok=True)
    done_path = output / "done.json"
    if done_path.exists():
        print(f"stage already complete: {stage}", flush=True)
        return
    calibration_state = torch.load(experiment_paths["calibration"], map_location="cpu")
    lm = load_lm(config, device)
    stats = ActivationStats.from_state_dict(calibration_state["activation_stats"], device)
    whitening = calibration_state["whitening"].to(device)
    ridge = float(calibration_state["ridge"])
    specs = stage_specs(config, stage)
    if stage != "screen":
        specs = _with_selected_decoder_weight(specs, experiment_paths["selection"])
    fleet = TrainingFleet(
        specs,
        input_dim=lm.model.config.n_embd,
        dictionary_size=config["sae"]["dictionary_size"],
        learning_rate=config["sae"]["learning_rate"],
        device=device,
        whitening=whitening,
        aux_weight=config["sae"]["aux_weight"],
        dead_after_steps=config["sae"]["dead_after_steps"],
        aux_k=config["sae"]["aux_k"],
    )
    range_name = {
        "screen": "screen",
        "confirmation": "confirmation",
        "robustness16": "robustness",
        "robustness64": "robustness",
    }[stage]
    batcher = make_batcher(config, experiment_paths, range_name, seed=config["seed"])
    token_budget = {
        "screen": config["training"]["screen_tokens"],
        "confirmation": config["training"]["confirmation_tokens"],
        "robustness16": config["training"]["robustness_tokens"],
        "robustness64": config["training"]["robustness_tokens"],
    }[stage]
    tokens_per_step = (
        config["training"]["sequence_length"]
        * config["training"]["sequences_per_batch"]
    )
    total_steps = math.ceil(token_budget / tokens_per_step)
    checkpoint_every = max(1, config["training"]["checkpoint_tokens"] // tokens_per_step)
    checkpoint_path = output / "checkpoint.pt"
    start_step, tokens_seen = 0, 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        start_step, tokens_seen = fleet.load_state_dict(state)
        batcher.load_generator_state(state["batcher_generator_state"])
        print(f"resuming {stage} at step {start_step:,}", flush=True)
    metrics_path = output / "training.jsonl"
    started = time.monotonic()
    for step in range(start_step, total_steps):
        progress = (step + 1) / total_steps
        warmup = config["sae"]["warmup_fraction"]
        if progress < warmup:
            lr_scale = progress / warmup
        else:
            lr_scale = 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))
        learning_rate = config["sae"]["learning_rate"] * lr_scale
        for optimizer in fleet.optimizers.values():
            optimizer.param_groups[0]["lr"] = learning_rate
        ids = batcher.batch()
        activation = lm.activations(ids).reshape(-1, lm.model.config.n_embd)
        activation = stats.normalize(activation)
        metrics = fleet.train_batch(
            activation,
            step=step + 1,
            ridge=ridge,
            group_size=config["geometry"]["group_size"],
            probes=config["geometry"]["probes"],
            probe_seed=config["seed"] + step,
        )
        tokens_seen += len(activation)
        if step % config["training"]["log_every_steps"] == 0 or step + 1 == total_steps:
            record = {
                "step": step + 1,
                "tokens_seen": tokens_seen,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.monotonic() - started,
                "models": metrics,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            summary = " ".join(
                f"{name}:nmse={value['nmse']:.4f},dec={value['decoder']:.4f}"
                for name, value in metrics.items()
            )
            print(f"{stage} {step + 1}/{total_steps} {summary}", flush=True)
        if (step + 1) % checkpoint_every == 0 or step + 1 == total_steps:
            state = fleet.state_dict(step=step + 1, tokens_seen=tokens_seen)
            state["batcher_generator_state"] = batcher.generator.get_state()
            save_torch(checkpoint_path, state)
    validation = validation_activations(config, experiment_paths, lm, stats)
    evaluation = evaluate_fleet(
        fleet,
        validation,
        ridge=ridge,
        group_size=config["geometry"]["group_size"],
        probes=config["geometry"]["probes"],
        seed=config["seed"] + 1_000_000,
    )
    save_json(output / "validation.json", evaluation)
    save_torch(output / "models.pt", fleet.export_models())
    save_json(done_path, {"stage": stage, "tokens_seen": tokens_seen, "validation": evaluation})
    if stage == "screen":
        baseline_nmse = evaluation["mse_s0"]["nmse"]
        candidates = [
            (name, value)
            for name, value in evaluation.items()
            if name.startswith("dpsae_") and value["nmse"] <= 1.10 * baseline_nmse
        ]
        if not candidates:
            candidates = [
                (name, value) for name, value in evaluation.items() if name.startswith("dpsae_")
            ]
        selected_name, selected_metrics = min(candidates, key=lambda item: item[1]["decoder"])
        selected_spec = next(spec for spec in specs if spec.name == selected_name)
        save_json(
            experiment_paths["selection"],
            {
                "selected_model": selected_name,
                "selected_decoder_weight": selected_spec.decoder_weight,
                "selection_rule": "minimum validation decoder distortion under 1.10x MSE NMSE",
                "metrics": selected_metrics,
                "all_validation": evaluation,
            },
        )


def analyze(
    config: dict, experiment_paths: dict[str, Path], device: torch.device
) -> None:
    analysis_path = experiment_paths["output"] / "analysis.json"
    if analysis_path.exists():
        print(f"analysis already exists: {analysis_path}", flush=True)
        return
    calibration_state = torch.load(experiment_paths["calibration"], map_location="cpu")
    lm = load_lm(config, device)
    stats = ActivationStats.from_state_dict(calibration_state["activation_stats"], device)
    examples = build_examples(config, lm.tokenizer)
    state_cache_path = experiment_paths["output"] / "ioi_state_activations.pt"
    if state_cache_path.exists():
        state_data = torch.load(state_cache_path, map_location="cpu")
    else:
        state_data = collect_state_activations(
            lm,
            stats,
            examples,
            batch_size=config["ioi"]["batch_size"],
        )
        save_torch(state_cache_path, state_data)

    result = {"protocol": {"model": config["model_name"], "layer": config["layer"]}}
    for stage in ("confirmation", "robustness16", "robustness64"):
        model_path = experiment_paths["output"] / stage / "models.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"missing trained models for {stage}: {model_path}")
        payloads = torch.load(model_path, map_location="cpu")
        stage_path = experiment_paths["output"] / f"analysis_{stage}.json"
        stage_result = load_partial_analysis(stage_path, set(payloads))
        for name, payload in payloads.items():
            if name in stage_result:
                print(f"analysis already complete: {stage}/{name}", flush=True)
                continue
            print(f"analyzing {stage}/{name}", flush=True)
            model = load_sae(payload, input_dim=lm.model.config.n_embd, device=device)
            stage_result[name] = analyze_model(
                config,
                lm,
                stats,
                model,
                state_data,
                examples,
                token_path=experiment_paths["tokens"],
                causal=stage == "confirmation",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            save_json(stage_path, stage_result)
        result[stage] = stage_result
    save_json(analysis_path, result)


def plot_results(config: dict, experiment_paths: dict[str, Path]) -> None:
    import matplotlib.pyplot as plt

    from dpsae.plot_style import COLORS, apply_paper_style, clean_axis, savefig

    analysis = json.loads((experiment_paths["output"] / "analysis.json").read_text())
    validation = json.loads(
        (experiment_paths["output"] / "confirmation" / "validation.json").read_text()
    )
    apply_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.0))
    method_style = {
        "mse": (COLORS["mse"], "MSE"),
        "dpsae": (COLORS["isotropic"], "Isotropic DPSAE"),
        "whitening": (COLORS["whitened"], "Whitening"),
    }

    for method, (color, label) in method_style.items():
        names = [name for name in validation if name.startswith(method)]
        axes[0].scatter(
            [validation[name]["nmse"] for name in names],
            [validation[name]["decoder"] for name in names],
            color=color,
            label=label,
        )
    axes[0].set_title("Natural-text validation")
    axes[0].set_xlabel("Reconstruction NMSE")
    axes[0].set_ylabel("Decoder distortion")

    confirmation = analysis["confirmation"]
    for method, (color, label) in method_style.items():
        names = [name for name in confirmation if name.startswith(method)]
        curves = []
        for name in names:
            curve = confirmation[name]["sparse_probe_curve"]
            x = [row["features"] for row in curve]
            y = [row["accuracy"] for row in curve]
            curves.append(y)
            axes[1].plot(x, y, color=color, alpha=0.22, linewidth=0.8)
        median = np.median(np.asarray(curves), axis=0)
        axes[1].plot(x, median, color=color, label=label, linewidth=2)
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Held-out duplicate-state decoding")
    axes[1].set_xlabel("Frozen sparse features")
    axes[1].set_ylabel("Test accuracy")

    positions = np.arange(3)
    for offset, (method, (color, label)) in enumerate(method_style.items()):
        names = [name for name in confirmation if name.startswith(method)]
        values = [confirmation[name]["features_to_80pct_dense"] for name in names]
        values = [np.nan if value is None else value for value in values]
        axes[2].scatter(
            np.full(len(values), offset), values, color=color, alpha=0.8, label=label
        )
    axes[2].set_xticks(positions, ["MSE", "DPSAE", "White"], rotation=20)
    axes[2].set_yscale("log", base=2)
    axes[2].set_title("Feature concentration")
    axes[2].set_ylabel("Features to 80% dense probe")

    for method, (color, label) in method_style.items():
        names = [name for name in confirmation if name.startswith(method)]
        for name in names:
            causal = confirmation[name]["causal_frontier"]
            collateral = confirmation[name]["collateral_frontier"]
            axes[3].plot(
                [row["collateral_kl"] for row in collateral],
                [row["abc_patch_effect"] for row in causal],
                color=color,
                alpha=0.22,
                linewidth=0.8,
            )
        x_values = np.median(
            [[row["collateral_kl"] for row in confirmation[name]["collateral_frontier"]]
             for name in names],
            axis=0,
        )
        y_values = np.median(
            [[row["abc_patch_effect"] for row in confirmation[name]["causal_frontier"]]
             for name in names],
            axis=0,
        )
        axes[3].plot(x_values, y_values, color=color, label=label, linewidth=2)
    axes[3].set_title("Causal specificity frontier")
    axes[3].set_xlabel("Collateral KL on natural text")
    axes[3].set_ylabel("ABC patch effect on IOI logit diff.")

    for axis in axes:
        clean_axis(axis)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.78, wspace=0.38)
    savefig(fig, experiment_paths["output"] / "figures" / "exp04_headline")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "calibrate",
            "screen",
            "confirmation",
            "robustness16",
            "robustness64",
            "analyze",
            "plot",
            "all",
        ],
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "exp04_ioi_mechanism.json"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config, smoke=args.smoke)
    experiment_paths = paths(config, smoke=args.smoke)
    experiment_paths["output"].mkdir(parents=True, exist_ok=True)
    save_json(experiment_paths["output"] / "resolved_config.json", config)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    stages = (
        [
            "prepare",
            "calibrate",
            "screen",
            "confirmation",
            "robustness16",
            "robustness64",
            "analyze",
            "plot",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        print(f"=== {stage} ===", flush=True)
        if stage == "prepare":
            prepare(config, experiment_paths, device)
        elif stage == "calibrate":
            calibrate(config, experiment_paths, device)
        elif stage in {"screen", "confirmation", "robustness16", "robustness64"}:
            train_stage(config, experiment_paths, device, stage)
        elif stage == "analyze":
            analyze(config, experiment_paths, device)
        elif stage == "plot":
            plot_results(config, experiment_paths)
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    main()
