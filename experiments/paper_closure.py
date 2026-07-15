#!/usr/bin/env python3
"""Evaluation and screening stages that close the remaining paper TODOs.

The initial stage evaluates the already-trained Experiment 4 screen fleet on
the untouched Experiment 4b natural-text test cache.  It deliberately trains
nothing: the goal is to decide which decoder weights, if any, deserve a new
100M-token confirmation run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from dpsae.exp04b_natural_text import (
    bootstrap_paired_reduction_interval,
    exact_decoder_sweep,
)
from dpsae.corpus import MemmapTokenBatcher, TokenRange
from dpsae.language_model import ActivationStats, GPT2ActivationModel
from dpsae.language_training import SAETrainSpec, TrainingFleet
from dpsae.mech_analysis import load_sae


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MODELS = ROOT / "artifacts/exp04_ioi_mechanism/screen/models.pt"
DEFAULT_CACHE = ROOT / "artifacts/exp04b_confirmatory/natural_selection.pt"
DEFAULT_STATIC = ROOT / "artifacts/exp04b_confirmatory/static_calibration.pt"
DEFAULT_CONFIG = ROOT / "configs/exp04b_confirmatory.json"
DEFAULT_OUTPUT = ROOT / "artifacts/paper_closure/frontier_selection.json"
DEFAULT_SOURCE_TOKENS = ROOT / "artifacts/exp04_ioi_mechanism/fineweb_gpt2_tokens.bin"
DEFAULT_SOURCE_CALIBRATION = ROOT / "artifacts/exp04_ioi_mechanism/calibration.pt"
DEFAULT_NEW_SCREEN = ROOT / "artifacts/paper_closure/frontier_screen_common"
DEFAULT_FRONTIER_INPUT = ROOT / "artifacts/paper_closure/frontier_common_selection.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def repository_state(root: Path = ROOT) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, status = "unknown", []
    return {"revision": revision, "dirty": bool(status), "status": status}


def _input_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _ensure_resources(
    output: Path,
    *,
    minimum_free_gib: float,
    device: torch.device,
    gpu_memory_fraction: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output.parent).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f} GiB"
        )
    if device.type == "cuda":
        if not 0 < gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must lie in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "device": str(device),
        "free_gib_at_start": free_gib,
        "minimum_free_gib_guard": minimum_free_gib,
        "gpu_memory_fraction_cap": gpu_memory_fraction if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
    }


@torch.inference_mode()
def reconstruct(model, activations: Tensor, *, batch_tokens: int = 4096) -> Tensor:
    chunks = []
    flat = activations.flatten(0, 1)
    for batch in flat.split(batch_tokens):
        reconstruction, _ = model(batch.float(), use_threshold=True)
        chunks.append(reconstruction)
    return torch.cat(chunks).reshape_as(activations)


def _screen_models(
    payloads: Mapping[str, Mapping[str, Any]], *, evaluation_seed: int | None = None
) -> list[str]:
    names = []
    for name, payload in payloads.items():
        spec = payload.get("spec", {})
        if evaluation_seed is not None and int(spec.get("seed", -1)) != evaluation_seed:
            continue
        if spec.get("method") == "mse" or name.startswith("dpsae_w"):
            names.append(name)
    mse = [name for name in names if payloads[name]["spec"].get("method") == "mse"]
    if len(mse) != 1:
        raise ValueError("screen fleet must contain exactly one MSE baseline")
    if not any(name.startswith("dpsae_w") for name in names):
        raise ValueError("screen fleet contains no decoder-weight candidates")
    return sorted(names, key=lambda name: (name != mse[0], name))


def _learning_rate(source: Mapping[str, Any], step: int, total_steps: int) -> float:
    progress = step / total_steps
    warmup = float(source["sae"]["warmup_fraction"])
    if progress < warmup:
        scale = progress / warmup
    else:
        scale = 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))
    return float(source["sae"]["learning_rate"]) * scale


def _frontier_specs(weights: list[float], seeds: list[int]) -> list[SAETrainSpec]:
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("decoder weights must be nonempty and unique")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("decoder weights must be finite and positive")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty, unique, and nonnegative")
    specs = []
    for seed in seeds:
        specs.append(SAETrainSpec(f"mse_s{seed}", "mse", seed, 32))
        specs.extend(
            SAETrainSpec(
                f"dpsae_w{weight:g}_s{seed}",
                "dpsae",
                seed,
                32,
                decoder_weight=weight,
            )
            for weight in weights
        )
    return specs


def _checkpoint_specs_match(state: Mapping[str, Any], specs: list[SAETrainSpec]) -> None:
    if state.get("specs") != [asdict(spec) for spec in specs]:
        raise RuntimeError("checkpoint specs differ from the frozen frontier screen")


def _checkpoint_contract_match(
    state: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    if "run_contract" not in state:
        raise RuntimeError("checkpoint predates the immutable run contract")
    if state["run_contract"] != contract:
        raise RuntimeError("checkpoint run contract differs from this invocation")


def _trim_jsonl(path: Path, maximum_step: int) -> None:
    if not path.exists():
        return
    records = [
        line
        for line in path.read_text().splitlines()
        if int(json.loads(line)["step"]) <= maximum_step
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in records))
    temporary.replace(path)


def frontier_train_screen(args: argparse.Namespace) -> None:
    """Train only the missing small-gamma candidates on the original screen stream."""

    started_at = time.time()
    device = torch.device(args.device)
    resources = _ensure_resources(
        args.new_screen,
        minimum_free_gib=args.minimum_free_gib,
        device=device,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )
    config = json.loads(args.config.read_text())
    source = json.loads((ROOT / config["source_config"]).read_text())
    specs = _frontier_specs(
        [float(value) for value in args.decoder_weights],
        [int(value) for value in args.seeds],
    )
    if any(spec.k != int(source["sae"]["primary_k"]) for spec in specs):
        raise ValueError("frontier specs must use the frozen primary k")
    calibration = torch.load(
        args.source_calibration, map_location="cpu", weights_only=False
    )
    lm = GPT2ActivationModel.from_pretrained(
        source["model_name"], layer=int(source["layer"]), device=device
    )
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    method_threshold_multipliers = None
    supplied_method_multipliers = (
        args.jump_relu_threshold_lr_multiplier_mse,
        args.jump_relu_threshold_lr_multiplier_dpsae,
    )
    if any(value is not None for value in supplied_method_multipliers):
        if args.sparsity_mode != "jump_relu" or any(
            value is None for value in supplied_method_multipliers
        ):
            raise ValueError(
                "method-specific threshold multipliers require JumpReLU and both methods"
            )
        method_threshold_multipliers = {
            "mse": args.jump_relu_threshold_lr_multiplier_mse,
            "dpsae": args.jump_relu_threshold_lr_multiplier_dpsae,
        }
    fleet = TrainingFleet(
        specs,
        input_dim=int(lm.model.config.n_embd),
        dictionary_size=int(source["sae"]["dictionary_size"]),
        learning_rate=float(source["sae"]["learning_rate"]),
        device=device,
        aux_weight=float(source["sae"]["aux_weight"]),
        dead_after_steps=int(source["sae"]["dead_after_steps"]),
        aux_k=int(source["sae"]["aux_k"]),
        sparsity_mode=args.sparsity_mode,
        jump_relu_init_threshold=args.jump_relu_init_threshold,
        jump_relu_init_mode=args.jump_relu_init_mode,
        jump_relu_bandwidth=args.jump_relu_bandwidth,
        jump_relu_sparsity_weight=args.jump_relu_sparsity_weight,
        jump_relu_threshold_lr_multiplier=args.jump_relu_threshold_lr_multiplier,
        jump_relu_threshold_lr_multipliers_by_method=method_threshold_multipliers,
    )
    source_range = TokenRange(
        *source["corpus"]["ranges"][args.source_range_name]
    )
    data_seed = int(source["seed"]) if args.data_seed < 0 else int(args.data_seed)
    probe_seed_base = (
        int(source["seed"]) if args.probe_seed_base < 0 else int(args.probe_seed_base)
    )
    batcher = MemmapTokenBatcher(
        args.source_tokens,
        token_count=int(source["corpus"]["token_count"]),
        token_range=source_range,
        sequence_length=int(source["training"]["sequence_length"]),
        batch_size=int(source["training"]["sequences_per_batch"]),
        seed=data_seed,
    )
    token_budget = (
        int(source["training"]["screen_tokens"])
        if args.token_budget <= 0
        else int(args.token_budget)
    )
    tokens_per_step = int(source["training"]["sequence_length"]) * int(
        source["training"]["sequences_per_batch"]
    )
    total_steps = math.ceil(token_budget / tokens_per_step)
    checkpoint_every = max(
        1, int(source["training"]["checkpoint_tokens"]) // tokens_per_step
    )
    args.new_screen.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "repository": repository_state(),
        "inputs": {
            "source_tokens": _input_record(args.source_tokens),
            "source_calibration": _input_record(args.source_calibration),
            "config": _input_record(args.config),
            "source_config": _input_record(ROOT / config["source_config"]),
            "trainer": _input_record(Path(__file__)),
        },
        "specs": [asdict(spec) for spec in specs],
        "sparsity_mode": args.sparsity_mode,
        "sparsity_parameters": {
            "jump_relu_init_threshold": args.jump_relu_init_threshold,
            "jump_relu_init_mode": args.jump_relu_init_mode,
            "jump_relu_bandwidth": args.jump_relu_bandwidth,
            "jump_relu_sparsity_weight": args.jump_relu_sparsity_weight,
            "jump_relu_threshold_lr_multiplier": args.jump_relu_threshold_lr_multiplier,
            "jump_relu_threshold_lr_multiplier_mse": args.jump_relu_threshold_lr_multiplier_mse,
            "jump_relu_threshold_lr_multiplier_dpsae": args.jump_relu_threshold_lr_multiplier_dpsae,
        },
        "stream": {
            "range_name": args.source_range_name,
            "range": [source_range.start, source_range.stop],
            "data_seed": data_seed,
            "probe_seed_base": probe_seed_base,
            "token_budget": token_budget,
            "tokens_per_step": tokens_per_step,
            "total_steps": total_steps,
        },
    }
    checkpoint = args.new_screen / "checkpoint.pt"
    start_step, tokens_seen = 0, 0
    prior_training_seconds = 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        _checkpoint_specs_match(state, specs)
        _checkpoint_contract_match(state, run_contract)
        start_step, tokens_seen = fleet.load_state_dict(state)
        batcher.load_generator_state(state["batcher_generator_state"])
        prior_training_seconds = float(state.get("cumulative_training_seconds", 0.0))
    log_path = args.new_screen / "training.jsonl"
    _trim_jsonl(log_path, start_step)
    target_steps = total_steps
    if args.stop_after_steps > 0:
        target_steps = min(total_steps, start_step + args.stop_after_steps)
    started_training = time.monotonic()
    ridge = float(calibration["ridge"])
    for zero_step in range(start_step, target_steps):
        step = zero_step + 1
        learning_rate = _learning_rate(source, step, total_steps)
        fleet.set_learning_rate(learning_rate)
        ids = batcher.batch()
        activations = stats.normalize(lm.activations(ids)).flatten(0, 1)
        metrics = fleet.train_batch(
            activations,
            step=step,
            ridge=ridge,
            group_size=int(source["geometry"]["group_size"]),
            probes=int(source["geometry"]["probes"]),
            probe_seed=probe_seed_base + zero_step,
        )
        tokens_seen += len(activations)
        if zero_step % int(source["training"]["log_every_steps"]) == 0 or step == target_steps:
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.monotonic() - started_training,
                "models": metrics,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"frontier screen {step:,}/{total_steps:,}", flush=True)
        if step % checkpoint_every == 0 or step == target_steps:
            state = fleet.state_dict(step=step, tokens_seen=tokens_seen)
            state["batcher_generator_state"] = batcher.generator.get_state()
            state["run_contract"] = run_contract
            state["cumulative_training_seconds"] = (
                prior_training_seconds + time.monotonic() - started_training
            )
            atomic_torch(checkpoint, state)
    if device.type == "cuda":
        resources["peak_allocated_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        if resources["peak_allocated_gpu_gib"] > args.maximum_peak_gpu_gib:
            raise RuntimeError("frontier screen exceeded the peak GPU allocation guard")
    cumulative_training_seconds = prior_training_seconds + time.monotonic() - started_training
    common = {
        "repository": repository_state(),
        "inputs": run_contract["inputs"],
        "run_contract": run_contract,
        "specs": [asdict(spec) for spec in specs],
        "sparsity_mode": args.sparsity_mode,
        "stream": {
            "range_name": args.source_range_name,
            "range": [source_range.start, source_range.stop],
            "data_seed": data_seed,
            "probe_seed_base": probe_seed_base,
            "probe_seed_formula": "probe_seed_base + zero_based_step",
            "total_steps": total_steps,
        },
        "tokens_seen": tokens_seen,
        "resources": resources,
        "training_seconds_cumulative": cumulative_training_seconds,
        "training_seconds_this_invocation": time.monotonic() - started_training,
        "wall_seconds": time.time() - started_at,
    }
    if args.sparsity_mode == "jump_relu":
        common["sparsity_config"] = fleet.sparsity_config()
        common["realized_thresholds"] = fleet.jump_threshold_summary()
    if target_steps < total_steps:
        atomic_json(
            args.new_screen / "partial.json",
            {"complete": False, "step": target_steps, **common},
        )
        print(json.dumps({"complete": False, "step": target_steps}, indent=2))
        return
    atomic_torch(args.new_screen / "models.pt", fleet.export_models())
    atomic_json(
        args.new_screen / "done.json",
        {"complete": True, "step": total_steps, **common},
    )
    print(json.dumps({"complete": True, "models": str(args.new_screen / 'models.pt')}, indent=2))


def _primary_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["model"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("expected one exact row per model")
    return result


def paired_frontier(
    models: Mapping[str, Mapping[str, Any]],
    exact_rows: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    mse_names = [name for name, value in models.items() if value["method"] == "mse"]
    if len(mse_names) != 1:
        raise ValueError("models must contain exactly one MSE baseline")
    baseline_name = mse_names[0]
    baseline = models[baseline_name]
    baseline_numerator = torch.tensor(
        exact_rows[baseline_name]["numerator_by_group"], dtype=torch.float64
    )
    result = []
    for name, value in models.items():
        if value["method"] != "dpsae":
            continue
        candidate_numerator = torch.tensor(
            exact_rows[name]["numerator_by_group"], dtype=torch.float64
        )
        interval = bootstrap_paired_reduction_interval(
            baseline_numerator,
            candidate_numerator,
            samples=bootstrap_samples,
            seed=seed,
        )
        nmse_ratio = value["nmse"] / baseline["nmse"]
        result.append(
            {
                "baseline": baseline_name,
                "candidate": name,
                "decoder_weight": value["decoder_weight"],
                "nmse_ratio_to_mse": nmse_ratio,
                "nmse_change_percent": 100 * (nmse_ratio - 1),
                "exact_decoder_reduction": interval["estimate"],
                "exact_decoder_reduction_ci95": [interval["low"], interval["high"]],
                "strictly_dominates_mse": bool(
                    value["nmse"] <= baseline["nmse"] and interval["low"] > 0
                ),
            }
        )
    return sorted(result, key=lambda row: row["decoder_weight"])


def select_frontier_candidate(
    rows: list[Mapping[str, Any]], rule: Mapping[str, Any]
) -> Mapping[str, Any]:
    maximum_nmse = float(rule["maximum_nmse_ratio"])
    minimum_reduction = float(rule["minimum_exact_decoder_reduction"])
    order = list(rule.get("selection_order", ()))
    if order != ["smaller_decoder_weight", "lower_nmse"]:
        raise ValueError("frontier selection order must be frozen weight-first")
    qualifying = [
        row
        for row in rows
        if float(row["nmse_ratio_to_mse"]) <= maximum_nmse
        and float(row["exact_decoder_reduction"]) >= minimum_reduction
    ]
    if not qualifying:
        raise RuntimeError("no frontier candidate passes the frozen selection gate")
    return min(
        qualifying,
        key=lambda row: (
            float(row["decoder_weight"]),
            float(row["nmse_ratio_to_mse"]),
        ),
    )


def frontier_select(args: argparse.Namespace) -> None:
    frontier = json.loads(args.frontier_input.read_text())
    config = json.loads(args.config.read_text())
    selected = select_frontier_candidate(
        frontier["paired_frontier"], config["frontier"]["selection_rule"]
    )
    result = {
        "complete": True,
        "experiment": "paper_closure_frontier_selection",
        "selected_decoder_weight": selected["decoder_weight"],
        "selected_candidate": selected["candidate"],
        "selected_metrics": selected,
        "selection_rule": config["frontier"]["selection_rule"],
        "inputs": {
            "frontier": _input_record(args.frontier_input),
            "config": _input_record(args.config),
            "selector": _input_record(Path(__file__)),
        },
        "repository": repository_state(),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


@torch.inference_mode()
def frontier_existing(args: argparse.Namespace) -> None:
    started_at = time.time()
    device = torch.device(args.device)
    resources = _ensure_resources(
        args.output,
        minimum_free_gib=args.minimum_free_gib,
        device=device,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )
    config = json.loads(args.config.read_text())
    natural = config["natural_text"]
    source = json.loads((ROOT / config["source_config"]).read_text())
    payloads = torch.load(args.source_models, map_location="cpu", weights_only=False)
    if args.candidate_models is not None:
        candidates = torch.load(
            args.candidate_models, map_location="cpu", weights_only=False
        )
        overlap = set(payloads) & set(candidates)
        if overlap:
            raise ValueError(f"candidate models overlap source fleet: {sorted(overlap)}")
        payloads.update(candidates)
    model_names = _screen_models(payloads, evaluation_seed=args.evaluation_seed)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    static = torch.load(args.static, map_location="cpu", weights_only=False)

    activation_tokens = int(natural["activation_tokens"])
    exact_tokens = int(natural["exact_tokens"])
    if cache["activations"].numel() == 0 or cache["input_ids"].numel() == 0:
        raise ValueError("natural test cache is empty")
    if cache["activations"].shape[:2] != cache["input_ids"].shape:
        raise ValueError("activation and token cache shapes disagree")
    if cache["input_ids"].numel() != activation_tokens:
        raise ValueError("natural cache token count differs from the frozen config")
    if exact_tokens > activation_tokens or exact_tokens % cache["input_ids"].shape[1]:
        raise ValueError("exact token budget must select complete cached sequences")

    activations = cache["activations"].to(device).float()
    exact_sequences = exact_tokens // activations.shape[1]
    reconstructions: dict[str, Tensor] = {}
    model_reports: dict[str, dict[str, Any]] = {}
    original_energy = activations.square().sum().clamp_min(1e-12)
    for name in model_names:
        print(f"evaluating {name}", flush=True)
        payload = payloads[name]
        model = load_sae(payload, input_dim=activations.shape[-1], device=device).eval()
        reconstruction = reconstruct(model, activations)
        nmse = float((reconstruction - activations).square().sum() / original_energy)
        spec = dict(payload["spec"])
        model_reports[name] = {
            "method": spec["method"],
            "seed": int(spec["seed"]),
            "decoder_weight": float(spec.get("decoder_weight", 0.0)),
            "nmse": nmse,
            "l0_inference": float(
                (model.encode(activations[:1].flatten(0, 1), use_threshold=True) != 0)
                .sum(1)
                .float()
                .mean()
            ),
        }
        reconstructions[name] = reconstruction[:exact_sequences]
        del model, reconstruction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    exact_rows_list = exact_decoder_sweep(
        activations[:exact_sequences],
        reconstructions,
        cache["input_ids"][:exact_sequences],
        ridges=[float(static["ridge"])],
        group_sizes=[int(source["geometry"]["group_size"])],
        groupings=["contiguous"],
        eos_token_id=int(cache["eos_token_id"]),
        max_groups=int(natural["exact_max_groups"]),
        bootstrap_samples=int(natural["bootstrap_samples"]),
        seed=int(natural["test_seed"]),
    )
    exact_rows = _primary_rows(exact_rows_list)
    for name, row in exact_rows.items():
        model_reports[name]["exact_decoder_distortion"] = row["decoder_distortion"]
        model_reports[name]["exact_decoder_ci95"] = [row["ci_low"], row["ci_high"]]

    paired = paired_frontier(
        model_reports,
        exact_rows,
        bootstrap_samples=int(natural["bootstrap_samples"]),
        seed=int(natural["test_seed"]),
    )
    if device.type == "cuda":
        resources["peak_allocated_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        if resources["peak_allocated_gpu_gib"] > args.maximum_peak_gpu_gib:
            raise RuntimeError(
                f"peak allocation {resources['peak_allocated_gpu_gib']:.2f} GiB exceeded "
                f"guard {args.maximum_peak_gpu_gib:.2f} GiB"
            )
    finished_at = time.time()
    result = {
        "complete": True,
        "experiment": "paper_closure_frontier_existing",
        "purpose": "exact selection audit of reusable 25M-token gamma screen",
        "repository": repository_state(),
        "inputs": {
            "source_models": _input_record(args.source_models),
            "candidate_models": (
                None if args.candidate_models is None else _input_record(args.candidate_models)
            ),
            "natural_test_cache": _input_record(args.cache),
            "static_calibration": _input_record(args.static),
            "config": _input_record(args.config),
            "evaluator": _input_record(Path(__file__)),
        },
        "protocol": {
            "split": args.split_label,
            "evaluation_seed": args.evaluation_seed,
            "activation_tokens": activation_tokens,
            "exact_tokens": exact_tokens,
            "group_size": int(source["geometry"]["group_size"]),
            "ridge": float(static["ridge"]),
            "bootstrap_samples": int(natural["bootstrap_samples"]),
            "seed": int(natural["test_seed"]),
            "promotion_rule": "advance only Pareto-relevant weights to fresh 100M confirmation",
        },
        "models": model_reports,
        "paired_frontier": paired,
        "exact_rows": exact_rows_list,
        "resources": resources,
        "started_unix": started_at,
        "finished_unix": finished_at,
        "wall_seconds": finished_at - started_at,
    }
    atomic_json(args.output, result)
    print(json.dumps({"complete": True, "output": str(args.output), "paired": paired}, indent=2))


def self_test() -> None:
    models = {
        "mse_s0": {"method": "mse", "nmse": 1.0, "decoder_weight": 0.0},
        "dpsae_w0.1_s0": {"method": "dpsae", "nmse": 0.9, "decoder_weight": 0.1},
        "dpsae_w0.2_s0": {"method": "dpsae", "nmse": 1.1, "decoder_weight": 0.2},
    }
    rows = {
        "mse_s0": {"numerator_by_group": [2.0, 2.0]},
        "dpsae_w0.1_s0": {"numerator_by_group": [1.0, 1.0]},
        "dpsae_w0.2_s0": {"numerator_by_group": [0.5, 0.5]},
    }
    result = paired_frontier(models, rows, bootstrap_samples=100, seed=1)
    assert len(result) == 2
    assert math.isclose(result[0]["exact_decoder_reduction"], 0.5)
    assert result[0]["strictly_dominates_mse"]
    assert not result[1]["strictly_dominates_mse"]
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "frontier-existing",
            "frontier-train-screen",
            "frontier-select",
            "self-test",
        ],
    )
    parser.add_argument("--source-models", type=Path, default=DEFAULT_SOURCE_MODELS)
    parser.add_argument("--candidate-models", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-label", default="exp04b natural selection [180M,185M)")
    parser.add_argument("--source-tokens", type=Path, default=DEFAULT_SOURCE_TOKENS)
    parser.add_argument(
        "--source-calibration", type=Path, default=DEFAULT_SOURCE_CALIBRATION
    )
    parser.add_argument("--new-screen", type=Path, default=DEFAULT_NEW_SCREEN)
    parser.add_argument("--frontier-input", type=Path, default=DEFAULT_FRONTIER_INPUT)
    parser.add_argument(
        "--decoder-weights",
        type=float,
        nargs="+",
        default=[0.03125, 0.0625, 0.09375, 0.125],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument(
        "--source-range-name",
        choices=("screen", "confirmation", "robustness", "validation"),
        default="screen",
    )
    parser.add_argument("--data-seed", type=int, default=-1)
    parser.add_argument("--probe-seed-base", type=int, default=-1)
    parser.add_argument(
        "--sparsity-mode",
        choices=["batch_topk", "token_topk", "jump_relu"],
        default="batch_topk",
    )
    parser.add_argument("--jump-relu-init-threshold", type=float, default=0.001)
    parser.add_argument(
        "--jump-relu-init-mode",
        choices=["fixed", "topk_quantile"],
        default="topk_quantile",
    )
    parser.add_argument("--jump-relu-bandwidth", type=float, default=0.001)
    parser.add_argument("--jump-relu-sparsity-weight", type=float, default=1.0)
    parser.add_argument(
        "--jump-relu-threshold-lr-multiplier", type=float, default=32.0
    )
    parser.add_argument("--jump-relu-threshold-lr-multiplier-mse", type=float)
    parser.add_argument("--jump-relu-threshold-lr-multiplier-dpsae", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.08)
    parser.add_argument("--maximum-peak-gpu-gib", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "frontier-train-screen":
        frontier_train_screen(args)
    elif args.stage == "frontier-select":
        frontier_select(args)
    else:
        frontier_existing(args)


if __name__ == "__main__":
    main()
