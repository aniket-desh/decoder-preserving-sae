#!/usr/bin/env python3
"""Clean matched-quality robustness, frozen-LM fidelity, and systems evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from dpsae.exp04b_natural_text import (
    apply_geometry_groups,
    bootstrap_paired_reduction_interval,
    exact_decoder_sweep,
    geometry_group_indices,
)
from dpsae.language_model import ActivationStats, GPT2ActivationModel
from dpsae.language_training import SAETrainSpec, TrainingFleet
from dpsae.mech_analysis import load_sae


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("robustness", "frozen", "overhead", "all"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--confirmation-cache", type=Path, required=True)
    parser.add_argument("--frozen-cache", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--training-done", type=Path, required=True)
    parser.add_argument("--confirmation-summary", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_closure.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-weight", type=float, default=0.03125)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-peak-gpu-gib", type=float, default=24.0)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--frozen-batch-sequences", type=int, default=8)
    parser.add_argument("--frozen-maximum-sequences", type=int, default=0)
    parser.add_argument("--overhead-warmup-steps", type=int, default=10)
    parser.add_argument("--overhead-timed-steps", type=int, default=100)
    parser.add_argument("--overhead-rounds", type=int, default=3)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def repository_state() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def completed_output_is_current(
    path: Path, *, inputs: Mapping[str, Any], repository: Mapping[str, Any]
) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        return False
    if payload.get("inputs") != inputs or payload.get("repository") != repository:
        raise RuntimeError(f"refusing to reuse stale complete output: {path}")
    return True


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text())
    source = json.loads((ROOT / config["source_config"]).read_text())
    return config, source


def validate_protocol_inputs(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> None:
    manifest = json.loads(args.run_manifest.read_text())
    if not manifest.get("complete") or manifest.get("repository") != repository:
        raise ValueError("run manifest does not match the current clean repository")
    external = manifest.get("external_inputs", {})
    for name, path in (
        ("activation_calibration", args.calibration),
        ("static_calibration", args.static),
    ):
        if external.get(name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"{name} differs from the immutable run manifest")
    done = json.loads(args.training_done.read_text())
    if not done.get("complete"):
        raise ValueError("confirmation training is not complete")
    if done.get("repository") != repository:
        raise ValueError("confirmation training revision differs from the evaluator")
    expected_training = tuple(int(value) for value in source["corpus"]["ranges"]["confirmation"])
    observed_training = tuple(int(value) for value in done["stream"]["range"])
    if observed_training != expected_training or done["stream"].get("range_name") != "confirmation":
        raise ValueError("confirmation models used the wrong training interval")
    confirmation_summary = json.loads(args.confirmation_summary.read_text())
    if not confirmation_summary.get("complete") or not confirmation_summary.get("gate_passed"):
        raise ValueError("clean confirmation did not pass the frozen matched-quality gate")
    if confirmation_summary.get("repository") != repository:
        raise ValueError("confirmation summary revision differs from the evaluator")
    for name, path in (
        ("models", args.models),
        ("training_done", args.training_done),
        ("cache", args.confirmation_cache),
        ("config", args.config),
        ("run_manifest", args.run_manifest),
    ):
        if confirmation_summary["inputs"][name]["sha256"] != sha256_file(path):
            raise ValueError(f"confirmation summary input changed: {name}")

    confirmation_cache = torch.load(
        args.confirmation_cache, map_location="cpu", weights_only=False
    )
    frozen_cache = torch.load(args.frozen_cache, map_location="cpu", weights_only=False)
    calibration_sha256 = sha256_file(args.calibration)
    for name, cache in (("confirmation", confirmation_cache), ("frozen", frozen_cache)):
        if cache.get("repository") != repository:
            raise ValueError(f"{name} cache was not generated by the current clean revision")
        if cache.get("normalized_with_sha256") != calibration_sha256:
            raise ValueError(f"{name} cache used another activation normalization")
    confirmation_interval = validate_natural_cache(
        confirmation_cache, config, split="selection"
    )
    frozen_interval = validate_natural_cache(frozen_cache, config, split="test")
    disjoint_intervals(
        {
            "confirmation_training": expected_training,
            "confirmation_evaluation": confirmation_interval,
            "frozen_evaluation": frozen_interval,
        }
    )
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=False)
    static = torch.load(args.static, map_location="cpu", weights_only=False)
    expected_model = str(source["model_name"])
    expected_layer = int(source["layer"])
    for name, payload in (("activation calibration", calibration), ("static calibration", static)):
        if payload.get("model_name") != expected_model or int(payload.get("layer", -1)) != expected_layer:
            raise ValueError(f"{name} targets another model or layer")
    args._confirmation_interval = confirmation_interval
    args._frozen_interval = frozen_interval


def prepare_resources(
    output: Path,
    *,
    device: torch.device,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output.parent).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f} GiB"
        )
    if device.type == "cuda":
        if not 0 < gpu_memory_fraction <= 1:
            raise ValueError("GPU memory fraction must lie in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "device": str(device),
        "free_gib_at_start": free_gib,
        "minimum_free_gib_guard": minimum_free_gib,
        "gpu_memory_fraction_cap": gpu_memory_fraction if device.type == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }


def enforce_peak_guard(resources: Mapping[str, Any], maximum_peak_gpu_gib: float) -> None:
    peak = resources.get("peak_allocated_gpu_gib")
    if peak is not None and float(peak) > maximum_peak_gpu_gib:
        raise RuntimeError(
            f"peak {float(peak):.2f} GiB exceeded guard {maximum_peak_gpu_gib:.2f} GiB"
        )


def selected_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    selected_weight: float,
    *,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Mapping[str, Any]]:
    selected = {}
    by_pair: dict[tuple[int, str], str] = {}
    for name, payload in payloads.items():
        spec = payload["spec"]
        method = str(spec["method"])
        if method == "mse":
            pass
        elif method == "dpsae" and math.isclose(
            float(spec.get("decoder_weight", 0)), selected_weight, rel_tol=0, abs_tol=1e-12
        ):
            pass
        else:
            continue
        key = (int(spec["seed"]), method)
        if key in by_pair:
            raise ValueError(f"duplicate selected model for {key}: {by_pair[key]}, {name}")
        by_pair[key] = name
        selected[name] = payload
    seeds = sorted({seed for seed, _ in by_pair})
    if expected_seeds is not None and seeds != sorted(int(seed) for seed in expected_seeds):
        raise ValueError(f"selected fleet seeds {seeds} differ from expected {list(expected_seeds)}")
    if not seeds or any((seed, method) not in by_pair for seed in seeds for method in ("mse", "dpsae")):
        raise ValueError("selected fleet must contain exactly one MSE/DPSAE pair per seed")
    return selected


def validate_natural_cache(
    cache: Mapping[str, Any], config: Mapping[str, Any], *, split: str
) -> tuple[int, int]:
    if split not in {"selection", "test"}:
        raise ValueError(f"unknown natural cache split: {split}")
    fresh = config["fresh_corpus"]
    expected_relative = [int(value) for value in fresh[f"{split}_range"]]
    offset = int(fresh["token_offset"])
    if cache.get("split") != split:
        raise ValueError(f"expected {split} cache, found {cache.get('split')!r}")
    if [int(value) for value in cache.get("token_range", ())] != expected_relative:
        raise ValueError(f"{split} cache has the wrong relative token range")
    if int(cache.get("token_offset", -1)) != offset:
        raise ValueError(f"{split} cache has the wrong token offset")
    input_ids = cache.get("input_ids")
    activations = cache.get("activations")
    starts = cache.get("starts")
    if not isinstance(input_ids, Tensor) or not isinstance(activations, Tensor):
        raise ValueError(f"{split} cache is missing tensor payloads")
    if activations.shape[:2] != input_ids.shape or not isinstance(starts, Tensor):
        raise ValueError(f"{split} cache shapes are inconsistent")
    if starts.shape != (len(input_ids),):
        raise ValueError(f"{split} cache start positions are inconsistent")
    absolute = (offset + expected_relative[0], offset + expected_relative[1])
    sequence_length = input_ids.shape[1]
    if len(starts) and (
        int(starts.min()) < absolute[0]
        or int(starts.max()) + sequence_length > absolute[1]
    ):
        raise ValueError(f"{split} cache contains a sequence outside its frozen range")
    return absolute


def disjoint_intervals(intervals: Mapping[str, tuple[int, int]]) -> None:
    names = list(intervals)
    for index, left_name in enumerate(names):
        left = intervals[left_name]
        if left[0] >= left[1]:
            raise ValueError(f"empty interval for {left_name}: {left}")
        for right_name in names[index + 1 :]:
            right = intervals[right_name]
            if max(left[0], right[0]) < min(left[1], right[1]):
                raise ValueError(
                    f"protocol intervals overlap: {left_name}={left}, {right_name}={right}"
                )


def pair_names(payloads: Mapping[str, Mapping[str, Any]]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for name, payload in payloads.items():
        spec = payload["spec"]
        result.setdefault(int(spec["seed"]), {})[str(spec["method"])] = name
    return result


def one_factor_settings(
    config: Mapping[str, Any], source: Mapping[str, Any], static: Mapping[str, Any]
) -> list[dict[str, Any]]:
    natural = config["natural_text"]
    base_ridge = float(static["ridge"])
    base_size = int(source["geometry"]["group_size"])
    settings = [
        {
            "audit_axis": "ridge",
            "setting_label": f"dof={fraction}",
            "setting_value": float(fraction),
            "ridge": float(value["ridge"]),
            "group_size": base_size,
            "grouping": "contiguous",
        }
        for fraction, value in static["ridges_by_dof_fraction"].items()
    ]
    settings.extend(
        {
            "audit_axis": "group_size",
            "setting_label": f"n={int(size)}",
            "setting_value": int(size),
            "ridge": float(static["ridges_by_group_size"][str(int(size))]["ridge"]),
            "group_size": int(size),
            "grouping": "contiguous",
        }
        for size in natural["group_sizes"]
    )
    settings.extend(
        {
            "audit_axis": "grouping",
            "setting_label": str(grouping).replace("_", " "),
            "setting_value": str(grouping),
            "ridge": base_ridge,
            "group_size": base_size,
            "grouping": str(grouping),
        }
        for grouping in natural["groupings"]
    )
    return settings


@torch.inference_mode()
def reconstruction_metrics(
    model,
    activations: Tensor,
    *,
    device: torch.device,
    exact_tokens: int,
    batch_tokens: int = 2048,
) -> tuple[dict[str, float], Tensor]:
    squared_error = 0.0
    squared_energy = 0.0
    l0_sum = 0.0
    token_count = 0
    exact_chunks = []
    for batch in activations.flatten(0, 1).split(batch_tokens):
        batch = batch.to(device).float()
        reconstruction, code = model(batch, use_threshold=True)
        squared_error += float((reconstruction - batch).square().sum())
        squared_energy += float(batch.square().sum())
        l0_sum += float((code != 0).sum())
        if token_count < exact_tokens:
            take = min(len(batch), exact_tokens - token_count)
            exact_chunks.append(reconstruction[:take].detach())
        token_count += len(batch)
    return (
        {
            "nmse": squared_error / max(squared_energy, 1e-30),
            "l0_inference": l0_sum / token_count,
            "activation_tokens": token_count,
        },
        torch.cat(exact_chunks),
    )


@torch.inference_mode()
def run_robustness(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    resources: dict[str, Any],
) -> None:
    output = args.output_dir / "robustness.json"
    if completed_output_is_current(
        output, inputs=args._inputs, repository=args._repository
    ):
        return
    started = time.time()
    cache = torch.load(args.confirmation_cache, map_location="cpu", weights_only=False)
    static = torch.load(args.static, map_location="cpu", weights_only=False)
    activations = cache["activations"]
    exact_tokens = int(config["natural_text"]["exact_tokens"])
    exact_sequences = exact_tokens // activations.shape[1]
    exact_original = activations[:exact_sequences].to(args._device).float()
    reconstructions = {}
    reports = {}
    for name, payload in payloads.items():
        print(f"robustness reconstruction: {name}", flush=True)
        model = load_sae(
            dict(payload), input_dim=activations.shape[-1], device=args._device
        ).eval()
        report, reconstruction = reconstruction_metrics(
            model,
            activations,
            device=args._device,
            exact_tokens=exact_tokens,
        )
        reports[name] = {"spec": dict(payload["spec"]), **report}
        reconstructions[name] = reconstruction.reshape_as(exact_original)
        del model
        if args._device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    settings = one_factor_settings(config, source, static)
    resolved_settings = []
    for index, setting in enumerate(settings):
        axis = str(setting["audit_axis"])
        ridge = float(setting["ridge"])
        group_size = int(setting["group_size"])
        grouping = str(setting["grouping"])
        print(
            f"robustness {axis}: ridge={ridge:.5g}, n={group_size}, {grouping}",
            flush=True,
        )
        maximum_groups = exact_tokens // group_size
        setting_rows = exact_decoder_sweep(
            exact_original,
            reconstructions,
            cache["input_ids"][:exact_sequences],
            ridges=[ridge],
            group_sizes=[group_size],
            groupings=[grouping],
            eos_token_id=int(cache["eos_token_id"]),
            max_groups=maximum_groups,
            bootstrap_samples=args.bootstrap_samples,
            seed=int(config["natural_text"]["test_seed"]) + index,
        )
        indices = geometry_group_indices(
            cache["input_ids"][:exact_sequences],
            group_size,
            grouping,
            seed=int(config["natural_text"]["test_seed"]) + index,
            eos_token_id=int(cache["eos_token_id"]),
        )
        if len(indices) != maximum_groups:
            raise ValueError(
                f"{setting['setting_label']} used {len(indices)} groups, expected {maximum_groups}"
            )
        grouped_original = apply_geometry_groups(exact_original, indices).float()
        singular_sq = torch.linalg.svdvals(grouped_original).square()
        realized_dof = (
            singular_sq / (singular_sq + group_size * ridge)
        ).sum(-1) / group_size
        resolved = {
            **setting,
            "evaluated_groups": len(indices),
            "evaluated_tokens": len(indices) * group_size,
            "realized_dof_fraction_mean": float(realized_dof.mean()),
            "realized_dof_fraction_min": float(realized_dof.min()),
            "realized_dof_fraction_max": float(realized_dof.max()),
        }
        resolved_settings.append(resolved)
        for row in setting_rows:
            row["audit_axis"] = axis
            row["setting_label"] = setting["setting_label"]
            row["setting_value"] = setting["setting_value"]
            row["realized_dof_fraction_mean"] = resolved[
                "realized_dof_fraction_mean"
            ]
            row["evaluated_tokens"] = resolved["evaluated_tokens"]
        rows.extend(setting_rows)

    paired = []
    pairs = pair_names(payloads)
    for index, setting in enumerate(resolved_settings):
        axis = str(setting["audit_axis"])
        ridge = float(setting["ridge"])
        group_size = int(setting["group_size"])
        grouping = str(setting["grouping"])
        matches = {
            row["model"]: row
            for row in rows
            if row["audit_axis"] == axis
            if math.isclose(float(row["ridge"]), ridge, rel_tol=0, abs_tol=1e-12)
            and int(row["group_size"]) == group_size
            and row["grouping"] == grouping
        }
        for seed, names in pairs.items():
            baseline = matches[names["mse"]]
            candidate = matches[names["dpsae"]]
            interval = bootstrap_paired_reduction_interval(
                torch.tensor(baseline["numerator_by_group"]),
                torch.tensor(candidate["numerator_by_group"]),
                samples=args.bootstrap_samples,
                seed=int(config["natural_text"]["test_seed"]) + index,
            )
            paired.append(
                {
                    "seed": seed,
                    "baseline": names["mse"],
                    "candidate": names["dpsae"],
                    "audit_axis": axis,
                    "setting_label": setting["setting_label"],
                    "setting_value": setting["setting_value"],
                    "ridge": ridge,
                    "group_size": group_size,
                    "grouping": grouping,
                    "realized_dof_fraction_mean": setting[
                        "realized_dof_fraction_mean"
                    ],
                    "evaluated_tokens": setting["evaluated_tokens"],
                    "decoder_reduction_vs_mse": interval["estimate"],
                    "decoder_reduction_ci95": [interval["low"], interval["high"]],
                    "nmse_reduction_vs_mse": 1
                    - reports[names["dpsae"]]["nmse"] / reports[names["mse"]]["nmse"],
                }
            )
    if args._device.type == "cuda":
        resources["peak_allocated_gpu_gib"] = (
            torch.cuda.max_memory_allocated(args._device) / 2**30
        )
    enforce_peak_guard(resources, args.maximum_peak_gpu_gib)
    atomic_json(
        output,
        {
            "complete": True,
            "experiment": "exp08_matched_quality_robustness",
            "models": reports,
            "settings": resolved_settings,
            "paired_reductions": paired,
            "exact_rows": rows,
            "protocol": {
                "selected_decoder_weight": args.selected_weight,
                "expected_seeds": list(config["frontier"]["confirmation_seeds"]),
                "bootstrap_samples": args.bootstrap_samples,
                "exact_tokens": exact_tokens,
                "cache_absolute_interval": list(args._confirmation_interval),
                "one_factor_at_a_time": True,
                "constant_token_budget_across_settings": True,
            },
            "inputs": args._inputs,
            "repository": args._repository,
            "resources": resources,
            "wall_seconds": time.time() - started,
        },
    )


def aggregate_frozen_rows(
    common_rows: Sequence[Mapping[str, float]],
    model_rows: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if len(common_rows) != len(model_rows) or not common_rows:
        raise ValueError("frozen rows must be nonempty and sequence-aligned")
    tokens = sum(row["tokens"] for row in common_rows)
    original_nll = sum(row["original_nll"] for row in common_rows)
    mean_nll = sum(row["mean_nll"] for row in common_rows)
    reconstructed_nll = sum(row["reconstructed_nll"] for row in model_rows)
    denominator = mean_nll - original_nll
    if denominator <= 0:
        raise ValueError("mean-activation ablation must worsen pooled next-token loss")
    return {
        "original_cross_entropy": original_nll / tokens,
        "mean_ablation_cross_entropy": mean_nll / tokens,
        "reconstruction_cross_entropy": reconstructed_nll / tokens,
        "cross_entropy_increase": (reconstructed_nll - original_nll) / tokens,
        "loss_recovered": 1 - (reconstructed_nll - original_nll) / denominator,
        "original_to_reconstruction_kl": sum(row["kl"] for row in model_rows) / tokens,
        "top1_agreement_with_original": sum(row["agreement"] for row in model_rows)
        / tokens,
        "reconstruction_next_token_accuracy": sum(
            row["reconstructed_correct"] for row in model_rows
        )
        / tokens,
        "original_next_token_accuracy": sum(
            row["original_correct"] for row in common_rows
        )
        / tokens,
        "activation_nmse": sum(row["reconstruction_sse"] for row in model_rows)
        / max(sum(row["activation_energy"] for row in common_rows), 1e-30),
        "inference_l0": sum(row["l0_count"] for row in model_rows)
        / max(sum(row["activation_tokens"] for row in common_rows), 1e-30),
        "tokens": tokens,
        "activation_tokens": sum(row["activation_tokens"] for row in common_rows),
        "sequences": len(common_rows),
    }


def _interval(values: Tensor) -> list[float]:
    finite = values[torch.isfinite(values)]
    if not len(finite):
        raise ValueError("bootstrap statistic has no finite draws")
    return [float(finite.quantile(0.025)), float(finite.quantile(0.975))]


def bootstrap_frozen_pair(
    common_rows: Sequence[Mapping[str, float]],
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not (
        len(common_rows) == len(baseline_rows) == len(candidate_rows) and common_rows
    ):
        raise ValueError("paired frozen rows must be nonempty and aligned")
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(common_rows), (samples, len(common_rows)), generator=generator)

    def values(rows: Sequence[Mapping[str, float]], key: str) -> Tensor:
        tensor = torch.tensor([row[key] for row in rows], dtype=torch.float64)
        return tensor[draws].sum(1)

    original = values(common_rows, "original_nll")
    mean = values(common_rows, "mean_nll")
    tokens = values(common_rows, "tokens")
    denominator = mean - original
    valid = denominator > 0
    baseline_nll = values(baseline_rows, "reconstructed_nll")
    candidate_nll = values(candidate_rows, "reconstructed_nll")
    baseline_recovered = torch.full_like(denominator, torch.nan)
    candidate_recovered = torch.full_like(denominator, torch.nan)
    baseline_recovered[valid] = 1 - (baseline_nll[valid] - original[valid]) / denominator[valid]
    candidate_recovered[valid] = 1 - (
        candidate_nll[valid] - original[valid]
    ) / denominator[valid]
    baseline_kl = values(baseline_rows, "kl") / tokens
    candidate_kl = values(candidate_rows, "kl") / tokens
    baseline_ce_increase = (baseline_nll - original) / tokens
    candidate_ce_increase = (candidate_nll - original) / tokens
    baseline_agreement = values(baseline_rows, "agreement") / tokens
    candidate_agreement = values(candidate_rows, "agreement") / tokens
    baseline_accuracy = values(baseline_rows, "reconstructed_correct") / tokens
    candidate_accuracy = values(candidate_rows, "reconstructed_correct") / tokens
    activation_energy = values(common_rows, "activation_energy")
    baseline_nmse = values(baseline_rows, "reconstruction_sse") / activation_energy
    candidate_nmse = values(candidate_rows, "reconstruction_sse") / activation_energy
    activation_tokens = values(common_rows, "activation_tokens")
    baseline_l0 = values(baseline_rows, "l0_count") / activation_tokens
    candidate_l0 = values(candidate_rows, "l0_count") / activation_tokens
    return {
        "loss_recovered_difference_dpsae_minus_mse_ci95": _interval(
            candidate_recovered - baseline_recovered
        ),
        "kl_difference_dpsae_minus_mse_ci95": _interval(candidate_kl - baseline_kl),
        "cross_entropy_increase_difference_dpsae_minus_mse_ci95": _interval(
            candidate_ce_increase - baseline_ce_increase
        ),
        "top1_agreement_difference_dpsae_minus_mse_ci95": _interval(
            candidate_agreement - baseline_agreement
        ),
        "next_token_accuracy_difference_dpsae_minus_mse_ci95": _interval(
            candidate_accuracy - baseline_accuracy
        ),
        "activation_nmse_ratio_dpsae_to_mse_ci95": _interval(
            candidate_nmse / baseline_nmse
        ),
        "inference_l0_difference_dpsae_minus_mse_ci95": _interval(
            candidate_l0 - baseline_l0
        ),
        "valid_loss_recovered_draw_fraction": float(valid.double().mean()),
    }


def _next_token_statistics(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    log_prob = F.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -log_prob.gather(-1, targets[:, :, None]).squeeze(-1)
    correct = log_prob.argmax(-1).eq(targets)
    return log_prob, nll.sum(1), correct.sum(1)


@torch.inference_mode()
def run_frozen_fidelity(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    resources: dict[str, Any],
) -> None:
    output = args.output_dir / "frozen_fidelity.json"
    if completed_output_is_current(
        output, inputs=args._inputs, repository=args._repository
    ):
        return
    started = time.time()
    cache = torch.load(args.frozen_cache, map_location="cpu", weights_only=False)
    input_ids = cache["input_ids"]
    maximum = args.frozen_maximum_sequences
    if maximum > 0:
        input_ids = input_ids[:maximum]
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=False)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], args._device)
    lm = GPT2ActivationModel.from_pretrained(
        source["model_name"], layer=int(source["layer"]), device=args._device
    )
    models = {
        name: load_sae(
            dict(payload), input_dim=int(lm.model.config.n_embd), device=args._device
        ).eval()
        for name, payload in payloads.items()
    }
    common_rows: list[dict[str, float]] = []
    model_rows: dict[str, list[dict[str, float]]] = {name: [] for name in models}

    def mean_replacement(hidden: Tensor) -> Tensor:
        return stats.mean.reshape(1, 1, -1).expand_as(hidden)

    for start in range(0, len(input_ids), args.frozen_batch_sequences):
        ids = input_ids[start : start + args.frozen_batch_sequences]
        targets = ids[:, 1:].to(args._device)
        original_logits = lm.logits(ids)
        original_log_prob, original_nll, original_correct = _next_token_statistics(
            original_logits, targets
        )
        original_prob = original_log_prob.exp()
        original_top1 = original_log_prob.argmax(-1)
        del original_logits
        normalized_hidden = stats.normalize(lm.activations(ids))
        activation_energy = normalized_hidden.square().sum(dim=(1, 2))
        mean_logits = lm.logits(ids, replacement=mean_replacement)
        _, mean_nll, _ = _next_token_statistics(mean_logits, targets)
        del mean_logits
        tokens_per_sequence = float(targets.shape[1])
        for row in range(len(ids)):
            common_rows.append(
                {
                    "sequence": float(start + row),
                    "tokens": tokens_per_sequence,
                    "original_nll": float(original_nll[row]),
                    "mean_nll": float(mean_nll[row]),
                    "original_correct": float(original_correct[row]),
                    "activation_energy": float(activation_energy[row]),
                    "activation_tokens": float(normalized_hidden.shape[1]),
                }
            )
        for name, model in models.items():
            shape = normalized_hidden.shape
            reconstruction, code = model(
                normalized_hidden.reshape(-1, shape[-1]), use_threshold=True
            )
            reconstruction = reconstruction.reshape(shape)
            code = code.reshape(shape[0], shape[1], -1)

            def replacement(_hidden: Tensor, value: Tensor = reconstruction) -> Tensor:
                if _hidden.shape != value.shape:
                    raise ValueError("frozen replacement shape changed within a batch")
                return stats.denormalize(value)

            reconstructed_logits = lm.logits(
                ids, replacement=replacement
            )
            reconstructed_log_prob, reconstructed_nll, reconstructed_correct = (
                _next_token_statistics(reconstructed_logits, targets)
            )
            kl = (
                original_prob * (original_log_prob - reconstructed_log_prob)
            ).sum(-1).sum(1)
            agreement = reconstructed_log_prob.argmax(-1).eq(original_top1).sum(1)
            for row in range(len(ids)):
                model_rows[name].append(
                    {
                        "sequence": float(start + row),
                        "reconstructed_nll": float(reconstructed_nll[row]),
                        "kl": float(kl[row]),
                        "agreement": float(agreement[row]),
                        "reconstructed_correct": float(reconstructed_correct[row]),
                        "reconstruction_sse": float(
                            (reconstruction[row] - normalized_hidden[row]).square().sum()
                        ),
                        "l0_count": float((code[row] != 0).sum()),
                    }
                )
            del reconstructed_logits, reconstructed_log_prob, reconstruction, code, replacement
        print(f"frozen fidelity: {min(start + len(ids), len(input_ids))}/{len(input_ids)}", flush=True)

    reports = {
        name: {
            "spec": dict(payloads[name]["spec"]),
            **aggregate_frozen_rows(common_rows, rows),
        }
        for name, rows in model_rows.items()
    }
    paired = []
    for seed, names in pair_names(payloads).items():
        baseline = reports[names["mse"]]
        candidate = reports[names["dpsae"]]
        row = {
            "seed": seed,
            "baseline": names["mse"],
            "candidate": names["dpsae"],
            "loss_recovered_difference_dpsae_minus_mse": candidate["loss_recovered"]
            - baseline["loss_recovered"],
            "kl_difference_dpsae_minus_mse": candidate[
                "original_to_reconstruction_kl"
            ]
            - baseline["original_to_reconstruction_kl"],
            "cross_entropy_increase_difference_dpsae_minus_mse": candidate[
                "cross_entropy_increase"
            ]
            - baseline["cross_entropy_increase"],
            "top1_agreement_difference_dpsae_minus_mse": candidate[
                "top1_agreement_with_original"
            ]
            - baseline["top1_agreement_with_original"],
            "next_token_accuracy_difference_dpsae_minus_mse": candidate[
                "reconstruction_next_token_accuracy"
            ]
            - baseline["reconstruction_next_token_accuracy"],
            "activation_nmse_ratio_dpsae_to_mse": candidate["activation_nmse"]
            / baseline["activation_nmse"],
            "inference_l0_difference_dpsae_minus_mse": candidate["inference_l0"]
            - baseline["inference_l0"],
        }
        row.update(
            bootstrap_frozen_pair(
                common_rows,
                model_rows[names["mse"]],
                model_rows[names["dpsae"]],
                samples=args.bootstrap_samples,
                seed=int(config["natural_text"]["test_seed"]) + seed,
            )
        )
        paired.append(row)
    if args._device.type == "cuda":
        resources["peak_allocated_gpu_gib"] = (
            torch.cuda.max_memory_allocated(args._device) / 2**30
        )
    enforce_peak_guard(resources, args.maximum_peak_gpu_gib)
    atomic_json(
        output,
        {
            "complete": True,
            "experiment": "exp08_frozen_language_model_fidelity",
            "purpose": "external evaluation not optimized by the refitted-readout objective",
            "split": {
                "name": cache.get("split"),
                "token_range": cache.get("token_range"),
                "token_offset": cache.get("token_offset"),
                "sequences": len(input_ids),
                "sequence_length": input_ids.shape[1],
                "absolute_interval": list(args._frozen_interval),
            },
            "mean_ablation": "replace the normalized activation with zero, then denormalize",
            "protocol": {
                "selected_decoder_weight": args.selected_weight,
                "expected_seeds": list(config["frontier"]["confirmation_seeds"]),
                "bootstrap_samples": args.bootstrap_samples,
                "batch_sequences": args.frozen_batch_sequences,
                "maximum_sequences": args.frozen_maximum_sequences,
                "consumes_previously_sealed_final_range": True,
                "same_split_activation_nmse_and_l0_reported": True,
            },
            "models": reports,
            "paired_differences": paired,
            "per_sequence": {"common": common_rows, "models": model_rows},
            "inputs": args._inputs,
            "repository": args._repository,
            "resources": resources,
            "wall_seconds": time.time() - started,
        },
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    method: str,
    activations: Tensor,
    *,
    source: Mapping[str, Any],
    ridge: float,
    selected_weight: float,
    device: torch.device,
    warmup_steps: int,
    timed_steps: int,
    round_index: int,
) -> dict[str, Any]:
    batch_tokens = int(source["training"]["sequence_length"]) * int(
        source["training"]["sequences_per_batch"]
    )
    if len(activations) < batch_tokens:
        raise ValueError("overhead cache is smaller than one training batch")
    spec = SAETrainSpec(
        f"{method}_benchmark",
        method,
        0,
        int(source["sae"]["primary_k"]),
        decoder_weight=selected_weight if method == "dpsae" else 0.0,
    )
    fleet = TrainingFleet(
        [spec],
        input_dim=activations.shape[1],
        dictionary_size=int(source["sae"]["dictionary_size"]),
        learning_rate=float(source["sae"]["learning_rate"]),
        device=device,
        aux_weight=float(source["sae"]["aux_weight"]),
        dead_after_steps=int(source["sae"]["dead_after_steps"]),
        aux_k=int(source["sae"]["aux_k"]),
        sparsity_mode="batch_topk",
    )
    batches = len(activations) // batch_tokens
    final_metrics = None
    for zero_step in range(warmup_steps):
        batch = activations[(zero_step % batches) * batch_tokens : (zero_step % batches + 1) * batch_tokens]
        final_metrics = fleet.train_batch(
            batch,
            step=zero_step + 1,
            ridge=ridge,
            group_size=int(source["geometry"]["group_size"]),
            probes=int(source["geometry"]["probes"]),
            probe_seed=2027080100 + zero_step,
        )
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for offset in range(timed_steps):
        zero_step = warmup_steps + offset
        batch = activations[(zero_step % batches) * batch_tokens : (zero_step % batches + 1) * batch_tokens]
        final_metrics = fleet.train_batch(
            batch,
            step=zero_step + 1,
            ridge=ridge,
            group_size=int(source["geometry"]["group_size"]),
            probes=int(source["geometry"]["probes"]),
            probe_seed=2027080100 + zero_step,
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "method": method,
        "round": round_index,
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "batch_tokens": batch_tokens,
        "elapsed_seconds": elapsed,
        "milliseconds_per_step": 1000 * elapsed / timed_steps,
        "activation_tokens_per_second": timed_steps * batch_tokens / elapsed,
        "peak_allocated_gpu_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else None
        ),
        "final_metrics": final_metrics[spec.name] if final_metrics is not None else None,
    }


def run_overhead(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    resources: dict[str, Any],
) -> None:
    output = args.output_dir / "training_overhead.json"
    if completed_output_is_current(
        output, inputs=args._inputs, repository=args._repository
    ):
        return
    started = time.time()
    cache = torch.load(args.confirmation_cache, map_location="cpu", weights_only=False)
    static = torch.load(args.static, map_location="cpu", weights_only=False)
    activations = cache["activations"].flatten(0, 1).to(args._device).float()
    rows = []
    for round_index in range(args.overhead_rounds):
        order = ("mse", "dpsae") if round_index % 2 == 0 else ("dpsae", "mse")
        for method in order:
            print(f"overhead round {round_index + 1}: {method}", flush=True)
            rows.append(
                benchmark_once(
                    method,
                    activations,
                    source=source,
                    ridge=float(static["ridge"]),
                    selected_weight=args.selected_weight,
                    device=args._device,
                    warmup_steps=args.overhead_warmup_steps,
                    timed_steps=args.overhead_timed_steps,
                    round_index=round_index,
                )
            )
            if args._device.type == "cuda":
                torch.cuda.empty_cache()
    summaries = {}
    for method in ("mse", "dpsae"):
        selected = [row for row in rows if row["method"] == method]
        summaries[method] = {
            "median_milliseconds_per_step": statistics.median(
                row["milliseconds_per_step"] for row in selected
            ),
            "median_activation_tokens_per_second": statistics.median(
                row["activation_tokens_per_second"] for row in selected
            ),
            "median_peak_allocated_gpu_gib": statistics.median(
                row["peak_allocated_gpu_gib"] for row in selected
            ),
        }
    resources["peak_allocated_gpu_gib"] = max(
        row["peak_allocated_gpu_gib"] for row in rows
    )
    enforce_peak_guard(resources, args.maximum_peak_gpu_gib)
    atomic_json(
        output,
        {
            "complete": True,
            "experiment": "exp08_cached_activation_training_overhead",
            "scope": "isolated SAE objective and optimizer step on cached normalized activations",
            "raw_rounds": rows,
            "summary": summaries,
            "dpsae_over_mse_step_time_ratio": summaries["dpsae"][
                "median_milliseconds_per_step"
            ]
            / summaries["mse"]["median_milliseconds_per_step"],
            "dpsae_minus_mse_peak_gpu_gib": summaries["dpsae"][
                "median_peak_allocated_gpu_gib"
            ]
            - summaries["mse"]["median_peak_allocated_gpu_gib"],
            "protocol": {
                "selected_decoder_weight": args.selected_weight,
                "warmup_steps": args.overhead_warmup_steps,
                "timed_steps": args.overhead_timed_steps,
                "rounds": args.overhead_rounds,
                "alternating_method_order": True,
                "shared_lm_forward_excluded": True,
            },
            "inputs": args._inputs,
            "repository": args._repository,
            "resources": resources,
            "wall_seconds": time.time() - started,
        },
    )


def common_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "models": input_record(args.models),
        "confirmation_cache": input_record(args.confirmation_cache),
        "frozen_cache": input_record(args.frozen_cache),
        "static_calibration": input_record(args.static),
        "activation_calibration": input_record(args.calibration),
        "confirmation_training_done": input_record(args.training_done),
        "confirmation_summary": input_record(args.confirmation_summary),
        "run_manifest": input_record(args.run_manifest),
        "config": input_record(args.config),
        "evaluator": input_record(Path(__file__)),
    }


def main() -> None:
    args = parse_args()
    args._device = torch.device(args.device)
    if args.selected_weight <= 0 or args.bootstrap_samples < 1:
        raise ValueError("selected weight and bootstrap samples must be positive")
    if min(
        args.frozen_batch_sequences,
        args.overhead_warmup_steps,
        args.overhead_timed_steps,
        args.overhead_rounds,
    ) < 1:
        raise ValueError("batch and overhead counts must be positive")
    state = repository_state()
    if state["dirty"] and not args.allow_dirty:
        raise RuntimeError(f"exp08 requires a clean repository: {state['status']}")
    config, source = load_configs(args.config)
    validate_protocol_inputs(args, config, source, state)
    args._repository = state
    args._inputs = common_inputs(args)
    all_payloads = torch.load(args.models, map_location="cpu", weights_only=False)
    payloads = selected_payloads(
        all_payloads,
        args.selected_weight,
        expected_seeds=config["frontier"]["confirmation_seeds"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = ("robustness", "frozen", "overhead") if args.stage == "all" else (args.stage,)
    for stage in stages:
        resources = prepare_resources(
            args.output_dir / f"{stage}.json",
            device=args._device,
            gpu_memory_fraction=args.gpu_memory_fraction,
            minimum_free_gib=args.minimum_free_gib,
        )
        if stage == "robustness":
            run_robustness(args, config, source, payloads, resources)
        elif stage == "frozen":
            run_frozen_fidelity(args, config, source, payloads, resources)
        else:
            run_overhead(args, source, resources)


if __name__ == "__main__":
    main()
