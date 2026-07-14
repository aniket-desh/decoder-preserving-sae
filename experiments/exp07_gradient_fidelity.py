#!/usr/bin/env python3
"""Audit the finite-probe decoder-loss gradient at frozen SAE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dpsae.language_training import sampled_decoder_loss_from_reference
from dpsae.mech_analysis import load_sae
from dpsae.task_fidelity import (
    exact_relative_gradients,
    fixed_radius_targets,
    ridge_gradient_factors,
    sampled_relative_gradients,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp07_task_fidelity.json"


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


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path) -> dict[str, Any]:
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


def percentile(values: list[float], probability: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability))


def log_slope(x: list[int], y: list[float]) -> float:
    log_x = torch.tensor(x, dtype=torch.float64).log()
    log_y = torch.tensor(y, dtype=torch.float64).clamp_min(1e-30).log()
    centered = log_x - log_x.mean()
    return float((centered * (log_y - log_y.mean())).sum() / centered.square().sum())


def gradient_metrics(samples: Tensor, exact: Tensor) -> dict[str, Tensor]:
    reduce_dims = tuple(range(1, samples.ndim))
    exact_norm = exact.square().sum().sqrt().clamp_min(1e-30)
    dot = (samples * exact.unsqueeze(0)).sum(dim=reduce_dims)
    norm = samples.square().sum(dim=reduce_dims).sqrt().clamp_min(1e-30)
    relative_error = (
        (samples - exact.unsqueeze(0)).square().sum(dim=reduce_dims).sqrt() / exact_norm
    )
    return {
        "dot": dot,
        "cosine": dot / (norm * exact_norm),
        "norm_ratio": norm / exact_norm,
        "relative_error": relative_error,
    }


def mean_gradient_metrics(mean: Tensor, exact: Tensor) -> dict[str, float]:
    exact_norm = exact.square().sum().sqrt().clamp_min(1e-30)
    mean_norm = mean.square().sum().sqrt().clamp_min(1e-30)
    dot = (mean * exact).sum()
    return {
        "cosine": float(dot / (mean_norm * exact_norm)),
        "norm_ratio": float(mean_norm / exact_norm),
        "relative_error": float((mean - exact).square().sum().sqrt() / exact_norm),
    }


@torch.inference_mode()
def training_reconstruction(model, activations: Tensor) -> Tensor:
    reconstruction, _ = model(activations.float(), use_threshold=False)
    return reconstruction


def target_covariances(targets: Tensor, probe_counts: list[int]) -> dict[int, Tensor]:
    result = {}
    covariance = torch.zeros(
        *targets.shape[:3],
        targets.shape[2],
        dtype=targets.dtype,
        device=targets.device,
    )
    wanted = set(probe_counts)
    for index in range(targets.shape[-1]):
        column = targets[..., index]
        covariance.add_(column.unsqueeze(-1) * column.unsqueeze(-2))
        count = index + 1
        if count in wanted:
            result[count] = covariance.clone()
    if set(result) != wanted:
        raise ValueError("probe counts exceed the generated target bank")
    return result


def autograd_spot_check(
    original: Tensor,
    reconstructed: Tensor,
    *,
    ridge: float,
    probes: int,
    seed: int,
    clamp_min: float,
) -> dict[str, float | int]:
    generator = torch.Generator(device=original.device).manual_seed(seed)
    targets, clamp_hits = fixed_radius_targets(
        1,
        original.shape[0],
        original.shape[1],
        probes,
        generator=generator,
        device=original.device,
        dtype=torch.float32,
        clamp_min=clamp_min,
    )
    targets = targets[0]
    covariance = targets @ targets.mT
    factors = ridge_gradient_factors(original.float(), reconstructed.float(), ridge=ridge)
    _, analytic, denominator = sampled_relative_gradients(
        factors,
        covariance.unsqueeze(0),
        probes=probes,
    )
    variable = reconstructed.float().detach().requires_grad_(True)
    loss = sampled_decoder_loss_from_reference(
        original.float(),
        variable,
        targets,
        ridge=ridge,
    )
    loss.backward()
    autograd = variable.grad
    analytic = analytic[0]
    exact_norm = autograd.square().sum().sqrt().clamp_min(1e-30)
    analytic_norm = analytic.square().sum().sqrt().clamp_min(1e-30)
    return {
        "target_rms_clamp_hits": clamp_hits,
        "denominator": float(denominator[0]),
        "loss": float(loss),
        "relative_error": float((analytic - autograd).square().sum().sqrt() / exact_norm),
        "cosine": float((analytic * autograd).sum() / (analytic_norm * exact_norm)),
    }


def run_model(
    config: dict[str, Any],
    output: Path,
    model_name: str,
    *,
    device: torch.device,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    started = time.time()
    settings = config["gradient_fidelity"]
    models_path = ROOT / settings["models"]
    cache_path = ROOT / settings["cache"]
    output.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f}")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)

    payloads = torch.load(models_path, map_location="cpu", weights_only=False)
    if model_name not in payloads:
        raise KeyError(f"unknown model {model_name}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    tokens_per_batch = int(settings["tokens_per_batch"])
    group_size = int(settings["group_size"])
    groups = tokens_per_batch // group_size
    batches = int(settings["batches"])
    if tokens_per_batch % group_size:
        raise ValueError("tokens per batch must divide into complete geometry groups")
    flat = cache["activations"].flatten(0, 1)
    if batches * tokens_per_batch > len(flat):
        raise ValueError("gradient batches exceed the held-out activation cache")
    model = load_sae(payloads[model_name], input_dim=flat.shape[-1], device=device)
    model.eval()
    probe_counts = [int(value) for value in settings["probe_counts"]]
    maximum_probes = max(probe_counts)
    banks = int(settings["banks"])
    bank_chunk = int(settings["bank_chunk"])
    if banks % bank_chunk:
        raise ValueError("banks must divide evenly by bank_chunk")
    ridge = float(settings["ridge"])
    raw: dict[str, Any] = {}
    batch_rows = []
    total_target_clamp_hits = 0
    total_denominator_clamp_hits = 0
    autograd_result = None

    for batch_index in range(batches):
        print(f"{model_name}: batch {batch_index + 1}/{batches}", flush=True)
        start = batch_index * tokens_per_batch
        activations = flat[start : start + tokens_per_batch].to(device).float()
        reconstructed = training_reconstruction(model, activations)
        original_groups = activations.reshape(groups, group_size, -1).double()
        reconstructed_groups = reconstructed.reshape_as(original_groups).double()
        factors = ridge_gradient_factors(
            original_groups,
            reconstructed_groups,
            ridge=ridge,
        )
        exact_gram, exact_reconstruction = exact_relative_gradients(factors)
        exact_by_space = {
            "row_gram": exact_gram,
            "reconstruction": exact_reconstruction,
        }
        accumulators = {
            count: {
                space: {
                    "sampled_sum": torch.zeros_like(exact),
                    "fixed_sum": torch.zeros_like(exact),
                    "sampled": {key: [] for key in ("dot", "cosine", "norm_ratio", "relative_error")},
                    "fixed": {key: [] for key in ("dot", "cosine", "norm_ratio", "relative_error")},
                }
                for space, exact in exact_by_space.items()
            }
            for count in probe_counts
        }
        denominators = {count: [] for count in probe_counts}
        generator = torch.Generator(device=device).manual_seed(
            int(settings["probe_seed"]) + batch_index
        )
        for bank_start in range(0, banks, bank_chunk):
            targets, clamp_hits = fixed_radius_targets(
                bank_chunk,
                groups,
                group_size,
                maximum_probes,
                generator=generator,
                device=device,
                dtype=torch.float64,
                clamp_min=float(settings["target_rms_clamp"]),
            )
            total_target_clamp_hits += clamp_hits
            covariances = target_covariances(targets, probe_counts)
            for count in probe_counts:
                sampled_gram, sampled_reconstruction, denominator = sampled_relative_gradients(
                    factors,
                    covariances[count],
                    probes=count,
                )
                fixed_scale = denominator / (
                    count * factors["source_energy"].clamp_min(1e-12)
                )
                fixed_gram = sampled_gram * fixed_scale[:, None, None, None]
                fixed_reconstruction = (
                    sampled_reconstruction * fixed_scale[:, None, None, None]
                )
                total_denominator_clamp_hits += int(
                    (denominator <= float(settings["denominator_clamp"])).sum()
                )
                denominators[count].append(denominator.detach().cpu())
                for space, sampled, fixed in (
                    ("row_gram", sampled_gram, fixed_gram),
                    ("reconstruction", sampled_reconstruction, fixed_reconstruction),
                ):
                    exact = exact_by_space[space]
                    sampled_metrics = gradient_metrics(sampled, exact)
                    fixed_metrics = gradient_metrics(fixed, exact)
                    accumulator = accumulators[count][space]
                    accumulator["sampled_sum"].add_(sampled.sum(0))
                    accumulator["fixed_sum"].add_(fixed.sum(0))
                    for key in sampled_metrics:
                        accumulator["sampled"][key].append(sampled_metrics[key].detach().cpu())
                        accumulator["fixed"][key].append(fixed_metrics[key].detach().cpu())
                del covariances[count], sampled_gram, sampled_reconstruction
                del fixed_gram, fixed_reconstruction

        rows_by_count = {}
        raw_batch = {}
        for count in probe_counts:
            denominator = torch.cat(denominators[count]).double()
            row = {
                "model": model_name,
                "batch": batch_index,
                "probes": count,
                "denominator_mean": float(denominator.mean()),
                "denominator_cv": float(denominator.std(unbiased=True) / denominator.mean().clamp_min(1e-30)),
                "spaces": {},
            }
            raw_batch[str(count)] = {"denominator": denominator.float()}
            for space, exact in exact_by_space.items():
                accumulator = accumulators[count][space]
                sampled_mean = accumulator["sampled_sum"] / banks
                fixed_mean = accumulator["fixed_sum"] / banks
                sampled_raw = {
                    key: torch.cat(values).double()
                    for key, values in accumulator["sampled"].items()
                }
                fixed_raw = {
                    key: torch.cat(values).double()
                    for key, values in accumulator["fixed"].items()
                }
                sampled_mean_metrics = mean_gradient_metrics(sampled_mean, exact)
                fixed_mean_metrics = mean_gradient_metrics(fixed_mean, exact)
                sampled_rmse = float(sampled_raw["relative_error"].square().mean().sqrt())
                fixed_rmse = float(fixed_raw["relative_error"].square().mean().sqrt())
                row["spaces"][space] = {
                    "mean_gradient": sampled_mean_metrics,
                    "fixed_denominator_mean_gradient": fixed_mean_metrics,
                    "individual_positive_dot_fraction": float((sampled_raw["dot"] > 0).double().mean()),
                    "individual_cosine_median": float(sampled_raw["cosine"].median()),
                    "individual_cosine_p10": float(torch.quantile(sampled_raw["cosine"], 0.10)),
                    "relative_rmse": sampled_rmse,
                    "fixed_denominator_relative_rmse": fixed_rmse,
                    "fixed_denominator_mc_resolution": fixed_rmse / math.sqrt(banks),
                }
                raw_batch[str(count)][space] = {
                    "sampled": {key: value.float() for key, value in sampled_raw.items()},
                    "fixed": {key: value.float() for key, value in fixed_raw.items()},
                }
            rows_by_count[count] = row
            batch_rows.append(row)

        slope_counts = [count for count in probe_counts if 2 <= count <= 32]
        for space in exact_by_space:
            slope = log_slope(
                slope_counts,
                [rows_by_count[count]["spaces"][space]["relative_rmse"] for count in slope_counts],
            )
            for count in probe_counts:
                rows_by_count[count]["spaces"][space]["rmse_log_slope_m2_to_m32"] = slope
        raw[str(batch_index)] = raw_batch

        if (
            model_name == settings["autograd_model"]
            and batch_index == int(settings["autograd_batch"])
        ):
            autograd_result = autograd_spot_check(
                original_groups,
                reconstructed_groups,
                ridge=ridge,
                probes=int(settings["autograd_probes"]),
                seed=int(settings["probe_seed"]) + 1_000_000,
                clamp_min=float(settings["target_rms_clamp"]),
            )
        del activations, reconstructed, original_groups, reconstructed_groups, factors
        del exact_gram, exact_reconstruction, accumulators
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_path = output / f"gradient_fidelity_{model_name}_raw.pt"
    atomic_torch(raw_path, raw)
    resources = {
        "device": str(device),
        "free_gib_at_start": free_gib,
        "minimum_free_gib_guard": minimum_free_gib,
        "gpu_memory_fraction_cap": gpu_memory_fraction if device.type == "cuda" else None,
        "peak_allocated_gpu_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    payload = {
        "complete": True,
        "experiment": "finite_probe_gradient_fidelity",
        "model": model_name,
        "spec": payloads[model_name]["spec"],
        "batch_rows": batch_rows,
        "target_rms_clamp_hits": total_target_clamp_hits,
        "denominator_clamp_hits": total_denominator_clamp_hits,
        "autograd_spot_check": autograd_result,
        "protocol": {
            **settings,
            "precision": "float64 analytic audit; float32 autograd spot check",
            "support": "training BatchTopK over each complete 2048-token batch",
            "probe_banks_reused_across_models": True,
            "probe_counts_are_paired_prefixes": True,
        },
        "inputs": {
            "models": input_record(models_path),
            "cache": input_record(cache_path),
            "config": input_record(Path(config["_config_path"])),
            "evaluator": input_record(Path(__file__)),
            "task_fidelity_module": input_record(ROOT / "src/dpsae/task_fidelity.py"),
            "raw_metrics": input_record(raw_path),
        },
        "repository": repository_state(),
        "resources": resources,
        "wall_seconds": time.time() - started,
    }
    atomic_json(output / f"gradient_fidelity_{model_name}.json", payload)
    return payload


def model_names(settings: dict[str, Any]) -> list[str]:
    names = []
    for seed in settings["seeds"]:
        names.append(settings["baseline_template"].format(seed=seed))
        names.append(settings["candidate_template"].format(seed=seed))
    return names


def checkpoint_gate(payload: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    gates = settings["gates"]
    primary = int(settings["autograd_probes"])
    rows = [row for row in payload["batch_rows"] if int(row["probes"]) == primary]
    denominator_cvs = [float(row["denominator_cv"]) for row in rows]
    checks: dict[str, bool] = {
        "target_rms_clamp_never_fired": payload["target_rms_clamp_hits"] == 0,
        "denominator_clamp_never_fired": payload["denominator_clamp_hits"] == 0,
        "median_denominator_cv": percentile(denominator_cvs, 0.50) <= float(gates["maximum_median_denominator_cv"]),
        "p90_denominator_cv": percentile(denominator_cvs, 0.90) <= float(gates["maximum_batch_p90_denominator_cv"]),
    }
    summaries = {}
    for space in ("row_gram", "reconstruction"):
        cosine = [float(row["spaces"][space]["mean_gradient"]["cosine"]) for row in rows]
        norm_ratio = [float(row["spaces"][space]["mean_gradient"]["norm_ratio"]) for row in rows]
        bias = [float(row["spaces"][space]["mean_gradient"]["relative_error"]) for row in rows]
        positive_dot = [float(row["spaces"][space]["individual_positive_dot_fraction"]) for row in rows]
        fixed_error = [float(row["spaces"][space]["fixed_denominator_mean_gradient"]["relative_error"]) for row in rows]
        fixed_resolution = [float(row["spaces"][space]["fixed_denominator_mc_resolution"]) for row in rows]
        slopes = [float(row["spaces"][space]["rmse_log_slope_m2_to_m32"]) for row in rows]
        summaries[space] = {
            "median_mean_gradient_cosine": percentile(cosine, 0.50),
            "p10_mean_gradient_cosine": percentile(cosine, 0.10),
            "median_norm_ratio": percentile(norm_ratio, 0.50),
            "median_relative_mean_bias": percentile(bias, 0.50),
            "p90_relative_mean_bias": percentile(bias, 0.90),
            "pooled_individual_positive_dot_fraction": sum(positive_dot) / len(positive_dot),
            "maximum_fixed_denominator_mean_error_over_resolution": max(
                error / max(float(gates["maximum_fixed_denominator_mean_error"]), resolution)
                for error, resolution in zip(fixed_error, fixed_resolution)
            ),
            "median_rmse_log_slope": percentile(slopes, 0.50),
        }
        checks[f"{space}_median_cosine"] = summaries[space]["median_mean_gradient_cosine"] >= float(gates["minimum_median_mean_gradient_cosine"])
        checks[f"{space}_p10_cosine"] = summaries[space]["p10_mean_gradient_cosine"] >= float(gates["minimum_batch_p10_mean_gradient_cosine"])
        checks[f"{space}_median_norm_ratio"] = float(gates["minimum_median_norm_ratio"]) <= summaries[space]["median_norm_ratio"] <= float(gates["maximum_median_norm_ratio"])
        checks[f"{space}_median_bias"] = summaries[space]["median_relative_mean_bias"] <= float(gates["maximum_median_relative_mean_bias"])
        checks[f"{space}_p90_bias"] = summaries[space]["p90_relative_mean_bias"] <= float(gates["maximum_batch_p90_relative_mean_bias"])
        checks[f"{space}_positive_dot"] = summaries[space]["pooled_individual_positive_dot_fraction"] >= float(gates["minimum_positive_dot_fraction"])
        checks[f"{space}_fixed_denominator"] = summaries[space]["maximum_fixed_denominator_mean_error_over_resolution"] <= 1.0
        checks[f"{space}_rmse_slope"] = float(gates["minimum_rmse_log_slope"]) <= summaries[space]["median_rmse_log_slope"] <= float(gates["maximum_rmse_log_slope"])
    if payload["autograd_spot_check"] is not None:
        spot = payload["autograd_spot_check"]
        checks["autograd_relative_error"] = float(spot["relative_error"]) <= float(gates["maximum_autograd_relative_error"])
        checks["autograd_cosine"] = float(spot["cosine"]) >= float(gates["minimum_autograd_cosine"])
    return {
        "model": payload["model"],
        "method": payload["spec"]["method"],
        "probes": primary,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "summaries": summaries,
        "denominator_cv": {
            "median": percentile(denominator_cvs, 0.50),
            "p90": percentile(denominator_cvs, 0.90),
        },
        "autograd_spot_check": payload["autograd_spot_check"],
    }


def summarize(config: dict[str, Any], output: Path) -> dict[str, Any]:
    settings = config["gradient_fidelity"]
    names = model_names(settings)
    payloads = [
        json.loads((output / f"gradient_fidelity_{name}.json").read_text())
        for name in names
    ]
    checkpoints = [checkpoint_gate(payload, settings) for payload in payloads]
    dpsae = [row for row in checkpoints if row["method"] == "dpsae"]
    result = {
        "complete": True,
        "experiment": "finite_probe_gradient_fidelity_summary",
        "checkpoints": checkpoints,
        "full_dpsae_fidelity_gate": all(row["all_checks_pass"] for row in dpsae),
        "interpretation_boundary": (
            "The implemented gradient is unbiased for the expected finite-probe "
            "self-normalized objective. Passing this audit supports numerical "
            "alignment with the identity-target gradient at m=16; it is not an "
            "exact finite-m unbiasedness theorem for the identity-target ratio."
        ),
        "inputs": {
            name: input_record(output / f"gradient_fidelity_{name}.json")
            for name in names
        },
        "repository": repository_state(),
    }
    atomic_json(output / "gradient_fidelity_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "summarize"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    config["_config_path"] = str(args.config.resolve())
    output = args.output or ROOT / config["output"]
    if args.mode == "run":
        if not args.model_name:
            raise ValueError("--model-name is required in run mode")
        if args.model_name not in model_names(config["gradient_fidelity"]):
            raise ValueError("model is not part of the frozen gradient audit")
        result = run_model(
            config,
            output,
            args.model_name,
            device=torch.device(args.device),
            gpu_memory_fraction=args.gpu_memory_fraction,
            minimum_free_gib=args.minimum_free_gib,
        )
        print(json.dumps({"complete": result["complete"], "model": result["model"]}, indent=2))
    else:
        result = summarize(config, output)
        print(json.dumps({"full_dpsae_fidelity_gate": result["full_dpsae_fidelity_gate"]}, indent=2))


if __name__ == "__main__":
    main()
