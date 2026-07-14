#!/usr/bin/env python3
"""Run the post-4b valid-target and equal-IOI-effect diagnostic.

Claims tested
--------------
1. Existing SAE reconstructions preserve a continuous IOI-related direction that
   is exactly linear in the original block-8 activations. The direction is fixed
   from the ranking split as the unit clean-minus-ABC mean difference. Ridge and
   sparse feature choices use only the intermediate selection split.
2. The existing natural-text collateral curves can be compared at an IOI effect
   frozen from the common MSE/DPSAE selection support. Test interpolation is
   reported only for models whose test curves contain that target.

Falsifiers
----------
- The dense original activation fails the preregistered test R2 gate of 0.8.
- DPSAE does not improve the valid target relative to paired MSE reconstructions.
- Equal-effect interpolation is unsupported, or its paired KL differences do
  not consistently favor DPSAE.

This script is evaluation-only. It loads the immutable IOI cache, trained SAE
payloads, and existing zero-ablation curves; it writes one new JSON artifact and
does not load the language model or modify any source artifacts/checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from dpsae.exp04b_execution import encode_confirmatory_states
from dpsae.mech_analysis import binary_auc, load_sae


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "artifacts" / "exp04b_confirmatory"
DEFAULT_OUTPUT = ROOT / "artifacts" / "exp04b_valid_target_equal_effect"
RIDGE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FEATURE_COUNTS = (1, 2, 4, 8, 16, 32, 64)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    return {"revision": revision, "dirty": dirty}


def method_name(name: str) -> str:
    if name.startswith("dpsae"):
        return "dpsae"
    if name.startswith("whitening"):
        return "whitening"
    if name.startswith("spectral"):
        return "spectral"
    return "mse"


def stack_pair(pair: Mapping[str, Tensor]) -> Tensor:
    return torch.cat([pair["positive"], pair["negative"]]).float()


def stack_encoded(pair: tuple[Tensor, Tensor]) -> Tensor:
    return torch.cat(pair).float()


def r2_score(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.float().flatten().cpu()
    target = target.float().flatten().cpu()
    denominator = (target - target.mean()).square().sum()
    if denominator <= 0:
        raise ValueError("target must vary")
    return float(1 - (prediction - target).square().sum() / denominator)


def pearson(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.float().flatten().cpu()
    target = target.float().flatten().cpu()
    x = prediction - prediction.mean()
    y = target - target.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denominator) if denominator > 0 else 0.0


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.float().flatten().cpu()
    target = target.float().flatten().cpu()
    return {
        "r2": r2_score(prediction, target),
        "pearson": pearson(prediction, target),
        "mae": float((prediction - target).abs().mean()),
        "rmse": float((prediction - target).square().mean().sqrt()),
    }


def _standardized_design(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x = x.float()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    standardized = (x - mean) / scale
    design = torch.cat(
        [standardized, torch.ones(len(x), 1, device=x.device, dtype=x.dtype)], 1
    )
    return mean, scale, design


def _solve_ridge(design: Tensor, target: Tensor, ridge: float) -> Tensor:
    penalty = torch.eye(design.shape[1], device=design.device, dtype=design.dtype)
    penalty[-1, -1] = 0
    gram = design.mT @ design + len(design) * ridge * penalty
    rhs = design.mT @ target.float().to(design.device)
    return torch.linalg.solve(gram, rhs)


def _predict_ridge(x: Tensor, mean: Tensor, scale: Tensor, weights: Tensor) -> Tensor:
    standardized = (x.float() - mean) / scale
    return standardized @ weights[:-1] + weights[-1]


def select_ridge(
    train_x: Tensor,
    train_y: Tensor,
    selection_x: Tensor,
    selection_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    *,
    ridge_grid: Sequence[float] = RIDGE_GRID,
) -> tuple[dict[str, Any], Tensor]:
    device = train_x.device
    train_y = train_y.to(device).float()
    selection_y = selection_y.to(device).float()
    test_y = test_y.to(device).float()
    mean, scale, design = _standardized_design(train_x)
    rows = []
    best: tuple[float, float, Tensor] | None = None
    for ridge in ridge_grid:
        weights = _solve_ridge(design, train_y, float(ridge))
        prediction = _predict_ridge(selection_x, mean, scale, weights)
        metrics = regression_metrics(prediction, selection_y)
        rows.append({"ridge": float(ridge), **metrics})
        candidate = (float(metrics["r2"]), -float(ridge), weights)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    selected_ridge = -best[1]
    test_prediction = _predict_ridge(test_x, mean, scale, best[2]).detach().cpu()
    return {
        "selection_curve": rows,
        "selected_ridge": selected_ridge,
        "selection": max(rows, key=lambda row: (row["r2"], -row["ridge"])),
        "test": regression_metrics(test_prediction, test_y),
    }, test_prediction


def rank_by_abs_correlation(codes: Tensor, target: Tensor) -> Tensor:
    codes = codes.float()
    target = target.float().to(codes.device)
    centered_codes = codes - codes.mean(0)
    centered_target = target - target.mean()
    numerator = (centered_codes * centered_target[:, None]).mean(0)
    denominator = (
        centered_codes.square().mean(0).sqrt()
        * centered_target.square().mean().sqrt()
    )
    score = torch.where(denominator > 0, numerator / denominator, 0)
    return score.abs().argsort(descending=True, stable=True)


def sparse_target_reports(
    train_codes: Tensor,
    selection_codes: Tensor,
    test_codes: Tensor,
    train_target: Tensor,
    selection_target: Tensor,
    test_target: Tensor,
    ranking: Tensor,
    *,
    feature_counts: Sequence[int] = FEATURE_COUNTS,
) -> tuple[dict[str, Any], Tensor]:
    maximum = max(feature_counts)
    selected = ranking[:maximum]
    train = train_codes[:, selected]
    selection = selection_codes[:, selected]
    test = test_codes[:, selected]
    rows: list[dict[str, Any]] = []
    predictions: dict[int, Tensor] = {}
    for count in feature_counts:
        report, prediction = select_ridge(
            train[:, :count],
            train_target,
            selection[:, :count],
            selection_target,
            test[:, :count],
            test_target,
        )
        rows.append({"features": int(count), **report})
        predictions[int(count)] = prediction
    selected_row = max(
        rows,
        key=lambda row: (
            row["selection"]["r2"],
            -row["features"],
            -row["selected_ridge"],
        ),
    )
    frozen = next(row for row in rows if row["features"] == maximum)
    return {
        "selection_curve": rows,
        "selected_by_validation": selected_row,
        "frozen_max_count": frozen,
    }, predictions[maximum]


def bootstrap_r2_difference(
    target: Tensor,
    baseline_prediction: Tensor,
    candidate_prediction: Tensor,
    *,
    seed: int,
    samples: int,
    chunk: int = 128,
) -> dict[str, Any]:
    target = target.float().flatten().cpu()
    baseline_sq = (baseline_prediction.float().flatten().cpu() - target).square()
    candidate_sq = (candidate_prediction.float().flatten().cpu() - target).square()
    improvement = baseline_sq - candidate_sq
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = []
    remaining = samples
    while remaining:
        size = min(chunk, remaining)
        indices = torch.randint(
            len(target), (size, len(target)), generator=generator
        )
        sampled_target = target[indices]
        denominator = (
            sampled_target - sampled_target.mean(1, keepdim=True)
        ).square().sum(1)
        numerator = improvement[indices].sum(1)
        draws.append(numerator / denominator.clamp_min(1e-12))
        remaining -= size
    bootstrap = torch.cat(draws)
    interval = torch.quantile(bootstrap, torch.tensor([0.025, 0.975]))
    estimate = r2_score(candidate_prediction, target) - r2_score(
        baseline_prediction, target
    )
    return {
        "estimand": "candidate_minus_paired_mse_test_r2",
        "estimate": estimate,
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_samples": samples,
        "examples": len(target),
    }


def joined_curve(model: Mapping[str, Any], *, test: bool) -> list[dict[str, Any]]:
    payload = model["duplicate_state"] if test else model
    natural = {int(row["features"]): row for row in payload["natural_zero_curve"]}
    rows = []
    for row in payload["ioi_zero_curve"]:
        count = int(row["features"])
        rows.append(
            {
                "features": count,
                "effect": float(row["ioi_effect"]),
                "kl": float(natural[count]["collateral_kl"]),
                "effect_by_example": row.get("effect_by_example"),
                "kl_by_sequence": natural[count].get("kl_by_sequence"),
            }
        )
    return sorted(rows, key=lambda row: row["features"])


def interpolation(curve: Sequence[Mapping[str, Any]], target: float) -> dict[str, Any] | None:
    ordered = sorted(curve, key=lambda row: (float(row["effect"]), int(row["features"])))
    if not ordered or target < ordered[0]["effect"] or target > ordered[-1]["effect"]:
        return None
    for row in ordered:
        if math.isclose(float(row["effect"]), target, rel_tol=0, abs_tol=1e-12):
            return {
                "left_features": int(row["features"]),
                "right_features": int(row["features"]),
                "alpha_right": 0.0,
                "effect": float(row["effect"]),
                "kl": float(row["kl"]),
            }
    for left, right in zip(ordered, ordered[1:]):
        left_effect, right_effect = float(left["effect"]), float(right["effect"])
        if left_effect <= target <= right_effect and right_effect > left_effect:
            alpha = (target - left_effect) / (right_effect - left_effect)
            return {
                "left_features": int(left["features"]),
                "right_features": int(right["features"]),
                "alpha_right": alpha,
                "effect": target,
                "kl": (1 - alpha) * float(left["kl"]) + alpha * float(right["kl"]),
            }
    return None


def apply_frozen_mixture(
    curve: Sequence[Mapping[str, Any]], interpolation_row: Mapping[str, Any]
) -> dict[str, float | int] | None:
    by_count = {int(row["features"]): row for row in curve}
    left = by_count.get(int(interpolation_row["left_features"]))
    right = by_count.get(int(interpolation_row["right_features"]))
    if left is None or right is None:
        return None
    alpha = float(interpolation_row["alpha_right"])
    return {
        "left_features": int(left["features"]),
        "right_features": int(right["features"]),
        "alpha_right": alpha,
        "effect": (1 - alpha) * float(left["effect"]) + alpha * float(right["effect"]),
        "kl": (1 - alpha) * float(left["kl"]) + alpha * float(right["kl"]),
    }


def _curve_row(curve: Sequence[Mapping[str, Any]], count: int) -> Mapping[str, Any]:
    matches = [row for row in curve if int(row["features"]) == count]
    if len(matches) != 1:
        raise ValueError(f"expected one curve row at feature count {count}")
    return matches[0]


def bootstrap_equal_effect_kl_difference(
    baseline_curve: Sequence[Mapping[str, Any]],
    baseline_interp: Mapping[str, Any],
    candidate_curve: Sequence[Mapping[str, Any]],
    candidate_interp: Mapping[str, Any],
    *,
    target: float,
    seed: int,
    samples: int,
    chunk: int = 128,
) -> dict[str, Any]:
    def arrays(curve: Sequence[Mapping[str, Any]], interp: Mapping[str, Any]):
        left = _curve_row(curve, int(interp["left_features"]))
        right = _curve_row(curve, int(interp["right_features"]))
        return tuple(
            torch.tensor(value, dtype=torch.float32)
            for value in (
                left["effect_by_example"],
                right["effect_by_example"],
                left["kl_by_sequence"],
                right["kl_by_sequence"],
            )
        )

    baseline = arrays(baseline_curve, baseline_interp)
    candidate = arrays(candidate_curve, candidate_interp)
    n_ioi, n_natural = len(baseline[0]), len(baseline[2])
    if any(len(values[0]) != n_ioi or len(values[2]) != n_natural for values in (baseline, candidate)):
        raise ValueError("paired curves have incompatible sample counts")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    differences = []
    valid = 0
    attempted = 0
    while attempted < samples:
        size = min(chunk, samples - attempted)
        ioi_indices = torch.randint(n_ioi, (size, n_ioi), generator=generator)
        natural_indices = torch.randint(
            n_natural, (size, n_natural), generator=generator
        )

        def interpolate_bootstrap(values: tuple[Tensor, ...]):
            effect_left = values[0][ioi_indices].mean(1)
            effect_right = values[1][ioi_indices].mean(1)
            denominator = effect_right - effect_left
            same = denominator.abs() < 1e-12
            alpha = torch.where(
                same,
                torch.zeros_like(denominator),
                (target - effect_left) / denominator,
            )
            supported = same | ((alpha >= 0) & (alpha <= 1))
            kl_left = values[2][natural_indices].mean(1)
            kl_right = values[3][natural_indices].mean(1)
            return kl_left + alpha * (kl_right - kl_left), supported

        baseline_kl, baseline_supported = interpolate_bootstrap(baseline)
        candidate_kl, candidate_supported = interpolate_bootstrap(candidate)
        supported = baseline_supported & candidate_supported
        if supported.any():
            differences.append((candidate_kl - baseline_kl)[supported])
            valid += int(supported.sum())
        attempted += size
    if not differences:
        return {
            "estimand": "candidate_minus_paired_mse_equal_effect_kl",
            "valid_bootstrap_samples": 0,
            "attempted_bootstrap_samples": samples,
        }
    bootstrap = torch.cat(differences)
    interval = torch.quantile(bootstrap, torch.tensor([0.025, 0.975]))
    return {
        "estimand": "candidate_minus_paired_mse_equal_effect_kl",
        "estimate": float(candidate_interp["kl"] - baseline_interp["kl"]),
        "ci95": [float(interval[0]), float(interval[1])],
        "valid_bootstrap_samples": valid,
        "attempted_bootstrap_samples": samples,
        "valid_fraction": valid / samples,
        "conditional_on": "frozen target and point-estimate test interpolation brackets",
    }


def equal_effect_report(
    selection_models: Mapping[str, Any],
    test_models: Mapping[str, Any],
    *,
    support_quantile: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    core_names = [
        name
        for name, value in selection_models.items()
        if value["method"] in {"mse", "dpsae"}
    ]
    selection_curves = {
        name: joined_curve(value, test=False) for name, value in selection_models.items()
    }
    test_curves = {name: joined_curve(value, test=True) for name, value in test_models.items()}
    lower = max(min(row["effect"] for row in selection_curves[name]) for name in core_names)
    upper = min(max(row["effect"] for row in selection_curves[name]) for name in core_names)
    if not lower < upper:
        return {
            "status": "unsupported_no_common_core_selection_support",
            "selection_support": [lower, upper],
        }
    target = lower + support_quantile * (upper - lower)
    by_model: dict[str, Any] = {}
    for name in sorted(test_models):
        selection_interp = interpolation(selection_curves[name], target)
        test_interp = interpolation(test_curves[name], target)
        by_model[name] = {
            "method": test_models[name]["method"],
            "seed": int(test_models[name]["spec"]["seed"]),
            "selection_interpolation": selection_interp,
            "selection_frozen_mixture_on_test": (
                apply_frozen_mixture(test_curves[name], selection_interp)
                if selection_interp is not None
                else None
            ),
            "test_equal_effect_interpolation": test_interp,
            "test_support": [
                min(row["effect"] for row in test_curves[name]),
                max(row["effect"] for row in test_curves[name]),
            ],
        }

    mse_by_seed = {
        int(value["spec"]["seed"]): name
        for name, value in test_models.items()
        if value["method"] == "mse"
    }
    paired = []
    for index, (name, value) in enumerate(sorted(test_models.items())):
        if value["method"] == "mse":
            continue
        model_seed = int(value["spec"]["seed"])
        baseline_name = mse_by_seed[model_seed]
        baseline_interp = by_model[baseline_name]["test_equal_effect_interpolation"]
        candidate_interp = by_model[name]["test_equal_effect_interpolation"]
        row: dict[str, Any] = {
            "baseline": baseline_name,
            "candidate": name,
            "method": value["method"],
            "seed": model_seed,
            "supported": baseline_interp is not None and candidate_interp is not None,
        }
        if row["supported"]:
            row["candidate_minus_mse_kl"] = float(
                candidate_interp["kl"] - baseline_interp["kl"]
            )
            row["bootstrap"] = bootstrap_equal_effect_kl_difference(
                test_curves[baseline_name],
                baseline_interp,
                test_curves[name],
                candidate_interp,
                target=target,
                seed=seed + 1000 + index,
                samples=bootstrap_samples,
            )
        paired.append(row)
    return {
        "status": "complete",
        "selection_support_core_mse_dpsae": [lower, upper],
        "support_quantile": support_quantile,
        "frozen_target_ioi_effect": target,
        "interpretation": {
            "selection_frozen_mixture_on_test": (
                "applies validation interpolation weights unchanged; test effects need not match"
            ),
            "test_equal_effect_interpolation": (
                "descriptive test-frontier comparison at the validation-frozen target"
            ),
        },
        "by_model": by_model,
        "paired_test_equal_effect": paired,
    }


def target_summary(cache: Mapping[str, Any], direction: Tensor) -> tuple[dict, dict[str, Tensor]]:
    targets: dict[str, Tensor] = {}
    summary: dict[str, Any] = {}
    for split in ("ranking", "selection", "test"):
        positive = cache[split]["positive"].float()
        negative = cache[split]["negative"].float()
        target = torch.cat([positive @ direction, negative @ direction]).cpu()
        labels = torch.cat([torch.ones(len(positive)), -torch.ones(len(negative))])
        targets[split] = target
        pooled_std = torch.sqrt(
            0.5
            * (
                (positive @ direction).var(unbiased=False)
                + (negative @ direction).var(unbiased=False)
            )
        ).clamp_min(1e-12)
        summary[split] = {
            "examples": len(target),
            "mean": float(target.mean()),
            "std": float(target.std(unbiased=False)),
            "positive_mean": float((positive @ direction).mean()),
            "negative_mean": float((negative @ direction).mean()),
            "standardized_mean_difference": float(
                ((positive @ direction).mean() - (negative @ direction).mean())
                / pooled_std
            ),
            "clean_vs_abc_auc": binary_auc(target, labels),
        }
    return summary, targets


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    required = {
        "cache": input_dir / "ioi_confirmatory_cache.pt",
        "models": input_dir / "baseline_confirm" / "models.pt",
        "selection": input_dir / "ioi_selection_models.json",
        "test": input_dir / "ioi_test_models.json",
        "config": ROOT / "configs" / "exp04b_confirmatory.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < args.minimum_free_gib * (1 << 30):
        raise RuntimeError(
            f"refusing to run with only {free_bytes / (1 << 30):.2f} GiB free"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    cache = torch.load(required["cache"], map_location="cpu")
    payloads = torch.load(required["models"], map_location="cpu")
    selection_models = read_json(required["selection"])
    test_models = read_json(required["test"])

    ranking_positive = cache["ranking"]["positive"].float()
    ranking_negative = cache["ranking"]["negative"].float()
    direction = ranking_positive.mean(0) - ranking_negative.mean(0)
    direction_norm = direction.norm()
    if not torch.isfinite(direction_norm) or direction_norm <= 0:
        raise RuntimeError("ranking clean-minus-ABC direction is invalid")
    direction = direction / direction_norm
    target_stats, targets = target_summary(cache, direction)

    original = {split: stack_pair(cache[split]).to(device) for split in targets}
    dense_gate, dense_prediction = select_ridge(
        original["ranking"],
        targets["ranking"],
        original["selection"],
        targets["selection"],
        original["test"],
        targets["test"],
    )
    dense_gate["threshold"] = args.dense_gate_r2
    dense_gate["passed"] = dense_gate["test"]["r2"] >= args.dense_gate_r2
    if not dense_gate["passed"]:
        raise RuntimeError(
            f"dense gate failed: test R2={dense_gate['test']['r2']:.6f}"
        )

    model_results: dict[str, Any] = {}
    predictions_reconstruction: dict[str, Tensor] = {}
    predictions_sparse64: dict[str, Tensor] = {}
    input_dim = ranking_positive.shape[1]
    for name, payload in payloads.items():
        print(f"[{datetime.now().isoformat(timespec='seconds')}] evaluating {name}", flush=True)
        model = load_sae(payload, input_dim=input_dim, device=device)
        encoded = encode_confirmatory_states(model, cache)
        reconstruction = {
            split: stack_encoded(encoded[split]["reconstructions"]).to(device)
            for split in targets
        }
        reconstruction_report, reconstruction_prediction = select_ridge(
            reconstruction["ranking"],
            targets["ranking"],
            reconstruction["selection"],
            targets["selection"],
            reconstruction["test"],
            targets["test"],
        )

        codes_cpu = {
            split: stack_encoded(encoded[split]["codes"]) for split in targets
        }
        ranking = rank_by_abs_correlation(
            codes_cpu["ranking"].to(device), targets["ranking"].to(device)
        ).cpu()
        sparse_report, sparse_prediction = sparse_target_reports(
            codes_cpu["ranking"][:, ranking[: max(FEATURE_COUNTS)]].to(device),
            codes_cpu["selection"][:, ranking[: max(FEATURE_COUNTS)]].to(device),
            codes_cpu["test"][:, ranking[: max(FEATURE_COUNTS)]].to(device),
            targets["ranking"],
            targets["selection"],
            targets["test"],
            torch.arange(max(FEATURE_COUNTS), device=device),
        )
        model_results[name] = {
            "method": method_name(name),
            "seed": int(payload["spec"]["seed"]),
            "spec": payload["spec"],
            "reconstruction_dense": reconstruction_report,
            "sparse_target": sparse_report,
            "top64_feature_indices": ranking[: max(FEATURE_COUNTS)].tolist(),
        }
        predictions_reconstruction[name] = reconstruction_prediction
        predictions_sparse64[name] = sparse_prediction
        del model, encoded, reconstruction, codes_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()
            peak_gib = torch.cuda.max_memory_allocated(device) / (1 << 30)
            if peak_gib > args.maximum_peak_gpu_gib:
                raise RuntimeError(
                    f"peak allocated GPU memory {peak_gib:.2f} GiB exceeds guard"
                )

    mse_by_seed = {
        value["seed"]: name
        for name, value in model_results.items()
        if value["method"] == "mse"
    }
    paired_valid_target = []
    for index, (name, value) in enumerate(sorted(model_results.items())):
        if value["method"] == "mse":
            continue
        baseline = mse_by_seed[value["seed"]]
        paired_valid_target.append(
            {
                "baseline": baseline,
                "candidate": name,
                "method": value["method"],
                "seed": value["seed"],
                "reconstruction_dense": bootstrap_r2_difference(
                    targets["test"],
                    predictions_reconstruction[baseline],
                    predictions_reconstruction[name],
                    seed=args.seed + index,
                    samples=args.bootstrap_samples,
                ),
                "sparse_top64": bootstrap_r2_difference(
                    targets["test"],
                    predictions_sparse64[baseline],
                    predictions_sparse64[name],
                    seed=args.seed + 100 + index,
                    samples=args.bootstrap_samples,
                ),
            }
        )

    equal_effect = equal_effect_report(
        selection_models,
        test_models,
        support_quantile=args.equal_effect_support_quantile,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    finished = datetime.now(timezone.utc)
    result = {
        "complete": True,
        "experiment": "exp04b_valid_target_equal_effect",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "wall_seconds": (finished - started).total_seconds(),
        "repository": git_revision(ROOT),
        "protocol": {
            "target": "unit ranking-split clean-minus-ABC mean-difference projection",
            "target_formula": "t(h)=h^T w, w=(mean_clean-mean_abc)/norm(mean_clean-mean_abc)",
            "target_training_split": "ranking only",
            "states_per_split": "clean and ABC stacked",
            "ridge_selection_split": "selection only",
            "ridge_grid": list(RIDGE_GRID),
            "sparse_feature_ranking": "absolute ranking-split target correlation",
            "feature_counts": list(FEATURE_COUNTS),
            "dense_gate_test_r2": args.dense_gate_r2,
            "equal_effect_target": (
                "90% by default through common MSE/DPSAE validation support"
            ),
            "equal_effect_test_status": (
                "descriptive frontier interpolation at validation-frozen target"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in required.items()
        },
        "resources": {
            "device": str(device),
            "torch_version": torch.__version__,
            "minimum_free_gib_guard": args.minimum_free_gib,
            "free_gib_at_start": free_bytes / (1 << 30),
            "gpu_memory_fraction_cap": (
                args.gpu_memory_fraction if device.type == "cuda" else None
            ),
            "peak_allocated_gpu_gib": (
                torch.cuda.max_memory_allocated(device) / (1 << 30)
                if device.type == "cuda"
                else None
            ),
        },
        "valid_target": {
            "direction_norm_before_normalization": float(direction_norm),
            "direction_sha256": hashlib.sha256(
                direction.numpy().tobytes()
            ).hexdigest(),
            "target_stats": target_stats,
            "original_dense_gate": dense_gate,
            "models": model_results,
            "paired_test_r2": paired_valid_target,
        },
        "equal_effect": equal_effect,
    }
    atomic_json(output_dir / "result.json", result)
    return result


def self_test() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(120, 6, generator=generator)
    w = torch.randn(6, generator=generator)
    y = x @ w
    report, prediction = select_ridge(
        x[:60], y[:60], x[60:90], y[60:90], x[90:], y[90:]
    )
    assert report["test"]["r2"] > 0.999
    assert r2_score(prediction, y[90:]) > 0.999
    curve = [
        {"features": 1, "effect": 0.2, "kl": 0.01},
        {"features": 2, "effect": 0.8, "kl": 0.05},
    ]
    row = interpolation(curve, 0.5)
    assert row is not None and math.isclose(row["kl"], 0.03)
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense-gate-r2", type=float, default=0.8)
    parser.add_argument("--equal-effect-support-quantile", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027071417)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.08)
    parser.add_argument("--maximum-peak-gpu-gib", type=float, default=6.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 0 < args.equal_effect_support_quantile < 1:
        parser.error("equal-effect-support-quantile must lie in (0,1)")
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    result = run(args)
    print(json.dumps({
        "complete": result["complete"],
        "result": str(args.output / "result.json"),
        "wall_seconds": result["wall_seconds"],
        "dense_gate": result["valid_target"]["original_dense_gate"]["test"],
        "equal_effect_status": result["equal_effect"]["status"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
