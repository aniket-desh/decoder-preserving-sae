"""Outcome-independent selection rules for the matched-NMSE spectral control."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .language_training import SAETrainSpec


def _validate_protocol(
    *,
    beta_grid: Sequence[float],
    target_nmse_ratio: float,
    matching_tolerance: float,
    decoder_reduction_margin: float,
) -> tuple[float, ...]:
    betas = tuple(float(beta) for beta in beta_grid)
    if not betas or len(set(betas)) != len(betas):
        raise ValueError("beta_grid must contain unique values")
    if any(not math.isfinite(beta) or beta <= 0 for beta in betas):
        raise ValueError("beta_grid values must be finite and positive")
    if not math.isfinite(target_nmse_ratio) or target_nmse_ratio <= 0:
        raise ValueError("target_nmse_ratio must be finite and positive")
    if not math.isfinite(matching_tolerance) or matching_tolerance < 0:
        raise ValueError("matching_tolerance must be finite and nonnegative")
    if not math.isfinite(decoder_reduction_margin) or decoder_reduction_margin < 0:
        raise ValueError("decoder_reduction_margin must be finite and nonnegative")
    return betas


def screen_specs(
    *,
    k: int,
    seed: int,
    decoder_weight: float,
    beta_grid: Sequence[float],
) -> list[SAETrainSpec]:
    """Build a paired MSE/DPSAE anchor and the spectral coefficient grid."""

    betas = _validate_protocol(
        beta_grid=beta_grid,
        target_nmse_ratio=1.0,
        matching_tolerance=0.0,
        decoder_reduction_margin=0.0,
    )
    if not math.isfinite(decoder_weight) or decoder_weight <= 0:
        raise ValueError("decoder_weight must be finite and positive")
    specs = [
        SAETrainSpec(f"mse_s{seed}", "mse", seed, k),
        SAETrainSpec(
            f"dpsae_s{seed}",
            "dpsae",
            seed,
            k,
            decoder_weight=float(decoder_weight),
        ),
    ]
    specs.extend(
        SAETrainSpec(
            f"spectral_b{format(beta, '.12g')}_s{seed}",
            "spectral",
            seed,
            k,
            loss_weight=beta,
        )
        for beta in betas
    )
    return specs


def confirmation_specs(
    *,
    k: int,
    seeds: Sequence[int],
    decoder_weight: float,
    spectral_beta: float,
) -> list[SAETrainSpec]:
    """Build the three paired confirmatory fleets after a positive screen."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("confirmation seeds must be nonempty and unique")
    if not math.isfinite(decoder_weight) or decoder_weight <= 0:
        raise ValueError("decoder_weight must be finite and positive")
    if not math.isfinite(spectral_beta) or spectral_beta <= 0:
        raise ValueError("spectral_beta must be finite and positive")
    result: list[SAETrainSpec] = []
    for seed in seeds:
        result.extend(
            [
                SAETrainSpec(f"mse_s{seed}", "mse", seed, k),
                SAETrainSpec(
                    f"dpsae_s{seed}",
                    "dpsae",
                    seed,
                    k,
                    decoder_weight=float(decoder_weight),
                ),
                SAETrainSpec(
                    f"spectral_s{seed}",
                    "spectral",
                    seed,
                    k,
                    loss_weight=float(spectral_beta),
                ),
            ]
        )
    return result


def _metric(metrics: Mapping[str, Mapping[str, Any]], name: str, key: str) -> float:
    try:
        value = float(metrics[name][key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} requires numeric {key}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} has invalid {key}")
    return value


def select_matched_spectral(
    metrics: Mapping[str, Mapping[str, Any]],
    specs: Sequence[SAETrainSpec],
    *,
    split: str,
    target_nmse_ratio: float,
    matching_tolerance: float,
    decoder_reduction_margin: float,
) -> dict[str, Any]:
    """Apply the sealed match and advancement rule on the selection split.

    The closest spectral NMSE ratio inside the inclusive tolerance is selected;
    exact distance ties resolve to the smaller coefficient. It advances only
    when its decoder-distortion reduction is no more than ``margin`` below the
    paired high-weight DPSAE anchor.
    """

    if split != "selection":
        raise ValueError("the spectral screen must be selected on selection data")
    betas = _validate_protocol(
        beta_grid=[spec.loss_weight for spec in specs if spec.method == "spectral"],
        target_nmse_ratio=target_nmse_ratio,
        matching_tolerance=matching_tolerance,
        decoder_reduction_margin=decoder_reduction_margin,
    )
    mse_specs = [spec for spec in specs if spec.method == "mse"]
    dpsae_specs = [spec for spec in specs if spec.method == "dpsae"]
    spectral_specs = [spec for spec in specs if spec.method == "spectral"]
    if len(mse_specs) != 1 or len(dpsae_specs) != 1 or len(spectral_specs) != len(betas):
        raise ValueError("screen requires one MSE, one DPSAE, and the spectral grid")
    mse, dpsae = mse_specs[0], dpsae_specs[0]
    if dpsae.seed != mse.seed or any(spec.seed != mse.seed for spec in spectral_specs):
        raise ValueError("all screen models must share the paired seed")

    mse_nmse = _metric(metrics, mse.name, "nmse")
    mse_decoder = _metric(metrics, mse.name, "decoder")
    if mse_nmse <= 0 or mse_decoder <= 0:
        raise ValueError("paired MSE denominators must be positive")
    dpsae_nmse = _metric(metrics, dpsae.name, "nmse")
    dpsae_decoder = _metric(metrics, dpsae.name, "decoder")
    dpsae_reduction = 1.0 - dpsae_decoder / mse_decoder

    candidates = []
    for spec in spectral_specs:
        nmse = _metric(metrics, spec.name, "nmse")
        decoder = _metric(metrics, spec.name, "decoder")
        ratio = nmse / mse_nmse
        distance = abs(ratio - target_nmse_ratio)
        candidates.append(
            {
                "spec": asdict(spec),
                "metrics": {"nmse": nmse, "decoder": decoder},
                "nmse_ratio": ratio,
                "target_distance": distance,
                "within_tolerance": distance <= matching_tolerance + 1e-12,
                "decoder_reduction": 1.0 - decoder / mse_decoder,
            }
        )

    matched = [candidate for candidate in candidates if candidate["within_tolerance"]]
    report: dict[str, Any] = {
        "complete": True,
        "selected_on": split,
        "rule": {
            "target_nmse_ratio": float(target_nmse_ratio),
            "matching_tolerance": float(matching_tolerance),
            "tie_break": "smaller_beta",
            "decoder_reduction_margin": float(decoder_reduction_margin),
            "advance_if": "dpsae_reduction - spectral_reduction <= margin",
        },
        "paired_mse": {
            "spec": asdict(mse),
            "metrics": {"nmse": mse_nmse, "decoder": mse_decoder},
        },
        "dpsae_anchor": {
            "spec": asdict(dpsae),
            "metrics": {"nmse": dpsae_nmse, "decoder": dpsae_decoder},
            "nmse_ratio": dpsae_nmse / mse_nmse,
            "decoder_reduction": dpsae_reduction,
        },
        "candidates": candidates,
        "selected": None,
        "advance": False,
        "status": "no_matching_candidate",
    }
    if not matched:
        return report

    selected = min(
        matched,
        key=lambda candidate: (
            round(candidate["target_distance"], 12),
            candidate["spec"]["loss_weight"],
        ),
    )
    gap = dpsae_reduction - float(selected["decoder_reduction"])
    advance = gap <= decoder_reduction_margin + 1e-12
    report.update(
        selected=selected,
        advance=advance,
        status="advance" if advance else "noncompetitive_match",
        decoder_reduction_gap=gap,
    )
    return report


def summarize_confirmation(
    metrics: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    """Return per-seed paired ratios and reductions without adding a new gate."""

    rows = []
    for seed in seeds:
        mse_name = f"mse_s{seed}"
        mse_nmse = _metric(metrics, mse_name, "nmse")
        mse_decoder = _metric(metrics, mse_name, "decoder")
        if mse_nmse <= 0 or mse_decoder <= 0:
            raise ValueError("paired MSE denominators must be positive")
        row: dict[str, Any] = {"seed": int(seed)}
        for method in ("dpsae", "spectral"):
            name = f"{method}_s{seed}"
            nmse = _metric(metrics, name, "nmse")
            decoder = _metric(metrics, name, "decoder")
            row[method] = {
                "nmse": nmse,
                "decoder": decoder,
                "nmse_ratio": nmse / mse_nmse,
                "decoder_reduction": 1.0 - decoder / mse_decoder,
            }
        row["spectral_minus_dpsae_reduction"] = (
            row["spectral"]["decoder_reduction"]
            - row["dpsae"]["decoder_reduction"]
        )
        rows.append(row)
    return {"complete": True, "seeds": rows, "confirmatory_gate": None}
