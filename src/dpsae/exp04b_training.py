"""Training-fleet construction and selection rules for Experiment 4b."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .language_training import SAETrainSpec


STATIC_BASELINES = ("whitening", "spectral")
NMSE_CAP_RATIO = 1.10


def _beta_name(beta: float) -> str:
    return format(beta, ".12g")


def _validate_betas(beta_grid: Sequence[float]) -> tuple[float, ...]:
    betas = tuple(float(beta) for beta in beta_grid)
    if not betas:
        raise ValueError("beta_grid must not be empty")
    if any(not math.isfinite(beta) or beta <= 0 for beta in betas):
        raise ValueError("beta_grid values must be finite and strictly positive")
    if len(set(betas)) != len(betas):
        raise ValueError("beta_grid values must be unique")
    return betas


def screen_specs(
    *,
    k: int,
    seed: int,
    dpsae_weight: float,
    beta_grid: Sequence[float],
) -> list[SAETrainSpec]:
    """Build the paired MSE, fixed-DPSAE, and static-baseline screen."""

    if not math.isfinite(dpsae_weight) or dpsae_weight <= 0:
        raise ValueError("dpsae_weight must be finite and strictly positive")
    betas = _validate_betas(beta_grid)
    specs = [
        SAETrainSpec(f"mse_s{seed}", "mse", seed, k),
        SAETrainSpec(
            f"dpsae_s{seed}",
            "dpsae",
            seed,
            k,
            decoder_weight=float(dpsae_weight),
        ),
    ]
    specs.extend(
        SAETrainSpec(
            f"{method}_b{_beta_name(beta)}_s{seed}",
            method,
            seed,
            k,
            loss_weight=beta,
        )
        for method in STATIC_BASELINES
        for beta in betas
    )
    return specs


def _finite_metric(metrics: Mapping[str, Any], key: str, model_name: str) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{model_name} requires a numeric {key} metric") from error
    if not math.isfinite(value):
        raise ValueError(f"{model_name} has nonfinite {key}")
    if value < 0:
        raise ValueError(f"{model_name} has negative {key}")
    return value


def select_static_baselines(
    metrics: Mapping[str, Mapping[str, Any]],
    specs: Sequence[SAETrainSpec],
    *,
    split: str,
) -> dict[str, Any]:
    """Select each static baseline on fresh data under the paired-MSE NMSE cap.

    Selection is performed separately for whitening and spectral candidates. A
    method with no candidate satisfying the cap is retained in the report with
    ``status == "no_qualifying_candidate"`` and is omitted from confirmation.
    """

    if split != "selection":
        raise ValueError("static baselines must be selected on the fresh selection split")

    mse_specs = [spec for spec in specs if spec.method == "mse"]
    if len(mse_specs) != 1:
        raise ValueError("screen specs must contain exactly one paired MSE model")
    mse_spec = mse_specs[0]
    if mse_spec.name not in metrics:
        raise ValueError(f"missing selection metrics for {mse_spec.name}")
    mse_nmse = _finite_metric(metrics[mse_spec.name], "nmse", mse_spec.name)
    mse_decoder = _finite_metric(metrics[mse_spec.name], "decoder", mse_spec.name)
    nmse_cap = NMSE_CAP_RATIO * mse_nmse

    report: dict[str, Any] = {
        "selected_on": split,
        "nmse_cap_ratio": NMSE_CAP_RATIO,
        "nmse_cap": nmse_cap,
        "paired_mse": {
            "name": mse_spec.name,
            "nmse": mse_nmse,
            "decoder": mse_decoder,
        },
        "baselines": {},
    }
    for method in STATIC_BASELINES:
        candidates = [spec for spec in specs if spec.method == method]
        scored = []
        for spec in candidates:
            if spec.seed != mse_spec.seed:
                raise ValueError(f"{spec.name} is not paired to MSE seed {mse_spec.seed}")
            if spec.name not in metrics:
                raise ValueError(f"missing selection metrics for {spec.name}")
            nmse = _finite_metric(metrics[spec.name], "nmse", spec.name)
            decoder = _finite_metric(metrics[spec.name], "decoder", spec.name)
            scored.append((spec, nmse, decoder))

        qualifying = [row for row in scored if row[1] <= nmse_cap]
        method_report: dict[str, Any] = {
            "status": "no_qualifying_candidate",
            "candidate_count": len(scored),
            "qualifying_count": len(qualifying),
            "candidates": [
                {
                    "spec": asdict(spec),
                    "metrics": {"nmse": nmse, "decoder": decoder},
                    "qualifies": nmse <= nmse_cap,
                }
                for spec, nmse, decoder in scored
            ],
            "selected_spec": None,
            "selected_metrics": None,
        }
        if qualifying:
            spec, nmse, decoder = min(
                qualifying,
                key=lambda row: (row[2], row[1], row[0].loss_weight, row[0].name),
            )
            method_report.update(
                status="selected",
                selected_spec=asdict(spec),
                selected_metrics={"nmse": nmse, "decoder": decoder},
            )
        report["baselines"][method] = method_report
    return report


def confirmation_specs(
    *,
    k: int,
    seeds: Sequence[int],
    dpsae_weight: float,
    selection: Mapping[str, Any],
) -> list[SAETrainSpec]:
    """Build confirmation specs, omitting static baselines that missed the gate."""

    if not seeds:
        raise ValueError("confirmation seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("confirmation seeds must be unique")
    if not math.isfinite(dpsae_weight) or dpsae_weight <= 0:
        raise ValueError("dpsae_weight must be finite and strictly positive")
    if selection.get("selected_on") != "selection":
        raise ValueError("confirmation requires a fresh-selection baseline report")

    selected_weights: dict[str, float] = {}
    baseline_reports = selection.get("baselines")
    if not isinstance(baseline_reports, Mapping):
        raise ValueError("selection report is missing baselines")
    for method in STATIC_BASELINES:
        method_report = baseline_reports.get(method)
        if not isinstance(method_report, Mapping):
            raise ValueError(f"selection report is missing {method}")
        status = method_report.get("status")
        if status == "no_qualifying_candidate":
            continue
        if status != "selected":
            raise ValueError(f"invalid {method} selection status: {status}")
        selected_spec = method_report.get("selected_spec")
        if not isinstance(selected_spec, Mapping):
            raise ValueError(f"selected {method} baseline is missing its spec")
        if selected_spec.get("method") != method:
            raise ValueError(f"selected {method} spec has the wrong method")
        try:
            weight = float(selected_spec["loss_weight"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"selected {method} spec has no numeric loss weight") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"selected {method} loss weight must be finite and positive")
        selected_weights[method] = weight

    specs = []
    for seed in seeds:
        specs.extend(
            [
                SAETrainSpec(f"mse_s{seed}", "mse", seed, k),
                SAETrainSpec(
                    f"dpsae_s{seed}",
                    "dpsae",
                    seed,
                    k,
                    decoder_weight=float(dpsae_weight),
                ),
            ]
        )
        specs.extend(
            SAETrainSpec(
                f"{method}_s{seed}",
                method,
                seed,
                k,
                loss_weight=selected_weights[method],
            )
            for method in STATIC_BASELINES
            if method in selected_weights
        )
    return specs


def _stable_seed(base_seed: int, stage: str, stream: str, replicate: int) -> int:
    payload = f"{base_seed}:{stage}:{stream}:{replicate}".encode()
    value = int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big")
    return value % (2**31 - 1)


@dataclass(frozen=True)
class StageSeeds:
    """Independent deterministic seeds for data order and decoder probes."""

    stage: str
    replicate: int
    data_order: int
    probe_sequence: int


def stage_seeds(base_seed: int, stage: str, *, replicate: int = 0) -> StageSeeds:
    if not stage:
        raise ValueError("stage must not be empty")
    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    return StageSeeds(
        stage=stage,
        replicate=replicate,
        data_order=_stable_seed(base_seed, stage, "data_order", replicate),
        probe_sequence=_stable_seed(base_seed, stage, "probe_sequence", replicate),
    )


def probe_seed_for_step(seeds: StageSeeds, step: int) -> int:
    """Return the deterministic probe seed for one zero-indexed training step."""

    if step < 0:
        raise ValueError("step must be nonnegative")
    return _stable_seed(seeds.probe_sequence, seeds.stage, "probe_step", step)


def confirmation_replicate_config(
    config: Mapping[str, Any], *, replicate: int
) -> dict[str, Any]:
    """Copy a config and attach independent confirmation-replicate randomness."""

    if replicate <= 0:
        raise ValueError("a changed confirmation replicate must be greater than zero")
    try:
        base_seed = int(config["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config requires an integer seed") from error
    result = copy.deepcopy(dict(config))
    seeds = stage_seeds(base_seed, "confirmation", replicate=replicate)
    result["randomness"] = asdict(seeds)
    return result
