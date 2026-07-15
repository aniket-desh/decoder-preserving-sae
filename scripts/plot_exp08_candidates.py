#!/usr/bin/env python3
"""Render review-only Exp08 candidate figures from sealed result artifacts.

The script is deliberately downstream of every experimental gate.  It validates
the complete input bundle before rendering and never writes into ``paper/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from dpsae.plot_style import (
    METHOD_STYLES,
    NEUTRAL,
    SEMANTIC,
    clean_axis,
    figure_size,
    paper_context,
    save_figure,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = (0, 1, 2)
STRUCTURED_METHODS = (
    "task_prior",
    "isotropic",
    "weighted_mse",
    "permuted_prior",
)
METHOD_KEYS = {
    "task_prior": "task_prior",
    "isotropic": "isotropic",
    "weighted_mse": "weighted_mse",
    "permuted_prior": "permuted_prior",
    "dpsae": "isotropic",
    "whitening": "whitened",
    "spectral": "spectral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--structured-baseline-dir", type=Path, required=True)
    parser.add_argument("--static-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required candidate-figure input is missing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    file_record(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed JSON input: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    file_record(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = set(required) - fieldnames
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"required table is empty: {path}")
    return rows


def number(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite, got {result}")
    return result


def integer(value: Any, *, context: str) -> int:
    result = number(value, context=context)
    if not result.is_integer():
        raise ValueError(f"{context} must be an integer, got {result}")
    return int(result)


def require_complete(payload: Mapping[str, Any], *, name: str, experiment: str) -> None:
    if payload.get("complete") is not True:
        raise ValueError(f"{name} is incomplete")
    if payload.get("experiment") != experiment:
        raise ValueError(
            f"{name} has experiment={payload.get('experiment')!r}, expected {experiment!r}"
        )


def require_repository(
    payload: Mapping[str, Any], repository: Mapping[str, Any], *, name: str
) -> None:
    if payload.get("repository") != repository:
        raise ValueError(f"{name} was not produced by the Exp08 run-manifest revision")


def require_unique(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, name: str
) -> None:
    observed = [tuple(row[key] for key in keys) for row in rows]
    if len(set(observed)) != len(observed):
        raise ValueError(f"{name} has duplicate rows for key {tuple(keys)}")


def validate_output_location(experiment_root: Path, output_dir: Path) -> None:
    expected = (experiment_root / "candidate_figures").resolve()
    if output_dir.resolve() != expected:
        raise ValueError(
            "candidate figures must be written exactly to "
            f"{experiment_root / 'candidate_figures'}"
        )
    if "paper" in output_dir.resolve().parts:
        raise ValueError("candidate figures may not be written into paper/")


def validate_manifest_hash(
    run_manifest: Mapping[str, Any], key: str, path: Path
) -> None:
    try:
        expected = run_manifest["external_inputs"][key]["sha256"]
    except KeyError as error:
        raise ValueError(f"run manifest is missing external input {key!r}") from error
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{key} changed after the Exp08 run contract was created: {path}"
        )


def validate_code_hash(run_manifest: Mapping[str, Any], relative: str) -> None:
    path = ROOT / relative
    try:
        expected = run_manifest["code"][relative]["sha256"]
    except KeyError as error:
        raise ValueError(f"run manifest is missing code input {relative!r}") from error
    if sha256_file(path) != expected:
        raise ValueError(f"candidate renderer code changed after run-manifest creation: {path}")


def panel_heading(
    ax: plt.Axes,
    label: str,
    title: str,
    subtitle: str,
    *,
    y: float = 1.075,
    label_x: float = -0.17,
    title_x: float = 0.0,
) -> None:
    """Place every panel label, title, and subtitle on fixed baselines."""

    ax.text(
        label_x,
        y,
        label,
        transform=ax.transAxes,
        color=NEUTRAL["text"],
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        title_x,
        y,
        title,
        transform=ax.transAxes,
        color=NEUTRAL["text"],
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        title_x,
        y - 0.075,
        subtitle,
        transform=ax.transAxes,
        color=NEUTRAL["muted"],
        fontsize=6.5,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cannot summarize an empty or nonfinite sample")
    low, median, high = np.quantile(array, (0.1, 0.5, 0.9))
    return float(low), float(median), float(high)


def method_handle(method: str) -> Line2D:
    style = METHOD_STYLES[METHOD_KEYS.get(method, method)]
    return Line2D(
        [0],
        [0],
        color=style.color,
        marker=style.marker,
        linestyle=style.linestyle,
        linewidth=1.3,
        markersize=4.5,
        label=style.label,
    )


def forest(
    ax: plt.Axes,
    values: Mapping[tuple[str, str], Sequence[float]],
    *,
    methods: Sequence[str],
    metric_keys: Sequence[str],
    metric_labels: Sequence[str],
) -> None:
    if len(metric_keys) != len(metric_labels):
        raise ValueError("forest labels do not match metric keys")
    centers = np.arange(len(metric_keys), dtype=float)[::-1]
    offsets = np.linspace(0.25, -0.25, len(methods))
    for offset, method in zip(offsets, methods, strict=True):
        style = METHOD_STYLES[METHOD_KEYS.get(method, method)]
        for center, metric in zip(centers, metric_keys, strict=True):
            sample = list(values[(method, metric)])
            low, median, high = quantiles(sample)
            y = center + offset
            ax.scatter(
                sample,
                np.full(len(sample), y),
                s=9,
                marker=style.marker,
                facecolors=style.color,
                edgecolors="none",
                alpha=0.27,
                zorder=2,
            )
            ax.plot([low, high], [y, y], color=style.color, linewidth=1.7, zorder=3)
            ax.scatter(
                [median],
                [y],
                s=25,
                marker=style.marker,
                facecolors=style.color,
                edgecolors=NEUTRAL["white"],
                linewidths=0.45,
                zorder=4,
            )
    ax.axvline(0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.9)
    ax.set_yticks(centers, metric_labels)
    ax.tick_params(axis="y", length=0)
    clean_axis(ax)


def transformed_interval(
    estimate: float, interval: Sequence[float], *, sign: float, scale: float
) -> tuple[float, float, float]:
    if len(interval) != 2:
        raise ValueError("confidence interval must have two endpoints")
    low = number(interval[0], context="confidence interval lower endpoint")
    high = number(interval[1], context="confidence interval upper endpoint")
    if low > high:
        raise ValueError("confidence interval endpoints are reversed")
    center = sign * scale * estimate
    endpoints = sorted((sign * scale * low, sign * scale * high))
    return center, endpoints[0], endpoints[1]


def validate_inputs(
    experiment_root: Path,
    structured_baseline_dir: Path,
    static_baseline_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        "run_manifest": experiment_root / "run_manifest.json",
        "structured_sweep_paired": experiment_root
        / "synthetic_prior_sweep/paired_metrics.csv",
        "structured_sweep_metadata": experiment_root
        / "synthetic_prior_sweep/metadata.json",
        "gamma_sweep": experiment_root / "gamma_sweep_selection.json",
        "gamma_choice": experiment_root / "gamma_sweep_choice.json",
        "confirmation": experiment_root / "confirmation_summary.json",
        "frozen_fidelity": experiment_root / "evidence/frozen_fidelity.json",
        "robustness": experiment_root / "evidence/robustness.json",
        "task_spectrum": experiment_root
        / "task_spectrum/advantage_spectrum_summary.json",
        "static_baseline": static_baseline_path,
        "structured_metrics": structured_baseline_dir / "metrics.csv",
        "structured_group_metrics": structured_baseline_dir / "group_metrics.csv",
        "structured_metadata": structured_baseline_dir / "metadata.json",
        "structured_crossover": structured_baseline_dir / "crossover.csv",
    }
    for path in paths.values():
        file_record(path)

    run_manifest = read_json(paths["run_manifest"])
    require_complete(
        run_manifest,
        name="run manifest",
        experiment="exp08_experiment_figure_closure",
    )
    repository = run_manifest.get("repository")
    if not isinstance(repository, dict) or repository.get("dirty") is not False:
        raise ValueError("Exp08 run manifest must bind a clean repository")
    for relative in ("scripts/plot_exp08_candidates.py", "src/dpsae/plot_style.py"):
        validate_code_hash(run_manifest, relative)

    external_hashes = {
        "static_baseline_evaluation": paths["static_baseline"],
        "structured_baseline_metrics": paths["structured_metrics"],
        "structured_baseline_group_metrics": paths["structured_group_metrics"],
        "structured_baseline_metadata": paths["structured_metadata"],
        "structured_baseline_crossover": paths["structured_crossover"],
    }
    for key, path in external_hashes.items():
        validate_manifest_hash(run_manifest, key, path)

    sweep_meta = read_json(paths["structured_sweep_metadata"])
    require_complete(
        sweep_meta,
        name="structured task-prior sweep",
        experiment="exp02_prior_weight_sweep",
    )
    if sweep_meta.get("git_revision") != repository.get("revision"):
        raise ValueError("structured task-prior sweep used another code revision")
    semantics = sweep_meta.get("weight_parameterization", {}).get(
        "relative_weight", ""
    )
    if "not a predicted sparse transition" not in semantics:
        raise ValueError("structured sweep does not record the required scale-only semantics")
    reference = sweep_meta.get("config", {}).get("empirical_crossover", {}).get(
        "relative_weight_reference"
    )
    if reference != "separate_two_direction_crossover_not_a_sparse_transition":
        raise ValueError("structured sweep mislabeled the 2D reference as a sparse transition")

    gamma = read_json(paths["gamma_sweep"])
    require_complete(
        gamma,
        name="gamma sweep",
        experiment="paper_closure_frontier_existing",
    )
    require_repository(gamma, repository, name="gamma sweep")
    choice = read_json(paths["gamma_choice"])
    require_complete(
        choice,
        name="gamma selection",
        experiment="paper_closure_frontier_selection",
    )
    require_repository(choice, repository, name="gamma selection")
    confirmation = read_json(paths["confirmation"])
    require_complete(
        confirmation,
        name="clean confirmation",
        experiment="exp08_clean_confirmation_summary",
    )
    require_repository(confirmation, repository, name="clean confirmation")
    if confirmation.get("gate_passed") is not True:
        raise ValueError("clean confirmation did not pass the preregistered gate")

    frozen = read_json(paths["frozen_fidelity"])
    require_complete(
        frozen,
        name="frozen-language-model evaluation",
        experiment="exp08_frozen_language_model_fidelity",
    )
    require_repository(frozen, repository, name="frozen-language-model evaluation")
    robustness = read_json(paths["robustness"])
    require_complete(
        robustness,
        name="robustness evaluation",
        experiment="exp08_matched_quality_robustness",
    )
    require_repository(robustness, repository, name="robustness evaluation")
    spectrum = read_json(paths["task_spectrum"])
    require_complete(
        spectrum,
        name="taskwise spectrum",
        experiment="taskwise_decoder_advantage_spectrum_summary",
    )
    require_repository(spectrum, repository, name="taskwise spectrum")

    static = read_json(paths["static_baseline"])
    if static.get("complete") is not True:
        raise ValueError("static-control baseline is incomplete")
    static_repository = static.get("protocol", {}).get("repository", {})
    if static_repository.get("dirty") is not False:
        raise ValueError("static-control baseline did not come from a clean revision")

    structured_meta = read_json(paths["structured_metadata"])
    if structured_meta.get("smoke") is not False:
        raise ValueError("structured baseline must be a full run, not a smoke run")
    seeds = tuple(int(seed) for seed in structured_meta.get("config", {}).get("seeds", ()))
    if seeds != tuple(range(10)):
        raise ValueError("structured baseline must contain seeds 0 through 9")

    payloads = {
        "run_manifest": run_manifest,
        "structured_sweep_metadata": sweep_meta,
        "gamma_sweep": gamma,
        "gamma_choice": choice,
        "confirmation": confirmation,
        "frozen_fidelity": frozen,
        "robustness": robustness,
        "task_spectrum": spectrum,
        "static_baseline": static,
        "structured_metadata": structured_meta,
    }
    return payloads, paths


def structured_data(
    paths: Mapping[str, Path], payloads: Mapping[str, Any]
) -> dict[str, Any]:
    paired = read_csv(
        paths["structured_sweep_paired"],
        (
            "seed",
            "relative_weight",
            "protected_reduction_vs_mse",
            "nmse_reduction_vs_mse",
        ),
    )
    require_unique(paired, ("seed", "relative_weight"), name="structured sweep")
    sweep_config = payloads["structured_sweep_metadata"]["config"]
    expected_seeds = tuple(int(seed) for seed in sweep_config["seeds"])
    expected_weights = tuple(
        float(value) for value in sweep_config["empirical_crossover"]["relative_weights"]
    )
    observed = {
        (
            integer(row["seed"], context="structured sweep seed"),
            number(row["relative_weight"], context="structured sweep relative weight"),
        )
        for row in paired
    }
    expected = {(seed, weight) for seed in expected_seeds for weight in expected_weights}
    if observed != expected:
        raise ValueError("structured sweep does not contain every configured seed-weight pair")
    sweep_by_seed: dict[int, dict[float, float]] = defaultdict(dict)
    for row in paired:
        seed = integer(row["seed"], context="structured sweep seed")
        weight = number(row["relative_weight"], context="structured sweep weight")
        sweep_by_seed[seed][weight] = 100 * number(
            row["protected_reduction_vs_mse"],
            context="protected reduction versus MSE",
        )

    metrics = read_csv(
        paths["structured_metrics"],
        (
            "seed",
            "method",
            "test_nmse",
            "protected_decoder_distortion",
            "unrelated_decoder_distortion",
            "isotropic_decoder_distortion",
        ),
    )
    require_unique(metrics, ("seed", "method"), name="structured baseline metrics")
    metric_lookup = {
        (integer(row["seed"], context="structured seed"), row["method"]): row
        for row in metrics
    }
    required_methods = ("mse", *STRUCTURED_METHODS)
    expected_metric_keys = {
        (seed, method) for seed in range(10) for method in required_methods
    }
    if set(metric_lookup) != expected_metric_keys:
        raise ValueError("structured baseline metrics do not contain the complete paired fleet")

    metric_keys = (
        "test_nmse",
        "protected_decoder_distortion",
        "unrelated_decoder_distortion",
        "isotropic_decoder_distortion",
    )
    reductions: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed in range(10):
        baseline = metric_lookup[(seed, "mse")]
        for method in STRUCTURED_METHODS:
            candidate = metric_lookup[(seed, method)]
            for metric in metric_keys:
                base = number(baseline[metric], context=f"MSE {metric}")
                value = number(candidate[metric], context=f"{method} {metric}")
                if base <= 0:
                    raise ValueError(f"MSE {metric} must be positive")
                reductions[(method, metric)].append(100 * (1 - value / base))

    groups = read_csv(
        paths["structured_group_metrics"],
        ("seed", "method", "group", "matched_cosine", "support_f1"),
    )
    require_unique(groups, ("seed", "method", "group"), name="structured groups")
    protected = {
        (integer(row["seed"], context="structured group seed"), row["method"]): row
        for row in groups
        if row["group"] == "protected"
    }
    if set(protected) != expected_metric_keys:
        raise ValueError("protected-feature table does not contain the complete paired fleet")
    recovery: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed in range(10):
        baseline = protected[(seed, "mse")]
        for method in STRUCTURED_METHODS:
            candidate = protected[(seed, method)]
            for metric in ("matched_cosine", "support_f1"):
                recovery[(method, metric)].append(
                    100
                    * (
                        number(candidate[metric], context=f"{method} {metric}")
                        - number(baseline[metric], context=f"MSE {metric}")
                    )
                )
    return {
        "weights": expected_weights,
        "sweep_by_seed": dict(sweep_by_seed),
        "reductions": dict(reductions),
        "recovery": dict(recovery),
    }


def plot_structured(
    data: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    weights = np.asarray(data["weights"], dtype=float)
    sweep_matrix = np.asarray(
        [
            [data["sweep_by_seed"][seed][float(weight)] for weight in weights]
            for seed in sorted(data["sweep_by_seed"])
        ]
    )
    q10, median, q90 = np.quantile(sweep_matrix, (0.1, 0.5, 0.9), axis=0)
    style = METHOD_STYLES["task_prior"]
    with paper_context():
        fig, axes = plt.subplots(1, 3, figsize=figure_size("full", aspect=0.64))
        ax = axes[0]
        for row in sweep_matrix:
            ax.plot(weights, row, color=style.color, linewidth=0.6, alpha=0.2)
            ax.scatter(weights, row, color=style.color, marker=style.marker, s=7, alpha=0.2)
        ax.fill_between(weights, q10, q90, color=style.color, alpha=0.13, linewidth=0)
        ax.plot(
            weights,
            median,
            color=style.color,
            marker=style.marker,
            linewidth=1.8,
            markersize=4,
            zorder=4,
        )
        ax.axhline(0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.9)
        ax.axvline(1, color=NEUTRAL["reference"], linestyle=":", linewidth=0.9)
        ax.text(
            0.48,
            0.78,
            "2D scale reference\nnot a sparse threshold",
            transform=ax.transAxes,
            fontsize=5.8,
            color=NEUTRAL["muted"],
            ha="left",
            va="top",
        )
        ax.set_xscale("log", base=2)
        labeled_weights = (0.25, 0.5, 1.0, 2.0, 4.0)
        ax.set_xticks(labeled_weights, [f"{value:g}" for value in labeled_weights])
        ax.set_xlabel(r"$\beta/\beta_{\star,\mathrm{2D}}$ (scale only)")
        ax.set_ylabel("Protected distortion reduction (%)")
        clean_axis(ax)
        panel_heading(ax, "A", "Task-prior dose response", "10 seeds; median, q10–q90")

        forest(
            axes[1],
            data["reductions"],
            methods=STRUCTURED_METHODS,
            metric_keys=(
                "test_nmse",
                "protected_decoder_distortion",
                "unrelated_decoder_distortion",
                "isotropic_decoder_distortion",
            ),
            metric_labels=("NMSE", "Protected", "Unrelated", "Isotropic"),
        )
        axes[1].set_xlabel("Reduction vs paired MSE (%)")
        panel_heading(
            axes[1],
            "B",
            "Task-family distortion",
            "Seeds; median, q10–q90",
        )

        forest(
            axes[2],
            data["recovery"],
            methods=STRUCTURED_METHODS,
            metric_keys=("matched_cosine", "support_f1"),
            metric_labels=("Matched |cos|", "Support F1"),
        )
        axes[2].set_xlabel("Gain over paired MSE (pp)")
        panel_heading(
            axes[2],
            "C",
            "Protected recovery",
            "Seeds; median, q10–q90",
        )

        fig.legend(
            handles=[method_handle(method) for method in STRUCTURED_METHODS],
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.27, wspace=0.55)
        pdf, png = save_figure(fig, output_dir / "task_prior_candidates")
        plt.close(fig)
    summary = {
        "relative_weights": weights.tolist(),
        "protected_reduction_percent_median": median.tolist(),
        "protected_reduction_percent_q10": q10.tolist(),
        "protected_reduction_percent_q90": q90.tolist(),
        "relative_weight_one_semantics": (
            "separate 2D crossover reference used only as a scale; no predicted sparse transition"
        ),
    }
    return pdf, png, summary


def gamma_rows(payload: Mapping[str, Any]) -> list[dict[str, float]]:
    rows = payload.get("paired_frontier")
    if not isinstance(rows, list) or not rows:
        raise ValueError("gamma sweep has no paired frontier")
    result = []
    for row in rows:
        interval = row.get("exact_decoder_reduction_ci95")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("gamma sweep row has no two-sided decoder interval")
        result.append(
            {
                "weight": number(row.get("decoder_weight"), context="gamma weight"),
                "nmse_change": number(
                    row.get("nmse_change_percent"), context="gamma NMSE change"
                ),
                "decoder_reduction": 100
                * number(
                    row.get("exact_decoder_reduction"), context="gamma decoder reduction"
                ),
                "ci_low": 100 * number(interval[0], context="gamma decoder CI low"),
                "ci_high": 100 * number(interval[1], context="gamma decoder CI high"),
            }
        )
    require_unique(result, ("weight",), name="gamma sweep")
    return sorted(result, key=lambda row: row["weight"])


def confirmation_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("confirmation summary has no rows")
    seeds = tuple(integer(row.get("seed"), context="confirmation seed") for row in rows)
    if seeds != EXPECTED_SEEDS:
        raise ValueError("confirmation summary must contain seeds 0, 1, and 2")
    for row in rows:
        for method in ("mse", "dpsae"):
            report = row.get(method)
            if not isinstance(report, dict):
                raise ValueError(f"confirmation row is missing {method}")
            number(report.get("nmse"), context=f"confirmation {method} NMSE")
            number(
                report.get("exact_decoder_distortion"),
                context=f"confirmation {method} exact decoder distortion",
            )
    return rows


def static_control_rows(payload: Mapping[str, Any]) -> dict[str, list[dict[str, float]]]:
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("static-control baseline has no models")
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for report in models.values():
        spec = report.get("spec", {})
        method = spec.get("method")
        seed = integer(spec.get("seed"), context="static-control seed")
        if method in by_seed[seed]:
            raise ValueError(f"duplicate static-control method {method!r} for seed {seed}")
        by_seed[seed][method] = report
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise ValueError("static-control baseline must contain seeds 0, 1, and 2")
    result: dict[str, list[dict[str, float]]] = defaultdict(list)
    for seed in EXPECTED_SEEDS:
        reports = by_seed[seed]
        if set(reports) != {"mse", "dpsae", "whitening", "spectral"}:
            raise ValueError(f"static-control fleet is incomplete for seed {seed}")
        baseline = reports["mse"]
        base_nmse = number(
            baseline.get("sampled_primary", {}).get("nmse"),
            context="static MSE NMSE",
        )
        base_decoder = number(
            baseline.get("exact_identity_primary", {}).get("decoder_distortion"),
            context="static MSE decoder distortion",
        )
        for method in ("dpsae", "whitening", "spectral"):
            report = reports[method]
            nmse = number(
                report.get("sampled_primary", {}).get("nmse"),
                context=f"static {method} NMSE",
            )
            decoder = number(
                report.get("exact_identity_primary", {}).get("decoder_distortion"),
                context=f"static {method} decoder distortion",
            )
            result[method].append(
                {
                    "seed": seed,
                    "nmse_change": 100 * (nmse / base_nmse - 1),
                    "decoder_reduction": 100 * (1 - decoder / base_decoder),
                }
            )
    return dict(result)


def task_share_rows(payload: Mapping[str, Any]) -> list[dict[str, float]]:
    rows = payload.get("seed_summaries")
    if not isinstance(rows, list):
        raise ValueError("taskwise spectrum has no seed summaries")
    result = []
    for row in rows:
        seed = integer(row.get("seed"), context="task-spectrum seed")
        positive = number(
            row.get("mean_random_direction_material_positive_probability"),
            context="task-spectrum positive share",
        )
        negative = number(
            row.get("mean_random_direction_material_negative_probability"),
            context="task-spectrum negative share",
        )
        unresolved = 1 - positive - negative
        if min(positive, negative, unresolved) < -1e-9 or max(positive, negative) > 1:
            raise ValueError(f"task-spectrum shares are invalid for seed {seed}")
        result.append(
            {
                "seed": seed,
                "positive": 100 * positive,
                "negative": 100 * negative,
                "unresolved": 100 * max(0.0, unresolved),
            }
        )
    result.sort(key=lambda row: row["seed"])
    if tuple(row["seed"] for row in result) != EXPECTED_SEEDS:
        raise ValueError("task-spectrum summary must contain seeds 0, 1, and 2")
    return result


def plot_language_model(
    payloads: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    gamma = gamma_rows(payloads["gamma_sweep"])
    choice = payloads["gamma_choice"]
    selected_weight = number(
        choice.get("selected_decoder_weight"), context="selected gamma"
    )
    selected = [row for row in gamma if math.isclose(row["weight"], selected_weight)]
    if len(selected) != 1:
        raise ValueError("selected gamma does not identify exactly one sweep point")
    confirmation = confirmation_rows(payloads["confirmation"])
    controls = static_control_rows(payloads["static_baseline"])
    shares = task_share_rows(payloads["task_spectrum"])

    with paper_context():
        fig, axes = plt.subplots(2, 2, figsize=figure_size("full", aspect=0.86))
        axes_flat = axes.ravel()

        ax = axes_flat[0]
        style = METHOD_STYLES["isotropic"]
        x = np.asarray([row["nmse_change"] for row in gamma])
        y = np.asarray([row["decoder_reduction"] for row in gamma])
        lower = y - np.asarray([row["ci_low"] for row in gamma])
        upper = np.asarray([row["ci_high"] for row in gamma]) - y
        x_margin = max(0.4, 0.08 * max(float(np.ptp(x)), 1.0))
        y_margin = max(1.0, 0.08 * max(float(np.ptp(y)), 1.0))
        selection_rule = choice.get("selection_rule", {})
        gate_x = 100 * (
            number(
                selection_rule.get("maximum_nmse_ratio"),
                context="gamma maximum NMSE ratio",
            )
            - 1
        )
        gate_y = 100 * number(
            selection_rule.get("minimum_exact_decoder_reduction"),
            context="gamma minimum decoder reduction",
        )
        xlim = (min(float(x.min()) - x_margin, gate_x - x_margin), float(x.max()) + x_margin)
        ylim = (
            min(-y_margin, float((y - lower).min()) - y_margin),
            max(float((y + upper).max()) + y_margin, gate_y + y_margin),
        )
        ax.add_patch(
            Rectangle(
                (xlim[0], gate_y),
                gate_x - xlim[0],
                ylim[1] - gate_y,
                facecolor=SEMANTIC["primary"],
                edgecolor="none",
                alpha=0.08,
                zorder=0,
            )
        )
        ax.plot(x, y, color=style.color, linestyle=style.linestyle, linewidth=1.3)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((lower, upper)),
            color=style.color,
            marker=style.marker,
            linestyle="none",
            markersize=4,
            capsize=1.8,
            linewidth=0.8,
            zorder=3,
        )
        for index, row in enumerate(gamma):
            near_top = row["decoder_reduction"] > ylim[1] - 0.12 * (
                ylim[1] - ylim[0]
            )
            vertical_offset = -6 if near_top else (7 if index % 2 == 0 else -12)
            ax.annotate(
                f"{row['weight']:g}",
                (row["nmse_change"], row["decoder_reduction"]),
                xytext=(4, vertical_offset),
                textcoords="offset points",
                fontsize=5.5,
                color=NEUTRAL["muted"],
                va="top" if near_top else "baseline",
            )
        point = selected[0]
        ax.scatter(
            [point["nmse_change"]],
            [point["decoder_reduction"]],
            s=58,
            marker="o",
            facecolors="none",
            edgecolors=SEMANTIC["primary"],
            linewidths=1.2,
            zorder=4,
        )
        mse_style = METHOD_STYLES["mse"]
        ax.scatter(
            [0],
            [0],
            s=25,
            marker=mse_style.marker,
            color=mse_style.color,
            zorder=4,
        )
        ax.annotate(
            "MSE",
            (0, 0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=5.8,
            color=NEUTRAL["text"],
            ha="left",
            va="bottom",
        )
        ax.axvline(gate_x, color=NEUTRAL["reference"], linestyle=":", linewidth=0.8)
        ax.axhline(gate_y, color=NEUTRAL["reference"], linestyle=":", linewidth=0.8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("NMSE change vs MSE (%)")
        ax.set_ylabel("Exact decoder reduction (%)")
        clean_axis(ax)
        panel_heading(ax, "A", "Clean decoder-weight sweep", "Labels show γ; ring marks selected γ")

        ax = axes_flat[1]
        for index, row in enumerate(confirmation):
            baseline = row["mse"]
            candidate = row["dpsae"]
            x0 = 100 * number(baseline["nmse"], context="confirmation MSE NMSE")
            y0 = 100 * number(
                baseline["exact_decoder_distortion"], context="confirmation MSE decoder"
            )
            x1 = 100 * number(candidate["nmse"], context="confirmation DPSAE NMSE")
            y1 = 100 * number(
                candidate["exact_decoder_distortion"], context="confirmation DPSAE decoder"
            )
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    "arrowstyle": "->",
                    "color": SEMANTIC["primary"],
                    "linewidth": 1.1,
                    "alpha": 0.72,
                },
            )
            ax.scatter(x0, y0, color=METHOD_STYLES["mse"].color, marker="o", s=20, zorder=3)
            ax.scatter(x1, y1, color=style.color, marker=style.marker, s=23, zorder=4)
            ax.annotate(
                f"s{index}",
                (x1, y1),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.8,
                color=NEUTRAL["muted"],
            )
        ax.set_xlabel("Reconstruction NMSE (%)")
        ax.set_ylabel("Exact decoder distortion (%)")
        clean_axis(ax)
        panel_heading(ax, "B", "Clean matched-quality confirmation", "Each arrow connects a paired seed")
        ax.legend(
            handles=[method_handle("mse"), method_handle("dpsae")],
            loc="best",
            frameon=False,
            fontsize=6,
        )

        ax = axes_flat[2]
        for method in ("dpsae", "whitening", "spectral"):
            style_key = METHOD_KEYS[method]
            method_style = METHOD_STYLES[style_key]
            rows = controls[method]
            xs = [row["nmse_change"] for row in rows]
            ys = [row["decoder_reduction"] for row in rows]
            ax.scatter(
                xs,
                ys,
                color=method_style.color,
                marker=method_style.marker,
                s=17,
                alpha=0.42,
            )
            ax.scatter(
                [float(np.median(xs))],
                [float(np.median(ys))],
                color=method_style.color,
                marker=method_style.marker,
                edgecolors=NEUTRAL["white"],
                linewidths=0.5,
                s=37,
                zorder=4,
            )
        ax.axhline(0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.9)
        ax.axvline(0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.9)
        ax.set_xlabel("NMSE change vs paired MSE (%)")
        ax.set_ylabel("Decoder reduction vs paired MSE (%)")
        clean_axis(ax)
        panel_heading(ax, "C", "Static-loss controls", "Separate clean three-seed baseline fleet")
        ax.legend(
            handles=[method_handle(method) for method in ("dpsae", "whitening", "spectral")],
            loc="best",
            frameon=False,
            fontsize=5.8,
        )

        ax = axes_flat[3]
        seed_labels = [f"s{int(row['seed'])}" for row in shares]
        x_positions = np.arange(len(shares))
        positive = np.asarray([row["positive"] for row in shares])
        unresolved = np.asarray([row["unresolved"] for row in shares])
        negative = np.asarray([row["negative"] for row in shares])
        ax.bar(x_positions, positive, color=SEMANTIC["primary"], width=0.62, label="Favors DPSAE")
        ax.bar(
            x_positions,
            unresolved,
            bottom=positive,
            color=SEMANTIC["unresolved"],
            width=0.62,
            label="Unresolved",
        )
        ax.bar(
            x_positions,
            negative,
            bottom=positive + unresolved,
            color=SEMANTIC["negative"],
            width=0.62,
            label="Favors MSE",
        )
        ax.set_xticks(x_positions, seed_labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Random target directions (%)")
        clean_axis(ax)
        panel_heading(ax, "D", "Taskwise advantage shares", "5% materiality threshold; refitted readouts")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=3,
            frameon=False,
            fontsize=5.4,
        )

        fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.15, wspace=0.43, hspace=0.75)
        pdf, png = save_figure(fig, output_dir / "language_model_candidates")
        plt.close(fig)
    summary = {
        "selected_decoder_weight": selected_weight,
        "gamma_sweep": gamma,
        "confirmation": [
            {
                "seed": int(row["seed"]),
                "nmse_change_percent": number(
                    row["nmse_change_percent"], context="confirmation NMSE change"
                ),
                "exact_decoder_reduction_percent": 100
                * number(
                    row["exact_decoder_reduction"],
                    context="confirmation decoder reduction",
                ),
            }
            for row in confirmation
        ],
        "taskwise_shares_percent": shares,
    }
    return pdf, png, summary


def frozen_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("paired_differences")
    if not isinstance(rows, list):
        raise ValueError("frozen-language-model artifact has no paired differences")
    result = sorted(rows, key=lambda row: integer(row.get("seed"), context="frozen seed"))
    if tuple(integer(row["seed"], context="frozen seed") for row in result) != EXPECTED_SEEDS:
        raise ValueError("frozen-language-model artifact must contain seeds 0, 1, and 2")
    return result


def plot_frozen_fidelity(
    payload: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    rows = frozen_rows(payload)
    specs = (
        (
            "loss_recovered_difference_dpsae_minus_mse",
            "loss_recovered_difference_dpsae_minus_mse_ci95",
            1.0,
            100.0,
            "Loss recovered",
            "DPSAE minus MSE (pp)",
        ),
        (
            "kl_difference_dpsae_minus_mse",
            "kl_difference_dpsae_minus_mse_ci95",
            -1.0,
            1000.0,
            "Output KL",
            r"MSE minus DPSAE ($10^{-3}$ nats/token)",
        ),
        (
            "cross_entropy_increase_difference_dpsae_minus_mse",
            "cross_entropy_increase_difference_dpsae_minus_mse_ci95",
            -1.0,
            1000.0,
            "Excess cross-entropy",
            r"MSE minus DPSAE ($10^{-3}$ nats/token)",
        ),
        (
            "top1_agreement_difference_dpsae_minus_mse",
            "top1_agreement_difference_dpsae_minus_mse_ci95",
            1.0,
            100.0,
            "Top-1 agreement",
            "DPSAE minus MSE (pp)",
        ),
    )
    transformed: dict[str, list[dict[str, float]]] = {}
    with paper_context():
        fig, axes = plt.subplots(2, 2, figsize=figure_size("full", aspect=0.74))
        for label, ax, spec in zip("ABCD", axes.ravel(), specs, strict=True):
            key, ci_key, sign, scale, title, subtitle = spec
            panel_rows = []
            for row in rows:
                center, low, high = transformed_interval(
                    number(row.get(key), context=key),
                    row.get(ci_key, ()),
                    sign=sign,
                    scale=scale,
                )
                panel_rows.append(
                    {
                        "seed": integer(row["seed"], context="frozen seed"),
                        "estimate": center,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            transformed[key] = panel_rows
            centers = np.asarray([row["estimate"] for row in panel_rows])
            low = centers - np.asarray([row["ci_low"] for row in panel_rows])
            high = np.asarray([row["ci_high"] for row in panel_rows]) - centers
            x = np.arange(len(rows))
            ax.errorbar(
                x,
                centers,
                yerr=np.vstack((low, high)),
                color=SEMANTIC["primary"],
                marker=METHOD_STYLES["isotropic"].marker,
                linestyle="none",
                markersize=4.5,
                capsize=2,
                linewidth=1,
            )
            ax.axhline(0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.9)
            ax.set_xticks(x, [f"s{int(row['seed'])}" for row in panel_rows])
            ax.set_ylabel(subtitle)
            clean_axis(ax)
            panel_heading(
                ax,
                label,
                title,
                "Positive values favor DPSAE",
                label_x=0.0,
                title_x=0.14,
            )

        nmse_ratios = [
            number(
                row.get("activation_nmse_ratio_dpsae_to_mse"),
                context="frozen activation NMSE ratio",
            )
            for row in rows
        ]
        l0_differences = [
            number(
                row.get("inference_l0_difference_dpsae_minus_mse"),
                context="frozen inference L0 difference",
            )
            for row in rows
        ]
        fig.text(
            0.5,
            0.022,
            "Same-split activation NMSE ratios (DPSAE/MSE): "
            + ", ".join(f"s{seed} {value:.3f}" for seed, value in enumerate(nmse_ratios))
            + "\nInference L0 differences (DPSAE - MSE): "
            + ", ".join(f"s{seed} {value:+.2f}" for seed, value in enumerate(l0_differences)),
            ha="center",
            va="bottom",
            fontsize=6,
            color=NEUTRAL["muted"],
        )
        fig.subplots_adjust(left=0.13, right=0.98, top=0.87, bottom=0.16, wspace=0.39, hspace=0.72)
        pdf, png = save_figure(fig, output_dir / "frozen_fidelity_review")
        plt.close(fig)
    summary = {
        "review_status": "external criterion for author review; no manuscript claim inferred",
        "positive_axis_semantics": "positive values favor DPSAE for every panel",
        "transformed_metrics": transformed,
        "activation_nmse_ratio_dpsae_to_mse": nmse_ratios,
        "inference_l0_difference_dpsae_minus_mse": l0_differences,
    }
    return pdf, png, summary


def robustness_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload.get("paired_reductions")
    settings = payload.get("settings")
    if not isinstance(rows, list) or not isinstance(settings, list):
        raise ValueError("robustness artifact is missing settings or paired reductions")
    expected_seeds = tuple(int(seed) for seed in payload.get("protocol", {}).get("expected_seeds", ()))
    if expected_seeds != EXPECTED_SEEDS:
        raise ValueError("robustness artifact does not use the frozen seed set")
    setting_order: dict[str, list[str]] = defaultdict(list)
    setting_values: dict[tuple[str, str], Any] = {}
    for setting in settings:
        axis = str(setting.get("audit_axis"))
        label = str(setting.get("setting_label"))
        if axis not in {"ridge", "group_size", "grouping"}:
            raise ValueError(f"unknown robustness audit axis {axis!r}")
        if label in setting_order[axis]:
            raise ValueError(f"duplicate robustness setting {axis}/{label}")
        setting_order[axis].append(label)
        setting_values[(axis, label)] = setting.get("setting_value")
    require_unique(rows, ("seed", "audit_axis", "setting_label"), name="robustness rows")
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    observed_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        axis = str(row.get("audit_axis"))
        label = str(row.get("setting_label"))
        if label not in setting_order.get(axis, ()):
            raise ValueError(f"robustness row has undeclared setting {axis}/{label}")
        seed = integer(row.get("seed"), context="robustness seed")
        values[(axis, label)].append(
            100
            * number(
                row.get("decoder_reduction_vs_mse"),
                context="robustness decoder reduction",
            )
        )
        observed_seeds[(axis, label)].add(seed)
    for axis, labels in setting_order.items():
        for label in labels:
            if observed_seeds[(axis, label)] != set(EXPECTED_SEEDS):
                raise ValueError(f"robustness setting {axis}/{label} lacks all paired seeds")
    return {
        "setting_order": dict(setting_order),
        "setting_values": setting_values,
        "values": dict(values),
    }


def robustness_label(axis: str, value: Any) -> str:
    if axis == "ridge":
        return f"DoF {number(value, context='ridge setting value'):g}"
    if axis == "group_size":
        return f"n={integer(value, context='group-size setting value')}"
    return {
        "contiguous": "contig.",
        "shuffled": "shuffled",
        "document_balanced": "doc-bal.",
    }.get(str(value), str(value))


def plot_robustness(
    payload: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    data = robustness_data(payload)
    axes_order = ("ridge", "group_size", "grouping")
    titles = ("Ridge strength", "Geometry group size", "Token grouping")
    summaries: dict[str, Any] = {}
    with paper_context():
        fig, axes = plt.subplots(1, 3, figsize=figure_size("full", aspect=0.52))
        for panel, ax, axis, title in zip("ABC", axes, axes_order, titles, strict=True):
            labels = data["setting_order"].get(axis)
            if not labels:
                raise ValueError(f"robustness artifact has no {axis} settings")
            centers = np.arange(len(labels), dtype=float)[::-1]
            axis_summary = []
            for center, label in zip(centers, labels, strict=True):
                sample = data["values"][(axis, label)]
                low, median, high = quantiles(sample)
                ax.scatter(
                    sample,
                    np.full(len(sample), center),
                    color=SEMANTIC["primary"],
                    marker=METHOD_STYLES["isotropic"].marker,
                    s=14,
                    alpha=0.34,
                    zorder=2,
                )
                ax.plot([low, high], [center, center], color=SEMANTIC["primary"], linewidth=1.8)
                ax.scatter(
                    [median],
                    [center],
                    color=SEMANTIC["primary"],
                    marker=METHOD_STYLES["isotropic"].marker,
                    edgecolors=NEUTRAL["white"],
                    linewidths=0.5,
                    s=30,
                    zorder=4,
                )
                axis_summary.append(
                    {
                        "setting_label": label,
                        "setting_value": data["setting_values"][(axis, label)],
                        "raw_seed_reductions_percent": list(sample),
                        "q10": low,
                        "median": median,
                        "q90": high,
                    }
                )
            ax.axvline(0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.9)
            ax.set_yticks(
                centers,
                [
                    robustness_label(axis, data["setting_values"][(axis, label)])
                    for label in labels
                ],
            )
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel("Decoder reduction (%)")
            clean_axis(ax)
            panel_heading(ax, panel, title, "Seeds; median, q10–q90")
            summaries[axis] = axis_summary
        fig.subplots_adjust(left=0.12, right=0.99, top=0.80, bottom=0.20, wspace=0.58)
        pdf, png = save_figure(fig, output_dir / "robustness_appendix")
        plt.close(fig)
    return pdf, png, summaries


def main() -> None:
    args = parse_args()
    validate_output_location(args.experiment_root, args.output_dir)
    payloads, paths = validate_inputs(
        args.experiment_root,
        args.structured_baseline_dir,
        args.static_baseline,
    )
    structured = structured_data(paths, payloads)
    robustness_data(payloads["robustness"])
    # All schemas are validated before the first output is written.
    gamma_rows(payloads["gamma_sweep"])
    confirmation_rows(payloads["confirmation"])
    static_control_rows(payloads["static_baseline"])
    task_share_rows(payloads["task_spectrum"])
    frozen_rows(payloads["frozen_fidelity"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    for name, renderer in (
        ("task_prior", lambda: plot_structured(structured, args.output_dir)),
        ("language_model", lambda: plot_language_model(payloads, args.output_dir)),
        (
            "frozen_fidelity_review",
            lambda: plot_frozen_fidelity(payloads["frozen_fidelity"], args.output_dir),
        ),
        (
            "robustness_appendix",
            lambda: plot_robustness(payloads["robustness"], args.output_dir),
        ),
    ):
        pdf, png, summary = renderer()
        figures[name] = {"pdf": file_record(pdf), "png": file_record(png)}
        derived[name] = summary

    manifest = {
        "complete": True,
        "experiment": "exp08_candidate_figure_review",
        "status": "review_only_not_integrated_into_manuscript",
        "created_unix": time.time(),
        "repository": payloads["run_manifest"]["repository"],
        "relative_weight_semantics": {
            "one": "separate 2D crossover reference used only as a plotting scale",
            "sparse_transition_claim": False,
        },
        "metric_direction": {
            "task_prior": "positive means improvement relative to the paired MSE baseline",
            "language_model": "decoder reduction is positive when DPSAE has lower distortion",
            "frozen_fidelity": "every displayed axis is transformed so positive favors DPSAE",
            "robustness": "positive means lower exact decoder distortion than paired MSE",
        },
        "aggregation": {
            "structured": "raw paired seeds with median and 10th--90th percentiles",
            "language_model": "raw paired seed comparisons; bootstrap intervals where stored",
            "frozen_fidelity": "raw paired seeds with paired sequence-bootstrap intervals",
            "robustness": "raw paired seeds with median and 10th--90th percentiles",
        },
        "inputs": {name: file_record(path) for name, path in paths.items()},
        "renderer_inputs": {
            "plotter": file_record(Path(__file__)),
            "plot_style": file_record(ROOT / "src/dpsae/plot_style.py"),
        },
        "figures": figures,
        "derived_summaries": derived,
    }
    manifest_path = args.output_dir / "candidate_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({"complete": True, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
