#!/usr/bin/env python3
"""Paper-style figures for the completed Experiment 4b audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dpsae.plot_style import (
    COLORS,
    LABELS,
    LINESTYLES,
    MARKERS,
    apply_paper_style,
    clean_axis,
    savefig,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def style_method(method: str) -> str:
    return {
        "mse": "mse",
        "dpsae": "isotropic",
        "whitening": "whitened",
        "spectral": "spectral",
    }[method]


def model_method(result: dict) -> str:
    return style_method(result["spec"]["method"])


def selected_row(rows: list[dict], count: int) -> dict:
    return next(row for row in rows if row["features"] == count)


def plot_natural(artifact: Path) -> None:
    result = read_json(artifact / "natural_evaluation_source.json")
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))

    for name, model in result["models"].items():
        method = model_method(model)
        if method not in {"mse", "isotropic", "whitened"}:
            continue
        k = int(model["spec"]["k"])
        axes[0].scatter(
            model["sampled_primary"]["nmse"],
            model["exact_identity_primary"]["decoder_distortion"],
            s={16: 18, 32: 30, 64: 44}[k],
            marker=MARKERS[method],
            color=COLORS[method],
            alpha=0.72,
            label=LABELS[method],
        )
    for k in (16, 32, 64):
        points = [
            (
                model["sampled_primary"]["nmse"],
                model["exact_identity_primary"]["decoder_distortion"],
            )
            for model in result["models"].values()
            if model["spec"]["method"] == "dpsae" and int(model["spec"]["k"]) == k
        ]
        axes[0].annotate(
            f"k={k}",
            np.median(points, axis=0),
            xytext=(5, -10),
            textcoords="offset points",
            color=COLORS["isotropic"],
            fontsize=7,
        )
    axes[0].set_title("Fresh natural-text confirmation")
    axes[0].set_xlabel("Reconstruction NMSE")
    axes[0].set_ylabel("Exact decoder distortion")
    clean_axis(axes[0], xlog=True, ylog=True)

    exact = result["exact_identity_audit"]
    values = {
        (row["model"], row["ridge"], row["group_size"], row["grouping"]): row[
            "decoder_distortion"
        ]
        for row in exact
    }
    settings = []
    for row in exact:
        setting = (
            row["audit_axis"],
            row["ridge"],
            row["group_size"],
            row["grouping"],
        )
        if row["model"] == "mse_s0" and setting not in settings:
            settings.append(setting)
    settings.sort(key=lambda row: (row[0], row[2], row[3], row[1]))
    trajectories = []
    for seed in range(3):
        seed_values = []
        for _axis, ridge, size, grouping in settings:
            mse = values[(f"mse_s{seed}", ridge, size, grouping)]
            dpsae = values[(f"dpsae_s{seed}", ridge, size, grouping)]
            seed_values.append(100 * (1 - dpsae / mse))
        trajectories.append(seed_values)
        axes[1].plot(
            range(len(settings)), seed_values, color=COLORS["isotropic"], alpha=0.22
        )
    axes[1].plot(
        range(len(settings)),
        np.median(trajectories, axis=0),
        color=COLORS["isotropic"],
        marker=MARKERS["isotropic"],
        linewidth=2,
    )
    labels = [
        f"n={size}" if axis == "group_size" else grouping.replace("document_", "doc-")
        if axis == "grouping"
        else f"ridge={ridge:.2g}"
        for axis, ridge, size, grouping in settings
    ]
    axes[1].axhline(0, color=COLORS["theory"], linestyle=":", linewidth=1)
    axes[1].set_xticks(range(len(settings)), labels, rotation=35, ha="right")
    axes[1].set_title("Geometry audit at k=32")
    axes[1].set_ylabel("DPSAE reduction vs. MSE (%)")
    clean_axis(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.27, wspace=0.38)
    savefig(fig, artifact / "figures" / "exp04b_natural_confirmation")
    plt.close(fig)


def plot_ioi(artifact: Path) -> None:
    result = read_json(artifact / "ioi_confirmatory.json")
    models = result["test_models"]
    count = int(result["feature_count_selection"]["selection"]["feature_count"])
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
    methods = ("mse", "isotropic", "whitened", "spectral")
    for method in methods:
        matches = [value for value in models.values() if model_method(value) == method]
        if not matches:
            continue
        curves = []
        for model in matches:
            ioi = model["duplicate_state"]["ioi_zero_curve"]
            natural = model["duplicate_state"]["natural_zero_curve"]
            x = [row["collateral_kl"] for row in natural]
            y = [row["ioi_effect"] for row in ioi]
            curves.append((x, y))
            axes[0].plot(x, y, color=COLORS[method], alpha=0.2, linewidth=0.8)
        axes[0].plot(
            np.median([curve[0] for curve in curves], axis=0),
            np.median([curve[1] for curve in curves], axis=0),
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2,
            label=LABELS[method],
        )
    axes[0].set_title("Matched zero-ablation frontier")
    axes[0].set_xlabel("Natural-text collateral KL")
    axes[0].set_ylabel("IOI logit-difference effect")
    clean_axis(axes[0])

    active_methods = [
        method
        for method in methods
        if any(model_method(value) == method for value in models.values())
    ]
    for index, method in enumerate(active_methods):
        matches = [value for value in models.values() if model_method(value) == method]
        if not matches:
            continue
        r2 = [value["continuous_target"]["test"]["r2"] for value in matches]
        axes[1].scatter(
            np.full(len(r2), index),
            r2,
            color=COLORS[method],
            marker=MARKERS[method],
            alpha=0.75,
        )
        axes[1].plot(
            [index - 0.18, index + 0.18],
            [np.median(r2)] * 2,
            color=COLORS[method],
            linewidth=2,
        )
    dense = [value["continuous_target"]["original_dense_test"]["r2"] for value in models.values()]
    axes[1].axhline(
        np.median(dense), color=COLORS["theory"], linestyle=":", label="Dense original"
    )
    axes[1].set_xticks(
        range(len(active_methods)),
        [LABELS[method].replace("MSE + ", "") for method in active_methods],
        rotation=25,
        ha="right",
    )
    axes[1].set_title(f"Harder target at frozen m={count}")
    axes[1].set_ylabel("Final correct-minus-subject R²")
    clean_axis(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.27, wspace=0.38)
    savefig(fig, artifact / "figures" / "exp04b_ioi_confirmation")
    plt.close(fig)


def plot_training(artifact: Path) -> None:
    log = artifact / "baseline_confirm" / "training.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    methods = ("mse", "dpsae", "whitening", "spectral")
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for raw_method in methods:
        method = style_method(raw_method)
        names = [
            name
            for name in records[-1]["models"]
            if name.startswith(f"{raw_method}_")
        ]
        if not names:
            continue
        nmse_curves, dead_curves = [], []
        for name in names:
            x = [row["tokens_seen"] / 1e6 for row in records]
            nmse = [row["models"][name]["nmse"] for row in records]
            dead = [row["models"][name]["dead"] for row in records]
            nmse_curves.append(nmse)
            dead_curves.append(dead)
            axes[0].plot(x, nmse, color=COLORS[method], alpha=0.2, linewidth=0.7)
            axes[1].plot(x, dead, color=COLORS[method], alpha=0.2, linewidth=0.7)
        for axis, curves in zip(axes, (nmse_curves, dead_curves)):
            axis.plot(
                x,
                np.median(curves, axis=0),
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2,
                label=LABELS[method],
            )
    axes[0].set_title("Confirmation reconstruction")
    axes[0].set_ylabel("Training-batch NMSE")
    axes[1].set_title("Dictionary stability")
    axes[1].set_ylabel("Dead features")
    for axis in axes:
        axis.set_xlabel("Training tokens (millions)")
        clean_axis(axis, ylog=axis is axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    fig.subplots_adjust(top=0.77, bottom=0.2, wspace=0.35)
    savefig(fig, artifact / "figures" / "exp04b_training")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "exp04b_confirmatory",
    )
    args = parser.parse_args()
    plot_natural(args.artifact)
    plot_ioi(args.artifact)
    plot_training(args.artifact)


if __name__ == "__main__":
    main()
