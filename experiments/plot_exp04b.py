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


def sparsity_from_name(name: str) -> int:
    for value in (16, 64):
        if f"_k{value}_" in name:
            return value
    return 32


def plot_natural(artifact: Path) -> None:
    source = read_json(artifact / "natural_evaluation_source.json")
    confirmation = read_json(artifact / "natural_evaluation_baseline.json")
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.75))

    for model in confirmation["models"].values():
        method = model_method(model)
        axes[0].scatter(
            model["sampled_primary"]["nmse"],
            model["exact_identity_primary"]["decoder_distortion"],
            s=26,
            marker=MARKERS[method],
            color=COLORS[method],
            alpha=0.72,
            label=LABELS[method],
        )
    axes[0].set_title("Held-out confirmation (k=32)")
    axes[0].set_xlabel("Reconstruction NMSE")
    axes[0].set_ylabel("Exact decoder distortion")
    axes[0].set_xlim(0.0230, 0.0254)
    axes[0].set_ylim(0.0285, 0.0465)
    axes[0].set_xticks((0.023, 0.024, 0.025))
    axes[0].set_yticks((0.03, 0.04))
    clean_axis(axes[0])

    by_seed: dict[int, dict[int, float]] = {}
    for row in source["paired_reductions"]:
        if row["method"] != "dpsae":
            continue
        by_seed.setdefault(int(row["seed"]), {})[
            sparsity_from_name(row["candidate"])
        ] = 100 * row["exact_identity_reduction"]["estimate"]
    ks = (16, 32, 64)
    trajectories = [[by_seed[seed][k] for k in ks] for seed in sorted(by_seed)]
    for values in trajectories:
        axes[1].plot(ks, values, color=COLORS["isotropic"], alpha=0.22)
    axes[1].plot(
        ks,
        np.median(trajectories, axis=0),
        color=COLORS["isotropic"],
        marker=MARKERS["isotropic"],
        linewidth=2,
    )
    axes[1].axhline(0, color=COLORS["theory"], linestyle=":", linewidth=1)
    axes[1].set_xticks(ks)
    axes[1].set_title("Sparsity robustness")
    axes[1].set_xlabel("Active features k")
    axes[1].set_ylabel("DPSAE reduction vs. MSE (%)")
    clean_axis(axes[1])

    exact = source["exact_identity_audit"]
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
        axes[2].plot(
            range(len(settings)), seed_values, color=COLORS["isotropic"], alpha=0.22
        )
    axes[2].plot(
        range(len(settings)),
        np.median(trajectories, axis=0),
        color=COLORS["isotropic"],
        marker=MARKERS["isotropic"],
        linewidth=2,
    )
    labels = [
        (
            f"n={size}"
            if axis == "group_size"
            else grouping.replace("document_", "doc-")
            if axis == "grouping"
            else f"ridge={ridge:.2g}"
        )
        for axis, ridge, size, grouping in settings
    ]
    axes[2].axhline(0, color=COLORS["theory"], linestyle=":", linewidth=1)
    axes[2].set_xticks(range(len(settings)), labels, rotation=35, ha="right")
    axes[2].set_title("Geometry robustness (k=32)")
    axes[2].set_ylabel("DPSAE reduction vs. MSE (%)")
    clean_axis(axes[2])

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.29, wspace=0.42)
    savefig(fig, artifact / "figures" / "exp04b_natural_confirmation")
    plt.close(fig)


def plot_ioi(artifact: Path) -> None:
    result = read_json(artifact / "ioi_confirmatory.json")
    models = result["test_models"]
    count = int(result["feature_count_selection"]["selection"]["feature_count"])
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
    for row in result["paired_test_summary"]:
        method = style_method(row["method"])
        effect = row["ioi_effect_difference"]
        kl = row["natural_kl_difference"]
        x = kl["paired_difference"]
        y = effect["paired_difference"]
        axes[0].errorbar(
            x,
            y,
            xerr=[[x - kl["ci_low"]], [kl["ci_high"] - x]],
            yerr=[[y - effect["ci_low"]], [effect["ci_high"] - y]],
            color=COLORS[method],
            marker=MARKERS[method],
            alpha=0.72,
            capsize=2,
            linestyle="none",
            label=LABELS[method],
        )
        axes[0].annotate(
            f"s{row['seed']}",
            (x, y),
            xytext=(4, 3),
            textcoords="offset points",
            color=COLORS[method],
            fontsize=6,
        )
    axes[0].axhline(0, color=COLORS["theory"], linestyle=":", linewidth=1)
    axes[0].axvline(0, color=COLORS["theory"], linestyle=":", linewidth=1)
    axes[0].set_title(f"Frozen paired causal test (m={count})")
    axes[0].set_xlabel("Δ natural-text collateral KL\n(left is better)")
    axes[0].set_ylabel("Δ IOI ablation effect\n(up is better)")
    clean_axis(axes[0])

    active_methods = ("mse", "isotropic", "whitened", "spectral")
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
    dense = np.median(
        [value["continuous_target"]["original_dense_test"]["r2"] for value in models.values()]
    )
    dense_index = len(active_methods)
    axes[1].scatter(dense_index, dense, color=COLORS["theory"], marker="x")
    axes[1].axhline(0, color=COLORS["theory"], linestyle=":", linewidth=1)
    axes[1].set_xticks(
        range(len(active_methods) + 1),
        [LABELS[method].replace("MSE + ", "") for method in active_methods]
        + ["Dense original"],
        rotation=25,
        ha="right",
    )
    axes[1].set_title("Continuous-target diagnostic (failed)")
    axes[1].set_ylabel("Final correct-minus-subject R²")
    clean_axis(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=False
    )
    fig.subplots_adjust(top=0.78, bottom=0.31, wspace=0.42)
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
