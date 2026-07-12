"""Publication plotting style for DPSAE experiments.

The visual grammar is adapted from the user's NLA plotting preference: white
canvas, restrained colorblind-safe colors, marker-plus-line method identity,
light grids, shared legends, and paired PDF/PNG output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


METHOD_ORDER = ["mse", "isotropic", "whitened", "decoder_only"]
COLORS = {
    "mse": "#222222",
    "isotropic": "#0072B2",
    "whitened": "#009E73",
    "decoder_only": "#D55E00",
    "theory": "#777777",
    "random": "#AAAAAA",
}
MARKERS = {"mse": "o", "isotropic": "s", "whitened": "D", "decoder_only": "X"}
LINESTYLES = {"mse": "--", "isotropic": "-", "whitened": "-.", "decoder_only": ":"}
LABELS = {
    "mse": "MSE",
    "isotropic": "MSE + isotropic DPSAE",
    "whitened": "MSE + whitening",
    "decoder_only": "Decoder only",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax, *, ylog: bool = False, xlog: bool = False) -> None:
    if ylog:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.grid(True, which="major", color="#E6E6E6", linewidth=0.7)
    if ylog:
        ax.grid(True, which="minor", axis="y", color="#F2F2F2", linewidth=0.5)
    ax.tick_params(direction="out", length=3, width=0.8, colors="#444444")


def savefig(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05, dpi=300)
