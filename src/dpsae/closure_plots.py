"""Result-agnostic renderers for the remaining arXiv closure figures.

The module defines the expected plotting payload but does not derive it from
experiment artifacts.  A separate, post-audit aggregation step must supply the
numbers.  This separation prevents figure code from choosing among failed or
partial Exp10 roots and keeps the renderer usable before outcomes are opened.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dpsae.plot_style import (
    CATEGORICAL,
    NEUTRAL,
    SEMANTIC,
    clean_axis,
    figure_size,
    paper_context,
    save_figure,
)


PAYLOAD_SCHEMA_VERSION = 1
STAGES = ("residual", "reconstruction", "full_code", "k5", "k2", "k1")
STAGE_LABELS = {
    "residual": "Residual",
    "reconstruction": "Reconstruction",
    "full_code": "Full code",
    "k5": "5 features",
    "k2": "2 features",
    "k1": "1 feature",
}
METHOD_STYLE = {
    "mse": {
        "color": SEMANTIC["negative"],
        "marker": "o",
        "linestyle": "--",
        "label": "MSE",
    },
    "dpsae": {
        "color": SEMANTIC["primary"],
        "marker": "s",
        "linestyle": "-",
        "label": "DPSAE",
    },
}


PLOT_PAYLOAD_CONTRACT = {
    "schema_version": PAYLOAD_SCHEMA_VERSION,
    "complete": True,
    "release_manifest_sha256": "<sha256 from audited release manifest>",
    "figures": {
        "concept_ladder": {
            "available": "boolean; true only after final Exp10 audit",
            "records": [
                {
                    "method": "mse|dpsae",
                    "stage": "residual|reconstruction|full_code|k5|k2|k1",
                    "estimate": "finite AUROC",
                    "ci_low": "finite lower interval endpoint",
                    "ci_high": "finite upper interval endpoint",
                }
            ],
        },
        "frozen_network_noninferiority": {
            "available": "boolean",
            "noninferiority_margin": "finite ratio, normally 1.01",
            "records": [
                {
                    "seed": "integer",
                    "estimate": "finite DPSAE/MSE KL ratio",
                    "ci_low": "finite lower interval endpoint",
                    "ci_high": "finite upper interval endpoint",
                }
            ],
        },
        "static_nmse_control": {
            "available": "boolean",
            "target_nmse_low": "finite lower target-band edge",
            "target_nmse_high": "finite upper target-band edge",
            "records": [
                {
                    "candidate": "stable display identifier",
                    "method": "mse|dpsae|spectral",
                    "nmse_ratio": "finite x coordinate",
                    "decoder_reduction": "finite y coordinate",
                    "selected": "boolean",
                }
            ],
        },
    },
    "summary_table": {
        "available": "boolean",
        "rows": [
            {
                "experiment": "Exp09|Exp10|Exp11|Exp12",
                "endpoint": "short endpoint label",
                "estimate": "formatted value or NA",
                "ci_low": "formatted value or NA",
                "ci_high": "formatted value or NA",
                "gate": "frozen gate text",
                "status": "passed|failed|not-run|descriptive",
                "scope": "one-line interpretation boundary",
            }
        ],
    },
}


def _number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _interval(record: Mapping[str, Any], *, label: str) -> tuple[float, float, float]:
    estimate = _number(record.get("estimate"), label=f"{label} estimate")
    low = _number(record.get("ci_low"), label=f"{label} ci_low")
    high = _number(record.get("ci_high"), label=f"{label} ci_high")
    if low > estimate or estimate > high:
        raise ValueError(f"{label} interval does not contain its estimate")
    return estimate, low, high


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise ValueError("unsupported closure plotting-payload schema")
    if payload.get("complete") is not True:
        raise ValueError("closure plotting payload is incomplete")
    digest = payload.get("release_manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("plotting payload is not bound to a release manifest")
    figures = payload.get("figures")
    if not isinstance(figures, Mapping):
        raise ValueError("plotting payload has no figure blocks")
    unknown = set(figures) - {
        "concept_ladder",
        "frozen_network_noninferiority",
        "static_nmse_control",
    }
    if unknown:
        raise ValueError(f"unknown closure figure blocks: {sorted(unknown)}")
    table = payload.get("summary_table")
    if not isinstance(table, Mapping) or not isinstance(table.get("available"), bool):
        raise ValueError("plotting payload has no explicit summary-table availability")


def _available(block: Mapping[str, Any], *, name: str) -> bool:
    available = block.get("available")
    if not isinstance(available, bool):
        raise ValueError(f"{name} has no explicit availability gate")
    if not available and block.get("records") not in (None, []):
        raise ValueError(f"{name} is unavailable but contains result records")
    return available


def plot_concept_ladder(
    block: Mapping[str, Any], output: Path
) -> tuple[Path, Path]:
    if not _available(block, name="concept ladder"):
        raise ValueError("cannot render unavailable concept ladder")
    records = block.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("concept ladder records must be a sequence")
    keyed: dict[tuple[str, str], tuple[float, float, float]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("concept ladder rows must be objects")
        method, stage = str(row.get("method")), str(row.get("stage"))
        if method not in METHOD_STYLE or stage not in STAGES:
            raise ValueError("concept ladder method or stage drift")
        key = (method, stage)
        if key in keyed:
            raise ValueError(f"duplicate concept ladder row: {key}")
        keyed[key] = _interval(row, label=f"concept ladder {method}/{stage}")
    expected = {(method, stage) for method in METHOD_STYLE for stage in STAGES}
    if set(keyed) != expected:
        raise ValueError("concept ladder does not contain the complete method-stage grid")

    with paper_context():
        fig, ax = plt.subplots(figsize=figure_size("full", aspect=0.44))
        x = np.arange(len(STAGES))
        for method, style in METHOD_STYLE.items():
            values = [keyed[(method, stage)] for stage in STAGES]
            estimates = np.asarray([value[0] for value in values])
            errors = np.asarray(
                [[value[0] - value[1] for value in values], [value[2] - value[0] for value in values]]
            )
            ax.errorbar(
                x,
                estimates,
                yerr=errors,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.45,
                markersize=4.2,
                capsize=2,
                label=style["label"],
            )
        ax.set_xticks(x, [STAGE_LABELS[stage] for stage in STAGES])
        ax.set_ylabel("Held-out AUROC")
        clean_axis(ax)
        ax.legend(loc="best", frameon=False)
        fig.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.21)
        paths = save_figure(fig, output)
        plt.close(fig)
    return paths


def plot_frozen_network_noninferiority(
    block: Mapping[str, Any], output: Path
) -> tuple[Path, Path]:
    if not _available(block, name="frozen-network noninferiority"):
        raise ValueError("cannot render unavailable frozen-network figure")
    margin = _number(block.get("noninferiority_margin"), label="noninferiority margin")
    records = block.get("records")
    if not isinstance(records, Sequence) or not records:
        raise ValueError("frozen-network figure requires seed records")
    parsed = []
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("frozen-network rows must be objects")
        seed = int(row["seed"])
        parsed.append((seed, *_interval(row, label=f"frozen seed {seed}")))
    if len({row[0] for row in parsed}) != len(parsed):
        raise ValueError("frozen-network figure has duplicate seeds")
    parsed.sort()

    with paper_context():
        fig, ax = plt.subplots(figsize=figure_size("half", aspect=0.72))
        for index, (seed, estimate, low, high) in enumerate(parsed):
            ax.errorbar(
                estimate,
                index,
                xerr=[[estimate - low], [high - estimate]],
                color=SEMANTIC["primary"],
                marker="s",
                linestyle="none",
                capsize=2,
                markersize=4.2,
            )
        ax.axvline(1.0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.8)
        ax.axvline(margin, color=SEMANTIC["warning"], linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(parsed)), [f"Seed {row[0]}" for row in parsed])
        ax.set_xlabel("KL ratio, DPSAE / MSE")
        clean_axis(ax)
        fig.subplots_adjust(left=0.25, right=0.97, top=0.97, bottom=0.23)
        paths = save_figure(fig, output)
        plt.close(fig)
    return paths


def plot_static_nmse_control(
    block: Mapping[str, Any], output: Path
) -> tuple[Path, Path]:
    if not _available(block, name="static NMSE control"):
        raise ValueError("cannot render unavailable static-control figure")
    low = _number(block.get("target_nmse_low"), label="target NMSE low")
    high = _number(block.get("target_nmse_high"), label="target NMSE high")
    if low > high:
        raise ValueError("static-control NMSE band is reversed")
    records = block.get("records")
    if not isinstance(records, Sequence) or not records:
        raise ValueError("static-control figure requires records")
    style = {
        "mse": METHOD_STYLE["mse"],
        "dpsae": METHOD_STYLE["dpsae"],
        "spectral": {
            "color": CATEGORICAL["yellow"],
            "marker": "^",
            "linestyle": ":",
            "label": "Static spectral",
        },
    }
    parsed = []
    seen = set()
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("static-control rows must be objects")
        candidate = str(row.get("candidate"))
        method = str(row.get("method"))
        if not candidate or candidate in seen or method not in style:
            raise ValueError("static-control candidate identity drift")
        seen.add(candidate)
        parsed.append(
            (
                candidate,
                method,
                _number(row.get("nmse_ratio"), label=f"{candidate} NMSE ratio"),
                _number(
                    row.get("decoder_reduction"), label=f"{candidate} decoder reduction"
                ),
                bool(row.get("selected", False)),
            )
        )

    with paper_context():
        fig, ax = plt.subplots(figsize=figure_size("half", aspect=0.72))
        ax.axvspan(low, high, color=NEUTRAL["fill"], zorder=0)
        for method, method_style in style.items():
            rows = [row for row in parsed if row[1] == method]
            if not rows:
                continue
            ax.scatter(
                [row[2] for row in rows],
                [row[3] for row in rows],
                color=method_style["color"],
                marker=method_style["marker"],
                label=method_style["label"],
                s=20,
                zorder=2,
            )
            for candidate, _method, x, y, selected in rows:
                if selected:
                    ax.scatter(
                        [x],
                        [y],
                        facecolors="none",
                        edgecolors=method_style["color"],
                        marker="o",
                        s=54,
                        linewidths=0.9,
                        zorder=3,
                    )
                    ax.annotate(
                        candidate,
                        (x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color=NEUTRAL["text"],
                        fontsize=6.5,
                    )
        ax.axhline(0, color=NEUTRAL["reference"], linestyle=":", linewidth=0.8)
        ax.set_xlabel("NMSE ratio to MSE")
        ax.set_ylabel("Decoder reduction")
        clean_axis(ax)
        ax.legend(loc="best", frameon=False)
        fig.subplots_adjust(left=0.21, right=0.97, top=0.97, bottom=0.23)
        paths = save_figure(fig, output)
        plt.close(fig)
    return paths


SUMMARY_FIELDS = (
    "experiment",
    "endpoint",
    "estimate",
    "ci_low",
    "ci_high",
    "gate",
    "status",
    "scope",
)


def write_summary_table(block: Mapping[str, Any], output: Path) -> Path:
    if not _available(block, name="closure summary table"):
        raise ValueError("cannot write unavailable closure summary table")
    rows = block.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("closure summary table requires rows")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(SUMMARY_FIELDS):
            raise ValueError("closure summary-table row schema drift")
        normalized.append({field: str(row[field]) for field in SUMMARY_FIELDS})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    temporary.replace(output)
    return output
