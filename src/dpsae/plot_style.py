"""Aniket's Nord scientific plotting system.

The module separates portable visual defaults from project semantics.  The
``aniket-nord.mplstyle`` file controls typography and axes; this module assigns
stable colors, markers, and line styles to scientific roles and DPSAE methods.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


STYLE_PATH = Path(__file__).with_name("styles") / "aniket-nord.mplstyle"
FONT_DIR = Path(__file__).with_name("fonts") / "d-din"
FIGURE_FONT_FAMILY = "D-DIN"
CONDENSED_FONT_FAMILY = "D-DIN Condensed"
EXPANDED_FONT_FAMILY = "D-DIN Exp"
VENDORED_FONT_FILES = (
    "D-DIN.ttf",
    "D-DIN-Italic.ttf",
    "D-DIN-Bold.ttf",
    "D-DINCondensed.ttf",
    "D-DINCondensed-Bold.ttf",
    "D-DINExp.ttf",
    "D-DINExp-Italic.ttf",
    "D-DINExp-Bold.ttf",
)
VENDORED_FONT_SHA256 = {
    "D-DIN.ttf": "705bece88e33c8f86d0ace0c7d93ee143b745cba7a99643753a4f91c3c22e204",
    "D-DIN-Italic.ttf": "22100d8442e310a1add840c3cbb717b64b5b59f176c0061e440b753c09ed3a26",
    "D-DIN-Bold.ttf": "69cc46d24509802693a2a5f6e1b18bcad65f8baf5fd3c08b9bd364beb2a8bdd5",
    "D-DINCondensed.ttf": "724b48d534dbcde7a9f039bddc3a7344d4913de43726f3b7d7a56f0770a8ea6b",
    "D-DINCondensed-Bold.ttf": (
        "664e694799db84a910f08edc717916763a2e3f23ee44b4530769968768b34293"
    ),
    "D-DINExp.ttf": "ebb595323d0af86931cccc35ba232bee564ab034f62a8b41ef7c1617dd5111c1",
    "D-DINExp-Italic.ttf": (
        "bbab81e71d8707d0ebbe9f19f7be4d2ef7e04b0d983df36effb8d2cd3e2e887e"
    ),
    "D-DINExp-Bold.ttf": "ede3a43f2ed4c5658a607fe179f3ddc497dfe6645f29bfbb8b5fbc5b8831e0ad",
}


def register_vendored_fonts() -> tuple[Path, ...]:
    """Register the pinned D-DIN family without relying on host installation."""

    paths = tuple(FONT_DIR / filename for filename in VENDORED_FONT_FILES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing vendored D-DIN font files: {missing}")
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != VENDORED_FONT_SHA256[path.name]:
            raise RuntimeError(f"Vendored D-DIN font hash mismatch: {path}")
        font_manager.fontManager.addfont(path)
    return paths


register_vendored_fonts()

# The official Nord palette is the complete source of categorical figure colors.
NORD = {
    "polar_night_0": "#2E3440",
    "polar_night_1": "#3B4252",
    "polar_night_2": "#434C5E",
    "polar_night_3": "#4C566A",
    "snow_storm_0": "#D8DEE9",
    "snow_storm_1": "#E5E9F0",
    "snow_storm_2": "#ECEFF4",
    "frost_teal": "#8FBCBB",
    "frost_cyan": "#88C0D0",
    "frost_blue": "#81A1C1",
    "frost_deep": "#5E81AC",
    "aurora_red": "#BF616A",
    "aurora_orange": "#D08770",
    "aurora_yellow": "#EBCB8B",
    "aurora_green": "#A3BE8C",
    "aurora_purple": "#B48EAD",
}

CATEGORICAL = {
    # This order maximizes separation for the first five ordinary series.
    "blue": NORD["frost_deep"],
    "orange": NORD["aurora_orange"],
    "purple": NORD["aurora_purple"],
    "green": NORD["aurora_green"],
    "red": NORD["aurora_red"],
    "teal": NORD["frost_teal"],
    "yellow": NORD["aurora_yellow"],
}

NEUTRAL = {
    "text": NORD["polar_night_0"],
    "muted": NORD["polar_night_3"],
    "reference": NORD["polar_night_3"],
    "unresolved": NORD["snow_storm_0"],
    "grid": NORD["snow_storm_0"],
    "grid_minor": NORD["snow_storm_2"],
    "fill": NORD["snow_storm_2"],
    "white": "#FFFFFF",
}

SEMANTIC = {
    "primary": CATEGORICAL["blue"],
    "baseline": NEUTRAL["text"],
    "secondary": CATEGORICAL["teal"],
    "structured": CATEGORICAL["purple"],
    "negative": CATEGORICAL["red"],
    "warning": CATEGORICAL["orange"],
    "success": CATEGORICAL["green"],
    "reference": NEUTRAL["reference"],
    "unresolved": NEUTRAL["unresolved"],
}


@dataclass(frozen=True)
class MethodStyle:
    """Stable visual identity for one method across every figure."""

    color: str
    marker: str
    linestyle: str
    label: str


METHOD_STYLES = {
    "mse": MethodStyle(SEMANTIC["baseline"], "o", "--", "MSE"),
    "isotropic": MethodStyle(SEMANTIC["primary"], "s", "-", "MSE + isotropic DPSAE"),
    "whitened": MethodStyle(SEMANTIC["secondary"], "D", "-.", "MSE + whitening"),
    "spectral": MethodStyle(
        CATEGORICAL["yellow"], "^", ":", "MSE + static spectral"
    ),
    "decoder_only": MethodStyle(SEMANTIC["negative"], "X", ":", "Decoder only"),
    "task_prior": MethodStyle(SEMANTIC["structured"], "P", "-", "MSE + task-prior DPSAE"),
    "weighted_mse": MethodStyle(
        SEMANTIC["success"], "v", "-.", "MSE + frozen-task loss"
    ),
    "permuted_prior": MethodStyle(
        SEMANTIC["warning"], "X", ":", "MSE + permuted prior"
    ),
}

METHOD_ORDER = ["mse", "isotropic", "whitened", "spectral", "decoder_only"]
COLORS = {name: style.color for name, style in METHOD_STYLES.items()}
COLORS.update(
    {
        "theory": SEMANTIC["reference"],
        "random": NEUTRAL["unresolved"],
        "advantage": SEMANTIC["primary"],
        "disadvantage": SEMANTIC["negative"],
        "unresolved": SEMANTIC["unresolved"],
    }
)
MARKERS = {name: style.marker for name, style in METHOD_STYLES.items()}
LINESTYLES = {name: style.linestyle for name, style in METHOD_STYLES.items()}
LABELS = {name: style.label for name, style in METHOD_STYLES.items()}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "nord_frost", [NEUTRAL["white"], NORD["frost_cyan"], NORD["frost_deep"]]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nord_advantage", [NORD["aurora_red"], NEUTRAL["white"], NORD["frost_deep"]]
)

FIGURE_WIDTHS = {
    "half": 2.62,
    "full": 5.50,
    "wide": 7.00,
}


def figure_size(width: str | float = "full", *, aspect: float = 0.52) -> tuple[float, float]:
    """Return an exact physical figure size.

    ``aspect`` is height divided by width.  Pass the venue's measured text
    width as a float when it differs from the named profiles.
    """

    width_in = FIGURE_WIDTHS[width] if isinstance(width, str) else float(width)
    if width_in <= 0 or aspect <= 0:
        raise ValueError("Figure width and aspect must be positive")
    return width_in, width_in * aspect


@contextmanager
def paper_context() -> Iterator[None]:
    """Apply the style without leaking rcParams into other plots."""

    with mpl.rc_context():
        mpl.style.use(STYLE_PATH)
        yield


def apply_paper_style() -> None:
    """Apply the style globally for legacy experiment scripts."""

    mpl.style.use(STYLE_PATH)


def clean_axis(ax: Axes, *, ylog: bool = False, xlog: bool = False) -> None:
    """Apply consistent axis scales, spines, ticks, and grids."""

    if ylog:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL["muted"])
    ax.spines["bottom"].set_color(NEUTRAL["muted"])
    ax.grid(True, which="major", color=NEUTRAL["grid"], linewidth=0.55)
    if ylog or xlog:
        axis = "both" if ylog and xlog else ("y" if ylog else "x")
        ax.grid(True, which="minor", axis=axis, color=NEUTRAL["grid_minor"], linewidth=0.4)
    ax.tick_params(direction="out", length=3, width=0.7, colors=NEUTRAL["muted"])


def label_panels(
    axes: Axes | Sequence[Axes],
    labels: Sequence[str] | None = None,
    *,
    x: float = -0.13,
    y: float = 1.06,
) -> None:
    """Add unobtrusive panel labels in axes coordinates."""

    axes_list = [axes] if isinstance(axes, Axes) else list(axes)
    panel_labels = labels or tuple(chr(ord("A") + i) for i in range(len(axes_list)))
    if len(panel_labels) != len(axes_list):
        raise ValueError("Panel label count must match axes count")
    for ax, label in zip(axes_list, panel_labels, strict=True):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            color=NEUTRAL["text"],
            fontsize=mpl.rcParams["axes.titlesize"],
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    crop: bool = False,
) -> tuple[Path, Path]:
    """Save paired vector and raster outputs with deterministic dimensions."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    bbox = "tight" if crop else None
    fig.savefig(pdf_path, bbox_inches=bbox, facecolor="white")
    fig.savefig(png_path, bbox_inches=bbox, facecolor="white", dpi=dpi)
    return pdf_path, png_path


def savefig(fig: Figure, path: str | Path) -> None:
    """Compatibility wrapper for legacy figures that were designed for tight crops."""

    save_figure(fig, path, crop=True)


def contrast_ratio(foreground: str, background: str = "#FFFFFF") -> float:
    """Return the WCAG relative-luminance contrast ratio for two hex colors."""

    def relative_luminance(color: str) -> float:
        value = color.lstrip("#")
        if len(value) != 6:
            raise ValueError(f"Expected a six-digit hex color, got {color!r}")
        channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    first = relative_luminance(foreground)
    second = relative_luminance(background)
    light, dark = max(first, second), min(first, second)
    return (light + 0.05) / (dark + 0.05)
