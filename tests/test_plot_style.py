from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dpsae.plot_style import (  # noqa: E402
    CATEGORICAL,
    CONDENSED_FONT_FAMILY,
    EXPANDED_FONT_FAMILY,
    FIGURE_FONT_FAMILY,
    FONT_DIR,
    METHOD_STYLES,
    NORD,
    SEMANTIC,
    STYLE_PATH,
    VENDORED_FONT_FILES,
    VENDORED_FONT_SHA256,
    contrast_ratio,
    figure_size,
    paper_context,
    save_figure,
)


def test_categorical_colors_are_distinct_official_nord_accents() -> None:
    assert set(CATEGORICAL.values()) <= set(NORD.values())
    assert len(set(CATEGORICAL.values())) == len(CATEGORICAL)
    assert contrast_ratio(CATEGORICAL["blue"]) >= 3.0


def test_method_identities_are_redundant_and_unique() -> None:
    required = {"mse", "isotropic", "whitened", "task_prior", "permuted_prior"}
    assert required <= METHOD_STYLES.keys()
    signatures = {
        (style.color, style.marker, style.linestyle) for style in METHOD_STYLES.values()
    }
    assert len(signatures) == len(METHOD_STYLES)
    assert len({style.color for style in METHOD_STYLES.values()}) == len(METHOD_STYLES)
    assert METHOD_STYLES["isotropic"].color == SEMANTIC["primary"]


def test_paper_context_is_local() -> None:
    previous_family = list(plt.rcParams["font.family"])
    with paper_context():
        assert STYLE_PATH.exists()
        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert plt.rcParams["font.sans-serif"][0] == FIGURE_FONT_FAMILY
        assert plt.rcParams["axes.unicode_minus"] is False
    assert plt.rcParams["font.family"] == previous_family


def test_vendored_d_din_family_is_registered() -> None:
    assert {path.name for path in FONT_DIR.glob("*.ttf")} == set(VENDORED_FONT_FILES)
    assert set(VENDORED_FONT_SHA256) == set(VENDORED_FONT_FILES)
    registered = {
        font.name
        for font in matplotlib.font_manager.fontManager.ttflist
        if Path(font.fname).parent == FONT_DIR
    }
    assert registered == {
        FIGURE_FONT_FAMILY,
        CONDENSED_FONT_FAMILY,
        EXPANDED_FONT_FAMILY,
    }
    expected_faces = {
        ("normal", "normal"): "D-DIN.ttf",
        ("italic", "normal"): "D-DIN-Italic.ttf",
        ("normal", "bold"): "D-DIN-Bold.ttf",
    }
    for (style, weight), filename in expected_faces.items():
        resolved = matplotlib.font_manager.findfont(
            matplotlib.font_manager.FontProperties(
                family=FIGURE_FONT_FAMILY,
                style=style,
                weight=weight,
            ),
            fallback_to_default=False,
        )
        assert Path(resolved) == FONT_DIR / filename


def test_exact_size_export(tmp_path) -> None:
    width, height = figure_size(4.0, aspect=0.5)
    with paper_context():
        fig, ax = plt.subplots(figsize=(width, height))
        ax.plot([0, 1], [0, 1], color=SEMANTIC["primary"])
        pdf_path, png_path = save_figure(fig, tmp_path / "smoke")
        plt.close(fig)

    assert pdf_path.exists()
    image = plt.imread(png_path)
    assert image.shape[:2] == (600, 1200)
