# Aniket Nord scientific figures

> **Current publication contract:** use `docs/notes/portable_figure_style_protocol.md`. It supersedes the older clauses below wherever they conflict, especially the rules for no in-figure titles or panel letters and the finalized red-MSE/blue-primary identity.

This is the default visual system for DPSAE and the portable starting point for future projects. It combines Nord's cool visual identity with the figure grammar that makes strong scientific papers feel coherent: a white canvas, one dominant method color, pale structural fills, direct annotations, restrained legends, and a small number of accents whose meanings do not change between figures.

The canonical files are:

- `src/dpsae/styles/aniket-nord.mplstyle` for portable Matplotlib defaults;
- `src/dpsae/plot_style.py` for semantic colors, method identities, colormaps, exact sizing, and export helpers;
- `scripts/install_plot_style.py` for installing `aniket-nord` into the current user's Matplotlib style library.

## The visual contract

Every figure should make its comparison legible before the caption is read. The primary method is Frost blue, the ordinary baseline is Polar Night, references and unresolved mass are neutral, and negative evidence is Aurora red. A method keeps the same color, marker, and line style everywhere. Hue is redundant with shape or line style, so the comparison survives grayscale printing and common color-vision deficiencies.

Categorical marks use the official Nord accents directly. Assign them as ordinary series colors, choosing well-separated hues before adjacent Frost shades; markers and line styles remain redundant so pale Nord colors and grayscale printing do not erase method identity.

| Scientific role | Color | Hex | Use |
| --- | --- | --- | --- |
| Primary method or positive advantage | Frost deep blue | `#5E81AC` | DPSAE, positive effects, selected result |
| Baseline and text | Polar Night | `#2E3440` | MSE, labels, structural outlines |
| Secondary control | Frost teal | `#8FBCBB` | orthogonal controls |
| Structured or task-aware method | Aurora purple | `#B48EAD` | task-prior DPSAE |
| Negative evidence | Aurora red | `#BF616A` | harm, regression, negative side of a diverging scale |
| Warning or correspondence control | Aurora orange | `#D08770` | permuted priors and cautions |
| Successful supervised control | Aurora green | `#A3BE8C` | frozen-task controls |
| Unresolved or unavailable | Snow Storm gray | `#D8DEE9` | indeterminate mass and missing estimates |
| Reference or theory | Polar Night gray | `#4C566A` | gates, chance lines, analytic references |

Do not use a rainbow map. Ordered nonnegative quantities use `SEQUENTIAL_CMAP`, which moves from white through Frost cyan to deep blue with increasing luminance contrast. Signed quantities use `DIVERGING_CMAP`, centered on white, with red for negative and blue for positive. A meaningful zero must be visually neutral.

## Physical size and typography

Design at the width where the figure will be printed. Do not make a 7-inch figure and ask LaTeX to shrink it to 5.5 inches, because that silently turns 8-point labels into 6-point labels.

```python
from dpsae.plot_style import figure_size, paper_context, save_figure

with paper_context():
    fig, ax = plt.subplots(figsize=figure_size("full", aspect=0.42))
    ...
    save_figure(fig, "figures/result")
```

The built-in widths are 5.50 inches for the current ICLR text block, 2.62 inches for a half-width panel, and 7.00 inches for a wide preprint figure. Pass a measured width directly for another venue. At final size, ordinary and axis-label text is 8.25 pt, ticks and legends are 7.75 pt, compact annotations are at least 7.5 pt, and the rare in-figure title is 9 pt. The style loads the repository's pinned D-DIN Regular, Italic, and Bold files directly, so local, RunPod, and CI renders use the same glyphs; DejaVu Sans is retained only as the unsupported-math fallback, and tick labels use the supported ASCII minus rather than D-DIN's absent Unicode-minus glyph. D-DIN Condensed is restricted to irreducible short labels, while D-DIN Expanded is restricted to sparse display accents and never appears on axes, ticks, legends, or ordinary annotations. Final exports are vector PDF plus exact-size 300-DPI PNG; the canonical exporter does not use a tight crop.

For lightweight work in any repository, install the portable style once:

```bash
PYTHONPATH=src python3 scripts/install_plot_style.py
```

Then use `plt.style.use("aniket-nord")`. The semantic Python layer is still preferred for papers because it fixes what each color means.

## Figure grammar

- One figure answers one scientific question. Panels may show the mechanism, primary estimate, and boundary of the claim, but they should form one argument.
- Do not use in-figure panel titles; use short axis labels with units and put experimental detail in the caption rather than inside the axes.
- Show raw paired observations when their pairing matters. Use arrows, connecting lines, or aligned points so the reader does not have to infer the comparison.
- Show uncertainty in the same visual unit as the estimate. State whether intervals cover seeds, held-out groups, samples, or Monte Carlo directions.
- Prefer direct labels for one or two series. Use one shared frameless legend when several methods recur, ordered baseline, primary method, then controls.
- Use pale fills to explain structure or uncertainty and saturated strokes for data. Large saturated rectangles should be rare.
- Do not add `A`, `B`, or `C` panel letters. Refer to panels as `Left`, `Center`, and `Right` in the caption.
- Keep top and right spines absent, grids light, and reference lines visually subordinate to measured data.
- Never encode scientific importance through color saturation alone. Line width, marker size, and z-order should emphasize the declared primary comparison.

## Stable method identities

| Method | Color | Marker | Line |
| --- | --- | --- | --- |
| MSE | Polar Night | circle | dashed |
| Isotropic DPSAE | Frost blue | square | solid |
| Whitening | Frost teal | diamond | dash-dot |
| Static spectral control | Aurora yellow | triangle | dotted |
| Decoder only | Aurora red | X | dotted |
| Task-prior DPSAE | Aurora purple | filled plus | solid |
| Frozen-task loss | Aurora green | down triangle | dash-dot |
| Permuted prior | Aurora orange | X | dotted |

## Release checklist

- [ ] The figure has one claim and the caption states its scope.
- [ ] Physical dimensions match the destination text or column width.
- [ ] Text remains readable at 100% PDF zoom and in the compiled paper.
- [ ] Ordinary Latin text resolves to the vendored D-DIN files without host substitution.
- [ ] The same methods use the same semantic identities as every other figure.
- [ ] No comparison depends on hue alone.
- [ ] Error bars or bands identify their sampling unit.
- [ ] Log axes exclude zero and document any display floor.
- [ ] Sequential and diverging maps respect the ordering and zero of the quantity.
- [ ] PDF text remains vector text and the PNG is exactly 300 DPI.
- [ ] The committed data regenerate the committed figure.
- [ ] The final PDF and PNG have been visually inspected for clipping, overlap, and false emphasis.
