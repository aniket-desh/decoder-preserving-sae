# Decoder-Preserving Sparse Autoencoders

This repository contains the implementation and experiments for testing whether sparse autoencoders preserve more useful structure when their reconstruction objective includes disagreement between optimal regularized linear decoders. The initial scope is decoder preservation, ordinary activation reconstruction, and sparsity; activation-manifold extensions are outside the first experimental claim.

## First research question

Holding the SAE architecture, dictionary width, sparsity budget, data, and optimizer fixed, does adding a Harvey-style decoder-preservation term improve held-out decodable-information retention or downstream fidelity beyond an ordinary reconstruction objective?

The first controlled comparison is:

1. BatchTopK SAE with activation MSE.
2. The same BatchTopK SAE with activation MSE plus decoder distance.
3. A decoder-heavy or decoder-only ablation, included to expose coordinate drift and downstream incompatibility rather than treated as the default method.

## Repository map

- `src/dpsae/decoder_distance.py` contains the differentiable reference objective.
- `tests/` checks invariances and basic numerical behavior before model training.
- `configs/` contains versioned experiment configurations.
- `scripts/` contains executable research and validation entry points.

## Local setup

The reference implementation targets Python 3.11+ and PyTorch. With `uv` installed:

```bash
uv sync --extra dev
uv run pytest
```

## Figures

The project uses the Nord scientific style in `src/dpsae/plot_style.py`. It fixes semantic method identities, colorblind-redundant markers and line styles, venue-aware physical sizing, and paired PDF/PNG export. See `docs/plotting_style.md` for the visual contract.

Install the portable Matplotlib layer once for use in any repository:

```bash
PYTHONPATH=src python3 scripts/install_plot_style.py
```

After installation, lightweight plots can use `plt.style.use("aniket-nord")`; paper figures should use the semantic helpers so colors retain the same scientific meaning across projects.
