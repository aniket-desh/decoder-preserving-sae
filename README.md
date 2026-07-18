# Decoder-Preserving Sparse Autoencoders

This repository contains the implementation and experiment code for Decoder-Preserving Sparse Autoencoders (DPSAEs). A DPSAE augments a matched sparse-autoencoder objective with disagreement between optimal regularized linear readouts fitted to the original and reconstructed activations.

The supported claim is deliberately narrow: at fixed sparsity and matched reconstruction quality, decoder preservation can improve held-out refittable readout fidelity. The repository also retains the boundary tests that prevent this result from being restated as uniform frozen-task preservation or better sparse concept discovery.

## Quick start

The reference environment uses Python 3.11 and `uv`:

```bash
uv sync --extra dev --extra experiments
bash scripts/check.sh
```

The first two controlled experiments run without model checkpoints or private data:

```bash
bash scripts/run_exp01_local.sh
bash scripts/run_exp02_local.sh
```

Each writes machine-readable tables under `experiments/outputs/`, figures under `experiments/figures/`, and a log under `experiments/logs/`. Those generated directories are intentionally ignored by Git.

## Repository map

- `src/dpsae/` is the reusable scientific library: decoder distance, SAE implementations, language-model activation and training utilities, task-fidelity operators, benchmark adapters, plotting, and release validation.
- `configs/` contains the versioned inputs for every retained experiment. Reported values should always resolve to a config, source revision, seed, and machine-readable artifact.
- `experiments/` contains scientific command-line entry points. [experiments/README.md](experiments/README.md) separates headline workflows from appendix and negative-result controls.
- `scripts/` contains a small runner and artifact-tool surface. Provider-specific backup, watcher, and queue-management helpers are excluded from the public tree.
- `tests/` covers the objective, training and evaluation utilities, retained experiment contracts, and release checks.
- `results/paper_audits/` contains compact, versioned audit summaries. Checkpoints, activation caches, per-example outputs, and full run trees remain external.

## Core workflows

The local runners cover the controlled synthetic and estimator studies. Full language-model reproduction requires the immutable token caches and checkpoint bundles described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
# Exact GPT-2 training and evaluation fleet used for the paper-facing result
bash scripts/run_exp08_gpu_runpod.sh
bash scripts/run_exp08_synthetic_runpod.sh

# Fresh frozen-network boundary test
bash scripts/run_exp09_frozen_network_runpod.sh status

# Inspect the standard-concept pilot stages
PYTHONPATH=src python3 experiments/exp10_concept_discovery.py --help

# Matched-NMSE static-control screen
PYTHONPATH=src python3 experiments/exp11_static_matched_nmse.py --help
```

The expensive runners fail closed when required configs, source hashes, caches, checkpoints, hardware, or approval markers do not match. Use the individual Python entry points with `--help` to inspect stages without starting a run.

## Artifact boundary

Large generated files do not belong in Git. The paper-facing release is a hash-audited artifact tree whose policy is `configs/arxiv_release_closure.json`; its source bundle preserves the exact experiment-time revision, including operational harnesses that are intentionally absent from this cleaned public surface. The current tree keeps the scientific implementation, configs, audit logic, aggregation code, and figure renderer needed to inspect or reproduce the results.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the input layout, validation commands, and the distinction between a local smoke reproduction and the full paper run.

## Figures

All generated plots use the Nord scientific style in `src/dpsae/plot_style.py`. It fixes semantic method identities, colorblind-redundant markers and line styles, D-DIN figure typography, venue-aware sizing, and paired PDF/PNG export. The plotting contract is recorded in `docs/plotting_style.md`.

Install the portable Matplotlib style for use outside this repository with:

```bash
PYTHONPATH=src python3 scripts/install_plot_style.py
```
