#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="/tmp/dpsae-matplotlib"

mkdir -p experiments/logs experiments/outputs/exp01_isotropic_spectral experiments/figures

uv run --extra experiments python -u experiments/exp01_isotropic_spectral.py \
  --config configs/exp01_isotropic_spectral.json \
  --output-dir experiments/outputs/exp01_isotropic_spectral \
  --figures-dir experiments/figures \
  2>&1 | tee experiments/logs/exp01_isotropic_spectral.log
