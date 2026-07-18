#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="/tmp/dpsae-matplotlib"

mkdir -p experiments/logs experiments/outputs/exp02_structured_prior experiments/figures

uv run --extra experiments python -u experiments/exp02_structured_prior.py \
  --config configs/exp02_structured_prior.json \
  --output-dir experiments/outputs/exp02_structured_prior \
  --figures-dir experiments/figures \
  2>&1 | tee experiments/logs/exp02_structured_prior.log
