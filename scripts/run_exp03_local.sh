#!/usr/bin/env bash
set -euo pipefail

export PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin:/usr/bin:/bin"
export PYTHONPATH="src"
export MPLBACKEND="Agg"
export MPLCONFIGDIR="/tmp/dpsae-matplotlib"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p experiments/logs experiments/outputs/exp03_estimator_scaling experiments/figures

python3 -u experiments/exp03_estimator_scaling.py \
  --config configs/exp03_estimator_scaling.json \
  --output-dir experiments/outputs/exp03_estimator_scaling \
  --figures-dir experiments/figures \
  2>&1 | tee experiments/logs/exp03_estimator_scaling.log
