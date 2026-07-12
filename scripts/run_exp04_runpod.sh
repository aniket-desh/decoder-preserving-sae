#!/usr/bin/env bash
set -euo pipefail

cd /workspace/decoder-preserving-sae
mkdir -p artifacts/exp04_ioi_mechanism/logs
export PYTHONPATH=src
export HF_HOME=/workspace/huggingface
export TOKENIZERS_PARALLELISM=true
python3 -u experiments/exp04_ioi_mechanism.py all \
  --config configs/exp04_ioi_mechanism.json \
  2>&1 | tee -a artifacts/exp04_ioi_mechanism/logs/full_run.log
