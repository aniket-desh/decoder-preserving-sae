#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/paper_closure/logs artifacts/exp04b_mechanism_attribution
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
PYTHON=.venv/bin/python

"$PYTHON" -u experiments/exp04b_mechanism_attribution.py prepare \
  --natural-cache artifacts/exp04b_confirmatory/natural_test.pt \
  --models artifacts/exp04b_confirmatory/baseline_confirm/models.pt \
  --reconstruction-dir artifacts/exp04b_mechanism_attribution/reconstructions \
  --output artifacts/exp04b_mechanism_attribution/prepare.json \
  --device cpu \
  > artifacts/paper_closure/logs/mechanism_reconstructions.log 2>&1

"$PYTHON" -u experiments/exp04b_mechanism_attribution.py tangent \
  --natural-cache artifacts/exp04b_confirmatory/natural_test.pt \
  --static-calibration artifacts/exp04b_confirmatory/static_calibration.pt \
  --reconstruction-dir artifacts/exp04b_mechanism_attribution/reconstructions \
  --output artifacts/exp04b_mechanism_attribution/tangent.json \
  > artifacts/paper_closure/logs/mechanism_tangent.log 2>&1
