#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT="$ROOT/artifacts/exp04b_confirmatory"
MODEL_REPO="${EXP04B_MODEL_REPO:-aniketdesh/decoder-preserving-sae-exp04b}"
DATASET_REPO="${EXP04B_DATASET_REPO:-aniketdesh/decoder-preserving-sae-exp04b-data}"

cd "$ROOT"
mkdir -p "$ARTIFACT/logs"
export PYTHONPATH=src
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export TOKENIZERS_PARALLELISM=true

run_stage() {
  local log="$1"
  shift
  "$@" 2>&1 | tee -a "$ARTIFACT/logs/$log.log"
  "$ROOT/scripts/backup_exp04b_hf.sh" "$MODEL_REPO" "$DATASET_REPO" || true
}

for stage in prepare-tail cache-natural calibrate-static; do
  run_stage "$stage" python3 -u experiments/exp04b_confirmatory.py "$stage"
done
run_stage natural-source python3 -u experiments/exp04b_confirmatory.py \
  natural-evaluate --fleet source
for stage in baseline-screen baseline-confirm; do
  run_stage "$stage" python3 -u experiments/exp04b_confirmatory.py "$stage"
done
run_stage natural-baseline python3 -u experiments/exp04b_confirmatory.py \
  natural-evaluate --fleet baseline
for stage in prepare selection test; do
  run_stage "ioi-$stage" python3 -u experiments/exp04b_ioi_confirmatory.py "$stage"
done
