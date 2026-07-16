#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/workspace/SAEBench/.venv/bin/python}"
SAEBENCH_ROOT="${SAEBENCH_ROOT:-/workspace/SAEBench}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/dpsae-restored/exp06_generality/pythia-block8}"
CONFIG="${CONFIG:-$ROOT/configs/exp10_concept_discovery.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/exp10_concept_discovery}"
MODEL_CACHE="${MODEL_CACHE:-/workspace/exp10-model-cache}"
COLD_CACHE_PROVENANCE="${COLD_CACHE_PROVENANCE:-$OUTPUT_ROOT/cold_cache_timing_provenance.json}"
LOG_ROOT="$OUTPUT_ROOT/logs"
SESSION="${SESSION:-exp10-timing}"
MODE="${1:-}"

if [[ -n "$MODE" && "$MODE" != "--worker" ]]; then
  echo "usage: $0 [--worker]" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/src:$ROOT:$SAEBENCH_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=true
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HOST_CPUS="$(nproc)"
WORKER_THREADS="$((HOST_CPUS / 4))"
if [[ "$WORKER_THREADS" -lt 1 ]]; then
  WORKER_THREADS=1
fi
export OMP_NUM_THREADS="$WORKER_THREADS"
export MKL_NUM_THREADS="$WORKER_THREADS"
export OPENBLAS_NUM_THREADS="$WORKER_THREADS"
export NUMEXPR_NUM_THREADS="$WORKER_THREADS"

mkdir -p "$LOG_ROOT" "$MODEL_CACHE"

if [[ ! -x "$PYTHON" ]]; then
  echo "sealed exp10 Python is missing or not executable: $PYTHON" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "exp10 timing smoke requires a clean repository revision" >&2
  exit 2
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 4 ]]; then
  echo "exp10 timing smoke requires exactly four visible GPUs" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "$name" != *A40* ]]; then
    echo "exp10 timing smoke is frozen for 4xA40; observed $name" >&2
    exit 2
  fi
done

if [[ "$MODE" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 2
  fi
  tmux new-session -d -s "$SESSION" \
    "env PYTHON='$PYTHON' SAEBENCH_ROOT='$SAEBENCH_ROOT' CHECKPOINT_DIR='$CHECKPOINT_DIR' CONFIG='$CONFIG' OUTPUT_ROOT='$OUTPUT_ROOT' MODEL_CACHE='$MODEL_CACHE' COLD_CACHE_PROVENANCE='$COLD_CACHE_PROVENANCE' HF_HOME='$HF_HOME' bash '$ROOT/scripts/run_exp10_timing_smoke_a40.sh' --worker"
  echo "started exp10 timing preflight in tmux session $SESSION"
  echo "log: $LOG_ROOT/timing-smoke.log"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0
"$PYTHON" -u experiments/exp10_concept_discovery.py \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --saebench-root "$SAEBENCH_ROOT" \
  timing-preflight \
  --model-cache "$MODEL_CACHE" \
  --cold-cache-provenance "$COLD_CACHE_PROVENANCE" \
  --device cuda:0 \
  2>&1 | tee "$LOG_ROOT/timing-smoke.log"
