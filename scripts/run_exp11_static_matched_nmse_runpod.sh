#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"
if [[ "$STAGE" == "-h" || "$STAGE" == "--help" ]]; then
  echo "usage: $0 [validate|screen|confirm|finalize|all]"
  exit 0
fi
if [[ "$STAGE" != "--worker" && ! "$STAGE" =~ ^(validate|screen|confirm|finalize|all)$ ]]; then
  echo "usage: $0 [validate|screen|confirm|finalize|all]" >&2
  exit 2
fi
case "$STAGE" in
  all|screen) DEFAULT_SESSION="dpsae-spectral-screen" ;;
  confirm) DEFAULT_SESSION="dpsae-spectral-confirm" ;;
  *) DEFAULT_SESSION="dpsae-spectral-${STAGE}" ;;
esac
SESSION="${EXP11_SESSION:-$DEFAULT_SESSION}"
REFERENCE_ROOT="${EXP11_REFERENCE_ARTIFACT_ROOT:-/workspace/dpsae-restored/paper_closure/confirmation_common}"
LOG_DIR="$ROOT/artifacts/exp11_static_matched_nmse/logs"
PYTHON_BIN="${EXP11_PYTHON:-$ROOT/.venv/bin/python}"
GPU_ID="${EXP11_CUDA_VISIBLE_DEVICES:-3}"
HF_ROOT="${HF_HOME:-/workspace/huggingface}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

if [[ "$STAGE" == "--worker" ]]; then
  STAGE="${2:-all}"
  mkdir -p "$LOG_DIR"
  cd "$ROOT"
  export PYTHONPATH=.:src
  export HF_HOME="$HF_ROOT"
  export TOKENIZERS_PARALLELISM=true
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  "$PYTHON_BIN" -u experiments/exp11_static_matched_nmse.py \
    "$STAGE" \
    --reference-artifact-root "$REFERENCE_ROOT" \
    2>&1 | tee -a "$LOG_DIR/${STAGE}.log"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" \
  "EXP11_REFERENCE_ARTIFACT_ROOT='$REFERENCE_ROOT' EXP11_PYTHON='$PYTHON_BIN' EXP11_CUDA_VISIBLE_DEVICES='$GPU_ID' HF_HOME='$HF_ROOT' '$0' --worker '$STAGE'"
echo "started tmux session $SESSION"
echo "log: $LOG_DIR/${STAGE}.log"
