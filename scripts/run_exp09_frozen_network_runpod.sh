#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-status}"
if [[ "$MODE" != full && "$MODE" != smoke && "$MODE" != status ]]; then
  echo "usage: $0 {full|smoke|status}" >&2
  exit 2
fi

CONFIG="${EXP09_CONFIG:-configs/exp09_frozen_network.json}"
OUTPUT="${EXP09_OUTPUT:-artifacts/exp09_frozen_network}"
MODELS="${EXP09_MODELS:-artifacts/exp08_experiment_figure/confirmation/models.pt}"
CALIBRATION="${EXP09_CALIBRATION:-artifacts/exp04_ioi_mechanism/calibration.pt}"
TRAINING_DONE="${EXP09_TRAINING_DONE:-artifacts/exp08_experiment_figure/confirmation/done.json}"
CONFIRMATION_SUMMARY="${EXP09_CONFIRMATION_SUMMARY:-artifacts/exp08_experiment_figure/confirmation_summary.json}"
PYTHON="${DPSAE_PYTHON:-$ROOT/.venv/bin/python}"
HF_CACHE="${HF_HOME:-/workspace/huggingface}"
GPU="${EXP09_GPU:-0}"
GPU_MEMORY_FRACTION="${EXP09_GPU_MEMORY_FRACTION:-0.25}"
MAXIMUM_PEAK_GPU_GIB="${EXP09_MAXIMUM_PEAK_GPU_GIB:-12}"
MINIMUM_FREE_GIB="${EXP09_MINIMUM_FREE_GIB:-10}"
LOCAL_FILES_ONLY="${EXP09_LOCAL_FILES_ONLY:-0}"
SMOKE_CACHE="${EXP09_SMOKE_CACHE:-}"
SMOKE_SEQUENCES="${EXP09_SMOKE_SEQUENCES:-8}"
SMOKE_IOI_EXAMPLES="${EXP09_SMOKE_IOI_EXAMPLES:-16}"
OPEN_FRESH_RANGE="${EXP09_OPEN_FRESH_RANGE:-}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$ROOT/$OUTPUT"
fi
CONFIG_SHA="$(shasum -a 256 "$CONFIG" | awk '{print substr($1,1,8)}')"
SESSION="dpsae-exp09-frozen-${MODE}-${CONFIG_SHA}"
LOGS="$OUTPUT/logs"
STATUS_DIR="$OUTPUT/status"
LOG="$LOGS/${MODE}-${CONFIG_SHA}.log"
STATUS="$STATUS_DIR/${MODE}-${CONFIG_SHA}.status"

mkdir -p "$LOGS" "$STATUS_DIR"

if [[ "$MODE" == status ]]; then
  for path in "$STATUS_DIR"/*.status; do
    [[ -e "$path" ]] || continue
    printf '%s: ' "$(basename "$path")"
    sed -n '1p' "$path"
  done
  tmux list-sessions -F '#{session_name} #{session_attached} #{session_windows}' 2>/dev/null \
    | awk '/^dpsae-exp09-frozen-/ {print}' || true
  exit 0
fi

if [[ "${DPSAE_EXP09_IN_TMUX:-0}" != 1 ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    PANE_DEAD="$(tmux display-message -p -t "$SESSION" '#{pane_dead}')"
    if [[ "$PANE_DEAD" == 1 ]]; then
      tmux kill-session -t "$SESSION"
    else
      echo "already running: $SESSION"
      echo "log: $LOG"
      exit 0
    fi
  fi
  TMUX_ENVIRONMENT=(
    "DPSAE_EXP09_IN_TMUX=1"
    "DPSAE_PYTHON=$PYTHON"
    "HF_HOME=$HF_CACHE"
    "EXP09_CONFIG=$CONFIG"
    "EXP09_OUTPUT=$OUTPUT"
    "EXP09_MODELS=$MODELS"
    "EXP09_CALIBRATION=$CALIBRATION"
    "EXP09_TRAINING_DONE=$TRAINING_DONE"
    "EXP09_CONFIRMATION_SUMMARY=$CONFIRMATION_SUMMARY"
    "EXP09_GPU=$GPU"
    "EXP09_GPU_MEMORY_FRACTION=$GPU_MEMORY_FRACTION"
    "EXP09_MAXIMUM_PEAK_GPU_GIB=$MAXIMUM_PEAK_GPU_GIB"
    "EXP09_MINIMUM_FREE_GIB=$MINIMUM_FREE_GIB"
    "EXP09_LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
    "EXP09_SMOKE_CACHE=$SMOKE_CACHE"
    "EXP09_SMOKE_SEQUENCES=$SMOKE_SEQUENCES"
    "EXP09_SMOKE_IOI_EXAMPLES=$SMOKE_IOI_EXAMPLES"
    "EXP09_OPEN_FRESH_RANGE=$OPEN_FRESH_RANGE"
  )
  printf -v WORKER_COMMAND 'cd %q && env' "$ROOT"
  for assignment in "${TMUX_ENVIRONMENT[@]}"; do
    printf -v WORKER_COMMAND '%s %q' "$WORKER_COMMAND" "$assignment"
  done
  printf -v WORKER_COMMAND '%s bash %q %q >> %q 2>&1' \
    "$WORKER_COMMAND" "$ROOT/scripts/run_exp09_frozen_network_runpod.sh" "$MODE" "$LOG"
  tmux new-session -d -s "$SESSION" "$WORKER_COMMAND"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "started $SESSION"
  echo "log: $LOG"
  echo "status: $STATUS"
  exit 0
fi

write_status() {
  local value="$1"
  local temporary="${STATUS}.tmp"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$STATUS"
}

run_mode() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Exp09 requires a clean worktree" >&2
    return 1
  fi
  test -x "$PYTHON"
  test -s "$CONFIG"
  test -s "$MODELS"
  test -s "$CALIBRATION"
  test -s "$TRAINING_DONE"
  test -s "$CONFIRMATION_SUMMARY"

  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH=.:src
  export HF_HOME="$HF_CACHE"
  export TOKENIZERS_PARALLELISM=true
  export CUDA_VISIBLE_DEVICES="$GPU"

  local common=(
    --config "$CONFIG"
    --output-dir "$OUTPUT"
    --models "$MODELS"
    --calibration "$CALIBRATION"
    --training-done "$TRAINING_DONE"
    --confirmation-summary "$CONFIRMATION_SUMMARY"
    --device cuda:0
    --gpu-memory-fraction "$GPU_MEMORY_FRACTION"
    --maximum-peak-gpu-gib "$MAXIMUM_PEAK_GPU_GIB"
    --minimum-free-gib "$MINIMUM_FREE_GIB"
  )
  if [[ "$LOCAL_FILES_ONLY" == 1 ]]; then
    common+=(--local-files-only)
  fi

  if [[ "$MODE" == smoke ]]; then
    if [[ -z "$SMOKE_CACHE" || ! -s "$SMOKE_CACHE" ]]; then
      echo "smoke mode requires EXP09_SMOKE_CACHE pointing to already-opened data" >&2
      return 1
    fi
    "$PYTHON" -u experiments/exp09_frozen_network.py natural \
      "${common[@]}" \
      --smoke \
      --smoke-cache "$SMOKE_CACHE" \
      --maximum-sequences "$SMOKE_SEQUENCES"
    "$PYTHON" -u experiments/exp09_frozen_network.py ioi \
      "${common[@]}" \
      --smoke \
      --maximum-ioi-examples "$SMOKE_IOI_EXAMPLES"
    "$PYTHON" -u experiments/exp09_frozen_network.py validate \
      "${common[@]}" \
      --smoke
    return
  fi

  if [[ "$OPEN_FRESH_RANGE" != YES ]]; then
    echo "full mode requires EXP09_OPEN_FRESH_RANGE=YES after protocol review" >&2
    return 1
  fi
  "$PYTHON" -u experiments/exp09_frozen_network.py prepare "${common[@]}"
  "$PYTHON" -u experiments/exp09_frozen_network.py natural "${common[@]}"
  "$PYTHON" -u experiments/exp09_frozen_network.py ioi "${common[@]}"
  "$PYTHON" -u experiments/exp09_frozen_network.py validate "${common[@]}"
}

write_status running
if run_mode; then
  write_status succeeded
else
  code=$?
  write_status "failed:$code"
  exit "$code"
fi
