#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$SCRIPT_ROOT}"
PYTHON="${PYTHON:-/workspace/SAEBench/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/exp13_concept_confirmation.json}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT/configs/exp10_concept_discovery.json}"
EXP12_CONFIG="${EXP12_CONFIG:-$ROOT/configs/exp12_pythia_fresh_confirmation.json}"
EXP12_ROOT="${EXP12_ROOT:-/workspace/dpsae-runs/20260716/exp12_pythia_fresh_confirmation/fresh-pairs-v1}"
PILOT_ROOT="${PILOT_ROOT:-/workspace/dpsae-runs/20260716/exp10_concept_discovery/pythia160m-block8-s0-pilot-v2}"
PILOT_AUDIT="${PILOT_AUDIT:-$PILOT_ROOT/artifact_audit_final.json}"
SOURCE_CACHE_READY="${SOURCE_CACHE_READY:-$PILOT_ROOT/cache_ready.json}"
MODEL_CACHE="${MODEL_CACHE:-/workspace/dpsae-runs/20260716/exp10_concept_discovery/shared-model-cache-v1}"
SAEBENCH_ROOT="${SAEBENCH_ROOT:-/workspace/SAEBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/dpsae-runs/20260716/exp13_concept_confirmation/fresh-pairs-v1}"
SESSION_PREFIX="${SESSION_PREFIX:-exp13}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-7.5}"
RUNNER="$ROOT/experiments/exp13_concept_confirmation.py"
THIS_SCRIPT="$ROOT/scripts/run_exp13_concept_confirmation_4xa40.sh"

if [[ "${EXP13_USER_APPROVED:-}" != "YES" ]]; then
  echo "Exp13 is paid confirmatory concept evaluation; set EXP13_USER_APPROVED=YES only after explicit approval." >&2
  exit 2
fi

if [[ "${EXP13_INSIDE_TMUX:-0}" != "1" ]]; then
  launcher="$SESSION_PREFIX-launch"
  for suffix in launch gpu0 gpu1 gpu2 gpu3 finalize; do
    session="$SESSION_PREFIX-$suffix"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "tmux session already exists: $session" >&2
      exit 2
    fi
  done
  if [[ -f "$OUTPUT_ROOT/abort_requested.json" ]]; then
    echo "Exp13 output root already contains an abort marker; use a fresh run ID." >&2
    exit 2
  fi
  if [[ -f "$OUTPUT_ROOT/freeze_failed.json" ]]; then
    echo "Exp13 output root already contains a freeze failure; use a fresh run ID." >&2
    exit 2
  fi
  if [[ -f "$OUTPUT_ROOT/artifact_audit_final.json" ]]; then
    echo "Exp13 output root already has a final audit; completed run IDs are immutable." >&2
    exit 2
  fi
  mkdir -p "$OUTPUT_ROOT/logs"
  launch=(
    env
    EXP13_INSIDE_TMUX=1
    EXP13_USER_APPROVED=YES
    ROOT="$ROOT"
    PYTHON="$PYTHON"
    CONFIG="$CONFIG"
    BASE_CONFIG="$BASE_CONFIG"
    EXP12_CONFIG="$EXP12_CONFIG"
    EXP12_ROOT="$EXP12_ROOT"
    PILOT_ROOT="$PILOT_ROOT"
    PILOT_AUDIT="$PILOT_AUDIT"
    SOURCE_CACHE_READY="$SOURCE_CACHE_READY"
    MODEL_CACHE="$MODEL_CACHE"
    SAEBENCH_ROOT="$SAEBENCH_ROOT"
    OUTPUT_ROOT="$OUTPUT_ROOT"
    SESSION_PREFIX="$SESSION_PREFIX"
    TIMEOUT_HOURS="$TIMEOUT_HOURS"
    bash "$THIS_SCRIPT"
  )
  printf -v launch_command '%q ' "${launch[@]}"
  tmux new-session -d -s "$launcher" "$launch_command"
  tmux set-window-option -t "$launcher":0 remain-on-exit on
  echo "Started tmux session $launcher; it will freeze the contract and launch the four workers plus finalizer."
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs" "$MODEL_CACHE"
exec > >(tee -a "$OUTPUT_ROOT/logs/launcher.log") 2>&1
cd "$ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "Pinned SAEBench Python is missing or not executable: $PYTHON" >&2
  exit 2
fi
if [[ "$(cd "$(dirname "$PYTHON")/.." && pwd)" != "$(cd "$SAEBENCH_ROOT/.venv" && pwd)" ]]; then
  echo "Exp13 requires the pinned SAEBench virtual environment." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Exp13 requires a clean, committed repository revision." >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for the Exp13 wall-clock backstop." >&2
  exit 2
fi

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#gpu_names[@]}" -ne 4 ]]; then
  echo "Exp13 requires exactly four visible GPUs; found ${#gpu_names[@]}." >&2
  exit 2
fi
for name in "${gpu_names[@]}"; do
  if [[ "$name" != *A40* ]]; then
    echo "Exp13 requires four A40 GPUs; found '$name'." >&2
    exit 2
  fi
done

timeout_seconds="$($PYTHON -c \
  'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and 0 < value <= 7.5; print(math.ceil(value * 3600))' \
  "$TIMEOUT_HOURS")"
finalizer_wait_seconds="$((timeout_seconds - 120))"
if [[ "$finalizer_wait_seconds" -le 0 ]]; then
  echo "Exp13 timeout must leave two minutes for retained failure finalization." >&2
  exit 2
fi
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=true
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cpu_budget_json="$($PYTHON -m dpsae.cpu_quota json --workers 4 --output "$OUTPUT_ROOT/cpu_budget.json")"
effective_cpus="$($PYTHON -c 'import json,sys; print(json.loads(sys.argv[1])["effective_cpu_count"])' "$cpu_budget_json")"
worker_threads="$($PYTHON -c 'import json,sys; print(json.loads(sys.argv[1])["threads_per_worker"])' "$cpu_budget_json")"
export LOKY_MAX_CPU_COUNT="$effective_cpus"
export OMP_NUM_THREADS="$worker_threads"
export MKL_NUM_THREADS="$worker_threads"
export OPENBLAS_NUM_THREADS="$worker_threads"
export NUMEXPR_NUM_THREADS="$worker_threads"

common=("$PYTHON" -u "$RUNNER" --config "$CONFIG" --output-root "$OUTPUT_ROOT")
"${common[@]}" freeze \
  --base-config "$BASE_CONFIG" \
  --exp12-config "$EXP12_CONFIG" \
  --exp12-root "$EXP12_ROOT" \
  --pilot-root "$PILOT_ROOT" \
  --pilot-audit "$PILOT_AUDIT" \
  --source-cache-ready "$SOURCE_CACHE_READY" \
  --model-cache "$MODEL_CACHE" \
  --saebench-root "$SAEBENCH_ROOT"

for suffix in gpu0 gpu1 gpu2 gpu3 finalize; do
  session="$SESSION_PREFIX-$suffix"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
  fi
done

base_env=(
  env
  PYTHONPATH="$PYTHONPATH"
  PYTHONDONTWRITEBYTECODE=1
  TOKENIZERS_PARALLELISM=true
  HF_HOME="$HF_HOME"
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  LOKY_MAX_CPU_COUNT="$LOKY_MAX_CPU_COUNT"
  OMP_NUM_THREADS="$OMP_NUM_THREADS"
  MKL_NUM_THREADS="$MKL_NUM_THREADS"
  OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS"
  NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS"
)

for index in 0 1 2 3; do
  session="$SESSION_PREFIX-gpu$index"
  command=(
    timeout --signal=TERM --kill-after=60 "${timeout_seconds}s"
    "${base_env[@]}"
    CUDA_VISIBLE_DEVICES="$index"
    "$PYTHON" -u "$RUNNER"
    --config "$CONFIG"
    --output-root "$OUTPUT_ROOT"
    run-worker --worker-index "$index" --device cuda
  )
  printf -v process_command '%q ' "${command[@]}"
  marker=(
    "${base_env[@]}"
    CUDA_VISIBLE_DEVICES=""
    "$PYTHON" -u "$RUNNER"
    --config "$CONFIG"
    --output-root "$OUTPUT_ROOT"
    retain-failure --stage worker --worker-index "$index"
  )
  printf -v marker_command '%q ' "${marker[@]}"
  printf -v root_q '%q' "$ROOT"
  printf -v log_q '%q' "$OUTPUT_ROOT/logs/gpu$index.log"
  # shellcheck disable=SC2016  # Expansions belong to the generated tmux shell.
  printf -v body \
    'cd %s || exit 2
set +e
%s > >(tee -a %s) 2>&1
code=$?
set -e
if [[ $code -ne 0 ]]; then
  %s --message "worker %s tmux command exited $code"
fi
exit "$code"
' \
    "$root_q" "$process_command" "$log_q" "$marker_command" "$index"
  printf -v body_q '%q' "$body"
  tmux new-session -d -s "$session" "bash -lc $body_q"
  tmux set-window-option -t "$session":0 remain-on-exit on
done

finalizer=(
  timeout --signal=TERM --kill-after=60 "${timeout_seconds}s"
  "${base_env[@]}"
  CUDA_VISIBLE_DEVICES=""
  "$PYTHON" -u "$RUNNER"
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  finalize --wait-seconds "$finalizer_wait_seconds"
)
printf -v finalizer_command '%q ' "${finalizer[@]}"
finalizer_marker=(
  "${base_env[@]}"
  CUDA_VISIBLE_DEVICES=""
  "$PYTHON" -u "$RUNNER"
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  retain-failure --stage finalizer
)
printf -v finalizer_marker_command '%q ' "${finalizer_marker[@]}"
printf -v root_q '%q' "$ROOT"
printf -v finalizer_log_q '%q' "$OUTPUT_ROOT/logs/finalize.log"
# shellcheck disable=SC2016  # Expansions belong to the generated tmux shell.
printf -v finalizer_body \
  'cd %s || exit 2
set +e
%s > >(tee -a %s) 2>&1
code=$?
set -e
if [[ $code -ne 0 ]]; then
  %s --message "finalizer tmux command exited $code"
fi
exit "$code"
' \
  "$root_q" "$finalizer_command" "$finalizer_log_q" "$finalizer_marker_command"
printf -v finalizer_body_q '%q' "$finalizer_body"
tmux new-session -d -s "$SESSION_PREFIX-finalize" "bash -lc $finalizer_body_q"
tmux set-window-option -t "$SESSION_PREFIX-finalize":0 remain-on-exit on

echo "Started $SESSION_PREFIX-gpu0 through $SESSION_PREFIX-gpu3 and $SESSION_PREFIX-finalize."
echo "All paid work is inside tmux; status is available through the Exp13 status subcommand."
