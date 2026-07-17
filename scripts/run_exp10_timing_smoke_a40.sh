#!/usr/bin/env bash
set -Eeuo pipefail

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
TIMING_CHILD_INDEX="${2:-}"

if [[ -n "$MODE" && "$MODE" != "--coordinator" && "$MODE" != "--timing-child" ]]; then
  echo "usage: $0 [--coordinator | --timing-child INDEX]" >&2
  exit 2
fi
if [[ "$MODE" == "--timing-child" && ! "$TIMING_CHILD_INDEX" =~ ^[0-3]$ ]]; then
  echo "timing child index must be one of 0, 1, 2, or 3" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/src:$ROOT:$SAEBENCH_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=true
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$MODE" == "--timing-child" ]]; then
  worker_index="$TIMING_CHILD_INDEX"
  worker_path="$OUTPUT_ROOT/timing_workers/worker_${worker_index}.json"
  exit_path="$OUTPUT_ROOT/timing_workers/exit_${worker_index}.json"
  temporary="$exit_path.tmp"
  child_command=(
    "$PYTHON" -u experiments/exp10_concept_discovery.py
    --config "$CONFIG"
    --output-root "$OUTPUT_ROOT"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --saebench-root "$SAEBENCH_ROOT"
    timing-worker
    --model-cache "$MODEL_CACHE"
  )
  if [[ -f "$COLD_CACHE_PROVENANCE" ]]; then
    child_command+=(--cold-cache-provenance "$COLD_CACHE_PROVENANCE")
  fi
  child_command+=(--worker-index "$worker_index" --device cuda:0)
  set +e
  "${child_command[@]}"
  code=$?
  set -e
  worker_hash=null
  if [[ -f "$worker_path" ]]; then
    worker_hash="$(sha256sum "$worker_path" | awk '{print $1}')"
  fi
  jq -n \
    --argjson worker_index "$worker_index" \
    --argjson exit_code "$code" \
    --arg worker_report_sha256 "$worker_hash" \
    '{schema_version: 1, complete: true, worker_index: $worker_index,
      exit_code: $exit_code,
      worker_report_sha256: (if $worker_report_sha256 == "null" then null else $worker_report_sha256 end)}' \
    > "$temporary"
  mv "$temporary" "$exit_path"
  exit "$code"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "sealed exp10 Python is missing or not executable: $PYTHON" >&2
  exit 2
fi
if [[ "$(cd "$(dirname "$PYTHON")/.." && pwd)" != "$(cd "$SAEBENCH_ROOT/.venv" && pwd)" ]]; then
  echo "exp10 requires the pinned SAEBench environment at $SAEBENCH_ROOT/.venv" >&2
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

mkdir -p "$LOG_ROOT" "$MODEL_CACHE"
CPU_BUDGET_JSON="$("$PYTHON" -m dpsae.cpu_quota json --workers 4 --output "$OUTPUT_ROOT/cpu_budget.json")"
EFFECTIVE_CPUS="$("$PYTHON" -c 'import json, sys; print(json.loads(sys.argv[1])["effective_cpu_count"])' "$CPU_BUDGET_JSON")"
WORKER_THREADS="$("$PYTHON" -c 'import json, sys; print(json.loads(sys.argv[1])["threads_per_worker"])' "$CPU_BUDGET_JSON")"
export LOKY_MAX_CPU_COUNT="$EFFECTIVE_CPUS"
export OMP_NUM_THREADS="$WORKER_THREADS"
export MKL_NUM_THREADS="$WORKER_THREADS"
export OPENBLAS_NUM_THREADS="$WORKER_THREADS"
export NUMEXPR_NUM_THREADS="$WORKER_THREADS"

COMMON=(
  "$PYTHON" -u experiments/exp10_concept_discovery.py
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --saebench-root "$SAEBENCH_ROOT"
)
CACHE_TIMING_ARGS=()
if [[ -f "$COLD_CACHE_PROVENANCE" ]]; then
  CACHE_TIMING_ARGS=(--cold-cache-provenance "$COLD_CACHE_PROVENANCE")
fi

if [[ "$MODE" != "--coordinator" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 2
  fi
  tmux new-session -d -s "$SESSION" \
    "env PYTHON='$PYTHON' SAEBENCH_ROOT='$SAEBENCH_ROOT' CHECKPOINT_DIR='$CHECKPOINT_DIR' CONFIG='$CONFIG' OUTPUT_ROOT='$OUTPUT_ROOT' MODEL_CACHE='$MODEL_CACHE' COLD_CACHE_PROVENANCE='$COLD_CACHE_PROVENANCE' HF_HOME='$HF_HOME' bash '$ROOT/scripts/run_exp10_timing_smoke_a40.sh' --coordinator"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "started four-worker exp10 timing coordinator in tmux session $SESSION"
  echo "log: $LOG_ROOT/timing-coordinator.log"
  exit 0
fi

exec > >(tee -a "$LOG_ROOT/timing-coordinator.log") 2>&1
for path in "$OUTPUT_ROOT/timing_smoke.json" "$OUTPUT_ROOT/timing_workers" "$OUTPUT_ROOT/timing_barrier"; do
  if [[ -e "$path" ]]; then
    echo "stale timing artifact exists; use a fresh timing output root: $path" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_ROOT/timing_workers" "$OUTPUT_ROOT/timing_barrier"
if ! command -v setsid >/dev/null; then
  echo "four-worker timing requires setsid for recursive process-group cleanup" >&2
  exit 2
fi

"${COMMON[@]}" freeze
CUDA_VISIBLE_DEVICES=0 "${COMMON[@]}" prepare-cache \
  --model-cache "$MODEL_CACHE" \
  "${CACHE_TIMING_ARGS[@]}" \
  --device cuda:0

worker_pids=()
cleanup_children() {
  local code="${1:-$?}"
  trap - EXIT INT TERM HUP
  if [[ "$code" -ne 0 ]]; then
    for pid in "${worker_pids[@]:-}"; do
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${worker_pids[@]:-}"; do
      kill -KILL -- "-$pid" 2>/dev/null || true
    done
    for pid in "${worker_pids[@]:-}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi
  exit "$code"
}
trap 'cleanup_children $?' EXIT
trap 'cleanup_children 130' INT
trap 'cleanup_children 143' TERM
trap 'cleanup_children 129' HUP

for worker_index in 0 1 2 3; do
  log_path="$LOG_ROOT/timing-worker-${worker_index}.log"
  setsid env \
    CUDA_VISIBLE_DEVICES="$worker_index" \
    PYTHON="$PYTHON" \
    SAEBENCH_ROOT="$SAEBENCH_ROOT" \
    CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    CONFIG="$CONFIG" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODEL_CACHE="$MODEL_CACHE" \
    COLD_CACHE_PROVENANCE="$COLD_CACHE_PROVENANCE" \
    HF_HOME="$HF_HOME" \
    SESSION="$SESSION" \
    LOKY_MAX_CPU_COUNT="$LOKY_MAX_CPU_COUNT" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" \
    bash "$ROOT/scripts/run_exp10_timing_smoke_a40.sh" --timing-child "$worker_index" \
    >> "$log_path" 2>&1 &
  worker_pids+=("$!")
done

"${COMMON[@]}" timing-start-barrier \
  --timeout-seconds "$(jq -r '.runtime.timing_smoke.barrier_timeout_seconds' "$CONFIG")"

worker_failure=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    worker_failure=1
  fi
done
if [[ "$worker_failure" -ne 0 ]]; then
  echo "one or more timing workers failed; inspect timing-worker logs and exit sentinels" >&2
  exit 1
fi

"${COMMON[@]}" timing-finalize
"${COMMON[@]}" timing-gate
echo "four-worker timing smoke complete: $OUTPUT_ROOT/timing_smoke.json"
