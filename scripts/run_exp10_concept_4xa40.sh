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
LOG_ROOT="$OUTPUT_ROOT/logs"

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

REPOSITORY_REVISION="$(git rev-parse HEAD)"
REPOSITORY_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "$REPOSITORY_STATUS" ]]; then
  echo "exp10 requires a clean repository revision" >&2
  exit 2
fi
printf '%s\n' "$REPOSITORY_REVISION" > "$OUTPUT_ROOT/repository-revision.txt"
"$PYTHON" -m pip freeze > "$OUTPUT_ROOT/environment-pip-freeze.txt"
nvidia-smi -q > "$OUTPUT_ROOT/nvidia-smi.txt"
sha256sum \
  "$CONFIG" \
  "$ROOT/experiments/exp10_concept_discovery.py" \
  "$ROOT/src/dpsae/saebench_adapter.py" \
  "$ROOT/scripts/audit_exp10_artifacts.py" \
  "$ROOT/scripts/run_exp10_concept_4xa40.sh" \
  "$ROOT/scripts/run_exp10_timing_smoke_a40.sh" \
  > "$OUTPUT_ROOT/deployed-source-sha256.txt"

if [[ ! -x "$PYTHON" ]]; then
  echo "sealed exp10 Python is missing or not executable: $PYTHON" >&2
  exit 2
fi
if [[ "$(cd "$(dirname "$PYTHON")/.." && pwd)" != "$(cd "$SAEBENCH_ROOT/.venv" && pwd)" ]]; then
  echo "exp10 requires the pinned SAEBench environment at $SAEBENCH_ROOT/.venv" >&2
  exit 2
fi

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 4 ]]; then
  echo "exp10 requires exactly four visible GPUs; observed ${#GPU_NAMES[@]}" >&2
  exit 2
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "$name" != *A40* ]]; then
    echo "exp10 is frozen for 4xA40; observed $name" >&2
    exit 2
  fi
done

COMMON=(
  "$PYTHON" -u experiments/exp10_concept_discovery.py
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --saebench-root "$SAEBENCH_ROOT"
)

"${COMMON[@]}" freeze | tee "$LOG_ROOT/freeze.log"
"${COMMON[@]}" timing-gate | tee "$LOG_ROOT/timing-gate.log"

for session in exp10-gpu0 exp10-gpu1 exp10-gpu2 exp10-gpu3 exp10-finalize; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 2
  fi
done

BASE="cd '$ROOT'; export PYTHONPATH='$PYTHONPATH' PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=true HF_HOME='$HF_HOME' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS='$OMP_NUM_THREADS' MKL_NUM_THREADS='$MKL_NUM_THREADS' OPENBLAS_NUM_THREADS='$OPENBLAS_NUM_THREADS' NUMEXPR_NUM_THREADS='$NUMEXPR_NUM_THREADS';"
RUNNER="'$PYTHON' -u experiments/exp10_concept_discovery.py --config '$CONFIG' --output-root '$OUTPUT_ROOT' --checkpoint-dir '$CHECKPOINT_DIR' --saebench-root '$SAEBENCH_ROOT'"
AUDITOR="'$PYTHON' -u scripts/audit_exp10_artifacts.py --config '$CONFIG' --output-root '$OUTPUT_ROOT'"

tmux new-session -d -s exp10-gpu0 "bash -lc \"$BASE export CUDA_VISIBLE_DEVICES=0; exec > >(tee -a '$LOG_ROOT/gpu0.log') 2>&1; $RUNNER run-worker --model-cache '$MODEL_CACHE' --worker-index 0 --cache-role wait --method mse --probe-seeds 2027071701 2027071702 2027071703 2027071704 2027071705 --companion-seeds 2027071701 2027071702 2027071703 --device cuda:0\""

tmux new-session -d -s exp10-gpu1 "bash -lc \"$BASE export CUDA_VISIBLE_DEVICES=1; exec > >(tee -a '$LOG_ROOT/gpu1.log') 2>&1; $RUNNER run-worker --model-cache '$MODEL_CACHE' --worker-index 1 --cache-role wait --method mse --probe-seeds 2027071706 2027071707 2027071708 2027071709 2027071710 --companion-seeds 2027071704 2027071705 2027071706 --device cuda:0\""

tmux new-session -d -s exp10-gpu2 "bash -lc \"$BASE export CUDA_VISIBLE_DEVICES=2; exec > >(tee -a '$LOG_ROOT/gpu2.log') 2>&1; $RUNNER run-worker --model-cache '$MODEL_CACHE' --worker-index 2 --cache-role wait --method dpsae --probe-seeds 2027071701 2027071702 2027071703 2027071704 2027071705 --companion-seeds 2027071707 2027071708 --device cuda:0\""

tmux new-session -d -s exp10-gpu3 "bash -lc \"$BASE export CUDA_VISIBLE_DEVICES=3; exec > >(tee -a '$LOG_ROOT/gpu3.log') 2>&1; $RUNNER run-worker --model-cache '$MODEL_CACHE' --worker-index 3 --cache-role wait --method dpsae --probe-seeds 2027071706 2027071707 2027071708 2027071709 2027071710 --companion-seeds 2027071709 2027071710 --device cuda:0\""

tmux new-session -d -s exp10-finalize "bash -lc \"$BASE exec > >(tee -a '$LOG_ROOT/finalize.log') 2>&1; $AUDITOR --phase pre-aggregate --wait-seconds 172800 && $RUNNER aggregate && $AUDITOR --phase final\""

tmux list-sessions -F '#{session_name}' | grep '^exp10-'
