#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$SCRIPT_ROOT}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/exp12_pythia_fresh_confirmation.json}"
RUNNER="$ROOT/experiments/exp12_pythia_fresh_confirmation.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/exp12_pythia_fresh_confirmation}"
TOKEN_CACHE="${TOKEN_CACHE:-$OUTPUT_ROOT/shared/fineweb_pythia_tokens.bin}"
PILOT_REPORT="${PILOT_REPORT:-$ROOT/artifacts/exp10_concept_discovery/advancement_report.json}"
SESSION_PREFIX="${SESSION_PREFIX:-exp12-pythia}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-8}"
PAIR_GPUS=(0 1 2)
COORDINATOR_GPU=3

cd "$ROOT"
if [[ "${EXP12_USER_APPROVED:-}" != "YES" ]]; then
  echo "Exp12 is behind the post-pilot user approval boundary; set EXP12_USER_APPROVED=YES only after explicit approval." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Exp12 requires a clean, committed repository revision." >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found at $PYTHON" >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1 || ! command -v bash >/dev/null 2>&1; then
  echo "GNU timeout and bash are required for the Exp12 wall-clock backstop." >&2
  exit 1
fi
TIMEOUT_SECONDS="$("$PYTHON" -c \
  'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and value > 0; print(math.ceil(value * 3600))' \
  "$TIMEOUT_HOURS")"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 4 ]]; then
  echo "Exp12 requires exactly four visible GPUs; found ${#GPU_NAMES[@]}." >&2
  exit 1
fi
for name in "${GPU_NAMES[@]}"; do
  if [[ "$name" != *A40* ]]; then
    echo "Exp12 requires four A40 GPUs; found '$name'." >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/nonreport_timing_smoke"
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=true
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON=(
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --token-cache "$TOKEN_CACHE"
  --pilot-report "$PILOT_REPORT"
  --timeout-hours "$TIMEOUT_HOURS"
)
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  COMMON+=(--local-files-only)
fi

# This exits with status 2 before opening the pilot report while any protocol
# choice remains null, and it refuses a dirty revision once the config is frozen.
"$PYTHON" "$RUNNER" preflight "${COMMON[@]}"

mapfile -t PAIR_SEEDS < <(
  "$PYTHON" -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["confirmation"]["pair_seeds"], sep="\n")' \
    "$CONFIG"
)
if [[ "${#PAIR_SEEDS[@]}" -ne 3 ]]; then
  echo "Frozen Exp12 config must contain exactly three pair seeds." >&2
  exit 1
fi

SESSIONS=("$SESSION_PREFIX-coordinator")
for seed in "${PAIR_SEEDS[@]}"; do
  SESSIONS+=("$SESSION_PREFIX-pair-$seed")
done
for session in "${SESSIONS[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

LOCAL_FLAG=""
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FLAG="--local-files-only"
fi
tmux new-session -d -s "$SESSION_PREFIX-coordinator" \
  "cd '$ROOT' && timeout --signal=TERM --kill-after=60 '${TIMEOUT_SECONDS}s' bash -lc \"set -o pipefail; CUDA_VISIBLE_DEVICES='$COORDINATOR_GPU' PYTHONPATH='$ROOT/src:$ROOT' '$PYTHON' -u '$RUNNER' timing-smoke --config '$CONFIG' --output-root '$OUTPUT_ROOT' --token-cache '$TOKEN_CACHE' --pilot-report '$PILOT_REPORT' --timeout-hours '$TIMEOUT_HOURS' $LOCAL_FLAG 2>&1 | tee '$OUTPUT_ROOT/nonreport_timing_smoke/timing-smoke.log' && CUDA_VISIBLE_DEVICES='$COORDINATOR_GPU' PYTHONPATH='$ROOT/src:$ROOT' '$PYTHON' -u '$RUNNER' coordinator --config '$CONFIG' --output-root '$OUTPUT_ROOT' --token-cache '$TOKEN_CACHE' --pilot-report '$PILOT_REPORT' --timeout-hours '$TIMEOUT_HOURS' $LOCAL_FLAG 2>&1 | tee '$OUTPUT_ROOT/logs/coordinator.log'\""

for index in "${!PAIR_SEEDS[@]}"; do
  seed="${PAIR_SEEDS[$index]}"
  gpu="${PAIR_GPUS[$index]}"
  session="$SESSION_PREFIX-pair-$seed"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && set -o pipefail && timeout --signal=TERM --kill-after=60 '${TIMEOUT_SECONDS}s' bash -lc \"CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/src:$ROOT' '$PYTHON' -u '$RUNNER' wait-shared --pair-seed '$seed' --config '$CONFIG' --output-root '$OUTPUT_ROOT' --token-cache '$TOKEN_CACHE' --pilot-report '$PILOT_REPORT' --timeout-hours '$TIMEOUT_HOURS' $LOCAL_FLAG && CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/src:$ROOT' '$PYTHON' -u '$RUNNER' train-pair --pair-seed '$seed' --config '$CONFIG' --output-root '$OUTPUT_ROOT' --token-cache '$TOKEN_CACHE' --pilot-report '$PILOT_REPORT' --timeout-hours '$TIMEOUT_HOURS' $LOCAL_FLAG\" 2>&1 | tee '$OUTPUT_ROOT/logs/pair-$seed.log'"
done

echo "Started ${SESSIONS[*]}"
echo "GPU $COORDINATOR_GPU first runs a nonreport 2M-token timing/memory smoke."
echo "The full cache and three report pairs start only if its blind projection gate passes."
echo "The coordinator stops after the maturity decision if no common checkpoint passes."
echo "A passing run ends at a concept-authorization artifact; it does not run concept evaluation."
