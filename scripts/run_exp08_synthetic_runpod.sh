#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "exp08 synthetic sweep requires a clean worktree" >&2
  exit 1
fi

SOURCE_ROOT="$(realpath "${DPSAE_SOURCE_ROOT:-/workspace/decoder-preserving-sae}")"
PYTHON="${DPSAE_PYTHON:-$SOURCE_ROOT/.venv/bin/python}"
OUTPUT="artifacts/exp08_experiment_figure/synthetic_prior_sweep"
LOGS="artifacts/exp08_experiment_figure/logs"
SEED_OUTPUT="$OUTPUT/seeds"
PARALLEL_SEEDS="${DPSAE_SYNTHETIC_PARALLEL_SEEDS:-4}"
mkdir -p "$OUTPUT" "$SEED_OUTPUT" "$LOGS"
if ! [[ "$PARALLEL_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DPSAE_SYNTHETIC_PARALLEL_SEEDS must be a positive integer" >&2
  exit 1
fi
"$PYTHON" -u scripts/prepare_exp08_run.py \
  --source-root "$SOURCE_ROOT" \
  --output artifacts/exp08_experiment_figure/run_manifest.json \
  > "$LOGS/manifest_validation_synthetic.log" 2>&1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dpsae-mpl}"

cleanup_children() {
  status=$?
  if (( status != 0 )); then
    while read -r pid; do
      kill -TERM "$pid" 2>/dev/null || true
    done < <(jobs -pr)
    wait || true
  fi
  exit "$status"
}
trap cleanup_children EXIT
trap 'exit 130' INT TERM

running=0
for seed in 0 1 2 3 4 5 6 7 8 9; do
  "$PYTHON" -u experiments/exp02_prior_weight_sweep.py \
    --config configs/exp02_structured_prior.json \
    --output-dir "$SEED_OUTPUT/seed${seed}" \
    --seed "$seed" \
    > "$LOGS/synthetic_prior_sweep_seed${seed}.log" 2>&1 &
  running=$((running + 1))
  if (( running >= PARALLEL_SEEDS )); then
    wait -n
    running=$((running - 1))
  fi
done
wait

"$PYTHON" -u scripts/merge_exp02_prior_sweep.py \
  --input-dir "$SEED_OUTPUT" \
  --output-dir "$OUTPUT" \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  > "$LOGS/synthetic_prior_sweep_merge.log" 2>&1

test -s "$OUTPUT/paired_metrics.csv"
test -s "$OUTPUT/calibration.csv"
test -s "$OUTPUT/metadata.json"
