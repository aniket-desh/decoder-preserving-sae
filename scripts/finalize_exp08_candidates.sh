#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "exp08 candidate finalization requires a clean worktree" >&2
  exit 1
fi

SOURCE_ROOT="$(realpath "${DPSAE_SOURCE_ROOT:-/workspace/decoder-preserving-sae}")"
PYTHON="${DPSAE_PYTHON:-$SOURCE_ROOT/.venv/bin/python}"
OUTPUT="artifacts/exp08_experiment_figure"
STATUS="$OUTPUT/status"
MANIFEST="$OUTPUT/run_manifest.json"
CANDIDATES="$OUTPUT/candidate_figures"
POLL_SECONDS="${DPSAE_FINALIZER_POLL_SECONDS:-15}"
TIMEOUT_SECONDS="${DPSAE_FINALIZER_TIMEOUT_SECONDS:-21600}"

if [[ "$SOURCE_ROOT" == "$ROOT" ]]; then
  echo "DPSAE_SOURCE_ROOT must differ from the clean worktree" >&2
  exit 1
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DPSAE_FINALIZER_POLL_SECONDS must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DPSAE_FINALIZER_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi
test -x "$PYTHON"
test -s "$MANIFEST"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dpsae-mpl}"

"$PYTHON" -u scripts/prepare_exp08_run.py \
  --source-root "$SOURCE_ROOT" \
  --output "$MANIFEST"

echo "waiting for the GPU and synthetic Exp08 workers"
wait_started="$(date +%s)"
while true; do
  pending=0
  for worker in gpu synthetic; do
    status_file="$STATUS/$worker.status"
    if [[ -s "$status_file" ]]; then
      worker_status="$(sed -n '1p' "$status_file")"
    else
      worker_status=""
    fi
    case "$worker_status" in
      succeeded)
        ;;
      "" | running)
        pending=1
        ;;
      failed:*)
        upstream_code="${worker_status#failed:}"
        if ! [[ "$upstream_code" =~ ^[1-9][0-9]*$ ]] \
          || (( upstream_code > 255 )); then
          echo "$worker worker has invalid failure status: $worker_status" >&2
          exit 1
        fi
        echo "$worker worker ended with $worker_status" >&2
        exit "$upstream_code"
        ;;
      *)
        echo "$worker worker has invalid status: $worker_status" >&2
        exit 1
        ;;
    esac
  done
  if (( pending == 0 )); then
    break
  fi
  elapsed="$(( $(date +%s) - wait_started ))"
  if (( elapsed >= TIMEOUT_SECONDS )); then
    echo "workers did not finish within ${TIMEOUT_SECONDS}s" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "workers succeeded; rendering review-only candidate figures"
"$PYTHON" -u scripts/prepare_exp08_run.py \
  --source-root "$SOURCE_ROOT" \
  --output "$MANIFEST"
"$PYTHON" -u scripts/plot_exp08_candidates.py \
  --experiment-root "$OUTPUT" \
  --structured-baseline-dir \
    "$SOURCE_ROOT/experiments/outputs/exp02_structured_prior" \
  --static-baseline \
    "$SOURCE_ROOT/artifacts/exp04b_confirmatory/natural_evaluation_baseline.json" \
  --output-dir "$CANDIDATES"

for stem in \
  task_prior_candidates \
  language_model_candidates \
  frozen_fidelity_review \
  robustness_appendix; do
  test -s "$CANDIDATES/$stem.pdf"
  test -s "$CANDIDATES/$stem.png"
done
test -s "$CANDIDATES/candidate_manifest.json"

echo "candidate figures: $CANDIDATES"
