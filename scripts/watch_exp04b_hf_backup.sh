#!/usr/bin/env bash
set -uo pipefail

MODEL_REPO="${1:?usage: watch_exp04b_hf_backup.sh MODEL_REPO DATASET_REPO [INTERVAL_SECONDS]}"
DATASET_REPO="${2:?usage: watch_exp04b_hf_backup.sh MODEL_REPO DATASET_REPO [INTERVAL_SECONDS]}"
INTERVAL_SECONDS="${3:-900}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
while tmux has-session -t dpsae-exp04b 2>/dev/null; do
  "$ROOT/scripts/backup_exp04b_hf.sh" "$MODEL_REPO" "$DATASET_REPO" || true
  sleep "$INTERVAL_SECONDS"
done
"$ROOT/scripts/backup_exp04b_hf.sh" "$MODEL_REPO" "$DATASET_REPO"
