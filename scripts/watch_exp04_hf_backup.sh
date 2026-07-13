#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${1:?usage: watch_exp04_hf_backup.sh MODEL_REPO DATASET_REPO [ARTIFACT_DIR] [INTERVAL_SECONDS]}"
DATASET_REPO="${2:?usage: watch_exp04_hf_backup.sh MODEL_REPO DATASET_REPO [ARTIFACT_DIR] [INTERVAL_SECONDS]}"
ARTIFACT_DIR="${3:-artifacts/exp04_ioi_mechanism}"
INTERVAL_SECONDS="${4:-300}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while true; do
  "$SCRIPT_DIR/backup_exp04_hf.sh" "$MODEL_REPO" "$DATASET_REPO" "$ARTIFACT_DIR"
  if [[ -f "$ARTIFACT_DIR/analysis.json" && -f "$ARTIFACT_DIR/figures/exp04_headline.png" ]]; then
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
