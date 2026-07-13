#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${1:?usage: backup_exp04_hf.sh MODEL_REPO DATASET_REPO [ARTIFACT_DIR]}"
DATASET_REPO="${2:?usage: backup_exp04_hf.sh MODEL_REPO DATASET_REPO [ARTIFACT_DIR]}"
ARTIFACT_DIR="${3:-artifacts/exp04_ioi_mechanism}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is not available; activate the experiment environment first" >&2
  exit 1
fi

hf auth whoami >/dev/null
hf repos create "$MODEL_REPO" --private --exist-ok >/dev/null
hf repos create "$DATASET_REPO" --type dataset --private --exist-ok >/dev/null

upload_file() {
  local repo="$1"
  local repo_type="$2"
  local local_path="$3"
  local path_in_repo="$4"
  local message="$5"
  [[ -f "$local_path" ]] || return 0
  hf upload "$repo" "$local_path" "$path_in_repo" \
    --type "$repo_type" \
    --private \
    --commit-message "$message"
}

if [[ ! -f "$ARTIFACT_DIR/.hf_backup_dataset" ]]; then
  upload_file "$DATASET_REPO" dataset \
    "$ARTIFACT_DIR/fineweb_gpt2_tokens.bin" fineweb_gpt2_tokens.bin \
    "Back up the frozen FineWeb GPT-2 token cache"
  upload_file "$DATASET_REPO" dataset \
    "$ARTIFACT_DIR/resolved_config.json" resolved_config.json \
    "Record the exact experiment and corpus configuration"
  touch "$ARTIFACT_DIR/.hf_backup_dataset"
fi

if [[ ! -f "$ARTIFACT_DIR/.hf_backup_shared" ]]; then
  upload_file "$MODEL_REPO" model \
    "$ARTIFACT_DIR/calibration.pt" calibration.pt \
    "Back up activation statistics and ridge calibration"
  upload_file "$MODEL_REPO" model \
    "$ARTIFACT_DIR/screening_selection.json" screening_selection.json \
    "Back up the frozen decoder-weight selection"
  upload_file "$MODEL_REPO" model \
    "$ARTIFACT_DIR/resolved_config.json" resolved_config.json \
    "Record the exact experiment configuration"
  touch "$ARTIFACT_DIR/.hf_backup_shared"
fi

for stage in screen confirmation robustness16 robustness64; do
  marker="$ARTIFACT_DIR/.hf_backup_${stage}"
  if [[ -f "$ARTIFACT_DIR/$stage/done.json" && ! -f "$marker" ]]; then
    hf upload "$MODEL_REPO" "$ARTIFACT_DIR/$stage" "$stage" \
      --type model \
      --private \
      --exclude checkpoint.pt \
      --commit-message "Back up completed ${stage} stage"
    touch "$marker"
  fi
done

if [[ -f "$ARTIFACT_DIR/analysis.json" && ! -f "$ARTIFACT_DIR/.hf_backup_analysis" ]]; then
  for path in \
    analysis.json \
    analysis_confirmation.json \
    analysis_robustness16.json \
    analysis_robustness64.json \
    ioi_state_activations.pt; do
    upload_file "$MODEL_REPO" model \
      "$ARTIFACT_DIR/$path" "$path" \
      "Back up held-out IOI analysis artifacts"
  done
  touch "$ARTIFACT_DIR/.hf_backup_analysis"
fi

if [[ -f "$ARTIFACT_DIR/figures/exp04_headline.png" && ! -f "$ARTIFACT_DIR/.hf_backup_figures" ]]; then
  hf upload "$MODEL_REPO" "$ARTIFACT_DIR/figures" figures \
    --type model \
    --private \
    --commit-message "Back up final experiment figures"
  touch "$ARTIFACT_DIR/.hf_backup_figures"
fi
