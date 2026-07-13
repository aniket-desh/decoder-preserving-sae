#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${1:?usage: backup_exp04b_hf.sh MODEL_REPO DATASET_REPO [ARTIFACT_DIR]}"
DATASET_REPO="${2:?usage: backup_exp04b_hf.sh MODEL_REPO DATASET_REPO [ARTIFACT_DIR]}"
ARTIFACT_DIR="${3:-artifacts/exp04b_confirmatory}"
MARKERS="$ARTIFACT_DIR/.hf_backup"

mkdir -p "$MARKERS"
hf auth whoami >/dev/null
hf repos create "$MODEL_REPO" --private --exist-ok >/dev/null
hf repos create "$DATASET_REPO" --type dataset --private --exist-ok >/dev/null

upload_file() {
  local repo="$1" repo_type="$2" source="$3" target="$4" message="$5"
  [[ -f "$source" ]] || return 0
  hf upload "$repo" "$source" "$target" --type "$repo_type" --private \
    --commit-message "$message"
}

upload_once() {
  local marker="$1" repo="$2" repo_type="$3" source="$4" target="$5" message="$6"
  [[ -f "$source" && ! -f "$MARKERS/$marker" ]] || return 0
  upload_file "$repo" "$repo_type" "$source" "$target" "$message"
  touch "$MARKERS/$marker"
}

upload_changed() {
  local marker="$1" repo="$2" repo_type="$3" source="$4" target="$5" message="$6"
  [[ -f "$source" ]] || return 0
  local fingerprint
  fingerprint="$(stat -c '%Y:%s' "$source")"
  [[ ! -f "$MARKERS/$marker" || "$(<"$MARKERS/$marker")" != "$fingerprint" ]] || return 0
  upload_file "$repo" "$repo_type" "$source" "$target" "$message"
  printf '%s\n' "$fingerprint" > "$MARKERS/$marker"
}

for name in fineweb_gpt2_tail_tokens.bin fineweb_gpt2_tail_tokens.bin.json \
  natural_selection.pt natural_test.pt resolved_config.json; do
  upload_once "dataset-$name" "$DATASET_REPO" dataset "$ARTIFACT_DIR/$name" "$name" \
    "Back up Experiment 4b immutable data: $name"
done

for name in static_calibration.pt baseline_selection.json ioi_confirmatory_cache.pt \
  ioi_selection_models.json ioi_feature_count_selection.json ioi_test_models.json \
  ioi_confirmatory.json natural_evaluation_source.json natural_exact_audit_source.json \
  natural_evaluation_baseline.json natural_exact_audit_baseline.json resolved_config.json; do
  upload_once "model-$name" "$MODEL_REPO" model "$ARTIFACT_DIR/$name" "$name" \
    "Back up Experiment 4b artifact: $name"
done

for stage in baseline_screen baseline_confirm; do
  checkpoint="$ARTIFACT_DIR/$stage/checkpoint.pt"
  upload_changed "$stage-checkpoint" "$MODEL_REPO" model "$checkpoint" \
    "$stage/checkpoint.pt" "Update resumable Experiment 4b $stage checkpoint"
  if [[ -f "$ARTIFACT_DIR/$stage/done.json" && ! -f "$MARKERS/$stage-complete" ]]; then
    hf upload "$MODEL_REPO" "$ARTIFACT_DIR/$stage" "$stage" --type model --private \
      --exclude checkpoint.pt --commit-message "Back up completed Experiment 4b $stage"
    touch "$MARKERS/$stage-complete"
  fi
done
