#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-robustness16}"
TARGET_SESSION="${2:-dpsae-exp04-full}"
MODEL_REPO="${3:-}"
ARTIFACT_DIR="${4:-artifacts/exp04_ioi_mechanism}"
CHECKPOINT="$ARTIFACT_DIR/$STAGE/checkpoint.pt"

echo "waiting for atomic checkpoint: $CHECKPOINT"
until [[ -f "$CHECKPOINT" ]]; do
  if ! tmux has-session -t "$TARGET_SESSION" 2>/dev/null; then
    echo "training session ended before checkpoint appeared" >&2
    exit 1
  fi
  sleep 10
done

echo "checkpoint detected; stopping $TARGET_SESSION"
tmux send-keys -t "$TARGET_SESSION" C-c

for _ in $(seq 1 60); do
  tmux has-session -t "$TARGET_SESSION" 2>/dev/null || break
  sleep 1
done
if tmux has-session -t "$TARGET_SESSION" 2>/dev/null; then
  echo "training session did not stop within 60 seconds" >&2
  exit 1
fi

if [[ -n "$MODEL_REPO" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI is not available; checkpoint is safe locally but was not uploaded" >&2
    exit 1
  fi
  hf upload "$MODEL_REPO" "$CHECKPOINT" "paused/$STAGE/checkpoint.pt" \
    --type model \
    --private \
    --commit-message "Back up paused $STAGE checkpoint"
  hf upload "$MODEL_REPO" "$ARTIFACT_DIR/$STAGE/training.jsonl" \
    "paused/$STAGE/training.jsonl" \
    --type model \
    --private \
    --commit-message "Record paused $STAGE training progress"
fi

if tmux has-session -t dpsae-hf-backup 2>/dev/null; then
  tmux kill-session -t dpsae-hf-backup
fi
echo "pause complete"
