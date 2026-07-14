#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPO_ID="${1:-aniketdesh/decoder-preserving-sae-paper-closure-20260714}"
mkdir -p artifacts/paper_closure/logs

while tmux has-session -t dpsae-finalize-queue 2>/dev/null; do
  sleep 60
done

test -f artifacts/paper_closure/reproducibility_manifest.json
.venv/bin/hf auth whoami --format json
.venv/bin/hf repos create "$REPO_ID" \
  --type dataset \
  --private \
  --exist-ok
.venv/bin/hf upload-large-folder "$REPO_ID" artifacts \
  --type dataset \
  --private \
  --include "paper_closure/**" \
  --include "exp04b_mechanism_attribution/**" \
  --include "exp05_decoder_advantage_discovery/**" \
  --include "exp06_generality/**" \
  --exclude "**/checkpoint.pt" \
  --exclude "**/.hf_backup/**" \
  --num-workers 4 \
  --no-bars
.venv/bin/hf datasets info "$REPO_ID" --format json \
  > artifacts/paper_closure/backup_repository.json
.venv/bin/hf upload "$REPO_ID" \
  artifacts/paper_closure/backup_repository.json \
  paper_closure/backup_repository.json \
  --type dataset \
  --private \
  --commit-message "Record the paper-closure backup repository revision"
