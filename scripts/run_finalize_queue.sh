#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/paper_closure/logs
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src

while tmux has-session -t dpsae-generality-queue 2>/dev/null; do
  sleep 60
done

while tmux has-session -t dpsae-jumprelu-queue 2>/dev/null; do
  sleep 60
done

.venv/bin/python -u scripts/finalize_paper_closure.py \
  --output artifacts/paper_closure/reproducibility_manifest.json \
  > artifacts/paper_closure/logs/finalize_manifest.log 2>&1
