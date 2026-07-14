#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME=/workspace/huggingface
export TOKENIZERS_PARALLELISM=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
PYTHON=/workspace/decoder-preserving-sae/.venv/bin/python
OUTPUT=artifacts/joint_audit_20260714
LOGS="$OUTPUT/logs"
mkdir -p "$LOGS"

while tmux has-session -t dpsae-jumprelu-matched 2>/dev/null; do
  sleep 60
done

"$PYTHON" -u experiments/exp07_advantage_spectrum.py all \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --minimum-free-gib 20 \
  > "$LOGS/advantage_spectrum.log" 2>&1

MODELS=(
  mse_s0 dpsae_w0.03125_s0
  mse_s1 dpsae_w0.03125_s1
  mse_s2 dpsae_w0.03125_s2
)
for MODEL in "${MODELS[@]}"; do
  RESULT="$OUTPUT/gradient_fidelity_${MODEL}.json"
  if [[ -f "$RESULT" ]] && "$PYTHON" -c \
    'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["complete"])' \
    "$RESULT"; then
    continue
  fi
  "$PYTHON" -u experiments/exp07_gradient_fidelity.py run \
    --model-name "$MODEL" \
    --device cuda:0 \
    --gpu-memory-fraction 0.25 \
    --minimum-free-gib 20 \
    > "$LOGS/gradient_fidelity_${MODEL}.log" 2>&1
done

"$PYTHON" -u experiments/exp07_gradient_fidelity.py summarize \
  > "$LOGS/gradient_fidelity_summary.log" 2>&1
