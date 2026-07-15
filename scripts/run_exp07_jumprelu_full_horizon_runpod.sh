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

for WEIGHT in 2 4 8 16; do
  RUN="$OUTPUT/jumprelu_full_weight${WEIGHT}"
  if [[ -f "$RUN/done.json" ]] && "$PYTHON" -c \
    'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["complete"])' \
    "$RUN/done.json"; then
    continue
  fi
  "$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
    --config configs/exp04b_confirmatory.json \
    --sparsity-mode jump_relu \
    --jump-relu-threshold-lr-multiplier 16 \
    --jump-relu-sparsity-weight "$WEIGHT" \
    --decoder-weights 0.03125 \
    --seeds 0 \
    --token-budget 25000000 \
    --source-range-name confirmation \
    --data-seed 31415926 \
    --probe-seed-base 27182818 \
    --new-screen "$RUN" \
    --device cuda:0 \
    --gpu-memory-fraction 0.20 \
    --maximum-peak-gpu-gib 18 \
    --minimum-free-gib 20 \
    > "$LOGS/jumprelu_full_weight${WEIGHT}_train.log" 2>&1
done

"$PYTHON" -u experiments/exp07_jumprelu_calibration.py weight-grid \
  --device cuda:0 \
  --gpu-memory-fraction 0.20 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_full_weight_grid.log" 2>&1

GRID="$OUTPUT/jump_relu_full_horizon_weight_grid.json"
SELECTED_WEIGHT="$($PYTHON -c '
import json, sys
selection = json.load(open(sys.argv[1]))["selection"]
if selection is None:
    raise SystemExit("no full-horizon JumpReLU weight passed the decoder-blind gate")
print(selection["weight"])
' "$GRID")"

SCREEN="$OUTPUT/jumprelu_full_horizon_screen"
"$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
  --config configs/exp04b_confirmatory.json \
  --sparsity-mode jump_relu \
  --jump-relu-threshold-lr-multiplier 16 \
  --jump-relu-sparsity-weight "$SELECTED_WEIGHT" \
  --decoder-weights 0.03125 \
  --seeds 0 \
  --token-budget 25000000 \
  --source-range-name robustness \
  --data-seed 16180339 \
  --probe-seed-base 14142135 \
  --new-screen "$SCREEN" \
  --device cuda:0 \
  --gpu-memory-fraction 0.20 \
  --maximum-peak-gpu-gib 18 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_full_horizon_screen_train.log" 2>&1

"$PYTHON" -u experiments/exp07_jumprelu_calibration.py pair \
  --models "$SCREEN/models.pt" \
  --training-log "$SCREEN/training.jsonl" \
  --run-done "$SCREEN/done.json" \
  --label full_horizon_screen \
  --device cuda:0 \
  --gpu-memory-fraction 0.20 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_full_horizon_screen_l0.log" 2>&1

"$PYTHON" -c '
import json, sys
raise SystemExit(not json.load(open(sys.argv[1]))["advance"])
' "$OUTPUT/jump_relu_full_horizon_screen_l0_gate.json"

"$PYTHON" -u experiments/paper_closure.py frontier-existing \
  --source-models "$SCREEN/models.pt" \
  --cache artifacts/paper_closure/natural_selection.pt \
  --static artifacts/exp04b_confirmatory/static_calibration.pt \
  --config configs/paper_closure.json \
  --output "$OUTPUT/jumprelu_full_horizon_screen_outcomes.json" \
  --split-label "JumpReLU full-horizon calibrated screen [180M,185M)" \
  --evaluation-seed 0 \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --maximum-peak-gpu-gib 24 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_full_horizon_screen_outcomes.log" 2>&1
