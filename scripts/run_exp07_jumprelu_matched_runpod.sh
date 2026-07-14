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
GRID="$OUTPUT/jump_relu_controller_grid.json"
CALIBRATION_MODELS="$OUTPUT/jumprelu_calibration_confirmation/models.pt"
SCREEN_MODELS="$OUTPUT/jumprelu_matched_screen/models.pt"
mkdir -p "$LOGS"

while tmux has-session -t dpsae-jumprelu-grid 2>/dev/null; do
  sleep 30
done
test -f "$GRID"

read -r MSE_MULTIPLIER DPSAE_MULTIPLIER < <(
  "$PYTHON" -c '
import json, sys
value = json.load(open(sys.argv[1]))["selection"]
print(value["mse"]["interpolated_multiplier"], value["dpsae"]["interpolated_multiplier"])
' "$GRID"
)

"$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
  --config configs/exp04b_confirmatory.json \
  --sparsity-mode jump_relu \
  --jump-relu-threshold-lr-multiplier-mse "$MSE_MULTIPLIER" \
  --jump-relu-threshold-lr-multiplier-dpsae "$DPSAE_MULTIPLIER" \
  --decoder-weights 0.03125 \
  --seeds 0 \
  --token-budget 2000000 \
  --source-range-name confirmation \
  --data-seed 1329472341 \
  --probe-seed-base 1794246311 \
  --new-screen "$OUTPUT/jumprelu_calibration_confirmation" \
  --device cuda:0 \
  --gpu-memory-fraction 0.10 \
  --maximum-peak-gpu-gib 8 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_calibration_confirmation_train.log" 2>&1

"$PYTHON" -u experiments/exp07_jumprelu_calibration.py pair \
  --models "$CALIBRATION_MODELS" \
  --label calibration_confirmation \
  --device cuda:0 \
  --gpu-memory-fraction 0.20 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_calibration_confirmation_l0.log" 2>&1

"$PYTHON" -c '
import json, sys
raise SystemExit(not json.load(open(sys.argv[1]))["advance"])
' "$OUTPUT/jump_relu_calibration_confirmation_l0_gate.json"

"$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
  --config configs/exp04b_confirmatory.json \
  --sparsity-mode jump_relu \
  --jump-relu-threshold-lr-multiplier-mse "$MSE_MULTIPLIER" \
  --jump-relu-threshold-lr-multiplier-dpsae "$DPSAE_MULTIPLIER" \
  --decoder-weights 0.03125 \
  --seeds 0 \
  --token-budget 25000000 \
  --source-range-name screen \
  --data-seed 20260712 \
  --probe-seed-base 20260712 \
  --new-screen "$OUTPUT/jumprelu_matched_screen" \
  --device cuda:0 \
  --gpu-memory-fraction 0.20 \
  --maximum-peak-gpu-gib 18 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_matched_screen_train.log" 2>&1

"$PYTHON" -u experiments/exp07_jumprelu_calibration.py pair \
  --models "$SCREEN_MODELS" \
  --label matched_screen \
  --device cuda:0 \
  --gpu-memory-fraction 0.20 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_matched_screen_l0.log" 2>&1

"$PYTHON" -c '
import json, sys
raise SystemExit(not json.load(open(sys.argv[1]))["advance"])
' "$OUTPUT/jump_relu_matched_screen_l0_gate.json"

"$PYTHON" -u experiments/paper_closure.py frontier-existing \
  --source-models "$SCREEN_MODELS" \
  --cache artifacts/paper_closure/natural_selection.pt \
  --static artifacts/exp04b_confirmatory/static_calibration.pt \
  --config configs/paper_closure.json \
  --output "$OUTPUT/jumprelu_matched_screen_outcomes.json" \
  --split-label "JumpReLU matched-controller screen [190M,195M)" \
  --evaluation-seed 0 \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --maximum-peak-gpu-gib 24 \
  --minimum-free-gib 20 \
  > "$LOGS/jumprelu_matched_screen_outcomes.log" 2>&1
