#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "exp08 GPU queue requires a clean worktree" >&2
  exit 1
fi

SOURCE_ROOT="$(realpath "${DPSAE_SOURCE_ROOT:-/workspace/decoder-preserving-sae}")"
PYTHON="${DPSAE_PYTHON:-$SOURCE_ROOT/.venv/bin/python}"
OUTPUT="artifacts/exp08_experiment_figure"
LOGS="$OUTPUT/logs"
MANIFEST="$OUTPUT/run_manifest.json"
mkdir -p "$LOGS" artifacts/exp04_ioi_mechanism artifacts/exp04b_confirmatory artifacts/paper_closure
if [[ "$(realpath "$SOURCE_ROOT")" == "$(realpath "$ROOT")" ]]; then
  echo "DPSAE_SOURCE_ROOT must be an artifact/env tree separate from the clean worktree" >&2
  exit 1
fi
test -s "$MANIFEST"
"$PYTHON" -u scripts/prepare_exp08_run.py \
  --source-root "$SOURCE_ROOT" \
  --output "$MANIFEST" \
  > "$LOGS/manifest_validation_gpu.log" 2>&1

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export TOKENIZERS_PARALLELISM=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.:src

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if (( GPU_USED_MIB > 1024 )); then
  echo "GPU is already using ${GPU_USED_MIB} MiB; refusing to start the exp08 queue" >&2
  exit 1
fi

SOURCE_TOKENS="$SOURCE_ROOT/artifacts/exp04_ioi_mechanism/fineweb_gpt2_tokens.bin"
SOURCE_CALIBRATION="$SOURCE_ROOT/artifacts/exp04_ioi_mechanism/calibration.pt"
STATIC="$SOURCE_ROOT/artifacts/exp04b_confirmatory/static_calibration.pt"
TAIL_180="$SOURCE_ROOT/artifacts/exp04b_confirmatory/fineweb_gpt2_tail_tokens.bin"
TAIL_190="$SOURCE_ROOT/artifacts/paper_closure/fineweb_gpt2_tail_tokens.bin"
SELECTION_180="artifacts/exp04b_confirmatory/natural_selection.pt"
SELECTION_190="artifacts/paper_closure/natural_selection.pt"

for path in \
  "$PYTHON" "$SOURCE_TOKENS" "$SOURCE_CALIBRATION" "$STATIC" \
  "$TAIL_180" "$TAIL_180.json" "$TAIL_190" "$TAIL_190.json"; do
  test -e "$path"
done

ln -sfn "$SOURCE_TOKENS" artifacts/exp04_ioi_mechanism/fineweb_gpt2_tokens.bin
ln -sfn "$SOURCE_CALIBRATION" artifacts/exp04_ioi_mechanism/calibration.pt
ln -sfn "$STATIC" artifacts/exp04b_confirmatory/static_calibration.pt
ln -sfn "$TAIL_180" artifacts/exp04b_confirmatory/fineweb_gpt2_tail_tokens.bin
ln -sfn "$TAIL_180.json" artifacts/exp04b_confirmatory/fineweb_gpt2_tail_tokens.bin.json
ln -sfn "$TAIL_190" artifacts/paper_closure/fineweb_gpt2_tail_tokens.bin
ln -sfn "$TAIL_190.json" artifacts/paper_closure/fineweb_gpt2_tail_tokens.bin.json

echo "[1/9] Validate immutable tails and regenerate clean evaluation caches"
CACHE_STARTED="$(date +%s)"
"$PYTHON" -u experiments/exp04b_confirmatory.py prepare-tail \
  --config configs/exp04b_confirmatory.json \
  --device cuda:0 \
  > "$LOGS/prepare_tail_180.log" 2>&1
"$PYTHON" -u experiments/exp04b_confirmatory.py prepare-tail \
  --config configs/paper_closure.json \
  --device cuda:0 \
  > "$LOGS/prepare_tail_190.log" 2>&1
"$PYTHON" -u experiments/exp04b_confirmatory.py cache-natural \
  --config configs/exp04b_confirmatory.json \
  --natural-split selection \
  --device cuda:0 \
  > "$LOGS/cache_gamma_selection.log" 2>&1
"$PYTHON" -u experiments/exp04b_confirmatory.py cache-natural \
  --config configs/paper_closure.json \
  --natural-split all \
  --device cuda:0 \
  > "$LOGS/cache_confirmation_and_frozen.log" 2>&1
CACHE_WALL_SECONDS="$(( $(date +%s) - CACHE_STARTED ))"
printf '%s\n' "$CACHE_WALL_SECONDS" > "$OUTPUT/cache_wall_seconds.txt"

echo "[2/9] Repeat the full one-seed gamma sweep from a clean revision"
"$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
  --new-screen "$OUTPUT/gamma_sweep" \
  --decoder-weights 0.03125 0.0625 0.09375 0.125 0.25 0.5 1.0 \
  --seeds 0 \
  --token-budget 25000000 \
  --source-range-name screen \
  --source-tokens "$SOURCE_TOKENS" \
  --source-calibration "$SOURCE_CALIBRATION" \
  --sparsity-mode batch_topk \
  --device cuda:0 \
  --gpu-memory-fraction 0.35 \
  --maximum-peak-gpu-gib 30.0 \
  > "$LOGS/gamma_sweep_train.log" 2>&1

"$PYTHON" -u experiments/paper_closure.py frontier-existing \
  --source-models "$OUTPUT/gamma_sweep/models.pt" \
  --cache "$SELECTION_180" \
  --static "$STATIC" \
  --config configs/exp04b_confirmatory.json \
  --output "$OUTPUT/gamma_sweep_selection.json" \
  --split-label "clean gamma selection [180M,185M)" \
  --evaluation-seed 0 \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --maximum-peak-gpu-gib 24.0 \
  > "$LOGS/gamma_sweep_eval.log" 2>&1

"$PYTHON" -u experiments/paper_closure.py frontier-select \
  --frontier-input "$OUTPUT/gamma_sweep_selection.json" \
  --config configs/paper_closure.json \
  --output "$OUTPUT/gamma_sweep_choice.json" \
  > "$LOGS/gamma_sweep_select.log" 2>&1

"$PYTHON" -c 'import json,sys; value=json.load(open(sys.argv[1]))["selected_decoder_weight"]; assert value == 0.03125, value' \
  "$OUTPUT/gamma_sweep_choice.json"

echo "[3/9] Repeat the frozen three-seed 100M-token confirmation"
"$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
  --new-screen "$OUTPUT/confirmation" \
  --decoder-weights 0.03125 \
  --seeds 0 1 2 \
  --token-budget 100000000 \
  --source-range-name confirmation \
  --data-seed 1995652635 \
  --probe-seed-base 1584467719 \
  --source-tokens "$SOURCE_TOKENS" \
  --source-calibration "$SOURCE_CALIBRATION" \
  --sparsity-mode batch_topk \
  --device cuda:0 \
  --gpu-memory-fraction 0.35 \
  --maximum-peak-gpu-gib 30.0 \
  > "$LOGS/confirmation_train.log" 2>&1

echo "[4/9] Evaluate the clean confirmation on [190M,195M)"
for seed in 0 1 2; do
  "$PYTHON" -u experiments/paper_closure.py frontier-existing \
    --source-models "$OUTPUT/confirmation/models.pt" \
    --cache "$SELECTION_190" \
    --static "$STATIC" \
    --config configs/paper_closure.json \
    --output "$OUTPUT/confirmation_seed${seed}.json" \
    --split-label "clean confirmation [190M,195M)" \
    --evaluation-seed "$seed" \
    --device cuda:0 \
    --gpu-memory-fraction 0.25 \
    --maximum-peak-gpu-gib 24.0 \
    > "$LOGS/confirmation_seed${seed}_eval.log" 2>&1
done

echo "[5/9] Enforce the frozen matched-quality confirmation gate"
"$PYTHON" -u scripts/summarize_exp08_confirmation.py \
  --confirmation \
    "$OUTPUT/confirmation_seed0.json" \
    "$OUTPUT/confirmation_seed1.json" \
    "$OUTPUT/confirmation_seed2.json" \
  --training-done "$OUTPUT/confirmation/done.json" \
  --models "$OUTPUT/confirmation/models.pt" \
  --cache "$SELECTION_190" \
  --calibration "$SOURCE_CALIBRATION" \
  --config configs/paper_closure.json \
  --run-manifest "$MANIFEST" \
  --selected-weight 0.03125 \
  --output "$OUTPUT/confirmation_summary.json" \
  > "$LOGS/confirmation_summary.log" 2>&1

echo "[6/9] Evaluate robustness, frozen-LM fidelity, and isolated overhead"
"$PYTHON" -u experiments/exp08_language_evidence.py all \
  --models "$OUTPUT/confirmation/models.pt" \
  --confirmation-cache "$SELECTION_190" \
  --frozen-cache artifacts/paper_closure/natural_test.pt \
  --static "$STATIC" \
  --calibration "$SOURCE_CALIBRATION" \
  --training-done "$OUTPUT/confirmation/done.json" \
  --confirmation-summary "$OUTPUT/confirmation_summary.json" \
  --run-manifest "$MANIFEST" \
  --config configs/paper_closure.json \
  --output-dir "$OUTPUT/evidence" \
  --selected-weight 0.03125 \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --maximum-peak-gpu-gib 24.0 \
  > "$LOGS/language_evidence.log" 2>&1

echo "[7/9] Recompute the taskwise spectrum from clean checkpoints"
"$PYTHON" -u experiments/exp07_advantage_spectrum.py spectrum \
  --config configs/exp08_task_spectrum.json \
  --output "$OUTPUT/task_spectrum" \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --minimum-free-gib 20 \
  > "$LOGS/task_spectrum.log" 2>&1

echo "[8/9] Summarize measured GPU time"
"$PYTHON" -u scripts/summarize_exp08_compute.py \
  --root "$OUTPUT" \
  --run-manifest "$MANIFEST" \
  --output "$OUTPUT/compute_summary.json" \
  > "$LOGS/compute_summary.log" 2>&1

echo "[9/9] Validate the complete GPU artifact set"

test -s "$OUTPUT/evidence/robustness.json"
test -s "$OUTPUT/evidence/frozen_fidelity.json"
test -s "$OUTPUT/evidence/training_overhead.json"
test -s "$OUTPUT/task_spectrum/advantage_spectrum_summary.json"
test -s "$OUTPUT/confirmation_summary.json"
test -s "$OUTPUT/compute_summary.json"
