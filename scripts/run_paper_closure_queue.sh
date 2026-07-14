#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/paper_closure/logs
export HF_HOME=/workspace/huggingface
export TOKENIZERS_PARALLELISM=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
PYTHON=.venv/bin/python

while tmux has-session -t dpsae-token-topk-screen 2>/dev/null; do
  sleep 30
done

test -f artifacts/paper_closure/token_topk_screen/models.pt
test -f artifacts/paper_closure/natural_selection.pt

"$PYTHON" -u experiments/paper_closure.py frontier-existing \
  --source-models artifacts/paper_closure/token_topk_screen/models.pt \
  --cache artifacts/exp04b_confirmatory/natural_selection.pt \
  --static artifacts/exp04b_confirmatory/static_calibration.pt \
  --config configs/exp04b_confirmatory.json \
  --output artifacts/paper_closure/token_topk_selection.json \
  --split-label "exp04b natural selection [180M,185M)" \
  --evaluation-seed 0 \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --maximum-peak-gpu-gib 24.0 \
  > artifacts/paper_closure/logs/token_topk_eval.log 2>&1

"$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
  --new-screen artifacts/paper_closure/confirmation_common \
  --decoder-weights 0.03125 \
  --seeds 0 1 2 \
  --token-budget 100000000 \
  --source-range-name confirmation \
  --data-seed 1995652635 \
  --probe-seed-base 1584467719 \
  --sparsity-mode batch_topk \
  --device cuda:0 \
  --gpu-memory-fraction 0.35 \
  --maximum-peak-gpu-gib 30.0 \
  > artifacts/paper_closure/logs/confirmation_common.log 2>&1

for seed in 0 1 2; do
  "$PYTHON" -u experiments/paper_closure.py frontier-existing \
    --source-models artifacts/paper_closure/confirmation_common/models.pt \
    --cache artifacts/paper_closure/natural_selection.pt \
    --static artifacts/exp04b_confirmatory/static_calibration.pt \
    --config configs/paper_closure.json \
    --output "artifacts/paper_closure/confirmation_seed${seed}.json" \
    --split-label "paper closure confirmation [190M,195M)" \
    --evaluation-seed "$seed" \
    --device cuda:0 \
    --gpu-memory-fraction 0.25 \
    --maximum-peak-gpu-gib 24.0 \
    > "artifacts/paper_closure/logs/confirmation_seed${seed}_eval.log" 2>&1
done

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/paper_closure")
rows = []
for seed in (0, 1, 2):
    payload = json.loads((root / f"confirmation_seed{seed}.json").read_text())
    if len(payload["paired_frontier"]) != 1:
        raise RuntimeError(f"seed {seed} did not produce one paired frontier row")
    rows.append({"seed": seed, **payload["paired_frontier"][0]})
(root / "confirmation_summary.json").write_text(
    json.dumps({"complete": True, "rows": rows}, indent=2, sort_keys=True) + "\n"
)
PY
