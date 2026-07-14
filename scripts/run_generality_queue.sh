#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/exp06_generality/logs
export HF_HOME=/workspace/huggingface
export TOKENIZERS_PARALLELISM=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
PYTHON=.venv/bin/python

while tmux has-session -t dpsae-closure-queue 2>/dev/null; do
  sleep 60
done

"$PYTHON" -u experiments/exp06_generality.py all \
  --target gpt2-block4 \
  --gamma 0.03125 \
  --output artifacts/exp06_generality/gpt2-block4 \
  --token-cache artifacts/exp06_generality/gpt2-block4/fineweb_gpt2_tokens.bin \
  --device cuda:0 \
  --min-free-disk-gib 20 \
  --max-gpu-reserved-gib 30 \
  --max-gpu-fraction 0.35 \
  > artifacts/exp06_generality/logs/gpt2_block4.log 2>&1

"$PYTHON" -u experiments/exp06_generality.py all \
  --target pythia-block8 \
  --gamma 0.03125 \
  --output artifacts/exp06_generality/pythia-block8 \
  --token-cache artifacts/exp06_generality/pythia-block8/fineweb_pythia_tokens.bin \
  --device cuda:0 \
  --min-free-disk-gib 20 \
  --max-gpu-reserved-gib 30 \
  --max-gpu-fraction 0.35 \
  > artifacts/exp06_generality/logs/pythia_block8.log 2>&1

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/exp06_generality")
targets = {}
for key in ("gpt2-block4", "pythia-block8"):
    result = json.loads((root / key / "evaluation.json").read_text())
    targets[key] = {
        "complete": result["complete"],
        "nmse_ratio": result["nmse_ratio"],
        "paired_reduction": result["paired_reduction"],
        "screen_gate": result["screen_gate"],
    }
(root / "summary.json").write_text(
    json.dumps({"complete": True, "targets": targets}, indent=2, sort_keys=True)
    + "\n"
)
PY
