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
MARKER=artifacts/paper_closure/jumprelu_code_ready.json
INTEGRATION_PREFIX=artifacts/paper_closure/architecture_jump_relu_integration_tlr
INTEGRATION_SELECTION=artifacts/paper_closure/jump_relu_integration_selection.json
SCREEN=artifacts/paper_closure/architecture_jump_relu_screen

while tmux has-session -t dpsae-generality-queue 2>/dev/null; do
  sleep 60
done

# The local JumpReLU implementation cannot be copied while the generality
# process has imported the shared SAE modules. The orchestrator writes this
# marker only after transferring and hash-checking the complete code bundle.
while [[ ! -f "$MARKER" ]]; do
  sleep 60
done

mapfile -t MULTIPLIERS < <(
  "$PYTHON" -c \
    'import json; print(*json.load(open("configs/paper_closure.json"))["architecture"]["jump_relu_threshold_lr_multiplier_integration_grid"], sep="\n")'
)
SELECTED=""
for MULTIPLIER in "${MULTIPLIERS[@]}"; do
  LABEL="${MULTIPLIER//./p}"
  INTEGRATION="${INTEGRATION_PREFIX}${LABEL}"
  "$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
    --sparsity-mode jump_relu \
    --jump-relu-threshold-lr-multiplier "$MULTIPLIER" \
    --decoder-weights 0.03125 \
    --seeds 0 \
    --token-budget 2000000 \
    --new-screen "$INTEGRATION" \
    --gpu-memory-fraction 0.08 \
    --maximum-peak-gpu-gib 6 \
    --minimum-free-gib 20 \
    > "artifacts/paper_closure/logs/jump_relu_integration_tlr${LABEL}.log" 2>&1

  "$PYTHON" - "$INTEGRATION" "$MULTIPLIER" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
multiplier = float(sys.argv[2])
last = json.loads((root / "training.jsonl").read_text().splitlines()[-1])
models = last["models"]
finite = all(
    math.isfinite(float(value[key]))
    for value in models.values()
    for key in ("loss", "nmse", "decoder", "aux", "l0", "sparsity")
)
l0_ok = all(30.4 <= float(value["l0"]) <= 33.6 for value in models.values())
dead_ok = all(int(value["dead"]) <= 1638 for value in models.values())
result = {
    "complete": True,
    "advance_to_screen": bool(finite and l0_ok and dead_ok),
    "threshold_lr_multiplier": multiplier,
    "finite": finite,
    "l0_ok": l0_ok,
    "dead_fraction_ok": dead_ok,
    "last_step": last["step"],
    "models": models,
}
(root / "integration_gate.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
PY

  if "$PYTHON" -c \
    'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["advance_to_screen"])' \
    "$INTEGRATION/integration_gate.json"; then
    SELECTED="$MULTIPLIER"
    break
  fi
done

"$PYTHON" - "$INTEGRATION_SELECTION" "$SELECTED" "${MULTIPLIERS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
selected = sys.argv[2]
multipliers = [float(value) for value in sys.argv[3:]]
root = Path("artifacts/paper_closure")
config = json.loads(Path("configs/paper_closure.json").read_text())
candidates = []
for multiplier in multipliers:
    label = str(multiplier).replace(".", "p")
    path = root / f"architecture_jump_relu_integration_tlr{label}" / "integration_gate.json"
    if not path.exists():
        break
    gate = json.loads(path.read_text())
    candidates.append({"path": str(path), **gate})
result = {
    "complete": True,
    "selection_rule": "smallest threshold LR multiplier passing finite, L0, and dead-feature gates",
    "configured_grid": multipliers,
    "calibration_bracket": config["architecture"][
        "jump_relu_threshold_lr_calibration_bracket"
    ],
    "selected_threshold_lr_multiplier": float(selected) if selected else None,
    "candidates": candidates,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

if [[ -z "$SELECTED" ]]; then
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/paper_closure")
selection = json.loads((root / "jump_relu_integration_selection.json").read_text())
(root / "jump_relu_summary.json").write_text(
    json.dumps(
        {
            "complete": True,
            "screen_pass": False,
            "requires_confirmation": False,
            "status": "integration_grid_failed_matched_sparsity_gate",
            "integration_selection": selection,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
  exit 0
fi

"$PYTHON" -u experiments/paper_closure.py frontier-train-screen \
  --sparsity-mode jump_relu \
  --jump-relu-threshold-lr-multiplier "$SELECTED" \
  --decoder-weights 0.03125 \
  --seeds 0 \
  --token-budget 25000000 \
  --new-screen "$SCREEN" \
  --gpu-memory-fraction 0.20 \
  --maximum-peak-gpu-gib 18 \
  --minimum-free-gib 20 \
  > artifacts/paper_closure/logs/jump_relu_screen.log 2>&1

"$PYTHON" -u experiments/paper_closure.py frontier-existing \
  --source-models "$SCREEN/models.pt" \
  --cache artifacts/exp04b_confirmatory/natural_selection.pt \
  --static artifacts/exp04b_confirmatory/static_calibration.pt \
  --config configs/paper_closure.json \
  --output artifacts/paper_closure/jump_relu_selection.json \
  --split-label "exp04b natural selection [180M,185M)" \
  --evaluation-seed 0 \
  --device cuda:0 \
  --gpu-memory-fraction 0.25 \
  --maximum-peak-gpu-gib 24 \
  > artifacts/paper_closure/logs/jump_relu_eval.log 2>&1

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/paper_closure")
evaluation = json.loads((root / "jump_relu_selection.json").read_text())
row = evaluation["paired_frontier"][0]
models = evaluation["models"]
l0_ok = all(30.4 <= float(value["l0_inference"]) <= 33.6 for value in models.values())
reduction = float(row["exact_decoder_reduction"])
comparators = {
    "batch_topk": float(
        json.loads((root / "frontier_common_selection.json").read_text())[
            "paired_frontier"
        ][0]["exact_decoder_reduction"]
    ),
    "token_topk": float(
        json.loads((root / "token_topk_selection.json").read_text())[
            "paired_frontier"
        ][0]["exact_decoder_reduction"]
    ),
}
interactions = {
    key: reduction - value for key, value in comparators.items()
}
screen_pass = (
    l0_ok
    and float(row["nmse_ratio_to_mse"]) <= 1.10
    and reduction >= 0.10
)
requires_confirmation = screen_pass and max(abs(value) for value in interactions.values()) >= 0.10
summary = {
    "complete": True,
    "screen_pass": screen_pass,
    "requires_confirmation": requires_confirmation,
    "paired_frontier": row,
    "l0_ok": l0_ok,
    "interaction_against": interactions,
    "threshold_lr_multiplier": json.loads(
        (root / "jump_relu_integration_selection.json").read_text()
    )["selected_threshold_lr_multiplier"],
    "confirmation_rule": "confirm only if an absolute interaction is at least 0.10",
}
(root / "jump_relu_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
PY
