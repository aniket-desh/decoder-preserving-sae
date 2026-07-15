#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "exp08 launch requires a clean worktree" >&2
  exit 1
fi

SOURCE_ROOT="$(realpath "${DPSAE_SOURCE_ROOT:-/workspace/decoder-preserving-sae}")"
PYTHON="${DPSAE_PYTHON:-$SOURCE_ROOT/.venv/bin/python}"
OUTPUT=artifacts/exp08_experiment_figure
STATUS="$OUTPUT/status"
MANIFEST="$OUTPUT/run_manifest.json"
test -x "$PYTHON"
mkdir -p "$STATUS" "$OUTPUT/logs"
"$PYTHON" -u scripts/prepare_exp08_run.py \
  --source-root "$SOURCE_ROOT" \
  --output "$MANIFEST"
RUN_ID="$(
  "$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["contract_sha256"][:8])' \
    "$MANIFEST"
)"

GPU_SESSION="dpsae-expfig-gpu-$RUN_ID"
SYNTHETIC_SESSION="dpsae-expfig-synthetic-$RUN_ID"
FINALIZER_SESSION="dpsae-expfig-finalize-$RUN_ID"
for session in "$GPU_SESSION" "$SYNTHETIC_SESSION" "$FINALIZER_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

rm -f \
  "$STATUS/gpu.status" \
  "$STATUS/synthetic.status" \
  "$STATUS/finalize.status"
tmux new-session -d -s "$GPU_SESSION" \
  "printf 'running\n' > '$ROOT/$STATUS/gpu.status'; if bash '$ROOT/scripts/run_exp08_gpu_runpod.sh'; then printf 'succeeded\n' > '$ROOT/$STATUS/gpu.status'; else code=\$?; printf 'failed:%s\n' \"\$code\" > '$ROOT/$STATUS/gpu.status'; exit \"\$code\"; fi"
tmux set-option -t "$GPU_SESSION" remain-on-exit on
tmux new-session -d -s "$SYNTHETIC_SESSION" \
  "printf 'running\n' > '$ROOT/$STATUS/synthetic.status'; if bash '$ROOT/scripts/run_exp08_synthetic_runpod.sh'; then printf 'succeeded\n' > '$ROOT/$STATUS/synthetic.status'; else code=\$?; printf 'failed:%s\n' \"\$code\" > '$ROOT/$STATUS/synthetic.status'; exit \"\$code\"; fi"
tmux set-option -t "$SYNTHETIC_SESSION" remain-on-exit on
tmux new-session -d -s "$FINALIZER_SESSION" \
  "printf 'running\n' > '$ROOT/$STATUS/finalize.status'; if bash '$ROOT/scripts/finalize_exp08_candidates.sh' > '$ROOT/$OUTPUT/logs/finalize_candidates.log' 2>&1; then printf 'succeeded\n' > '$ROOT/$STATUS/finalize.status'; else code=\$?; printf 'failed:%s\n' \"\$code\" > '$ROOT/$STATUS/finalize.status'; exit \"\$code\"; fi"
tmux set-option -t "$FINALIZER_SESSION" remain-on-exit on

echo "started $GPU_SESSION, $SYNTHETIC_SESSION, and $FINALIZER_SESSION"
echo "status files: $STATUS/{gpu,synthetic,finalize}.status"
