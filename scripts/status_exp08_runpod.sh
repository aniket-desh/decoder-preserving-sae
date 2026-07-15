#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT=artifacts/exp08_experiment_figure
for worker in gpu synthetic finalize; do
  status_file="$OUTPUT/status/$worker.status"
  if [[ -s "$status_file" ]]; then
    printf '%s: ' "$worker"
    sed -n '1p' "$status_file"
  else
    echo "$worker: not started"
  fi
done

gpu_status="$(sed -n '1p' "$OUTPUT/status/gpu.status" 2>/dev/null || true)"
finalize_status="$(sed -n '1p' "$OUTPUT/status/finalize.status" 2>/dev/null || true)"

if [[ "$gpu_status" == succeeded && -f "$OUTPUT/compute_summary.json" ]]; then
  echo "compute summary: $OUTPUT/compute_summary.json"
fi

if [[ "$finalize_status" == succeeded \
  && -f "$OUTPUT/candidate_figures/candidate_manifest.json" ]]; then
  echo "candidate manifest: $OUTPUT/candidate_figures/candidate_manifest.json"
fi
