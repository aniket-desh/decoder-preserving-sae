#!/usr/bin/env bash
set -Eeuo pipefail

# Pod-resident supervisor for the already-frozen Exp10/Exp11 closure wave. It
# deliberately stops after the concept-pilot decision and never calls an API,
# uploads an artifact, or starts fresh confirmation.

ROOT="${ROOT:-/workspace/decoder-preserving-sae-concept-v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/dpsae-runs/20260716/exp10_concept_discovery/pythia160m-block8-s0-pilot-v2}"
MODEL_CACHE="${MODEL_CACHE:-/workspace/dpsae-runs/20260716/exp10_concept_discovery/shared-model-cache-v1}"
EXP11_ROOT="${EXP11_ROOT:-/workspace/decoder-preserving-sae/artifacts/exp11_static_matched_nmse}"
EXP11_SESSION="${EXP11_SESSION:-dpsae-spectral-screen-v2}"
TIMING_SESSION="${TIMING_SESSION:-exp10-timing-v2}"
PYTHON="${PYTHON:-/workspace/SAEBench/.venv/bin/python}"
SAEBENCH_ROOT="${SAEBENCH_ROOT:-/workspace/SAEBench}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/dpsae-restored/exp06_generality/pythia-block8}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
CONTROL_ROOT="${CONTROL_ROOT:-/workspace/dpsae-runs/20260716/control}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_EXP10_SECONDS="${MAX_EXP10_SECONDS:-14400}"

STATUS="$CONTROL_ROOT/steps1_4_status.json"
LOG="$CONTROL_ROOT/steps1_4.log"
mkdir -p "$CONTROL_ROOT"
exec > >(tee -a "$LOG") 2>&1

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FLEET_LAUNCHED=0

write_status() {
  local stage="$1"
  local state="$2"
  local detail="$3"
  local tmp="$STATUS.tmp"
  jq -n \
    --arg schema_version "1" \
    --arg started_utc "$STARTED_UTC" \
    --arg updated_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg stage "$stage" \
    --arg state "$state" \
    --arg detail "$detail" \
    --arg root "$ROOT" \
    --arg output_root "$OUTPUT_ROOT" \
    --arg root_revision "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)" \
    '{
      schema_version: ($schema_version | tonumber),
      started_utc: $started_utc,
      updated_utc: $updated_utc,
      stage: $stage,
      state: $state,
      detail: $detail,
      root: $root,
      output_root: $output_root,
      root_revision: $root_revision,
      approval_boundary: "stop_before_fresh_confirmation_api_or_backup"
    }' > "$tmp"
  mv "$tmp" "$STATUS"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $stage $state: $detail"
}

stop_exp10_fleet() {
  local session
  for session in exp10-gpu0 exp10-gpu1 exp10-gpu2 exp10-gpu3 exp10-finalize; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
    fi
  done
}

on_error() {
  local code=$?
  trap - ERR
  if [[ "$FLEET_LAUNCHED" == "1" ]]; then
    stop_exp10_fleet
  fi
  write_status "failed" "error" "exit=$code line=${BASH_LINENO[0]} command=${BASH_COMMAND}"
  exit "$code"
}
trap on_error ERR

session_record() {
  local session="$1"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    echo "missing"
    return
  fi
  tmux list-panes -t "$session" -F '#{pane_dead} #{pane_dead_status}' | head -n 1
}

wait_for_artifact() {
  local session="$1"
  local artifact="$2"
  local label="$3"
  local record
  while [[ ! -f "$artifact" ]]; do
    record="$(session_record "$session")"
    if [[ "$record" == "missing" ]]; then
      write_status "$label" "error" "session $session disappeared before $artifact"
      return 1
    fi
    if [[ "${record%% *}" == "1" ]]; then
      write_status "$label" "error" "session $session exited ${record#* } before $artifact"
      return 1
    fi
    write_status "$label" "waiting" "session=$session artifact=$artifact"
    sleep "$POLL_SECONDS"
  done
}

if [[ ! -x "$PYTHON" ]]; then
  echo "missing sealed Python: $PYTHON" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "autonomous supervisor requires a clean Exp10 checkout: $ROOT" >&2
  exit 2
fi
if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "da64fc2913cdce262b2718f4e5f47a7f97c8f33a" ]]; then
  echo "unexpected Exp10 revision in $ROOT" >&2
  exit 2
fi

TIMING_REPORT="$OUTPUT_ROOT/timing_smoke.json"
write_status "timing_gate" "running" "waiting for blind timing report"
wait_for_artifact "$TIMING_SESSION" "$TIMING_REPORT" "timing_gate"
if ! jq -e '.complete == true and .passed == true' "$TIMING_REPORT" >/dev/null; then
  projected="$(jq -r '.projection.projected_pod_hours // "missing"' "$TIMING_REPORT")"
  write_status "timing_gate" "halted" "blind timing gate failed; projected_pod_hours=$projected"
  exit 0
fi
write_status "timing_gate" "passed" "projected_pod_hours=$(jq -r '.projection.projected_pod_hours' "$TIMING_REPORT")"

EXP11_SUMMARY="$EXP11_ROOT/summary.json"
write_status "exp11" "running" "waiting for matched-NMSE screen and any gated confirmation"
wait_for_artifact "$EXP11_SESSION" "$EXP11_SUMMARY" "exp11"
if ! jq -e '.complete == true' "$EXP11_SUMMARY" >/dev/null; then
  echo "Exp11 summary is incomplete" >&2
  exit 1
fi

while true; do
  gpu3_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')"
  if [[ "$gpu3_memory" -le 512 ]]; then
    break
  fi
  write_status "gpu_release" "waiting" "GPU3 memory_used_mib=$gpu3_memory"
  sleep "$POLL_SECONDS"
done
write_status "gpu_release" "passed" "all four A40s available"

for session in exp10-gpu0 exp10-gpu1 exp10-gpu2 exp10-gpu3 exp10-finalize; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "refusing to overwrite existing tmux session: $session" >&2
    exit 2
  fi
done

write_status "concept_pilot" "launching" "starting four frozen Exp10 workers and finalizer"
(
  cd "$ROOT"
  env \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODEL_CACHE="$MODEL_CACHE" \
    PYTHON="$PYTHON" \
    SAEBENCH_ROOT="$SAEBENCH_ROOT" \
    CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    HF_HOME="$HF_HOME" \
    bash scripts/run_exp10_concept_4xa40.sh
)
FLEET_LAUNCHED=1

FINAL_AUDIT="$OUTPUT_ROOT/artifact_audit_final.json"
deadline="$(( $(date +%s) + MAX_EXP10_SECONDS ))"
while [[ ! -f "$FINAL_AUDIT" ]]; do
  record="$(session_record exp10-finalize)"
  if [[ "$record" == "missing" ]]; then
    write_status "concept_pilot" "error" "exp10-finalize disappeared before final audit"
    false
  fi
  if [[ "${record%% *}" == "1" ]]; then
    write_status "concept_pilot" "error" "exp10-finalize exited ${record#* } before final audit"
    false
  fi
  if (( $(date +%s) > deadline )); then
    write_status "concept_pilot" "error" "four-hour autonomous runtime ceiling exceeded"
    false
  fi
  write_status "concept_pilot" "running" "waiting for four workers, aggregate, and final integrity audit"
  sleep "$POLL_SECONDS"
done

if ! jq -e '.complete == true and .passed == true and .phase == "final"' "$FINAL_AUDIT" >/dev/null; then
  echo "Exp10 final audit did not pass" >&2
  false
fi

ADVANCEMENT="$OUTPUT_ROOT/advancement_report.json"
if [[ ! -f "$ADVANCEMENT" ]]; then
  echo "Exp10 advancement report is missing after final audit" >&2
  false
fi
advance="$(jq -r '.advance_fresh_confirmation' "$ADVANCEMENT")"
if [[ "$advance" != "true" && "$advance" != "false" ]]; then
  echo "invalid Exp10 advancement decision: $advance" >&2
  false
fi

write_status "approval_boundary" "complete" "advance_fresh_confirmation=$advance; stopped before fresh confirmation, API labeling, cross-experiment audit, or backup"
