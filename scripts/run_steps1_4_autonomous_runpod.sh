#!/usr/bin/env bash
set -Eeuo pipefail

# Pod-resident supervisor for the already-frozen Exp10/Exp11 closure wave. It
# deliberately stops after the concept-pilot decision and never calls an API,
# uploads an artifact, or starts fresh confirmation.

ROOT="${ROOT:-/workspace/decoder-preserving-sae-concept-v2}"
EXPECTED_ROOT_REVISION="${EXPECTED_ROOT_REVISION:?set EXPECTED_ROOT_REVISION to the deployed clean commit}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/dpsae-runs/20260716/exp10_concept_discovery/pythia160m-block8-s0-pilot-v2}"
MODEL_CACHE="${MODEL_CACHE:-/workspace/dpsae-runs/20260716/exp10_concept_discovery/shared-model-cache-v1}"
CONFIG="${CONFIG:-$ROOT/configs/exp10_concept_discovery.json}"
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
FLEET_LAUNCH_STARTED=0
WORKER_SESSIONS=(exp10-gpu0 exp10-gpu1 exp10-gpu2 exp10-gpu3)

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
      tmux kill-session -t "$session" 2>/dev/null || true
    fi
  done
}

on_error() {
  local code=$?
  trap - ERR
  if [[ "$FLEET_LAUNCH_STARTED" == "1" ]]; then
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
if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$EXPECTED_ROOT_REVISION" ]]; then
  echo "unexpected Exp10 revision in $ROOT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "missing frozen Exp10 config: $CONFIG" >&2
  exit 2
fi

RESOLVED_CONFIG="$OUTPUT_ROOT/resolved_config.json"
if [[ ! -f "$RESOLVED_CONFIG" ]]; then
  echo "missing resolved Exp10 contract: $RESOLVED_CONFIG" >&2
  exit 2
fi
EXPECTED_CONFIG_DIGEST="$(jq -er '.config_digest | select(type == "string" and length == 64)' "$RESOLVED_CONFIG")"
RESOLVED_ROOT_REVISION="$(jq -er '.repository.revision | select(type == "string")' "$RESOLVED_CONFIG")"
EXPECTED_CONFIG_SHA256="$(jq -er '.config_sha256 | select(type == "string" and length == 64)' "$RESOLVED_CONFIG")"
OBSERVED_CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
if [[ "$RESOLVED_ROOT_REVISION" != "$EXPECTED_ROOT_REVISION" ]]; then
  echo "resolved Exp10 contract belongs to revision $RESOLVED_ROOT_REVISION" >&2
  exit 2
fi
if [[ "$OBSERVED_CONFIG_SHA256" != "$EXPECTED_CONFIG_SHA256" ]]; then
  echo "resolved Exp10 contract config hash differs from $CONFIG" >&2
  exit 2
fi
EXPECTED_PROBE_SEED="$(jq -er '.runtime.timing_smoke.probe_seed | select(type == "number")' "$CONFIG")"
EXPECTED_TASK_COUNT="$(jq -er '.runtime.timing_smoke.task_count | select(type == "number")' "$CONFIG")"
EXPECTED_TIMING_TOPOLOGY="$(jq -er '.runtime.timing_smoke.topology_mode | select(type == "string")' "$CONFIG")"
EXPECTED_MEASURED_WORKERS="$(jq -er '.runtime.timing_smoke.measured_worker_count | select(type == "number")' "$CONFIG")"
EXPECTED_MAXIMUM_START_SKEW="$(jq -er '.runtime.timing_smoke.maximum_start_skew_seconds | select(type == "number")' "$CONFIG")"
EXPECTED_MAXIMUM_POD_HOURS="$(jq -er '.runtime.timing_smoke.maximum_projected_pod_hours | select(type == "number")' "$CONFIG")"
EXPECTED_MATRIX_FORMAT="$(jq -er '.benchmark.companion_full_code_matrix_format | select(type == "string")' "$CONFIG")"
EXPECTED_L2_OPTIMIZATION="$(jq -er '.benchmark.companion_l2_path_optimization | select(type == "string")' "$CONFIG")"
EXPECTED_COLD_C_JOBS="$(jq -er '.runtime.companion_full_code_cold_C_jobs_per_worker | select(type == "number")' "$CONFIG")"
EXPECTED_WORKER_COUNT="$(jq -er '.runtime.worker_count | select(type == "number")' "$CONFIG")"
EXPECTED_CGROUP_QUOTA_CORES="$(jq -er '.runtime.resource_identity.cgroup_quota_cores | select(type == "number")' "$CONFIG")"
EXPECTED_EFFECTIVE_CPU_COUNT="$(jq -er '.runtime.resource_identity.effective_cpu_count | select(type == "number")' "$CONFIG")"
EXPECTED_THREADS_PER_WORKER="$(jq -er '.runtime.resource_identity.threads_per_worker | select(type == "number")' "$CONFIG")"

OBSERVED_CPU_BUDGET="$(
  PYTHONPATH="$ROOT/src:$ROOT:$SAEBENCH_ROOT" \
    "$PYTHON" -m dpsae.cpu_quota json --workers "$EXPECTED_WORKER_COUNT"
)"
if ! jq -e \
  --argjson worker_count "$EXPECTED_WORKER_COUNT" \
  --argjson cgroup_quota_cores "$EXPECTED_CGROUP_QUOTA_CORES" \
  --argjson effective_cpu_count "$EXPECTED_EFFECTIVE_CPU_COUNT" \
  --argjson threads_per_worker "$EXPECTED_THREADS_PER_WORKER" \
  '
    .worker_count == $worker_count
    and .cgroup_quota_cores == $cgroup_quota_cores
    and .effective_cpu_count == $effective_cpu_count
    and .threads_per_worker == $threads_per_worker
    and (.visible_cpu_count | type == "number")
    and .visible_cpu_count >= $effective_cpu_count
  ' <<< "$OBSERVED_CPU_BUDGET" >/dev/null; then
  echo "live cgroup CPU budget differs from the frozen Exp10 runtime identity" >&2
  exit 2
fi

mapfile -t WORKER_DONE_PATHS < <(
  jq -r --arg output_root "$OUTPUT_ROOT" '
    .runtime.sparse_worker_shards
    | to_entries[]
    | $output_root
      + "/workers/worker_"
      + (.key | tostring)
      + "_"
      + .value.method
      + "_"
      + (.value.probe_seeds[0] | tostring)
      + "_"
      + (.value.probe_seeds[-1] | tostring)
      + ".json"
  ' "$CONFIG"
)
if [[ "${#WORKER_DONE_PATHS[@]}" -ne "${#WORKER_SESSIONS[@]}" ]]; then
  echo "frozen worker artifact mapping must contain exactly four workers" >&2
  exit 2
fi

TIMING_REPORT="$OUTPUT_ROOT/timing_smoke.json"
write_status "timing_gate" "running" "waiting for blind timing report"
wait_for_artifact "$TIMING_SESSION" "$TIMING_REPORT" "timing_gate"
if ! jq -e \
  --arg config_digest "$EXPECTED_CONFIG_DIGEST" \
  --arg matrix_format "$EXPECTED_MATRIX_FORMAT" \
  --arg l2_optimization "$EXPECTED_L2_OPTIMIZATION" \
  --argjson cold_c_jobs "$EXPECTED_COLD_C_JOBS" \
  --argjson probe_seed "$EXPECTED_PROBE_SEED" \
  --argjson task_count "$EXPECTED_TASK_COUNT" \
  --arg timing_topology "$EXPECTED_TIMING_TOPOLOGY" \
  --argjson measured_workers "$EXPECTED_MEASURED_WORKERS" \
  --argjson maximum_start_skew "$EXPECTED_MAXIMUM_START_SKEW" \
  --argjson maximum_pod_hours "$EXPECTED_MAXIMUM_POD_HOURS" \
  --argjson worker_count "$EXPECTED_WORKER_COUNT" \
  --argjson cgroup_quota_cores "$EXPECTED_CGROUP_QUOTA_CORES" \
  --argjson effective_cpu_count "$EXPECTED_EFFECTIVE_CPU_COUNT" \
  --argjson threads_per_worker "$EXPECTED_THREADS_PER_WORKER" \
  '
    .schema_version == 6
    and .complete == true
    and .config_digest == $config_digest
    and .probe_seed == $probe_seed
    and .task_count == $task_count
    and .measured_worker_count == $measured_workers
    and .measured_task_count == ($measured_workers * $task_count)
    and .topology.mode == $timing_topology
    and .topology.measured_worker_count == $measured_workers
    and .topology.tasks_per_worker == $task_count
    and .topology.same_task_set_per_worker == true
    and .topology.barrier_synchronized == true
    and .barrier.synchronized == true
    and .barrier.maximum_start_skew_seconds == $maximum_start_skew
    and (.barrier.observed_start_skew_seconds | type == "number")
    and .barrier.observed_start_skew_seconds <= $maximum_start_skew
    and (.barrier.ready_reports | length) == $measured_workers
    and (.timing_worker_reports | length) == $measured_workers
    and (.timing_worker_exit_sentinels | length) == $measured_workers
    and (.cgroup_cpu_stat_deltas | length) == $measured_workers
    and .names_and_concept_results_suppressed == true
    and .saved_concept_metric_count == 0
    and .companion_full_code_matrix_format == $matrix_format
    and .companion_l2_path_optimization == $l2_optimization
    and .companion_full_code_cold_C_jobs_per_worker == $cold_c_jobs
    and .runtime_resources.worker_count == $worker_count
    and .runtime_resources.cgroup_quota_cores == $cgroup_quota_cores
    and .runtime_resources.effective_cpu_count == $effective_cpu_count
    and .runtime_resources.threads_per_worker == $threads_per_worker
    and (.runtime_resources.visible_cpu_count | type == "number")
    and .runtime_resources.visible_cpu_count >= $effective_cpu_count
    and .runtime_resources.environment.LOKY_MAX_CPU_COUNT == ($effective_cpu_count | tostring)
    and .runtime_resources.environment.OMP_NUM_THREADS == ($threads_per_worker | tostring)
    and .runtime_resources.environment.MKL_NUM_THREADS == ($threads_per_worker | tostring)
    and .runtime_resources.environment.OPENBLAS_NUM_THREADS == ($threads_per_worker | tostring)
    and .runtime_resources.environment.NUMEXPR_NUM_THREADS == ($threads_per_worker | tostring)
    and .projection.aggregation == "slowest_measured_worker"
    and .projection.initialization_accounting == "maximum_pre_barrier_initialization_added_once"
    and (.projection.maximum_initialization_seconds | type == "number")
    and (.projection.projected_pod_hours | type == "number")
    and (.passed == (.projection.projected_pod_hours <= $maximum_pod_hours))
  ' "$TIMING_REPORT" >/dev/null; then
  write_status "timing_gate" "error" "blind timing report failed schema-v6/config/runtime identity checks"
  exit 1
fi
if ! jq -e '.passed == true' "$TIMING_REPORT" >/dev/null; then
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
  gpu_snapshot="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
  gpu_memory_by_index=()
  while IFS=',' read -r gpu_index gpu_memory; do
    gpu_index="${gpu_index//[[:space:]]/}"
    gpu_memory="${gpu_memory//[[:space:]]/}"
    if [[ ! "$gpu_index" =~ ^[0-3]$ || ! "$gpu_memory" =~ ^[0-9]+$ ]]; then
      echo "invalid four-GPU memory snapshot row: index=$gpu_index memory=$gpu_memory" >&2
      exit 2
    fi
    gpu_memory_by_index[$gpu_index]="$gpu_memory"
  done <<< "$gpu_snapshot"

  gpu_blockers=()
  gpu_summary=()
  for gpu_index in 0 1 2 3; do
    if [[ -z "${gpu_memory_by_index[$gpu_index]:-}" ]]; then
      echo "four-GPU memory snapshot omitted GPU $gpu_index" >&2
      exit 2
    fi
    gpu_memory="${gpu_memory_by_index[$gpu_index]}"
    gpu_summary+=("GPU${gpu_index}=${gpu_memory}MiB")
    if (( gpu_memory > 512 )); then
      gpu_blockers+=("GPU${gpu_index}=${gpu_memory}MiB")
    fi
  done
  if [[ "${#gpu_blockers[@]}" -eq 0 ]]; then
    break
  fi
  blocker_detail="$(IFS=', '; echo "${gpu_blockers[*]}")"
  write_status "gpu_release" "waiting" "blocked_by=$blocker_detail"
  sleep "$POLL_SECONDS"
done
gpu_detail="$(IFS=', '; echo "${gpu_summary[*]}")"
write_status "gpu_release" "passed" "all four A40s available; $gpu_detail"

for session in exp10-gpu0 exp10-gpu1 exp10-gpu2 exp10-gpu3 exp10-finalize; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "refusing to overwrite existing tmux session: $session" >&2
    exit 2
  fi
done

write_status "concept_pilot" "launching" "starting four frozen Exp10 workers and finalizer"
FLEET_LAUNCH_STARTED=1
(
  cd "$ROOT"
  env \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODEL_CACHE="$MODEL_CACHE" \
    CONFIG="$CONFIG" \
    PYTHON="$PYTHON" \
    SAEBENCH_ROOT="$SAEBENCH_ROOT" \
    CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    HF_HOME="$HF_HOME" \
    bash scripts/run_exp10_concept_4xa40.sh
)

FINAL_AUDIT="$OUTPUT_ROOT/artifact_audit_final.json"
deadline="$(( $(date +%s) + MAX_EXP10_SECONDS ))"
while [[ ! -f "$FINAL_AUDIT" ]]; do
  for worker_index in "${!WORKER_SESSIONS[@]}"; do
    worker_done="${WORKER_DONE_PATHS[$worker_index]}"
    if [[ -f "$worker_done" ]]; then
      continue
    fi
    worker_session="${WORKER_SESSIONS[$worker_index]}"
    record="$(session_record "$worker_session")"
    if [[ "$record" == "missing" ]]; then
      write_status "concept_pilot" "error" "$worker_session disappeared before $worker_done"
      false
    fi
    if [[ "${record%% *}" == "1" ]]; then
      write_status "concept_pilot" "error" "$worker_session exited ${record#* } before $worker_done"
      false
    fi
  done
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
