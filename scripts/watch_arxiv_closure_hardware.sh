#!/usr/bin/env bash
set -uo pipefail

LOG_ROOT="${DPSAE_MONITOR_ROOT:-/workspace/dpsae-runs/20260716/monitor}"
INTERVAL_SECONDS="${DPSAE_MONITOR_INTERVAL_SECONDS:-30}"
DISK_FREE_WARN_GIB="${DPSAE_DISK_FREE_WARN_GIB:-25}"
DISK_FREE_CRITICAL_GIB="${DPSAE_DISK_FREE_CRITICAL_GIB:-10}"
RAM_AVAILABLE_WARN_GIB="${DPSAE_RAM_AVAILABLE_WARN_GIB:-12}"
GPU_MEMORY_WARN_MIB="${DPSAE_GPU_MEMORY_WARN_MIB:-44000}"
GPU_TEMPERATURE_WARN_C="${DPSAE_GPU_TEMPERATURE_WARN_C:-88}"

METRICS_LOG="$LOG_ROOT/hardware.jsonl"
ALERTS_LOG="$LOG_ROOT/alerts.log"
STATUS_FILE="$LOG_ROOT/status.json"

mkdir -p "$LOG_ROOT"

json_escape() {
  sed 's/\\/\\\\/g; s/"/\\"/g'
}

integer_or_zero() {
  local value="${1:-0}"
  value="${value//[^0-9]/}"
  printf '%s' "${value:-0}"
}

oom_count() {
  local text=""
  if text="$(dmesg 2>/dev/null)"; then
    printf '%s\n' "$text" | grep -Eic 'oom-kill|out of memory|killed process' || true
    return
  fi
  if [[ -r /var/log/kern.log ]]; then
    grep -Eic 'oom-kill|out of memory|killed process' /var/log/kern.log || true
    return
  fi
  printf 'unavailable'
}

emit_alert() {
  local timestamp="$1"
  local level="$2"
  local message="$3"
  printf '%s level=%s %s\n' "$timestamp" "$level" "$message" | tee -a "$ALERTS_LOG" >&2
}

while true; do
  timestamp="$(date --iso-8601=seconds)"
  disk_line="$(df -Pk /workspace | awk 'NR==2 {print $2, $3, $4, $5}')"
  read -r disk_total_kib disk_used_kib disk_free_kib disk_used_percent <<< "$disk_line"
  disk_free_gib=$(( $(integer_or_zero "$disk_free_kib") / 1024 / 1024 ))

  mem_available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_free_kib="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  mem_available_gib=$(( $(integer_or_zero "$mem_available_kib") / 1024 / 1024 ))
  swap_free_gib=$(( $(integer_or_zero "$swap_free_kib") / 1024 / 1024 ))
  oom_events="$(oom_count)"
  load_average="$(cut -d' ' -f1-3 /proc/loadavg | json_escape)"
  sessions="$(tmux list-sessions -F '#{session_name}:#{session_windows}:#{session_attached}' 2>/dev/null | sort | paste -sd, - | json_escape)"

  gpu_json=""
  while IFS=',' read -r index name memory_used memory_total utilization temperature power; do
    index="$(integer_or_zero "$index")"
    name="$(printf '%s' "$name" | sed 's/^ *//; s/ *$//' | json_escape)"
    memory_used="$(integer_or_zero "$memory_used")"
    memory_total="$(integer_or_zero "$memory_total")"
    utilization="$(integer_or_zero "$utilization")"
    temperature="$(integer_or_zero "$temperature")"
    power="$(printf '%s' "$power" | sed 's/^ *//; s/ *$//' | json_escape)"
    [[ -n "$gpu_json" ]] && gpu_json+=","
    gpu_json+="{\"index\":$index,\"name\":\"$name\",\"memory_used_mib\":$memory_used,\"memory_total_mib\":$memory_total,\"utilization_percent\":$utilization,\"temperature_c\":$temperature,\"power_w\":\"$power\"}"
    if (( memory_used >= GPU_MEMORY_WARN_MIB )); then
      emit_alert "$timestamp" warning "gpu=$index memory_used_mib=$memory_used threshold_mib=$GPU_MEMORY_WARN_MIB"
    fi
    if (( temperature >= GPU_TEMPERATURE_WARN_C )); then
      emit_alert "$timestamp" warning "gpu=$index temperature_c=$temperature threshold_c=$GPU_TEMPERATURE_WARN_C"
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
  )

  if (( disk_free_gib <= DISK_FREE_CRITICAL_GIB )); then
    emit_alert "$timestamp" critical "disk_free_gib=$disk_free_gib threshold_gib=$DISK_FREE_CRITICAL_GIB"
  elif (( disk_free_gib <= DISK_FREE_WARN_GIB )); then
    emit_alert "$timestamp" warning "disk_free_gib=$disk_free_gib threshold_gib=$DISK_FREE_WARN_GIB"
  fi
  if (( mem_available_gib <= RAM_AVAILABLE_WARN_GIB )); then
    emit_alert "$timestamp" warning "ram_available_gib=$mem_available_gib threshold_gib=$RAM_AVAILABLE_WARN_GIB"
  fi
  if [[ "$oom_events" != unavailable ]] && (( oom_events > 0 )); then
    emit_alert "$timestamp" critical "kernel_oom_event_count=$oom_events"
  fi

  record="{\"timestamp\":\"$timestamp\",\"disk_total_kib\":$(integer_or_zero "$disk_total_kib"),\"disk_used_kib\":$(integer_or_zero "$disk_used_kib"),\"disk_free_kib\":$(integer_or_zero "$disk_free_kib"),\"disk_used_percent\":\"$disk_used_percent\",\"ram_available_kib\":$(integer_or_zero "$mem_available_kib"),\"swap_free_kib\":$(integer_or_zero "$swap_free_kib"),\"kernel_oom_event_count\":\"$oom_events\",\"load_average\":\"$load_average\",\"tmux_sessions\":\"$sessions\",\"gpus\":[$gpu_json]}"
  printf '%s\n' "$record" >> "$METRICS_LOG"
  temporary="${STATUS_FILE}.tmp"
  printf '%s\n' "$record" > "$temporary"
  mv "$temporary" "$STATUS_FILE"
  sleep "$INTERVAL_SECONDS"
done
