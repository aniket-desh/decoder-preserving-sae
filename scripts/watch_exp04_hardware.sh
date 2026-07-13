#!/usr/bin/env bash
set -uo pipefail

SESSION="${1:-dpsae-exp04-full}"
LOG="${2:-/workspace/dpsae-bootstrap/exp04-hardware.log}"
MEMORY_LIMIT_MIB="${MEMORY_LIMIT_MIB:-43000}"
TEMPERATURE_LIMIT_C="${TEMPERATURE_LIMIT_C:-88}"

mkdir -p "$(dirname "$LOG")"
while tmux has-session -t "$SESSION" 2>/dev/null; do
  read -r utilization memory temperature power < <(
    nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
      --format=csv,noheader,nounits | tr -d ','
  )
  printf '%s utilization_pct=%s memory_mib=%s temperature_c=%s power_w=%s\n' \
    "$(date --iso-8601=seconds)" "$utilization" "$memory" "$temperature" "$power" \
    >> "$LOG"
  if (( memory >= MEMORY_LIMIT_MIB || temperature >= TEMPERATURE_LIMIT_C )); then
    printf '%s safety_stop memory_limit_mib=%s temperature_limit_c=%s\n' \
      "$(date --iso-8601=seconds)" "$MEMORY_LIMIT_MIB" "$TEMPERATURE_LIMIT_C" \
      >> "$LOG"
    tmux send-keys -t "$SESSION" C-c
    exit 2
  fi
  sleep 60
done
printf '%s monitored_session_ended session=%s\n' \
  "$(date --iso-8601=seconds)" "$SESSION" >> "$LOG"
