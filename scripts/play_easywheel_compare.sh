#!/usr/bin/env bash
set -euo pipefail

# EasyWheel play script with editable default command speed.
# Usage:
#   bash scripts/play_easywheel_compare.sh
#   CHECKPOINT_FILE=logs/rsl_rl/wf_gensong_easy_wheel_only/<run_dir>/model_1800.pt \
#     bash scripts/play_easywheel_compare.sh
#
# Optional env vars:
#   TASK_ID, VIEWER, NUM_ENVS, DEVICE, PRINT_VEL_INTERVAL

TASK_ID="${TASK_ID:-Mjlab-Gensong-EasyWheel-Flat}"
VIEWER="${VIEWER:-native}"
NUM_ENVS="${NUM_ENVS:-32}"
DEVICE="${DEVICE:-cuda:0}"
PRINT_VEL_INTERVAL="${PRINT_VEL_INTERVAL:-50}"

# Default velocity command.
# Edit these three values directly when you want a different speed command.
CMD_VX="0.35"
CMD_VY="0.00"
CMD_WZ="0.00"

CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

find_latest_checkpoint() {
  local root="logs/rsl_rl/wf_gensong_easy_wheel_only"
  if [[ ! -d "$root" ]]; then
    return 1
  fi
  find "$root" -type f -name 'model_*.pt' -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1 {print $2}'
}

if [[ -z "$CHECKPOINT_FILE" ]]; then
  CHECKPOINT_FILE="$(find_latest_checkpoint || true)"
fi

if [[ -z "$CHECKPOINT_FILE" || ! -f "$CHECKPOINT_FILE" ]]; then
  echo "[ERROR] No checkpoint found for EasyWheel."
  echo "        Please set CHECKPOINT_FILE, e.g.:"
  echo "        CHECKPOINT_FILE=logs/rsl_rl/wf_gensong_easy_wheel_only/<run_dir>/model_1800.pt bash scripts/play_easywheel_compare.sh"
  exit 1
fi

echo "[INFO] EasyWheel play script"
echo "[INFO] task=$TASK_ID viewer=$VIEWER num_envs=$NUM_ENVS device=$DEVICE"
echo "[INFO] checkpoint=$CHECKPOINT_FILE"
echo
echo "[CMD] base_velocity: vx=$CMD_VX vy=$CMD_VY wz=$CMD_WZ"

uv run play "$TASK_ID" \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --viewer "$VIEWER" \
  --num-envs "$NUM_ENVS" \
  --device "$DEVICE" \
  --command-name base_velocity \
  --cmd-vx "$CMD_VX" \
  --cmd-vy "$CMD_VY" \
  --cmd-wz "$CMD_WZ" \
  --print-actual-vel \
  --print-vel-interval "$PRINT_VEL_INTERVAL"

echo
echo "[DONE] Play finished."
