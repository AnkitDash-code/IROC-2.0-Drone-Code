#!/usr/bin/env bash
set -euo pipefail

# Auto-detect Pixhawk/Cube serial endpoint and start MAVProxy.
# Priority:
# 1) /dev/serial/by-id/*Cube*if00, *Pixhawk*if00, *Holybro*if00, then generic *if00
# 2) /dev/ttyACM*
# 3) /dev/ttyUSB*
#
# Usage:
#   ./run_mavproxy.sh [--master /dev/ttyACM1] [--baudrate 57600] [--dry-run] [-- extra mavproxy args]

BAUDRATE="57600"
MASTER_OVERRIDE=""
DRY_RUN=0
OUTS=("udp:127.0.0.1:14550" "udp:127.0.0.1:14551")
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_mavproxy.sh [options] [-- extra mavproxy args]

Options:
  --master PATH         Override auto-detected serial device path.
  --baudrate N          Serial baudrate (default: 57600).
  --out DEST            Add MAVProxy --out destination (can be repeated).
  --dry-run             Print detected values and command, do not execute.
  --help, -h            Show this help.

Examples:
  ./run_mavproxy.sh
  ./run_mavproxy.sh --baudrate 115200
  ./run_mavproxy.sh --out udp:127.0.0.1:14552 -- --aircraft mylog
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --master)
      MASTER_OVERRIDE="${2-}"
      shift 2
      ;;
    --baudrate)
      BAUDRATE="${2-}"
      shift 2
      ;;
    --out)
      OUTS+=("${2-}")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

has_glob_matches() {
  local pattern="$1"
  compgen -G "$pattern" >/dev/null 2>&1
}

pick_by_id() {
  local patterns=(
    "/dev/serial/by-id/*Cube*if00"
    "/dev/serial/by-id/*Pixhawk*if00"
    "/dev/serial/by-id/*Holybro*if00"
    "/dev/serial/by-id/*if00"
  )
  local p
  for p in "${patterns[@]}"; do
    if has_glob_matches "$p"; then
      ls -1 $p 2>/dev/null | head -n 1
      return 0
    fi
  done
  return 1
}

pick_serial_port() {
  local by_id=""
  by_id="$(pick_by_id || true)"
  if [[ -n "$by_id" ]]; then
    echo "$by_id"
    return 0
  fi

  if has_glob_matches "/dev/ttyACM*"; then
    ls -1 /dev/ttyACM* 2>/dev/null | sort | head -n 1
    return 0
  fi

  if has_glob_matches "/dev/ttyUSB*"; then
    ls -1 /dev/ttyUSB* 2>/dev/null | sort | head -n 1
    return 0
  fi

  return 1
}

MASTER="${MASTER_OVERRIDE}"
if [[ -z "$MASTER" ]]; then
  MASTER="$(pick_serial_port || true)"
fi

if [[ -z "$MASTER" ]]; then
  echo "No Pixhawk/Cube serial device found."
  echo "Checked: /dev/serial/by-id/*if00, /dev/ttyACM*, /dev/ttyUSB*"
  exit 1
fi

if systemctl is-active ModemManager >/dev/null 2>&1; then
  echo "Warning: ModemManager is active and may interfere with Pixhawk/Cube serial ports."
  echo "Suggested: sudo systemctl stop ModemManager && sudo systemctl disable ModemManager"
fi

if ! command -v mavproxy.py >/dev/null 2>&1; then
  echo "mavproxy.py not found on PATH."
  exit 1
fi

CMD=("mavproxy.py" "--master=${MASTER}" "--baudrate=${BAUDRATE}")
for out in "${OUTS[@]}"; do
  CMD+=("--out" "$out")
done
CMD+=("--daemon")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Using master: ${MASTER}"
echo "Using baudrate: ${BAUDRATE}"
echo "Launching: ${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

exec "${CMD[@]}"
