#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for dashboard + CS20 bridge + non-ROS ArUco streamer
# Usage: ./run_all.sh start|stop|status|restart|logs

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$BASE_DIR/logs"
RUN_DIR="$BASE_DIR/run"
mkdir -p "$LOG_DIR" "$RUN_DIR"

# Source ROS setup (silently tolerate unset vars)
if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
  set -u
fi

PYBIN_CMD() {
  command -v python3.10 >/dev/null 2>&1 && echo "$(command -v python3.10)" || echo "$(command -v python3 || true)"
}

start_component() {
  name=$1; shift
  pidfile="$RUN_DIR/${name}.pid"
  logfile="$LOG_DIR/${name}.log"
  cmd=("$@")
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name already running (pid $pid)"
      return
    else
      rm -f "$pidfile"
    fi
  fi
  nohup "${cmd[@]}" > "$logfile" 2>&1 &
  echo $! > "$pidfile"
  sleep 0.2
  echo "$name started (pid $(cat "$pidfile")) -> $logfile"
}

stop_component() {
  name=$1
  pidfile="$RUN_DIR/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid $pid)"
      kill "$pid" || true
      sleep 0.5
      if kill -0 "$pid" 2>/dev/null; then
        echo "$name still running; killing -9"
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pidfile"
  else
    echo "$name not running"
  fi
}

case "${1-}" in
  start|restart)
    if [[ "${1-}" == restart ]]; then
      "$0" stop >/dev/null 2>&1 || true
    fi
    echo "Starting dashboard, CS20 bridge, and ArUco streamer..."

    # Dashboard (python3.10 preferred)
    PY=$(PYBIN_CMD)
    if [[ -z "$PY" ]]; then
      echo "No python3.* found on PATH"
      exit 1
    fi
    start_component dashboard "$PY" "$BASE_DIR/web_dashboard.py"

    # CS20 bridge (uses wrapper to pick correct python / source ROS)
    start_component bridge "$BASE_DIR/run_cs20_bridge.sh"

    # Wait for bridge to fully start to avoid SDK/device conflicts
    bridge_log="$LOG_DIR/bridge.log"
    echo "Waiting up to 8s for CS20 bridge to initialize..."
    started=0
    for i in {1..16}; do
      if grep -q "CS20 bridge started" "$bridge_log" 2>/dev/null; then
        started=1
        break
      fi
      sleep 0.5
    done
    if [[ $started -ne 1 ]]; then
      echo "Bridge did not report startup in time; skipping aruco_streamer to avoid device conflict. Check: $bridge_log"
    else
      # ArUco streamer (non-ROS)
      start_component aruco_streamer "$PY" "$BASE_DIR/optical_flow_cs20_lidar/optional_marker_landing/aruco_streamer.py" --dashboard http://localhost:5000/api/upload_frame
    fi

    echo "All components started. Check logs in $LOG_DIR"
    ;;

  stop)
    echo "Stopping all components..."
    stop_component aruco_streamer
    stop_component bridge
    stop_component dashboard
    ;;

  status)
    for n in dashboard bridge aruco_streamer; do
      pidfile="$RUN_DIR/${n}.pid"
      if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
          echo "$n running (pid $pid)"
        else
          echo "$n pidfile exists but not running"
        fi
      else
        echo "$n not running"
      fi
    done
    ;;

  logs)
    tail -n 200 "$LOG_DIR"/*.log || true
    ;;

  *)
    echo "Usage: $0 {start|stop|status|restart|logs}"
    exit 1
    ;;
esac

exit 0
