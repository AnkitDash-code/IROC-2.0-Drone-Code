#!/usr/bin/env bash
# Wrapper to run the CS20 bridge using a Python interpreter compatible with ROS2 Humble (Python 3.10).
# It prefers system Python binaries and supports --python and --dry-run for testing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$REPO_ROOT/optical_flow_cs20_lidar/cs20_lidar_bridge.py"

usage() {
  cat <<EOF
Usage: $0 [--python /path/to/python3.10] [--dry-run] [--] [args...]
Runs the CS20 bridge with a Python interpreter compatible with ROS2 Humble (Python 3.10).
--python PATH   Use the specified Python binary.
--dry-run       Print chosen python and exit (don't run the bridge).
Any remaining arguments are passed to the bridge script.
EOF
}

PYTHON_OVERRIDE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage; exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

# Source ROS setup if present (adds ROS packages to PYTHONPATH for the matching interpreter)
if [[ -f /opt/ros/humble/setup.bash ]]; then
  # Some setup scripts reference unset variables; disable 'set -u' temporarily
  set +u
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash
  set -u
fi

candidates=()
if [[ -n "$PYTHON_OVERRIDE" ]]; then
  candidates+=("$PYTHON_OVERRIDE")
fi
# prefer explicit system paths before generic names
candidates+=("/usr/bin/python3.10" "/usr/local/bin/python3.10" "/usr/bin/python3" "python3.10" "python3")

choose_python() {
  for p in "${candidates[@]}"; do
    if ! command -v "$p" >/dev/null 2>&1 && [[ ! -x "$p" ]]; then
      continue
    fi
    if [[ -x "$p" ]]; then
      py="$p"
    else
      py="$(command -v "$p" 2>/dev/null || true)"
      [[ -z "$py" ]] && continue
    fi
    ver="$("$py" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
    if [[ "$ver" == "3.10" ]]; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

PYBIN="$(choose_python || true)"
if [[ -z "$PYBIN" ]]; then
  echo "No Python 3.10 interpreter found."
  echo "Install python3.10 or pass --python /path/to/python3.10"
  exit 1
fi

echo "Using python: $PYBIN"
if [[ $DRY_RUN -eq 1 ]]; then
  exit 0
fi

exec "$PYBIN" "$SCRIPT" "$@"
