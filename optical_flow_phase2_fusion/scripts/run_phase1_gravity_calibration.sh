#!/usr/bin/env bash
set -euo pipefail

# Phase 1 Gravity Calibration runner
# Usage:
#   ./run_phase1_gravity_calibration.sh
#   ./run_phase1_gravity_calibration.sh /mavros/imu/data

TOPIC="${1:-/camera/camera/imu}"
LOG_DIR="log/phase1_gravity"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# ROS setup scripts may reference optional vars that are unset under `set -u`.
# Temporarily relax nounset while sourcing.
set +u
source /opt/ros/humble/setup.bash
set -u

mkdir -p "${LOG_DIR}"

echo "[Phase1] Starting gravity calibration"
echo "[Phase1] Topic   : ${TOPIC}"
echo "[Phase1] Log dir : ${LOG_DIR}"
echo "[Phase1] Python  : ${PYTHON_BIN}"
echo "[Phase1] Keep drone still on flat surface..."

if ! ros2 topic list | grep -qx "${TOPIC}"; then
  echo "[Phase1][WARN] Topic ${TOPIC} not found in ROS graph."
  echo "[Phase1][WARN] Start IMU publisher first, for example:"
  echo "[Phase1][WARN] ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true unite_imu_method:=2"
fi

"${PYTHON_BIN}" phase1_gravity_calibration.py \
  --topic "${TOPIC}" \
  --log-dir "${LOG_DIR}" \
  --window-size 120 \
  --settle-std-deg 0.15 \
  --min-settle-seconds 6.0 \
  --no-data-warn-seconds 3.0 \
  --no-data-timeout-seconds 20.0

echo "[Phase1] Done. See logs in ${LOG_DIR}"
