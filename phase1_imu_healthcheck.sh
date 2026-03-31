#!/usr/bin/env bash
set -euo pipefail

# Phase 1 IMU Health Check (Jetson)
# Usage:
#   ./phase1_imu_healthcheck.sh
#   ./phase1_imu_healthcheck.sh /camera/camera/imu
#
# What it checks:
# 1) RealSense USB presence
# 2) ROS topic existence
# 3) Publisher count on IMU topic
# 4) Non-zero IMU rate

TOPIC="${1:-/camera/camera/imu}"
HZ_TIMEOUT_SEC="${HZ_TIMEOUT_SEC:-8}"

PASS=0
FAIL=0

pass() {
  echo "[PASS] $*"
  PASS=$((PASS + 1))
}

fail() {
  echo "[FAIL] $*"
  FAIL=$((FAIL + 1))
}

info() {
  echo "[INFO] $*"
}

info "Phase 1 IMU Health Check"
info "Target topic: ${TOPIC}"

# ROS setup with nounset disabled during source (ROS scripts may use unset vars)
set +u
source /opt/ros/humble/setup.bash
set -u

# 1) USB device check
if lsusb | grep -Eiq "realsense|intel"; then
  pass "RealSense/Intel USB device detected by lsusb"
else
  fail "No RealSense/Intel USB device detected (check USB 3.0 cable/port)"
fi

# Optional detailed RealSense probe if tool is available
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  if rs-enumerate-devices >/dev/null 2>&1; then
    pass "rs-enumerate-devices succeeded"
  else
    fail "rs-enumerate-devices failed (camera/permissions issue possible)"
  fi
else
  info "rs-enumerate-devices not installed; skipping detailed probe"
fi

# 2) Topic existence
if ros2 topic list | grep -qx "${TOPIC}"; then
  pass "Topic exists: ${TOPIC}"
else
  fail "Topic not found: ${TOPIC}. Start IMU publisher first."
fi

# 3) Topic info / publisher count
PUB_COUNT=0
if ros2 topic list | grep -qx "${TOPIC}"; then
  TOPIC_INFO="$(ros2 topic info "${TOPIC}" 2>/dev/null || true)"
  PUB_COUNT="$(echo "${TOPIC_INFO}" | awk -F': ' '/Publisher count/ {print $2}' | tr -d '\r' | head -n1)"
  PUB_COUNT="${PUB_COUNT:-0}"

  if [[ "${PUB_COUNT}" =~ ^[0-9]+$ ]] && [[ "${PUB_COUNT}" -ge 1 ]]; then
    pass "Publisher count is ${PUB_COUNT}"
  else
    fail "Publisher count is ${PUB_COUNT} (expected >= 1)"
  fi
else
  info "Skipping publisher count (topic missing)"
fi

# 4) Non-zero Hz check (bounded runtime)
HZ_OUTPUT="$(timeout "${HZ_TIMEOUT_SEC}" ros2 topic hz "${TOPIC}" --window 20 2>&1 || true)"
HZ_LINE="$(echo "${HZ_OUTPUT}" | grep -E "average rate:" | tail -n1 || true)"
HZ_VALUE="$(echo "${HZ_LINE}" | awk '{print $3}' | tr -d '\r' || true)"

if [[ -n "${HZ_VALUE}" ]]; then
  # numeric and > 0
  if awk -v hz="${HZ_VALUE}" 'BEGIN {exit !(hz+0 > 0)}'; then
    pass "IMU publish rate detected: ${HZ_VALUE} Hz"
  else
    fail "IMU rate is zero/unusable: ${HZ_VALUE}"
  fi
else
  if echo "${HZ_OUTPUT}" | grep -qi "does not appear to be published yet"; then
    fail "No IMU messages observed on ${TOPIC}"
  else
    fail "Could not determine IMU rate (timeout ${HZ_TIMEOUT_SEC}s)"
  fi
fi

echo ""
echo "========== SUMMARY =========="
echo "PASS: ${PASS}"
echo "FAIL: ${FAIL}"

if [[ "${FAIL}" -eq 0 ]]; then
  echo "RESULT: HEALTH CHECK PASSED"
  echo "Next: run calibration"
  echo "  cd /home/jetson123/Drone"
  echo "  ./run_phase1_gravity_calibration.sh ${TOPIC}"
  exit 0
fi

echo "RESULT: HEALTH CHECK FAILED"
echo "Suggested fix order:"
echo "1) Start camera IMU publisher:"
echo "   ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true unite_imu_method:=2"
echo "2) Re-run this check"
echo "3) If still failing, unplug/replug camera and retry"
exit 1
