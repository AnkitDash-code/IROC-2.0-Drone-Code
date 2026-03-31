#!/usr/bin/env bash
set -euo pipefail

# Root-only deep test runner for:
# 1) d455_ipm_lk_velocity_node.py
# 2) d455_ipm_lk_deep_test_monitor.py

set +u
source /opt/ros/humble/setup.bash
set -u

IMAGE_TOPIC="${IMAGE_TOPIC:-}"
CAMINFO_TOPIC="${CAMINFO_TOPIC:-}"
RANGE_TOPIC="${RANGE_TOPIC:-/mavros/distance_sensor/rangefinder}"
IMU_TOPIC="${IMU_TOPIC:-}"
TWIST_TOPIC="${TWIST_TOPIC:-/mavros/vision_speed/speed_twist}"
QUALITY_TOPIC="${QUALITY_TOPIC:-/d455_ipm_lk_velocity_node/quality}"

LOG_DIR="${LOG_DIR:-.}"
mkdir -p "${LOG_DIR}"

VEL_LOG="${LOG_DIR}/d455_ipm_lk_velocity_node.log"
MON_LOG="${LOG_DIR}/d455_ipm_lk_deep_test_monitor.log"

if [[ -z "${IMAGE_TOPIC}" || -z "${CAMINFO_TOPIC}" || -z "${IMU_TOPIC}" ]]; then
  TOPICS="$(ros2 topic list || true)"

  if [[ -z "${IMAGE_TOPIC}" ]]; then
    if echo "${TOPICS}" | grep -qx "/camera/color/image_raw"; then
      IMAGE_TOPIC="/camera/color/image_raw"
    elif echo "${TOPICS}" | grep -qx "/camera/camera/color/image_raw"; then
      IMAGE_TOPIC="/camera/camera/color/image_raw"
    else
      IMAGE_TOPIC="/camera/color/image_raw"
    fi
  fi

  if [[ -z "${CAMINFO_TOPIC}" ]]; then
    if echo "${TOPICS}" | grep -qx "/camera/color/camera_info"; then
      CAMINFO_TOPIC="/camera/color/camera_info"
    elif echo "${TOPICS}" | grep -qx "/camera/camera/color/camera_info"; then
      CAMINFO_TOPIC="/camera/camera/color/camera_info"
    else
      CAMINFO_TOPIC="/camera/color/camera_info"
    fi
  fi

  if [[ -z "${IMU_TOPIC}" ]]; then
    if echo "${TOPICS}" | grep -qx "/camera/imu"; then
      IMU_TOPIC="/camera/imu"
    elif echo "${TOPICS}" | grep -qx "/camera/camera/imu"; then
      IMU_TOPIC="/camera/camera/imu"
    else
      IMU_TOPIC="/camera/imu"
    fi
  fi
fi

echo "[DeepTest] Topics:"
echo "  IMAGE_TOPIC=${IMAGE_TOPIC}"
echo "  CAMINFO_TOPIC=${CAMINFO_TOPIC}"
echo "  RANGE_TOPIC=${RANGE_TOPIC}"
echo "  IMU_TOPIC=${IMU_TOPIC}"
echo "  TWIST_TOPIC=${TWIST_TOPIC}"
echo "  QUALITY_TOPIC=${QUALITY_TOPIC}"

echo "[DeepTest] Starting velocity node..."
/usr/bin/python3 d455_ipm_lk_velocity_node.py --ros-args \
  -p image_topic:="${IMAGE_TOPIC}" \
  -p camera_info_topic:="${CAMINFO_TOPIC}" \
  -p range_topic:="${RANGE_TOPIC}" \
  -p output_topic:="${TWIST_TOPIC}" \
  >"${VEL_LOG}" 2>&1 &
PID_VEL=$!

echo "[DeepTest] Starting deep monitor..."
/usr/bin/python3 d455_ipm_lk_deep_test_monitor.py --ros-args \
  -p image_topic:="${IMAGE_TOPIC}" \
  -p range_topic:="${RANGE_TOPIC}" \
  -p imu_topic:="${IMU_TOPIC}" \
  -p twist_topic:="${TWIST_TOPIC}" \
  -p quality_topic:="${QUALITY_TOPIC}" \
  -p save_video:=true \
  -p show_window:=false \
  >"${MON_LOG}" 2>&1 &
PID_MON=$!

echo "[DeepTest] Running. PIDs: vel=${PID_VEL}, monitor=${PID_MON}"
echo "[DeepTest] Logs: ${VEL_LOG}, ${MON_LOG}"
echo "[DeepTest] CSV: ./d455_ipm_lk_deep_test.csv"
echo "[DeepTest] Video: ./d455_ipm_lk_debug.mp4"
echo "[DeepTest] Press Ctrl+C to stop."

cleanup() {
  set +e
  kill "${PID_MON}" >/dev/null 2>&1 || true
  kill "${PID_VEL}" >/dev/null 2>&1 || true
  wait "${PID_MON}" >/dev/null 2>&1 || true
  wait "${PID_VEL}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wait
