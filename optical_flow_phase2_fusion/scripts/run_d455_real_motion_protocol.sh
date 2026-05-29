#!/usr/bin/env bash
set -euo pipefail

# Real-motion protocol for D455 IPM+LK velocity validation.
# Phases:
# 1) static hold
# 2) forward translation
# 3) tilt-in-place

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/protocol_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

RUN_LOG="${OUT_DIR}/run_terminal.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

set +u
source /opt/ros/humble/setup.bash
set -u

IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/camera/color/image_raw}"
CAMINFO_TOPIC="${CAMINFO_TOPIC:-/camera/camera/color/camera_info}"
RANGE_TOPIC="${RANGE_TOPIC:-/mavros/distance_sensor/rangefinder}"
IMU_TOPIC="${IMU_TOPIC:-/camera/camera/imu}"
TWIST_TOPIC="${TWIST_TOPIC:-/mavros/vision_speed/speed_twist}"
QUALITY_TOPIC="${QUALITY_TOPIC:-/d455_ipm_lk_velocity_node/quality}"
REQUIRE_LIVE_RANGE="${REQUIRE_LIVE_RANGE:-0}"
REQUIRE_IMU="${REQUIRE_IMU:-0}"
SHOW_WINDOW="${SHOW_WINDOW:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
LOG_PERIOD_S="${LOG_PERIOD_S:-0.5}"

DUR_STATIC="${DUR_STATIC:-12}"
DUR_TRANSLATE="${DUR_TRANSLATE:-12}"
DUR_TILT="${DUR_TILT:-12}"

echo "[protocol] output: ${OUT_DIR}"
echo "[protocol] terminal transcript: ${RUN_LOG}"
echo "[protocol] checking range topic activity..."

DUMMY_RANGE_PID=""
if ! timeout 2 ros2 topic echo "${RANGE_TOPIC}" --once >/dev/null 2>&1; then
  if [[ "${REQUIRE_LIVE_RANGE}" == "1" ]]; then
    echo "[protocol] ERROR: live range required but no data on ${RANGE_TOPIC}" >&2
    exit 3
  fi
  echo "[protocol] no active range input on ${RANGE_TOPIC}; starting temporary 1.0 m publisher"
  ros2 topic pub -r 20 "${RANGE_TOPIC}" sensor_msgs/msg/Range "{header: {frame_id: base_link}, radiation_type: 1, field_of_view: 0.0, min_range: 0.1, max_range: 10.0, range: 1.0}" \
    >"${OUT_DIR}/dummy_range.log" 2>&1 &
  DUMMY_RANGE_PID=$!
else
  echo "[protocol] live range topic detected"
fi

echo "[protocol] checking imu topic activity..."
if ! timeout 2 ros2 topic echo "${IMU_TOPIC}" --once >/dev/null 2>&1; then
  if [[ "${REQUIRE_IMU}" == "1" ]]; then
    echo "[protocol] ERROR: live IMU required but no data on ${IMU_TOPIC}" >&2
    exit 4
  fi
  echo "[protocol] warning: no IMU data on ${IMU_TOPIC}; pitch diagnostics may be unavailable"
else
  echo "[protocol] live IMU topic detected"
fi

cleanup() {
  set +e
  [[ -n "${MON_PID:-}" ]] && kill "${MON_PID}" >/dev/null 2>&1 || true
  [[ -n "${VEL_PID:-}" ]] && kill "${VEL_PID}" >/dev/null 2>&1 || true
  [[ -n "${DUMMY_RANGE_PID}" ]] && kill "${DUMMY_RANGE_PID}" >/dev/null 2>&1 || true
  wait "${MON_PID:-}" >/dev/null 2>&1 || true
  wait "${VEL_PID:-}" >/dev/null 2>&1 || true
  wait "${DUMMY_RANGE_PID:-}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

run_phase() {
  local phase="$1"
  local seconds="$2"
  local csv_path="${OUT_DIR}/${phase}.csv"
  local video_path="${OUT_DIR}/${phase}_debug.mp4"

  echo "[protocol] phase=${phase} duration=${seconds}s"
  echo "[protocol] csv=${csv_path}"
  if [[ "${SAVE_VIDEO}" == "1" ]]; then
    echo "[protocol] video=${video_path}"
  fi

  /usr/bin/python3 "${ROOT_DIR}/d455_ipm_lk_velocity_node.py" --ros-args \
    -p image_topic:="${IMAGE_TOPIC}" \
    -p camera_info_topic:="${CAMINFO_TOPIC}" \
    -p range_topic:="${RANGE_TOPIC}" \
    -p output_topic:="${TWIST_TOPIC}" \
    -p process_hz:=15.0 \
    -p max_dt_s:=0.50 \
    -p ipm_blur_kernel:=7 \
    -p feature_quality_level:=0.02 \
    -p feature_min_distance:=10 \
    -p lk_win_size:=21 \
    -p stationary_flow_px:=0.35 \
    -p flow_mad_k:=3.0 \
    > >(tee -a "${OUT_DIR}/${phase}_vel.log") 2>&1 &
  VEL_PID=$!

  /usr/bin/python3 "${ROOT_DIR}/d455_ipm_lk_deep_test_monitor.py" --ros-args \
    -p image_topic:="${IMAGE_TOPIC}" \
    -p range_topic:="${RANGE_TOPIC}" \
    -p imu_topic:="${IMU_TOPIC}" \
    -p twist_topic:="${TWIST_TOPIC}" \
    -p quality_topic:="${QUALITY_TOPIC}" \
    -p save_video:=$([[ "${SAVE_VIDEO}" == "1" ]] && echo true || echo false) \
    -p show_window:=$([[ "${SHOW_WINDOW}" == "1" ]] && echo true || echo false) \
    -p log_period_s:="${LOG_PERIOD_S}" \
    -p video_path:="${video_path}" \
    -p csv_path:="${csv_path}" \
    > >(tee -a "${OUT_DIR}/${phase}_monitor.log") 2>&1 &
  MON_PID=$!

  echo "[operator] perform now: ${phase}"
  sleep "${seconds}"

  kill "${MON_PID}" >/dev/null 2>&1 || true
  kill "${VEL_PID}" >/dev/null 2>&1 || true
  wait "${MON_PID}" >/dev/null 2>&1 || true
  wait "${VEL_PID}" >/dev/null 2>&1 || true

  MON_PID=""
  VEL_PID=""

  if [[ -s "${csv_path}" ]]; then
    /usr/bin/python3 "${ROOT_DIR}/plot_d455_ipm_lk_deep_test.py" \
      --csv "${csv_path}" \
      --out "${OUT_DIR}/${phase}_plot.png" || true
  else
    echo "[protocol] warning: csv missing or empty for ${phase}, skipping plot"
  fi
}

run_phase "phase1_static" "${DUR_STATIC}"
run_phase "phase2_translate" "${DUR_TRANSLATE}"
run_phase "phase3_tilt" "${DUR_TILT}"

echo "[protocol] complete"
/usr/bin/python3 "${ROOT_DIR}/score_d455_real_motion_protocol.py" "${OUT_DIR}" | tee "${OUT_DIR}/score.txt"
echo "[protocol] score report: ${OUT_DIR}/score.txt"
echo "[protocol] done. all artifacts in: ${OUT_DIR}"
