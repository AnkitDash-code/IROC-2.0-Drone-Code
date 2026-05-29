#!/usr/bin/env bash
set -euo pipefail

# Jetson-only simple test wrapper (no MAVROS required).
# It delegates to run_d455_real_motion_protocol.sh with local topics and
# relaxed gates so you can test translation/tilt quickly.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer common RealSense topic names; override from shell if needed.
export IMAGE_TOPIC="${IMAGE_TOPIC:-/camera/camera/color/image_raw}"
export CAMINFO_TOPIC="${CAMINFO_TOPIC:-/camera/camera/color/camera_info}"

# Prefer fused IMU when present; keep user-provided IMU_TOPIC untouched.
if [[ -z "${IMU_TOPIC+x}" ]]; then
	IMU_TOPIC="/camera/camera/imu"
	if command -v ros2 >/dev/null 2>&1; then
		TOPICS="$(ros2 topic list 2>/dev/null || true)"
		if echo "${TOPICS}" | grep -qx "/imu/data"; then
			IMU_TOPIC="/imu/data"
		fi
	fi
fi
export IMU_TOPIC

# Use a local non-MAVROS range topic; protocol will auto-publish a dummy range
# when no live range source exists and REQUIRE_LIVE_RANGE=0.
export RANGE_TOPIC="${RANGE_TOPIC:-/d455/test/range}"

# Use local output topics for visibility in Jetson-only tests.
export TWIST_TOPIC="${TWIST_TOPIC:-/d455/ipm_lk/velocity}"
export QUALITY_TOPIC="${QUALITY_TOPIC:-/d455_ipm_lk_velocity_node/quality}"

# Jetson-only defaults: no strict live range/imu requirement.
export REQUIRE_LIVE_RANGE="${REQUIRE_LIVE_RANGE:-0}"
export REQUIRE_IMU="${REQUIRE_IMU:-0}"

# Keep logging enabled, and only show OpenCV window when display is available.
if [[ -z "${SHOW_WINDOW+x}" ]]; then
	if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
		SHOW_WINDOW="1"
	else
		SHOW_WINDOW="0"
	fi
fi

if [[ "${SHOW_WINDOW}" == "1" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
	echo "[jetson-test] No GUI display found; forcing SHOW_WINDOW=0 to prevent OpenCV Qt crash."
	SHOW_WINDOW="0"
fi

export SHOW_WINDOW
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export LOG_PERIOD_S="${LOG_PERIOD_S:-0.5}"

# Motion phase durations (seconds).
export DUR_STATIC="${DUR_STATIC:-12}"
export DUR_TRANSLATE="${DUR_TRANSLATE:-12}"
export DUR_TILT="${DUR_TILT:-12}"

exec bash "${ROOT_DIR}/run_d455_real_motion_protocol.sh"
