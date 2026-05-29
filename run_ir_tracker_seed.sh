#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/jetson123/Drone"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-drone}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:5000/api/upload_frame}"
MAVLINK_URL="${MAVLINK_URL:-udp:127.0.0.1:14550}"
SEED_IMAGE="${SEED_IMAGE:-/home/jetson123/Drone/aruco_tags/base_seed_1p5m.jpg}"
SEED_YAW_SIGN="${SEED_YAW_SIGN:-1.0}"
SEED_YAW_DEADBAND_DEG="${SEED_YAW_DEADBAND_DEG:-3.0}"
RANGEFINDER_RATE_HZ="${RANGEFINDER_RATE_HZ:-20.0}"
RANGEFINDER_MAX_AGE_S="${RANGEFINDER_MAX_AGE_S:-0.25}"
RANGEFINDER_POLL_TIMEOUT="${RANGEFINDER_POLL_TIMEOUT:-0.0}"
RANGEFINDER_MAX_MSGS="${RANGEFINDER_MAX_MSGS:-240}"

if [[ ! -d "$REPO_DIR" ]]; then
    echo "[ERROR] Repo directory not found: $REPO_DIR"
    exit 1
fi
cd "$REPO_DIR"

if ! command -v conda >/dev/null 2>&1; then
    if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
        export PATH="$HOME/miniconda3/bin:$PATH"
    elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
        export PATH="$HOME/anaconda3/bin:$PATH"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found in PATH."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

CMD=(
    python3 SynexensPythonSDK4_4.2.4.0_202504281506/ir_tracker.py
    --source uvc
    --video-device "$VIDEO_DEVICE"
    --uvc-flip hv
    --dashboard "$DASHBOARD_URL"
    --interval 0.02
    --camera-feed ir
    --image-mode clahe
    --track-alt-min-m 1.5
    --sticky-lock-forever
    --blob-min-area 50
    --ema-alpha 0.30
    --lock-max-jump-px 100
    --lock-hold-frames 20
    --enable-rate-limit
    --rate-limit-ms 120
    --auto-exposure-threshold
    --auto-threshold-percentile 99.2
    --capture-base-on-lock
    --base-lock-frames 8
    --hover-ema-alpha 0.2
    --imu-fusion
    --imu-fusion-alpha 0.92
    --post-timeout 0.1
    --post-retries 1
    --rangefinder-connection "$MAVLINK_URL"
    --rangefinder-wait-s 8
    --rangefinder-rate-hz "$RANGEFINDER_RATE_HZ"
    --rangefinder-max-age-s "$RANGEFINDER_MAX_AGE_S"
    --rangefinder-poll-timeout "$RANGEFINDER_POLL_TIMEOUT"
    --rangefinder-max-msgs "$RANGEFINDER_MAX_MSGS"
    --imu-source mavlink
    --imu-connection "$MAVLINK_URL"
    --imu-wait-s 5
    --lock-template-threshold 0.40
    --lock-seed-area-min-ratio 0.45
    --lock-seed-area-max-ratio 2.2
    --imu-fusion-filter kalman
    --imu-kf-q-angle 0.02
    --imu-kf-q-bias 0.003
    --imu-kf-r-measure 0.6
    --seed-image "$SEED_IMAGE"
    --seed-label base_1p5m
    --seed-orb-min-good 10
    --seed-orb-min-norm 0.01
    --seed-match-stride 1
    --seed-align-min-inliers 6
    --seed-align-deadband-px 14
    --seed-align-ransac-reproj 6
    --seed-yaw-sign "$SEED_YAW_SIGN"
    --seed-yaw-deadband-deg "$SEED_YAW_DEADBAND_DEG"
)

# Optional strict blob confirmation for seed match.
if [[ "${REQUIRE_SEED_BLOB:-0}" == "1" ]]; then
    CMD+=(--seed-require-blob --seed-blob-sim-threshold "${SEED_BLOB_SIM_THRESHOLD:-0.15}")
fi

# Pass any extra CLI args through to ir_tracker.py.
if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
fi

echo "[INFO] Activated conda env: $CONDA_ENV_NAME"
echo "[INFO] Running: ${CMD[*]}"
exec "${CMD[@]}"
