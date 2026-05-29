# Jetson D455 Simple Test Runbook

This runbook is for Jetson-only testing of D455 IPM + LK velocity.
No MAVROS is required.

## 1. Open 3 terminals

Terminal A: Camera stream
Terminal B: Optional IMU filter and topic checks
Terminal C: Test execution

## 2. Terminal A - Start ROS and RealSense camera

cd /home/jetson123/Drone
source /opt/ros/humble/setup.bash

# Start D455 color + depth + IMU
ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 align_depth.enable:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2 depth_module.decimation_filter.enable:=true

Keep this terminal running.

## 3. Terminal B - Optional IMU filter and topic validation

cd /home/jetson123/Drone
source /opt/ros/humble/setup.bash

# Optional but useful for stable orientation diagnostics
ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -r /imu/data_raw:=/camera/camera/imu -p use_mag:=false -p publish_tf:=false

If you do not want IMU filtering, skip the command above and just run checks below in a new terminal.

Topic checks (run one by one):

ros2 topic list | grep camera
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/imu
ros2 topic echo /camera/camera/color/camera_info --once

## 4. Terminal C - Run simple test protocol (no MAVROS)

cd /home/jetson123/Drone
source /opt/ros/humble/setup.bash

# If first time only
chmod +x run_d455_jetson_simple_test.sh

# Run complete protocol: static -> translate -> tilt
bash ./run_d455_jetson_simple_test.sh

This command automatically:
- Runs phase1_static, phase2_translate, phase3_tilt
- Shows live metrics in terminal
- Saves per-phase CSV logs
- Saves per-phase monitor and velocity logs
- Saves per-phase debug videos
- Saves per-phase plots
- Saves full terminal transcript
- Saves final score report

## 5. Operator actions during phases

When prompted with phase1_static:
- Keep drone/device stationary.

When prompted with phase2_translate:
- Move in a straight line (translation), keep tilt small.

When prompted with phase3_tilt:
- Tilt in place (pitch/roll), avoid translation.

## 6. Output location after every run

A new folder is created automatically:

/home/jetson123/Drone/protocol_YYYYMMDD_HHMMSS

Main artifacts inside that folder:
- run_terminal.log
- phase1_static.csv
- phase2_translate.csv
- phase3_tilt.csv
- phase1_static_debug.mp4
- phase2_translate_debug.mp4
- phase3_tilt_debug.mp4
- phase1_static_plot.png
- phase2_translate_plot.png
- phase3_tilt_plot.png
- phase1_static_monitor.log
- phase2_translate_monitor.log
- phase3_tilt_monitor.log
- phase1_static_vel.log
- phase2_translate_vel.log
- phase3_tilt_vel.log
- score.txt

## 7. Useful command variants

Run with longer phase duration (20 seconds each):

DUR_STATIC=20 DUR_TRANSLATE=20 DUR_TILT=20 bash ./run_d455_jetson_simple_test.sh

Disable on-screen window, keep video recording:

SHOW_WINDOW=0 SAVE_VIDEO=1 bash ./run_d455_jetson_simple_test.sh

Use strict live IMU requirement:

REQUIRE_IMU=1 bash ./run_d455_jetson_simple_test.sh

Use strict live IMU with explicit D455 IMU topic:

REQUIRE_IMU=1 IMU_TOPIC=/camera/camera/imu bash ./run_d455_jetson_simple_test.sh

Use strict live IMU with drone IMU topic (example):

REQUIRE_IMU=1 IMU_TOPIC=/drone/imu bash ./run_d455_jetson_simple_test.sh

Use custom camera topics (if your topic names differ):

IMAGE_TOPIC=/camera/color/image_raw CAMINFO_TOPIC=/camera/color/camera_info IMU_TOPIC=/camera/imu bash ./run_d455_jetson_simple_test.sh

## 8. Stop all test-related processes

# Stop RealSense launch
pkill -f realsense2_camera

# Stop IMU filter (if running)
pkill -f imu_filter_madgwick

# Stop protocol (if still running)
pkill -f run_d455_real_motion_protocol.sh
pkill -f d455_ipm_lk_velocity_node.py
pkill -f d455_ipm_lk_deep_test_monitor.py

## 9. Quick troubleshooting

If no camera image:
- Check cable and power, then run:
  ros2 topic list | grep /camera/camera/color/image_raw

If no IMU data:
- Check:
  ros2 topic list | grep /camera/camera/imu

## 10. What "real IMU" means

Real IMU means a live ROS IMU stream on the topic configured by IMU_TOPIC.
It is not dummy data and not a stale/inactive topic.

How the test checks this:
- REQUIRE_IMU=1 makes the protocol fail fast if no message arrives on IMU_TOPIC.

Recommended IMU source for this IPM+LK pipeline:
- Prefer D455 IMU (/camera/camera/imu), because it is physically mounted with the camera view used by optical flow.

Using drone flight-controller IMU:
- Possible if you set IMU_TOPIC to that stream.
- Only use it for tilt compensation if camera-to-body frame alignment and timing are well calibrated.
- If not calibrated, tilt leakage metrics may become misleading.

If protocol exits early:
- Open the latest run folder and inspect:
  run_terminal.log
  phase*_monitor.log
  phase*_vel.log

## 11. Run Phase 2 With GUI (Proper Sequence)

Use this when you want full Phase 2 runtime plus web dashboard visualization.

### A) Open 3 terminals

Terminal A: Camera + IMU source
Terminal B: Optional IMU filter
Terminal C: Phase 2 runtime

### B) Terminal A - Start RealSense pipeline

cd /home/jetson123/Drone
source /opt/ros/humble/setup.bash

ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 align_depth.enable:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2 depth_module.decimation_filter.enable:=true

Keep this terminal running.

### C) Terminal B - Optional Madgwick filter

cd /home/jetson123/Drone
source /opt/ros/humble/setup.bash

ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -r /imu/data_raw:=/camera/camera/imu -p use_mag:=false -p publish_tf:=false

If you do not need Madgwick, skip this terminal.

### D) Terminal C - Run Phase 2 node

cd /home/jetson123/Drone
source /opt/ros/humble/setup.bash

/usr/bin/python3 phase2_standalone.py

Why `/usr/bin/python3`:
- ROS Humble here is built for Python 3.10.
- If Conda/Miniforge Python 3.12 is active, `rclpy` import fails with
  `No module named rclpy._rclpy_pybind11`.

If your shell shows `(base)` (Conda active), run this first:

conda deactivate
source /opt/ros/humble/setup.bash
/usr/bin/python3 phase2_standalone.py

What happens automatically:
- MAVROS is launched
- Phase 2 node starts subscribing/publishing
- Flask web dashboard is started on port 5000

### E) Open GUI

From local machine:
- http://localhost:5000

From another device on same network:
- http://<jetson_ip>:5000

In terminal output, use the printed IP line from Phase 2 startup.

### F) Quick pre-flight checks in GUI

- FPS is stable (no long stalls)
- Keypoint count is non-zero and not collapsing
- Route trace updates while moving
- Bin readiness increases over time
- Relocalization messages appear during rotation

### G) Useful run variants

Run fusion script (recommended for error-reduction experiments):

/usr/bin/python3 phase2_fusion_standalone.py

This keeps the same GUI and MAVROS flow, but blends two velocity methods online.

Run AI frontend (if configured on your Jetson):

USE_AI_FRONTEND=1 python3 phase2_standalone.py

Force MSCKF backend:

VO_BACKEND=MSCKF python3 phase2_standalone.py

Auto backend selection:

VO_BACKEND=AUTO python3 phase2_standalone.py

### H) If GUI does not open

1. Check process logs for dashboard startup line mentioning port 5000.
2. Verify the port is listening:

  ss -ltnp | grep :5000

3. Confirm no stale process is occupying port 5000:

  pkill -f phase2_standalone.py
  pkill -f web_dashboard.py

4. Restart Terminal C command.

### H2) If you get `rclpy._rclpy_pybind11` import error

Cause:
- Wrong Python interpreter (usually Conda Python 3.12) is being used.

Quick fix:

which python3
/usr/bin/python3 -V

Expected:
- `/usr/bin/python3`
- `Python 3.10.x`

Then run:

source /opt/ros/humble/setup.bash
/usr/bin/python3 phase2_standalone.py

### I) Clean stop

Press Ctrl+C in Terminal C first, then stop RealSense/Madgwick terminals.

## 12. Manual Live Validator (Phase1 + Phase2) + GUI

If you are testing manually and do not want to prepare any protocol folder,
use `phase12_manual_live_suite.py`.

This script:
- Captures live samples from your running Phase 2 stack
- Guides you through Phase1 (static) and Phase2 (translate)
- Saves CSV files automatically in a new run folder
- Computes pass/fail and estimated error report
- Optionally opens GUI for easier interpretation

### A) Start your Phase 2 pipeline first

Run your normal Phase 2 node so these topics are active:
- `/mavros/vision_speed/speed_twist`
- `/camera/camera/imu`

Optional (recommended):
- Keep Phase 2 web dashboard active at `http://127.0.0.1:5000/api/state`
  so quality/tracked-flow metrics are included.

### B) Run manual live test (interactive)

```bash
cd /home/jetson123/Drone
/usr/bin/python3 phase12_manual_live_suite.py
```

If your shell is in Conda `(base)`, run:

```bash
conda deactivate
source /opt/ros/humble/setup.bash
/usr/bin/python3 phase12_manual_live_suite.py
```

Interactive flow:
1. Press Enter to start `phase1_static` and hold still.
2. Press Enter again to stop `phase1_static`.
3. Press Enter to start `phase2_translate` and move in straight translation.
4. Press Enter again to stop `phase2_translate`.

### C) Output files (auto-created)

A new folder is created automatically:

`/home/jetson123/Drone/manual_phase12_YYYYMMDD_HHMMSS`

Inside it:
- `phase1_static.csv`
- `phase2_translate.csv`
- `phase12_report.txt`
- `phase12_report.json`

### D) Launch GUI after manual capture

```bash
/usr/bin/python3 phase12_manual_live_suite.py --gui --host 0.0.0.0 --port 5060
```

Open:
- `http://<jetson_ip>:5060`
- `http://localhost:5060`

GUI features:
- Select any `manual_phase12_*` or `protocol_*` run
- Per-test PASS/FAIL for phase1 and phase2
- Estimated error fields (drift, RMS, p95/p99, cross-axis leakage)

### E) Useful options

```bash
/usr/bin/python3 phase12_manual_live_suite.py \
  --vel-topic /mavros/vision_speed/speed_twist \
  --imu-topic /camera/camera/imu \
  --dashboard-url http://127.0.0.1:5000/api/state \
  --sample-hz 20 \
  --output-root /home/jetson123/Drone
```

### F) Metrics this validator checks

Phase1 static:
- `q2_ratio`, forward/lateral drift mean, forward p95 and p99

Phase2 translate:
- `q2_ratio`, forward response mean/p95, forward dominance,
  outlier-heaviness guard, cross-axis leakage percent

## 13. In-Depth Analysis of 3 Recent Tests

This section documents these protocol runs:
- /home/jetson123/Drone/protocol_20260329_043536
- /home/jetson123/Drone/protocol_20260329_043916
- /home/jetson123/Drone/protocol_20260329_044112

### A) Evaluation Criteria Used by Scorer

Static pass requires all:
- vf_mean < 0.02 m/s
- vr_mean < 0.02 m/s
- vf_p95 < 0.05 m/s
- q2_ratio > 0.50

Translate pass requires all:
- q2_ratio >= 0.60
- vf_mean >= 0.02 m/s
- vf_p95 >= 0.04 m/s
- forward dominant (vf_mean > 1.5 x vr_mean)
- no outlier-heavy behavior (vf_max <= 6 x vf_p95)

Tilt pass requires all:
- pitch_rate_p95 >= 1.0 deg/s (IMU active)
- vf_mean < 0.03 m/s
- vf_p95 < 0.08 m/s

### B) Raw Result Summary

| Run | Static | Translate | Tilt | Overall |
|---|---|---|---|---|
| 043536 | PASS | PASS | FAIL | FAIL |
| 043916 | PASS | PASS | FAIL | FAIL |
| 044112 | PASS | PASS | PASS | PASS |

### C) Core Metrics by Run

| Run | Static n/q2 | Static vf_mean / vf_p95 | Translate n/q2 | Translate vf_mean / vf_p95 | Translate vr_mean | Tilt n/q2 | Tilt vf_mean / vf_p95 | Tilt pitch_rate_p95 |
|---|---|---|---|---|---|---|---|---|
| 043536 | 202 / 1.00 | 0.0016 / 0.0037 | 209 / 0.86 | 0.8197 / 2.7477 | 0.2401 | 206 / 0.81 | 0.3265 / 1.0077 | 29.97 |
| 043916 | 210 / 0.99 | 0.0021 / 0.0062 | 203 / 1.00 | 0.6172 / 1.6098 | 0.1656 | 196 / 0.51 | 0.0322 / 0.1579 | 11.23 |
| 044112 | 194 / 1.00 | 0.0064 / 0.0193 | 205 / 1.00 | 0.4156 / 1.0906 | 0.1821 | 195 / 0.99 | 0.0166 / 0.0487 | 11.72 |

### D) Derived Accuracy and Quality Metrics

Forward-dominance ratio (translate) = vf_mean / vr_mean:
- 043536: 3.41
- 043916: 3.73
- 044112: 2.28

Tilt threshold margins:
- Mean margin = 0.03 - vf_mean
- P95 margin = 0.08 - vf_p95

| Run | Tilt mean margin | Tilt p95 margin | Interpretation |
|---|---|---|---|
| 043536 | -0.2965 | -0.9277 | Strong leakage (far from pass) |
| 043916 | -0.0022 | -0.0779 | Near pass on mean, still fails p95 |
| 044112 | +0.0134 | +0.0313 | Pass with safety margin |

Tilt suppression index (relative leakage reduction) = 1 - (tilt_vf_mean / translate_vf_mean):
- 043536: 0.6016
- 043916: 0.9478
- 044112: 0.9601

Interpretation:
- Higher is better.
- Values near 1 indicate tilt produces little fake translation relative to true translation motion.

### E) Trend and Physical Meaning

Tilt leakage improved sharply over runs:
- vf_mean: 0.3265 -> 0.0322 -> 0.0166
- vf_p95: 1.0077 -> 0.1579 -> 0.0487

This means the estimator became progressively better at rejecting phantom lateral motion during pitch/roll, which is the main source of hover drift during attitude changes.

Translation became more controlled while keeping high quality:
- q2 improved to 1.00 in 043916 and 044112.
- Forward response stayed dominant over lateral response in all runs.

### F) Final Judgement

- 043536: not flight-ready (tilt leakage too high).
- 043916: close but not flight-ready (tilt p95 still above threshold).
- 044112: protocol-ready (all phases pass).

### G) Recommended Operating Profile (from best run)

- Use IMU_TOPIC=/imu/data and REQUIRE_IMU=1.
- Keep tilt smooth, in-place, moderate amplitude.
- Avoid hidden translation during tilt window.
- Use matte/high-texture floor to reduce swirly vector artifacts.
- Use SHOW_WINDOW=0 in headless sessions.
