# Drone Repository

This repository contains flight-control scripts, vision pipelines, Lidar processing, MAVLink utilities, trained model artifacts, and a ROS 2 package for drone perception/control experiments.

## Repository Scope

The codebase combines:
- MAVLink flight control utilities and mission scripts (manual and autonomous).
- Intel RealSense + YOLO based safe-spot detection workflows.
- Synexens Lidar SDK integration and RANSAC-based plane estimation.
- ROS 2 package scaffolding for vision-control nodes.
- Trained model artifacts and experiment outputs.

## Top-Level Structure

- `drone_control.py`: Core pymavlink helper functions (connect, mode, arm, takeoff, yaw, movement, land).
- `run_all.sh` / `run_all_foreground.py`: Unified launcher scripts to start all required nodes and the web dashboard.
- `web_dashboard.py`: Real-time monitoring dashboard and telemetry stream center.
- `guarded_mission.py`: Safety-first guided mission flow with manual confirmations and failsafes.
- `intel.py`, `intelligent_flight.py`, `intel_slant.py`, `test_yellow*.py`, `temp*.py`: Vision-driven autonomous mission variants.
- `detect.py`, `Detect.py`, `depth.py`, `dist.py`, `imagec.py`, `rs_li_test.py`: Camera/depth/detection utilities.
- `events.py`: Logging helper for mission events.
- `pcl.py`: Point cloud processing utilities.
- `PyMav/`: Standalone pymavlink examples and test scripts.
- `Lidar/`: Synexens SDK assets and `ransac.py` for plane-angle estimation.
- `optical_flow_cs20_lidar/`: CS20 Lidar bridge and IR tracking pipelines (`ir_tracker.py`, `cs20_lidar_bridge_simple.py`).
- `optional_marker_landing/`: Precision ArUco marker landing logic (`aruco_landing.py`, `aruco_streamer.py`).
- `drone_vision_control/`: ROS 2 Python package workspace content (package manifest, build/install artifacts, tests).
- `files_cv/`: Training outputs and weights for safe-spot detection models.
- `INSTALL_NETWORK_FIX.md` & `jetson-network-fix.sh`: Jetson network stability service routines.
- Model files (`*.pt`, `*.onnx`, `*.engine`) and telemetry logs (`mav.tlog`, `log/`) are also present.

## Main Workflows

### 1) Guarded Manual-Gated Flight
Use `guarded_mission.py` for a strict sequence:
1. Connect
2. Ground check
3. Arm
4. Takeoff (slow profile)
5. Hover countdown
6. LAND mode landing
7. Disarm

This script is intended as the safest baseline entry point in the repo.

### 2) Vision-Based Safe-Spot Missions
Scripts such as `intelligent_flight.py` and `intel.py` combine:
- RealSense RGB/depth frames
- YOLO model inference
- Position/yaw corrections via MAVLink
- Landing decisions based on detected safe regions

### 3) Lidar Surface Analysis
`Lidar/ransac.py` integrates Synexens depth data and estimates plane geometry for slope/surface-aware behavior.

### 4) ROS 2 Package Workflow
`drone_vision_control/` is a ROS 2 package with entry points listed in setup scripts and test files for style/compliance.

### 5) IR Tracking & Lidar Bridges
`optical_flow_cs20_lidar/ir_tracker.py` processes IR sensor data alongside `cs20_lidar_bridge_simple.py` to forward target vectors and Lidar obstacle avoidance data into the system, optionally combining with MSCKF estimators.

### 6) Web Dashboard & Unified Orchestration
Run the complete suite via `run_all.sh` or `run_all_foreground.py`. This starts the web UI (`web_dashboard.py`), handles stream relay points, coordinates the MAVLink pipelines, and visualizes the logs and real-time state in `dashboard.html`.

### 7) Precision ArUco Landing
`optional_marker_landing/aruco_landing.py` controls the exact final descent phase aligning to specific fiducial tags, driven by poses computed in `aruco_streamer.py`.

## Environment And Dependencies

This project does not currently contain a single unified dependency lockfile for the root scripts. Based on imports used across the repository, install at least:

- Python 3.10+
- pymavlink
- numpy
- opencv-python
- pyrealsense2
- ultralytics
- torch
- tqdm
- open3d

For ROS 2 package work, also install:
- ROS 2 (matching your distro)
- colcon
- rclpy and related message dependencies

For Synexens Lidar workflows:
- Use vendor SDK assets included under `Lidar/` and `SynexensPythonSDK4_4.2.4.0_202504281506/`
- Ensure native `.so` libraries are accessible in your runtime library path

## Quick Start

### A) Create Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pymavlink numpy opencv-python pyrealsense2 ultralytics torch tqdm open3d
```

### B) Run Guarded Mission
```bash
python3 guarded_mission.py --connect /dev/ttyACM0 --baud 57600
```

### C) Run a Vision Mission Example
```bash
python3 intelligent_flight.py
```

## Hardware/Runtime Notes

- Connection defaults in several scripts target `/dev/ttyACM0` at 57600 baud.
- Many scripts assume ArduPilot-compatible mode names (for example GUIDED, LOITER, LAND).
- RealSense workflows require camera access and supported firmware.
- Lidar workflows require Synexens SDK libraries and proper permissions.

## Safety Notes

- Test in SITL before real hardware.
- Keep RC/manual takeover available during all real flights.
- Review mode assumptions before arming.
- Validate geofence, failsafe, and battery limits in flight controller parameters.

## Development Notes

- There are generated/build artifacts inside `drone_vision_control/build/` and `drone_vision_control/install/`.
- Avoid editing generated files directly.
- Primary editable sources are top-level scripts, package source files, and config assets.

## Suggested Next Improvements

- Add a root `requirements.txt` for non-ROS scripts.
- Add a small launcher CLI to select mission profiles.
- Consolidate duplicate mission variants and standardize logging.
- Add SITL-focused integration tests for critical mission scripts.

## D455 Protocol Validation (Phase1 + Phase2)

After running a D455 protocol folder (`protocol_YYYYMMDD_HHMMSS`), use:

```bash
python3 phase12_test_suite.py
```

This validates:
- `phase1_static.csv`
- `phase2_translate.csv`

And writes:
- `phase12_report.txt`
- `phase12_report.json`

Optional GUI dashboard:

```bash
python3 phase12_test_suite.py --gui --host 0.0.0.0 --port 5060
```

Open: `http://<jetson_ip>:5060`

## Optional Edge Architecture: XFeat + LightGlue + TensorRT

The exact edge-robotics architecture discussed for this project can be added as an optional backend:

- XFeat feature extraction (fast on Jetson-class devices)
- LightGlue matching
- Native TensorRT deployment
- C++ implementation with Python bindings via PyBind

This architecture is attractive for embedded robotics because extraction and matching both run on GPU with low latency.

### Build Notes (Jetson)

The common workflow is CMake + Make (or Ninja). If the external project already provides `CMakeLists.txt`, the minimal build flow is:

```bash
git clone <repo-url>
cd <repo-dir>
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

If Python bindings are included, they are typically built in the same step and then imported from Python once the shared library is produced.

### Integration Strategy In This Repo

- Keep the existing MSCKF/VO path as fallback.
- Add runtime switch to select backend (`MSCKF`, `XFEAT_LIGHTGLUE_TRT`).
- Feed the selected backend output into the same MAVROS publish path to avoid touching flight-control wiring.

### Important Catch

This path is not pure Python. You need a native compile step on Jetson, and success depends on your CUDA/TensorRT toolchain versions matching the external project requirements.
