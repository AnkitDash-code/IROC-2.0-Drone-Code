Aruco landing helper
====================

This folder contains a lightweight ArUco-based landing helper for the CS20
downward camera. It provides two utilities:

- `calibrate_checkerboard.py` — calibrate the camera once from checkerboard
  images and produce `camera_calib.npz` (camera_matrix, dist_coeffs).
- `aruco_landing.py` — a ROS2 node that detects a 4-marker board and publishes
  pose estimates to MAVROS at `/mavros/vision_pose/pose`.

How it works
------------

1. When the drone is low (on the ground) the downward camera cannot see the
   markers; the node stays in `BLIND_CLIMB` and does not publish vision data.
2. Once the drone climbs above `altitude_threshold` (default 0.8 m) and the
   node sees the markers for `lock_frames_needed` consecutive frames (default 5),
   it switches to `VISION_HOVER` and starts publishing `PoseStamped` messages
   to MAVROS. The FCU will pick this up in the EKF as vision position data.
3. If vision is lost, the node falls back to `BLIND_HOVER` and stops publishing
   vision messages — the FCU will revert to optical-flow + IMU.

Quick start
-----------

1. Collect checkerboard photos from the same camera (IR/mono images).
   Put them into `optical_flow_cs20_lidar/optional_marker_landing/calib_images/`.
2. Run the calibrator (example):

```bash
python3 optical_flow_cs20_lidar/optional_marker_landing/calibrate_checkerboard.py \
  --images optical_flow_cs20_lidar/optional_marker_landing/calib_images \
  --rows 6 --cols 9 --square 0.024
```

3. Launch the CS20 bridge and MAVROS as your normal Phase2 workflow does.
4. Start the ArUco node (in a ROS2 sourced shell):

```bash
python3 optical_flow_cs20_lidar/optional_marker_landing/aruco_landing.py
```

Configuration
-------------

Environment variables and ROS2 parameters are supported. Useful env vars:

- `ARUCO_MARKER_LENGTH_M` — marker side length (m)
- `ARUCO_MARKER_SEPARATION_M` — center-to-center separation between markers (m)
- `ARUCO_ALTITUDE_THRESHOLD_M` — min altitude before enabling vision hand-off
- `ARUCO_LOCK_FRAMES` — consecutive frames required for a stable lock
- `ARUCO_PUBLISH_HZ` — publish rate to MAVROS

Notes
-----

- The node publishes `PoseStamped` in the `landing_pad` frame. You may want
  to adjust the frame ID or transform to match your flight controller's frame.
- Ensure your ArUco markers use the same dictionary as configured (4x4 variants
  are supported by default).
- This node is intentionally lightweight: pose is computed using `estimatePoseBoard`
  and only a yaw is passed in orientation — roll/pitch are left at zero to keep
  the EKF stable for downward-facing setups.
# Optional Marker Landing (Separated)

Marker landing is intentionally disabled in the default CS20 runtime.

This folder keeps the marker-related configuration separate so the main
workflow focuses on optical-flow and fusion accuracy checks.

To re-enable marker detection later, set in your launcher/runtime:

- `APRILTAG_ENABLED = True`
- `APRILTAG_DICT_NAME` (e.g. `DICT_4X4_100`)
- `APRILTAG_SIZE_M` (marker size in meters)

Current default CS20 mode keeps this off.
