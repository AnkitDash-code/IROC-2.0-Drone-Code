#!/usr/bin/env python3
"""
CS20 LiDAR Phase2 standalone launcher.

Reuses the validated RealSense Phase2 stack but replaces hardware input with
CS20 SDK bridge + IMU relay, publishing the same topics expected by Phase2.
"""

import os
import time

from optical_flow_realsense import phase2_standalone as rs
from .ir_optical_flow_estimator import CS20IROpticalFlowEstimator


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "True", "yes", "YES")


def cleanup_lidar():
    os.system(
        "pkill -9 -f 'optical_flow_cs20_lidar.cs20_lidar_bridge' && "
        "pkill -9 -f 'topic_tools relay /mavros/imu/data /camera/camera/imu' && "
        "pkill -9 -f 'imu_filter_madgwick' && "
        "pkill -9 -f 'static_transform_publisher' > /dev/null 2>&1"
    )


def launch_hardware_lidar():
    print("CS20: launching SDK bridge...")
    os.system(
        f"/usr/bin/python3 -m optical_flow_cs20_lidar.cs20_lidar_bridge >> {rs.LOG_FILE} 2>&1 &"
    )
    time.sleep(2)

    print("CS20: relaying MAVROS IMU -> /camera/camera/imu ...")
    os.system(
        f"ros2 run topic_tools relay /mavros/imu/data /camera/camera/imu >> {rs.LOG_FILE} 2>&1 &"
    )
    time.sleep(1)

    print("CS20: launching IMU filter...")
    os.system(
        f"ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args "
        f"-r /imu/data_raw:=/camera/camera/imu -p use_mag:=false -p publish_tf:=false >> {rs.LOG_FILE} 2>&1 &"
    )
    time.sleep(1)

    print("CS20: setting static transforms...")
    os.system(
        f"ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom >> {rs.LOG_FILE} 2>&1 &"
    )
    time.sleep(0.5)


def configure_cs20_runtime():
    # Force Phase2 to pick our CS20 mapper path.
    os.environ["VO_BACKEND"] = "MSCKF"
    rs.VO_BACKEND = "MSCKF"

    rs.MSCKFVelocityEstimator = CS20IROpticalFlowEstimator
    print("CS20 odometry mode: IR optical flow")

    # Optional toggle to request GPU path in estimators that support it.
    # (falls back to CPU automatically when unavailable)
    if os.environ.get("CS20_USE_GPU", "1") not in ("0", "false", "False"):
        os.environ["CS20_USE_GPU"] = "1"
    else:
        os.environ["CS20_USE_GPU"] = "0"

    # Disable marker landing path in main CS20 runtime for now.
    rs.APRILTAG_ENABLED = False
    rs.GLOBAL_RELOCALIZE_FALLBACK = False

    # Disable Phase1-style feature bin relocalization path for pure optical flow mode.
    class _NoOpFeatureExtractor:
        def __init__(self, method="ORB", n_features=200):
            del method, n_features

        def extract(self, image):
            del image
            return [], None

        def match(self, desc1, desc2):
            del desc1, desc2
            return 0

        def get_runtime_status(self):
            return {
                "backend": "DISABLED_NO_SLAM",
                "ai_enabled": False,
                "ai_fps": 0.0,
                "extract_ms": 0.0,
                "match_ms": 0.0,
            }

    class _NoOpBin:
        def __init__(self):
            self.descriptors = []

    class _NoOpCompassBinManager:
        def __init__(self, n_bins=8):
            self.n_bins = int(max(1, n_bins))
            self.bin_width = 360.0 / float(self.n_bins)
            self.bins = [_NoOpBin() for _ in range(self.n_bins)]

        def heading_to_bin(self, heading_deg):
            h = float(heading_deg) % 360.0
            return int(h // self.bin_width) % self.n_bins

        def store_features(self, heading_deg, keypoints, descriptors, timestamp):
            del heading_deg, keypoints, descriptors, timestamp

        def relocalize(self, heading_deg, query_descriptors, feature_extractor, query_rad_search=3):
            del heading_deg, query_descriptors, feature_extractor, query_rad_search
            return None, 0

    rs.FeatureExtractor = _NoOpFeatureExtractor
    rs.CompassBinManager = _NoOpCompassBinManager
    print("CS20 pure optical-flow mode: relocalization/bin pipeline disabled")

    # FCU IMU is mandatory in CS20 mode, but calibration gate is bypassed.
    _orig_init = rs.IMUPreintegrator.__init__

    def _patched_init_direct(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.gyro_bias_z = 0.0
        self.bias_calibrated = True
        self.spin_calibrated = True
        self.calib_state = "ready"
        self.yaw_deg = 0.0

    rs.IMUPreintegrator.__init__ = _patched_init_direct
    rs.IMU_CALIBRATION_SAMPLES = 1
    rs.USE_PIXHAWK_IMU_CORRECTION = True
    rs.USE_PIXHAWK_YAW_PRIMARY = True
    rs.USE_PIXHAWK_RATE_CORRECTION = True
    rs.USE_PIXHAWK_RATE_PRIMARY = True
    print("CS20 FCU IMU mandatory mode: calibration removed, fusion starts immediately")

    # Keep using LiDAR bridge launch/cleanup hooks.
    rs.cleanup = cleanup_lidar
    rs.launch_hardware = launch_hardware_lidar


def main():
    configure_cs20_runtime()
    rs.main()


if __name__ == "__main__":
    main()
