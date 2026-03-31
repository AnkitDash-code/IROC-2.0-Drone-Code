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
from .depth_icp_odometry import CS20DepthICPEstimator


def _env_true(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "True", "yes", "YES")


def _cs20_no_imu_enabled() -> bool:
    # Default to no-IMU in CS20 mode; set CS20_NO_IMU=0 to re-enable FCU IMU fusion.
    return _env_true("CS20_NO_IMU", "1")


def _cs20_direct_fcu_imu_enabled() -> bool:
    # When FCU IMU is used, bypass startup calibration and fuse immediately.
    # Set CS20_DIRECT_FCU_IMU=0 to restore original calibration-gated behavior.
    return _env_true("CS20_DIRECT_FCU_IMU", "1")


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

    if _cs20_no_imu_enabled():
        print("CS20: no-imu mode enabled; skipping IMU relay/filter")
    else:
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

    odom_mode = os.environ.get("CS20_ODOM_MODE", "OF").strip().upper()
    if odom_mode == "ICP":
        rs.MSCKFVelocityEstimator = CS20DepthICPEstimator
        print("CS20 odometry mode: ICP depth odometry")
    else:
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

    # Optional LiDAR-only mode: bypass IMU calibration gate in reused Phase2 stack.
    if _cs20_no_imu_enabled():
        _orig_init = rs.IMUPreintegrator.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.gyro_bias_z = 0.0
            self.bias_calibrated = True
            self.spin_calibrated = True
            self.calib_state = "ready"
            self.yaw_deg = 0.0

        rs.IMUPreintegrator.__init__ = _patched_init
        rs.USE_PIXHAWK_IMU_CORRECTION = False
        rs.USE_MADGWICK_YAW = False
        rs.USE_PIXHAWK_YAW_PRIMARY = False
        rs.USE_PIXHAWK_RATE_CORRECTION = False
        rs.USE_PIXHAWK_RATE_PRIMARY = False
        rs.IMU_CALIBRATION_SAMPLES = 1
        print("CS20 no-imu mode: IMU calibration bypass enabled")
    elif _cs20_direct_fcu_imu_enabled():
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
        # Keep FCU/Madgwick fusion toggles as configured in base Phase2; only remove gate.
        print("CS20 direct FCU IMU mode: calibration gate bypassed, fusion starts immediately")

    # Keep using LiDAR bridge launch/cleanup hooks.
    rs.cleanup = cleanup_lidar
    rs.launch_hardware = launch_hardware_lidar


def main():
    configure_cs20_runtime()
    rs.main()


if __name__ == "__main__":
    main()
