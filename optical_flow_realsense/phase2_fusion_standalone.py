#!/usr/bin/env python3
"""
Phase 2 Fusion Standalone

Fuses two visual velocity mappers online while preserving Phase 2 runtime UX.
"""

import csv
import math
import os
import threading
import time

import numpy as np

from .phase2_standalone import (
    MATPLOTLIB_AVAILABLE,
    MSCKFVelocityEstimator,
    XFeatLightGlueTRTBackend,
    XFEAT_BACKEND_IMPORT_ERROR,
    XFEAT_TRT_MODULE,
    XFEAT_TRT_NATIVE_MODULE_PATH,
    XFEAT_TRT_CONFIG_PATH,
    XFEAT_TRT_XFEAT_ENGINE_PATH,
    XFEAT_TRT_LIGHTGLUE_ENGINE_PATH,
    XFEAT_TRT_VEL_SCALE,
    XFEAT_TRT_FOCAL_PX,
    XFEAT_TRT_MIN_MATCHES,
    XFEAT_TRT_FRAME_STRIDE,
    XFEAT_TRT_REPO_PATH,
    XFEAT_TRT_TOP_K,
    DASHBOARD_AVAILABLE,
    cleanup,
    get_fcu_serial_device,
    request_mavros_stream_rate,
    MAVROS_STREAM_RATE_HZ,
    MAVROS_STREAM_RETRIES,
    MAVROS_BAUD,
    MAVROS_LOG_FILE,
    launch_hardware,
    get_ip,
    REALSENSE_FX,
    REALSENSE_FY,
    rclpy,
    ExternalShutdownException,
    Phase2Node,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


FUSION_ALPHA_MIN = float(os.environ.get("FUSION_ALPHA_MIN", "0.20"))
FUSION_ALPHA_MAX = float(os.environ.get("FUSION_ALPHA_MAX", "0.85"))
FUSION_ROUTE_CSV = os.environ.get("FUSION_ROUTE_CSV", "my_drone_route_fusion.csv")
FUSION_ROUTE_IMG = os.environ.get("FUSION_ROUTE_IMG", "my_drone_route_fusion.png")
FUSION_MAX_SPEED_MPS = float(os.environ.get("FUSION_MAX_SPEED_MPS", "1.4"))
FUSION_MAX_STEP_M = float(os.environ.get("FUSION_MAX_STEP_M", "0.07"))
FUSION_DEADBAND_MPS = float(os.environ.get("FUSION_DEADBAND_MPS", "0.008"))
FUSION_EMA_ALPHA = float(os.environ.get("FUSION_EMA_ALPHA", "0.55"))
FUSION_DISAGREE_GATE_MPS = float(os.environ.get("FUSION_DISAGREE_GATE_MPS", "0.55"))
FUSION_STATIONARY_FLOW_PX = float(os.environ.get("FUSION_STATIONARY_FLOW_PX", "0.12"))
FUSION_STATIONARY_DECAY = float(os.environ.get("FUSION_STATIONARY_DECAY", "0.88"))
FUSION_STATIONARY_MIN_FRAMES = int(os.environ.get("FUSION_STATIONARY_MIN_FRAMES", "3"))
FUSION_AXIS_DOM_RATIO = float(os.environ.get("FUSION_AXIS_DOM_RATIO", "2.3"))
FUSION_LEAK_ATTENUATION = float(os.environ.get("FUSION_LEAK_ATTENUATION", "0.70"))
FUSION_MIN_SPEED_RETENTION = float(os.environ.get("FUSION_MIN_SPEED_RETENTION", "0.55"))
FUSION_OPPOSE_COS_THRESH = float(os.environ.get("FUSION_OPPOSE_COS_THRESH", "-0.20"))


class FusionVelocityMapper:
    def __init__(self, primary_mapper, secondary_mapper=None):
        self.primary = primary_mapper
        self.secondary = secondary_mapper
        self._lock = threading.Lock()
        self._pos_x = 0.0
        self._pos_z = 0.0
        self._traj_x = [0.0]
        self._traj_z = [0.0]
        self._prev_ts = None
        self._vx_filt = 0.0
        self._vy_filt = 0.0
        self._stationary_low_flow_count = 0
        self._last_debug = {
            "source": "FUSION",
            "success": False,
            "flow_x": 0.0,
            "flow_y": 0.0,
            "alpha": 1.0,
            "primary": "unknown",
            "secondary": "none",
        }

    def _confidence(self, dbg):
        if not isinstance(dbg, dict) or (not bool(dbg.get("success", False))):
            return 0.0
        tracked = float(max(0, int(dbg.get("tracked", 0))))
        n_samples = float(max(0, int(dbg.get("n_samples", 0))))
        med_flow = float(max(0.0, float(dbg.get("median_flow_px", 0.0))))
        c_track = min(1.0, tracked / 120.0)
        c_samp = min(1.0, n_samples / 40.0)
        c_flow = min(1.0, med_flow / 0.8)
        return 0.45 * c_track + 0.35 * c_samp + 0.20 * c_flow

    def _alpha(self, p_conf, s_conf):
        den = p_conf + s_conf
        if den <= 1e-9:
            return 0.5
        a = p_conf / den
        return float(min(FUSION_ALPHA_MAX, max(FUSION_ALPHA_MIN, a)))

    def update(self, gray_image, depth_image_16uc1, tof_altitude_m, timestamp_sec, imu_gyro_xyz=(0.0, 0.0, 0.0), imu_gyro_z_rads=0.0, yaw_deg=None):
        p_vx, p_vy = self.primary.update(
            gray_image,
            depth_image_16uc1,
            tof_altitude_m,
            timestamp_sec,
            imu_gyro_xyz=imu_gyro_xyz,
            imu_gyro_z_rads=imu_gyro_z_rads,
            yaw_deg=yaw_deg,
        )
        p_dbg = self.primary.get_dashboard_payload().get("debug", {})

        if self.secondary is None:
            f_vx, f_vy = p_vx, p_vy
            alpha = 1.0
            s_vx, s_vy, s_dbg = 0.0, 0.0, {}
            spike_reject = False
            oppose_select = False
            p_conf, s_conf = 1.0, 0.0
        else:
            s_vx, s_vy = self.secondary.update(
                gray_image,
                depth_image_16uc1,
                tof_altitude_m,
                timestamp_sec,
                imu_gyro_xyz=imu_gyro_xyz,
                imu_gyro_z_rads=imu_gyro_z_rads,
                yaw_deg=yaw_deg,
            )
            s_dbg = self.secondary.get_dashboard_payload().get("debug", {})
            p_conf = self._confidence(p_dbg)
            s_conf = self._confidence(s_dbg)
            alpha = self._alpha(p_conf, s_conf)

            disagree = math.hypot(p_vx - s_vx, p_vy - s_vy)
            p_speed = math.hypot(p_vx, p_vy)
            s_speed = math.hypot(s_vx, s_vy)
            cos_ps = 1.0
            if p_speed > 1e-6 and s_speed > 1e-6:
                cos_ps = ((p_vx * s_vx) + (p_vy * s_vy)) / max(1e-9, p_speed * s_speed)
            oppose_select = cos_ps < FUSION_OPPOSE_COS_THRESH
            spike_reject = disagree > FUSION_DISAGREE_GATE_MPS
            if spike_reject or oppose_select:
                if p_conf >= s_conf:
                    alpha = 1.0
                    f_vx, f_vy = p_vx, p_vy
                else:
                    alpha = 0.0
                    f_vx, f_vy = s_vx, s_vy
            else:
                f_vx = alpha * p_vx + (1.0 - alpha) * s_vx
                f_vy = alpha * p_vy + (1.0 - alpha) * s_vy

            max_src_speed = max(p_speed, s_speed)
            fused_speed = math.hypot(f_vx, f_vy)
            min_keep_speed = FUSION_MIN_SPEED_RETENTION * max_src_speed
            if p_conf > 0.35 and s_conf > 0.35 and max_src_speed > 1e-6 and fused_speed < min_keep_speed:
                if fused_speed > 1e-9:
                    k = min_keep_speed / fused_speed
                    f_vx *= k
                    f_vy *= k

        p_flow = float(p_dbg.get("median_flow_px", 0.0))
        s_flow = float(s_dbg.get("median_flow_px", 0.0)) if self.secondary is not None else p_flow
        if p_flow < FUSION_STATIONARY_FLOW_PX and s_flow < FUSION_STATIONARY_FLOW_PX:
            self._stationary_low_flow_count += 1
        else:
            self._stationary_low_flow_count = 0
        if self._stationary_low_flow_count >= max(1, FUSION_STATIONARY_MIN_FRAMES):
            f_vx *= FUSION_STATIONARY_DECAY
            f_vy *= FUSION_STATIONARY_DECAY

        speed = math.hypot(f_vx, f_vy)
        if speed > FUSION_MAX_SPEED_MPS:
            k = FUSION_MAX_SPEED_MPS / max(speed, 1e-9)
            f_vx *= k
            f_vy *= k

        if abs(f_vx) < FUSION_DEADBAND_MPS:
            f_vx = 0.0
        if abs(f_vy) < FUSION_DEADBAND_MPS:
            f_vy = 0.0

        self._vx_filt = (FUSION_EMA_ALPHA * f_vx) + ((1.0 - FUSION_EMA_ALPHA) * self._vx_filt)
        self._vy_filt = (FUSION_EMA_ALPHA * f_vy) + ((1.0 - FUSION_EMA_ALPHA) * self._vy_filt)
        f_vx, f_vy = self._vx_filt, self._vy_filt

        ax = abs(f_vx)
        ay = abs(f_vy)
        if ax > (FUSION_AXIS_DOM_RATIO * max(ay, 1e-9)):
            f_vy *= FUSION_LEAK_ATTENUATION
        elif ay > (FUSION_AXIS_DOM_RATIO * max(ax, 1e-9)):
            f_vx *= FUSION_LEAK_ATTENUATION

        with self._lock:
            if self._prev_ts is not None:
                dt = max(0.0, min(0.2, float(timestamp_sec - self._prev_ts)))
                if dt > 0.0:
                    step = math.hypot(f_vx * dt, f_vy * dt)
                    if step > FUSION_MAX_STEP_M:
                        k = FUSION_MAX_STEP_M / max(step, 1e-9)
                        f_vx *= k
                        f_vy *= k
                    if yaw_deg is not None:
                        yr = math.radians(float(yaw_deg))
                        wx = math.cos(yr) * f_vx - math.sin(yr) * f_vy
                        wz = math.sin(yr) * f_vx + math.cos(yr) * f_vy
                    else:
                        wx, wz = f_vx, f_vy
                    self._pos_x += wx * dt
                    self._pos_z += wz * dt
                    self._traj_x.append(self._pos_x)
                    self._traj_z.append(self._pos_z)
            self._prev_ts = float(timestamp_sec)
            self._last_debug = {
                "source": "FUSION",
                "success": bool(p_dbg.get("success", False)) or bool(s_dbg.get("success", False)),
                "flow_x": float(f_vx),
                "flow_y": float(f_vy),
                "alpha": float(alpha),
                "primary": str(p_dbg.get("source", "primary")),
                "secondary": str(s_dbg.get("source", "secondary")) if self.secondary is not None else "none",
                "p_tracked": int(p_dbg.get("tracked", 0)),
                "s_tracked": int(s_dbg.get("tracked", 0)) if self.secondary is not None else 0,
                "p_median_flow_px": float(p_dbg.get("median_flow_px", 0.0)),
                "s_median_flow_px": float(s_dbg.get("median_flow_px", 0.0)) if self.secondary is not None else 0.0,
                "spike_reject": bool(spike_reject),
                "oppose_select": bool(oppose_select),
                "stationary_low_flow_count": int(self._stationary_low_flow_count),
                "p_flow_x": float(p_vx),
                "p_flow_y": float(p_vy),
                "s_flow_x": float(s_vx),
                "s_flow_y": float(s_vy),
            }

        return float(f_vx), float(f_vy)

    def set_intrinsics(self, fx, fy, cx, cy, source="external"):
        if hasattr(self.primary, "set_intrinsics"):
            self.primary.set_intrinsics(fx, fy, cx, cy, source=source)
        if self.secondary is not None and hasattr(self.secondary, "set_intrinsics"):
            self.secondary.set_intrinsics(fx, fy, cx, cy, source=source)

    def get_dashboard_payload(self):
        with self._lock:
            return {
                "tracks": {
                    "fused": list(zip(self._traj_x, self._traj_z)),
                    "pnp": [],
                    "fallback": [],
                },
                "debug": dict(self._last_debug),
            }

    def save_map(self, image_path=FUSION_ROUTE_IMG, csv_path=FUSION_ROUTE_CSV):
        with self._lock:
            if len(self._traj_x) < 2:
                print("Route map skipped: not enough trajectory points")
                return False

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x_m", "z_m"])
                for x, z in zip(self._traj_x, self._traj_z):
                    w.writerow([f"{x:.6f}", f"{z:.6f}"])
            print(f"Saved fusion route CSV: {csv_path}")

            if MATPLOTLIB_AVAILABLE and plt is not None:
                plt.figure(figsize=(8, 8))
                plt.plot(self._traj_x, self._traj_z, color="darkorange", linewidth=1.2, marker="o", markersize=1.5)
                plt.plot(0, 0, "*g", markersize=14, label="Start")
                plt.plot(self._traj_x[-1], self._traj_z[-1], "Xr", markersize=10, label="End")
                plt.title("Fusion 2-D Route Map")
                plt.xlabel("Left / Right (m)")
                plt.ylabel("Forward / Back (m)")
                plt.axis("equal")
                plt.grid(True)
                plt.legend()
                plt.savefig(image_path, dpi=200)
                plt.close()
                print(f"Saved fusion route image: {image_path}")
            return True


class Phase2FusionNode(Phase2Node):
    def __init__(self):
        super().__init__()

        primary = self.flow_mapper
        secondary = None

        if not isinstance(primary, MSCKFVelocityEstimator):
            secondary = MSCKFVelocityEstimator()
            self.node.get_logger().info("Fusion secondary mapper: MSCKF")
        elif XFeatLightGlueTRTBackend is not None:
            candidate = XFeatLightGlueTRTBackend(
                module_name=XFEAT_TRT_MODULE,
                native_module_path=XFEAT_TRT_NATIVE_MODULE_PATH,
                config_path=XFEAT_TRT_CONFIG_PATH,
                xfeat_engine_path=XFEAT_TRT_XFEAT_ENGINE_PATH,
                lightglue_engine_path=XFEAT_TRT_LIGHTGLUE_ENGINE_PATH,
                velocity_scale=XFEAT_TRT_VEL_SCALE,
                focal_px=XFEAT_TRT_FOCAL_PX,
                min_matches=XFEAT_TRT_MIN_MATCHES,
                frame_stride=XFEAT_TRT_FRAME_STRIDE,
                repo_path=XFEAT_TRT_REPO_PATH,
                top_k=XFEAT_TRT_TOP_K,
            )
            if getattr(candidate, "available", False):
                secondary = candidate
                self.node.get_logger().info("Fusion secondary mapper: XFEAT_LIGHTGLUE_TRT")
            else:
                self.node.get_logger().warn("Fusion secondary XFEAT unavailable; running primary-only")
        else:
            self.node.get_logger().warn(f"XFEAT import unavailable: {XFEAT_BACKEND_IMPORT_ERROR}")

        self.flow_mapper = FusionVelocityMapper(primary_mapper=primary, secondary_mapper=secondary)
        self.node.get_logger().info("Phase2 fusion mapper enabled")

    def save_route_outputs(self):
        if self.flow_mapper is None:
            return
        self.flow_mapper.save_map(FUSION_ROUTE_IMG, FUSION_ROUTE_CSV)


def main():
    cleanup()

    if DASHBOARD_AVAILABLE:
        from .web_dashboard import app as flask_app

        def run_flask():
            flask_app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print("Web Dashboard started on http://0.0.0.0:5000")
        time.sleep(1)

    print("Launching MAVROS...")
    fcu_device = get_fcu_serial_device()
    fcu_url = f"{fcu_device}:{MAVROS_BAUD}"
    os.system(f"ros2 launch mavros px4.launch fcu_url:={fcu_url} >> {MAVROS_LOG_FILE} 2>&1 &")
    time.sleep(8)
    request_mavros_stream_rate(MAVROS_STREAM_RATE_HZ, MAVROS_STREAM_RETRIES)
    time.sleep(2)

    launch_hardware()
    time.sleep(5)

    rclpy.init()
    node = Phase2FusionNode()

    jetson_ip = get_ip()
    print(f"PHASE 2 FUSION running on {jetson_ip} | fx={REALSENSE_FX}, fy={REALSENSE_FY}")

    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        print("Shutting down Phase 2 Fusion...")
    except ExternalShutdownException:
        print("ROS shutdown requested.")
    finally:
        try:
            node.save_route_outputs()
        except Exception as e:
            print(f"Fusion route map save failed: {e}")
        try:
            node.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        cleanup()


if __name__ == "__main__":
    main()
