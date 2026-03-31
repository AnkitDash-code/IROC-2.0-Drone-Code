#!/usr/bin/env python3
"""
CS20 depth ICP odometry estimator (optional).

This estimator converts depth frames into sparse 3D clouds and runs point-to-point
ICP between consecutive frames to estimate planar velocity.
"""

import csv
import math
import threading
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    o3d = None
    OPEN3D_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class CS20DepthICPEstimator:
    def __init__(
        self,
        fx: float = 240.0,
        fy: float = 240.0,
        cx: float = 160.0,
        cy: float = 120.0,
        max_speed_mps: float = 2.0,
        lowpass_alpha: float = 0.35,
        min_alt_m: float = 0.08,
        z_min_m: float = 0.08,
        z_max_m: float = 6.0,
        voxel_m: float = 0.04,
        icp_max_corr_m: float = 0.20,
        sample_stride: int = 3,
        min_points: int = 1200,
        use_gpu: bool = False,
    ):
        self.fx = float(max(1.0, fx))
        self.fy = float(max(1.0, fy))
        self.cx = float(cx)
        self.cy = float(cy)
        self.max_speed_mps = float(max(0.2, max_speed_mps))
        self.lowpass_alpha = float(np.clip(lowpass_alpha, 0.05, 0.95))
        self.min_alt_m = float(max(0.02, min_alt_m))
        self.z_min_m = float(max(0.02, z_min_m))
        self.z_max_m = float(max(self.z_min_m + 0.1, z_max_m))
        self.voxel_m = float(max(0.005, voxel_m))
        self.icp_max_corr_m = float(max(0.02, icp_max_corr_m))
        self.sample_stride = int(max(1, sample_stride))
        self.min_points = int(max(200, min_points))
        self.use_gpu = bool(use_gpu)

        self._lock = threading.Lock()
        self.prev_cloud = None
        self.prev_ts: Optional[float] = None

        self.vx_filt = 0.0
        self.vy_filt = 0.0

        self._pos_x = 0.0
        self._pos_z = 0.0
        self._traj_x = [0.0]
        self._traj_z = [0.0]
        self._last_ts = None

        self._last_debug = {
            "source": "CS20_DEPTH_ICP",
            "success": False,
            "flow_x": 0.0,
            "flow_y": 0.0,
            "alt": 0.0,
            "tracked": 0,
            "n_samples": 0,
            "median_flow_px": 0.0,
            "gyro_comp": False,
            "backend": "icp_gpu" if (self.use_gpu and OPEN3D_AVAILABLE) else "icp_cpu",
            "inlier_ratio": 0.0,
            "step_m": 0.0,
            "flow_px": 0.0,
            "rejected": False,
            "reject_reason": "",
        }

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float, source: str = "external") -> None:
        del source
        with self._lock:
            if fx > 1.0:
                self.fx = float(fx)
            if fy > 1.0:
                self.fy = float(fy)
            self.cx = float(cx)
            self.cy = float(cy)

    def _depth_to_cloud(self, depth16: np.ndarray):
        if not OPEN3D_AVAILABLE:
            return None, 0

        if depth16 is None:
            return None, 0

        d = depth16.astype(np.float32) / 1000.0
        h, w = d.shape[:2]

        v_idx = np.arange(0, h, self.sample_stride, dtype=np.int32)
        u_idx = np.arange(0, w, self.sample_stride, dtype=np.int32)
        vv, uu = np.meshgrid(v_idx, u_idx, indexing="ij")

        z = d[vv, uu]
        valid = (z > self.z_min_m) & (z < self.z_max_m)
        if not np.any(valid):
            return None, 0

        z = z[valid]
        uu = uu[valid].astype(np.float32)
        vv = vv[valid].astype(np.float32)

        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        pts = np.stack((x, y, z), axis=1)
        if pts.shape[0] < self.min_points:
            return None, int(pts.shape[0])

        pcd = o3d.geometry.PointCloud()  # type: ignore[attr-defined]
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))  # type: ignore[attr-defined]
        pcd = pcd.voxel_down_sample(self.voxel_m)
        return pcd, len(pcd.points)

    def update(
        self,
        gray_image: np.ndarray,
        depth_image_16uc1: np.ndarray,
        tof_altitude_m: float,
        timestamp_sec: float,
        imu_gyro_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        imu_gyro_z_rads: float = 0.0,
        yaw_deg: Optional[float] = None,
    ) -> Tuple[float, float]:
        del gray_image, imu_gyro_xyz, imu_gyro_z_rads

        with self._lock:
            alt = float(max(self.min_alt_m, tof_altitude_m))

            if not OPEN3D_AVAILABLE:
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": "open3d_missing",
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            if depth_image_16uc1 is None:
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": "depth_missing",
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            cloud, n_points = self._depth_to_cloud(depth_image_16uc1)
            if cloud is None:
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": f"few_points:{n_points}",
                    "n_samples": n_points,
                    "tracked": n_points,
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            if self.prev_cloud is None or self.prev_ts is None:
                self.prev_cloud = cloud
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": "bootstrap",
                    "n_samples": n_points,
                    "tracked": n_points,
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            dt = float(timestamp_sec - self.prev_ts)
            if dt <= 1e-4 or dt > 0.30:
                self.prev_cloud = cloud
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": f"bad_dt:{dt:.4f}",
                    "n_samples": n_points,
                    "tracked": n_points,
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            reg = o3d.pipelines.registration.registration_icp(  # type: ignore[attr-defined]
                self.prev_cloud,
                cloud,
                self.icp_max_corr_m,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # type: ignore[attr-defined]
            )

            T = reg.transformation
            tx = float(T[0, 3])
            tz = float(T[2, 3])

            # Scene motion is inverse of camera motion.
            v_right_raw = float(np.clip(-tx / dt, -self.max_speed_mps, self.max_speed_mps))
            v_forward_raw = float(np.clip(-tz / dt, -self.max_speed_mps, self.max_speed_mps))

            a = self.lowpass_alpha
            self.vx_filt = a * v_right_raw + (1.0 - a) * self.vx_filt
            self.vy_filt = a * v_forward_raw + (1.0 - a) * self.vy_filt

            step_m = 0.0
            if self._last_ts is not None:
                dt_pose = float(timestamp_sec - self._last_ts)
                if 0.0 < dt_pose < 0.5:
                    if yaw_deg is not None:
                        yr = math.radians(float(yaw_deg))
                        wx = math.cos(yr) * self.vx_filt - math.sin(yr) * self.vy_filt
                        wz = math.sin(yr) * self.vx_filt + math.cos(yr) * self.vy_filt
                    else:
                        wx, wz = self.vx_filt, self.vy_filt
                    dx_w = wx * dt_pose
                    dz_w = wz * dt_pose
                    step_m = float(math.hypot(dx_w, dz_w))
                    self._pos_x += dx_w
                    self._pos_z += dz_w
                    self._traj_x.append(self._pos_x)
                    self._traj_z.append(self._pos_z)
            self._last_ts = float(timestamp_sec)

            self.prev_cloud = cloud
            self.prev_ts = float(timestamp_sec)

            fitness = float(reg.fitness)
            rmse = float(reg.inlier_rmse)
            self._last_debug.update({
                "success": True,
                "rejected": False,
                "reject_reason": "",
                "flow_x": self.vx_filt,
                "flow_y": self.vy_filt,
                "tracked": n_points,
                "n_samples": n_points,
                "median_flow_px": rmse,
                "flow_px": rmse,
                "inlier_ratio": fitness,
                "alt": alt,
                "step_m": step_m,
            })

            return self.vx_filt, self.vy_filt

    def get_dashboard_payload(self) -> dict:
        with self._lock:
            return {
                "tracks": {"fused": list(zip(self._traj_x, self._traj_z)), "pnp": [], "fallback": []},
                "debug": dict(self._last_debug),
            }

    def save_map(self, image_path: str = "my_drone_route.png", csv_path: str = "my_drone_route.csv") -> bool:
        with self._lock:
            if len(self._traj_x) < 2:
                return False

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x_m", "z_m"])
                for x, z in zip(self._traj_x, self._traj_z):
                    w.writerow([f"{x:.6f}", f"{z:.6f}"])

            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(self._traj_x, self._traj_z, color="seagreen", linewidth=1.2)
                ax.plot(0.0, 0.0, "*g", markersize=14, label="Start")
                ax.plot(self._traj_x[-1], self._traj_z[-1], "Xr", markersize=10, label="End")
                ax.set_title("CS20 Depth ICP Route")
                ax.set_xlabel("Left / Right (m)")
                ax.set_ylabel("Forward / Back (m)")
                ax.set_aspect("equal")
                ax.grid(True)
                ax.legend()
                fig.savefig(image_path, dpi=200)
                plt.close(fig)
            return True
