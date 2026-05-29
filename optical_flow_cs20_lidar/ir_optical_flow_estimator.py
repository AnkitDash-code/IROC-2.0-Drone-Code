#!/usr/bin/env python3
"""
CS20 IR optical-flow velocity estimator.

Designed to be API-compatible with the existing Phase2 flow mapper interface:
- update(...)
- set_intrinsics(...)
- get_dashboard_payload()
- save_map(...)
"""

import csv
import math
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class CS20IROpticalFlowEstimator:
    def __init__(
        self,
        fx: float = 240.0,
        fy: float = 240.0,
        cx: float = 160.0,
        cy: float = 120.0,
        max_corners: int = 500,
        min_tracked: int = 40,
        max_speed_mps: float = 2.5,
        lowpass_alpha: float = 0.45,
        min_alt_m: float = 0.08,
        stationary_flow_px: float = 0.10,
        flow_mad_k: float = 2.5,
        use_cuda: bool = True,
    ):
        self.fx = float(max(1.0, fx))
        self.fy = float(max(1.0, fy))
        self.cx = float(cx)
        self.cy = float(cy)
        self.max_corners = int(max(100, max_corners))
        self.min_tracked = int(max(8, min_tracked))
        self.max_speed_mps = float(max(0.2, max_speed_mps))
        self.lowpass_alpha = float(np.clip(lowpass_alpha, 0.05, 0.95))
        self.min_alt_m = float(max(0.02, min_alt_m))
        self.stationary_flow_px = float(max(0.01, stationary_flow_px))
        self.flow_mad_k = float(max(1.2, flow_mad_k))

        self._lock = threading.Lock()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_ts: Optional[float] = None

        self.vx_filt = 0.0
        self.vy_filt = 0.0
        self._pos_x = 0.0
        self._pos_z = 0.0
        self._traj_x = [0.0]
        self._traj_z = [0.0]
        self._last_ts = None

        self.feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.cuda_enabled = False
        self.cuda_lk = None
        if use_cuda and hasattr(cv2, "cuda"):
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, "SparsePyrLKOpticalFlow_create"):
                    self.cuda_lk = cv2.cuda.SparsePyrLKOpticalFlow_create(
                        winSize=self.lk_params["winSize"],
                        maxLevel=self.lk_params["maxLevel"],
                        iters=30,
                        useInitialFlow=False,
                    )
                    self.cuda_enabled = self.cuda_lk is not None
            except Exception:
                self.cuda_lk = None
                self.cuda_enabled = False

        self._last_debug = {
            "source": "CS20_IR_OF",
            "success": False,
            "flow_x": 0.0,
            "flow_y": 0.0,
            "alt": 0.0,
            "tracked": 0,
            "n_samples": 0,
            "median_flow_px": 0.0,
            "gyro_comp": False,
            "backend": "cuda" if self.cuda_enabled else "cpu",
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

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        if pts is None:
            return None
        return pts.astype(np.float32)

    def _track(self, prev_gray: np.ndarray, gray: np.ndarray, prev_pts: np.ndarray):
        if self.cuda_enabled and self.cuda_lk is not None:
            try:
                g_prev = cv2.cuda_GpuMat()
                g_cur = cv2.cuda_GpuMat()
                g_prev.upload(prev_gray)
                g_cur.upload(gray)
                g_pts = cv2.cuda_GpuMat()
                g_pts.upload(prev_pts)
                g_next, g_status, _ = self.cuda_lk.calc(g_prev, g_cur, g_pts, None)
                next_pts = g_next.download()
                status = g_status.download()
                return next_pts, status
            except Exception:
                pass

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **self.lk_params)
        return next_pts, status

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
        del depth_image_16uc1, imu_gyro_xyz, imu_gyro_z_rads

        if gray_image is None:
            return self.vx_filt, self.vy_filt

        if len(gray_image.shape) == 3:
            gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_image

        with self._lock:
            alt = float(max(self.min_alt_m, tof_altitude_m))

            if self.prev_gray is None or self.prev_ts is None:
                self.prev_gray = gray.copy()
                self.prev_pts = self._detect_features(self.prev_gray)
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": "bootstrap",
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            dt = float(timestamp_sec - self.prev_ts)
            if dt <= 1e-4 or dt > 0.30:
                self.prev_gray = gray.copy()
                self.prev_pts = self._detect_features(self.prev_gray)
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": f"bad_dt:{dt:.4f}",
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            if self.prev_pts is None or len(self.prev_pts) < self.min_tracked:
                self.prev_pts = self._detect_features(self.prev_gray)

            if self.prev_pts is None or len(self.prev_pts) < self.min_tracked:
                self.prev_gray = gray.copy()
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": "no_features",
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            next_pts, status = self._track(self.prev_gray, gray, self.prev_pts)
            if next_pts is None or status is None:
                self.prev_gray = gray.copy()
                self.prev_pts = self._detect_features(self.prev_gray)
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": "lk_failed",
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            n = min(len(self.prev_pts), len(next_pts), len(status))
            if n <= 0:
                self.prev_gray = gray.copy()
                self.prev_pts = self._detect_features(self.prev_gray)
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": "lk_empty",
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            p0 = self.prev_pts[:n].reshape(-1, 2)
            p1 = next_pts[:n].reshape(-1, 2)
            ok = status[:n].ravel() == 1
            tracked = int(np.sum(ok))

            if tracked < self.min_tracked:
                self.prev_gray = gray.copy()
                self.prev_pts = self._detect_features(self.prev_gray)
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": False,
                    "rejected": True,
                    "reject_reason": f"low_tracks:{tracked}",
                    "tracked": tracked,
                    "alt": alt,
                })
                return self.vx_filt, self.vy_filt

            d = p1[ok] - p0[ok]
            mag = np.linalg.norm(d, axis=1)
            med_flow = float(np.median(mag)) if len(mag) else 0.0

            if med_flow < self.stationary_flow_px:
                self.vx_filt *= 0.70
                self.vy_filt *= 0.70
                if abs(self.vx_filt) < 0.01:
                    self.vx_filt = 0.0
                if abs(self.vy_filt) < 0.01:
                    self.vy_filt = 0.0
                self.prev_gray = gray.copy()
                self.prev_pts = p1[ok].reshape(-1, 1, 2).astype(np.float32)
                self.prev_ts = float(timestamp_sec)
                self._last_debug.update({
                    "success": True,
                    "rejected": False,
                    "reject_reason": "",
                    "flow_x": self.vx_filt,
                    "flow_y": self.vy_filt,
                    "tracked": tracked,
                    "n_samples": tracked,
                    "median_flow_px": med_flow,
                    "flow_px": med_flow,
                    "inlier_ratio": 1.0,
                    "alt": alt,
                    "step_m": 0.0,
                })
                return self.vx_filt, self.vy_filt

            dx = d[:, 0]
            dy = d[:, 1]
            med_dx = float(np.median(dx))
            med_dy = float(np.median(dy))
            mad_dx = float(np.median(np.abs(dx - med_dx))) + 1e-6
            mad_dy = float(np.median(np.abs(dy - med_dy))) + 1e-6

            inlier = (np.abs(dx - med_dx) < self.flow_mad_k * mad_dx) & (np.abs(dy - med_dy) < self.flow_mad_k * mad_dy)
            if int(np.sum(inlier)) >= self.min_tracked:
                dx = dx[inlier]
                dy = dy[inlier]

            dpx = float(np.median(dx))
            dpy = float(np.median(dy))

            vx_raw = -(dpx * alt) / (self.fx * dt)
            vy_raw = -(dpy * alt) / (self.fy * dt)
            vx_raw = float(np.clip(vx_raw, -self.max_speed_mps, self.max_speed_mps))
            vy_raw = float(np.clip(vy_raw, -self.max_speed_mps, self.max_speed_mps))

            a = self.lowpass_alpha
            self.vx_filt = a * vx_raw + (1.0 - a) * self.vx_filt
            self.vy_filt = a * vy_raw + (1.0 - a) * self.vy_filt

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

            self.prev_gray = gray.copy()
            self.prev_pts = p1[ok].reshape(-1, 1, 2).astype(np.float32)
            self.prev_ts = float(timestamp_sec)

            n_samples = int(len(dx))
            inlier_ratio = float(n_samples) / float(max(1, tracked))
            self._last_debug.update({
                "success": True,
                "rejected": False,
                "reject_reason": "",
                "flow_x": self.vx_filt,
                "flow_y": self.vy_filt,
                "tracked": tracked,
                "n_samples": n_samples,
                "median_flow_px": med_flow,
                "flow_px": med_flow,
                "inlier_ratio": inlier_ratio,
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
                ax.plot(self._traj_x, self._traj_z, color="royalblue", linewidth=1.2)
                ax.plot(0.0, 0.0, "*g", markersize=14, label="Start")
                ax.plot(self._traj_x[-1], self._traj_z[-1], "Xr", markersize=10, label="End")
                ax.set_title("CS20 IR Optical Flow Route")
                ax.set_xlabel("Left / Right (m)")
                ax.set_ylabel("Forward / Back (m)")
                ax.set_aspect("equal")
                ax.grid(True)
                ax.legend()
                fig.savefig(image_path, dpi=200)
                plt.close(fig)
            return True
