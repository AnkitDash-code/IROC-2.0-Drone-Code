"""
XFeat + LightGlue TensorRT backend adapter.

This is a thin integration layer for a native C++/PyBind backend.
It keeps the same public API shape used by Phase2Node flow mappers:
- update(...)
- get_dashboard_payload()
- save_map(...)

Expected native backend behavior:
- module importable by name (default: xfeat_lightglue_trt)
- exposes either:
  - class `XFeatLightGlueTRT` with `estimate(...)` or `update(...)`
  - function `create_backend()` returning an object with estimate/update
  - module-level function `estimate(...)` or `update(...)`

Returned estimate can be either:
- tuple/list: (vx, vy)
- dict: {"vx": ..., "vy": ..., ...}
"""

import csv
import importlib
import math
import os
import sys
import threading
from typing import Any, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class XFeatLightGlueTRTBackend:
    def __init__(
        self,
        module_name: str = "xfeat_lightglue_trt",
        native_module_path: Optional[str] = None,
        config_path: str = "config/xfeat_lightglue.yaml",
        xfeat_engine_path: str = "weights/xfeat_1_800_800.engine",
        lightglue_engine_path: str = "weights/lightglue_L6_1_800_800.engine",
        velocity_scale: float = 1.0,
        lowpass_alpha: float = 0.35,
        max_vel_mps: float = 4.0,
        focal_px: float = 420.0,
        min_matches: int = 24,
        frame_stride: int = 2,
        repo_path: str = "XFeat-Lightglue-TRT",
        top_k: int = 512,
    ):
        self.module_name = module_name
        self.native_module_path = native_module_path
        self.config_path = str(config_path)
        self.xfeat_engine_path = str(xfeat_engine_path)
        self.lightglue_engine_path = str(lightglue_engine_path)
        self.velocity_scale = float(velocity_scale)
        self.lowpass_alpha = float(np.clip(lowpass_alpha, 0.01, 0.99))
        self.max_vel_mps = float(max_vel_mps)
        self.focal_px = float(max(1.0, focal_px))
        self.min_matches = int(max(8, min_matches))
        self.frame_stride = int(max(1, frame_stride))
        self.repo_path = str(repo_path)
        self.top_k = int(max(128, top_k))

        self._lock = threading.Lock()
        self._native = None
        self._py_xfeat = None
        self.available = False
        self.backend_mode = "none"
        self._frame_idx = 0

        self.vx_filt = 0.0
        self.vy_filt = 0.0

        self._pos_x = 0.0
        self._pos_z = 0.0
        self._traj_x = [0.0]
        self._traj_z = [0.0]
        self._last_ts = None

        self._last_debug = {
            "source": "XFEAT_LIGHTGLUE_TRT",
            "success": False,
            "flow_x": 0.0,
            "flow_y": 0.0,
            "tracked": 0,
            "inliers": 0,
            "backend_available": False,
            "backend_module": module_name,
            "backend_error": "",
        }

        self._init_native_backend()

    def _init_native_backend(self) -> None:
        backend_error = ""
        try:
            search_paths = []
            if self.native_module_path:
                search_paths.append(os.path.abspath(self.native_module_path))
            # Common default when repo is cloned next to this file.
            search_paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "XFeat-Lightglue-TRT", "build")))
            # Optional cwd-relative build folder.
            search_paths.append(os.path.abspath(os.path.join("XFeat-Lightglue-TRT", "build")))

            for p in search_paths:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)

            mod = importlib.import_module(self.module_name)

            cfg = os.path.abspath(self.config_path)
            eng_x = os.path.abspath(self.xfeat_engine_path)
            eng_l = os.path.abspath(self.lightglue_engine_path)

            if hasattr(mod, "XFeatLightGlueTRT"):
                # Try TensorRT-aware constructor first, then fallback signatures.
                ctor = mod.XFeatLightGlueTRT
                try:
                    self._native = ctor(cfg, eng_x, eng_l, 1.0, self.focal_px)
                except Exception:
                    try:
                        self._native = ctor(cfg, eng_x, eng_l)
                    except Exception:
                        self._native = ctor()
            elif hasattr(mod, "create_backend"):
                self._native = mod.create_backend()
            else:
                self._native = mod

            self.available = True
            self.backend_mode = "native"
            self._last_debug["backend_available"] = True
            self._last_debug["backend_error"] = ""
            self._last_debug["backend_mode"] = self.backend_mode
            self._last_debug["native_config"] = cfg
            self._last_debug["native_xfeat_engine"] = eng_x
            self._last_debug["native_lightglue_engine"] = eng_l
        except Exception as e:
            self.available = False
            self._native = None
            backend_error = str(e)

        # Fallback: use the actual cloned repo Python implementation.
        if not self.available:
            try:
                self._init_python_repo_backend()
                self.available = True
                self.backend_mode = "python_repo"
                self._last_debug["backend_available"] = True
                self._last_debug["backend_error"] = ""
                self._last_debug["backend_mode"] = self.backend_mode
            except Exception as e2:
                self.available = False
                self.backend_mode = "none"
                self._py_xfeat = None
                self._last_debug["backend_available"] = False
                if backend_error:
                    self._last_debug["backend_error"] = f"native: {backend_error} | python_repo: {e2}"
                else:
                    self._last_debug["backend_error"] = str(e2)
                self._last_debug["backend_mode"] = self.backend_mode

    def _init_python_repo_backend(self) -> None:
        repo_abs = os.path.abspath(self.repo_path)
        if not os.path.isdir(repo_abs):
            local_repo_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), self.repo_path))
            if os.path.isdir(local_repo_abs):
                repo_abs = local_repo_abs
        scripts_dir = os.path.join(repo_abs, "scripts")
        if not os.path.isdir(scripts_dir):
            raise RuntimeError(f"repo scripts path not found: {scripts_dir}")

        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from modules.xfeat import XFeat  # type: ignore

        self._py_xfeat = XFeat(top_k=self.top_k)

    def _call_python_repo(self, gray_image: np.ndarray, timestamp_sec: float, tof_altitude_m: float) -> Tuple[float, float, int, int]:
        if self._py_xfeat is None:
            raise RuntimeError("python_repo backend unavailable")

        if gray_image is None:
            return 0.0, 0.0, 0, 0

        # Upstream parser expects HxWxC arrays; promote grayscale to 3 channels.
        if len(gray_image.shape) == 2:
            frame = np.repeat(gray_image[:, :, None], 3, axis=2)
        elif len(gray_image.shape) == 3 and gray_image.shape[2] == 1:
            frame = np.repeat(gray_image, 3, axis=2)
        else:
            frame = gray_image

        self._frame_idx += 1
        if (self._frame_idx % self.frame_stride) != 0:
            return self.vx_filt, self.vy_filt, 0, 0

        if not hasattr(self, "_prev_gray") or self._prev_gray is None:
            self._prev_gray = frame.copy()
            self._prev_ts = float(timestamp_sec)
            return 0.0, 0.0, 0, 0

        dt = float(timestamp_sec - getattr(self, "_prev_ts", timestamp_sec))
        if dt <= 1e-3:
            self._prev_gray = frame.copy()
            self._prev_ts = float(timestamp_sec)
            return 0.0, 0.0, 0, 0

        mk0, mk1 = self._py_xfeat.match_xfeat(self._prev_gray, frame, top_k=self.top_k)
        self._prev_gray = frame.copy()
        self._prev_ts = float(timestamp_sec)

        if mk0 is None or mk1 is None:
            return 0.0, 0.0, 0, 0

        n = int(min(len(mk0), len(mk1)))
        if n < self.min_matches:
            return 0.0, 0.0, n, 0

        flow = mk1[:n] - mk0[:n]
        dx_px = float(np.median(flow[:, 0]))
        dy_px = float(np.median(flow[:, 1]))

        alt_m = float(max(0.05, tof_altitude_m))
        m_per_px = alt_m / self.focal_px

        # Camera looking downward: translation is opposite to image flow.
        vx = (-dx_px * m_per_px) / dt
        vy = (-dy_px * m_per_px) / dt
        return vx, vy, n, n

    def _call_native(self, gray_image: np.ndarray, timestamp_sec: float) -> Tuple[float, float, int, int]:
        if self._native is None:
            return 0.0, 0.0, 0, 0

        call_kwargs = {
            "gray": gray_image,
            "timestamp": float(timestamp_sec),
        }

        result: Any = None

        # Try common method/function names used by PyBind wrappers.
        if hasattr(self._native, "estimate"):
            result = self._native.estimate(**call_kwargs)
        elif hasattr(self._native, "update"):
            result = self._native.update(**call_kwargs)
        elif hasattr(self._native, "infer"):
            result = self._native.infer(**call_kwargs)
        elif hasattr(self._native, "estimate"):
            result = self._native.estimate(gray_image, float(timestamp_sec))
        elif hasattr(self._native, "update"):
            result = self._native.update(gray_image, float(timestamp_sec))
        elif hasattr(self._native, "infer"):
            result = self._native.infer(gray_image, float(timestamp_sec))
        else:
            # Module-level fallback
            if hasattr(self._native, "estimate"):
                result = self._native.estimate(gray_image, float(timestamp_sec))
            elif hasattr(self._native, "update"):
                result = self._native.update(gray_image, float(timestamp_sec))
            else:
                raise RuntimeError("Native backend does not expose estimate/update")

        if isinstance(result, dict):
            vx = float(result.get("vx", result.get("vel_x", 0.0)))
            vy = float(result.get("vy", result.get("vel_y", 0.0)))
            tracked = int(result.get("tracked", 0))
            inliers = int(result.get("inliers", 0))
            return vx, vy, tracked, inliers

        if isinstance(result, (tuple, list)) and len(result) >= 2:
            vx = float(result[0])
            vy = float(result[1])
            tracked = int(result[2]) if len(result) > 2 else 0
            inliers = int(result[3]) if len(result) > 3 else 0
            return vx, vy, tracked, inliers

        raise RuntimeError("Native backend returned unsupported result format")

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

        with self._lock:
            if not self.available:
                # Keep stable zero output when backend is unavailable.
                self.vx_filt *= 0.8
                self.vy_filt *= 0.8
                self._last_debug.update({
                    "success": False,
                    "flow_x": self.vx_filt,
                    "flow_y": self.vy_filt,
                })
                return self.vx_filt, self.vy_filt

            try:
                if self.backend_mode == "native":
                    vx_raw, vy_raw, tracked, inliers = self._call_native(gray_image, timestamp_sec)
                elif self.backend_mode == "python_repo":
                    vx_raw, vy_raw, tracked, inliers = self._call_python_repo(gray_image, timestamp_sec, tof_altitude_m)
                else:
                    raise RuntimeError("no active backend mode")
                vx_raw *= self.velocity_scale
                vy_raw *= self.velocity_scale

                vx_raw = float(np.clip(vx_raw, -self.max_vel_mps, self.max_vel_mps))
                vy_raw = float(np.clip(vy_raw, -self.max_vel_mps, self.max_vel_mps))

                self.vx_filt = self.lowpass_alpha * vx_raw + (1.0 - self.lowpass_alpha) * self.vx_filt
                self.vy_filt = self.lowpass_alpha * vy_raw + (1.0 - self.lowpass_alpha) * self.vy_filt

                if self._last_ts is not None:
                    dt = float(timestamp_sec - self._last_ts)
                    if 0.0 < dt < 0.5:
                        if yaw_deg is not None:
                            yr = math.radians(float(yaw_deg))
                            wx_world = math.cos(yr) * self.vx_filt - math.sin(yr) * self.vy_filt
                            wz_world = math.sin(yr) * self.vx_filt + math.cos(yr) * self.vy_filt
                        else:
                            wx_world, wz_world = self.vx_filt, self.vy_filt
                        self._pos_x += wx_world * dt
                        self._pos_z += wz_world * dt
                        self._traj_x.append(self._pos_x)
                        self._traj_z.append(self._pos_z)

                self._last_ts = float(timestamp_sec)

                self._last_debug.update({
                    "success": True,
                    "flow_x": self.vx_filt,
                    "flow_y": self.vy_filt,
                    "tracked": tracked,
                    "inliers": inliers,
                    "backend_available": True,
                    "backend_mode": self.backend_mode,
                    "backend_error": "",
                })
            except Exception as e:
                self.vx_filt *= 0.8
                self.vy_filt *= 0.8
                self._last_debug.update({
                    "success": False,
                    "flow_x": self.vx_filt,
                    "flow_y": self.vy_filt,
                    "backend_available": False,
                    "backend_mode": self.backend_mode,
                    "backend_error": str(e),
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
                print("Route map skipped: not enough trajectory points")
                return False

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x_m", "z_m"])
                for x, z in zip(self._traj_x, self._traj_z):
                    writer.writerow([f"{x:.6f}", f"{z:.6f}"])
            print(f"Saved route CSV: {csv_path}")

            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(self._traj_x, self._traj_z, color="steelblue", linewidth=1.2, marker="o", markersize=1.5)
                ax.plot(0.0, 0.0, "*g", markersize=14, label="Start")
                ax.plot(self._traj_x[-1], self._traj_z[-1], "Xr", markersize=10, label="End")
                ax.set_title("XFeat-LightGlue-TRT 2-D Route Map")
                ax.set_xlabel("Left / Right (m)")
                ax.set_ylabel("Forward / Back (m)")
                ax.set_aspect("equal")
                ax.grid(True)
                ax.legend()
                fig.savefig(image_path, dpi=200)
                plt.close(fig)
                print(f"Saved route image: {image_path}")
            else:
                print("matplotlib unavailable; PNG not generated")
            return True
