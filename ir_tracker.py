import os
import time
import json
import io
import argparse
import threading
import socketserver
import http.server
import subprocess
import sys
from ctypes import POINTER, c_ushort, byref, cast, c_int
from queue import Queue, Empty, Full
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from PIL import Image
import math
from typing import Optional

try:
    from pymavlink import mavutil
except Exception:
    mavutil = None

from SynexensPythonSDK import (
    InitSDK,
    UnInitSDK,
    FindDevice,
    OpenDevice,
    CloseDevice,
    SYDeviceInfo,
    SYErrorCodeEnum,
    SYStreamTypeEnum,
    SYFrameTypeEnum,
    SYResolutionEnum,
    SYFrameData,
    SYIntrinsics,
    GetIntric,
    GetLastFrameData,
    SetFrameResolution,
    StartStreaming,
    StopStreaming,
    PrintErrorCode,
)


_synexens_device_id = -1
_synexens_sdk_initialized = False
RANGEFINDER_INTERVAL_US = 50000
IMU_INTERVAL_US = 20000


def initialize_lidar_stream(camera_feed: str = "depth") -> bool:
    global _synexens_device_id, _synexens_sdk_initialized

    if _synexens_sdk_initialized:
        return True

    print("Lidar Initializing...")
    errorCodeInitSDK = InitSDK()
    if errorCodeInitSDK != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        return False

    nDeviceCount = c_int()
    pDeviceInfo = (SYDeviceInfo * 1)()

    errorCodeFindDevice = FindDevice(byref(nDeviceCount), None)
    if errorCodeFindDevice == SYErrorCodeEnum.SYERRORCODE_SUCCESS and nDeviceCount.value > 0:
        errorCodeFindDevice = FindDevice(byref(nDeviceCount), pDeviceInfo)
        if errorCodeFindDevice == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            _synexens_device_id = pDeviceInfo[0].m_nDeviceID
        else:
            UnInitSDK()
            return False
    else:
        UnInitSDK()
        return False

    errorCodeOpenDevice = OpenDevice(pDeviceInfo[0])
    if errorCodeOpenDevice != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        UnInitSDK()
        return False

    resolution_enum = SYResolutionEnum.SYRESOLUTION_640_480

    errorCodeSetResolution = SetFrameResolution(
        _synexens_device_id,
        SYFrameTypeEnum.SYFRAMETYPE_DEPTH,
        resolution_enum,
    )
    if errorCodeSetResolution != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        CloseDevice(_synexens_device_id)
        UnInitSDK()
        return False

    wants_ir = camera_feed in ("ir", "ir+depth")
    if wants_ir:
        try:
            SetFrameResolution(_synexens_device_id, SYFrameTypeEnum.SYFRAMETYPE_IR, resolution_enum)
        except Exception:
            pass

    stream_type = SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR if wants_ir else SYStreamTypeEnum.SYSTREAMTYPE_DEPTH
    print("Lidar Stream: Starting streaming...")
    errorCodeStartStreaming = StartStreaming(_synexens_device_id, stream_type)
    if errorCodeStartStreaming != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
        PrintErrorCode("StartStreaming", errorCodeStartStreaming)
        CloseDevice(_synexens_device_id)
        UnInitSDK()
        return False

    time.sleep(2)
    _synexens_sdk_initialized = True
    print("Lidar Initialized and streaming started successfully.")
    return True


def stop_lidar_stream() -> None:
    global _synexens_device_id, _synexens_sdk_initialized
    if _synexens_sdk_initialized and _synexens_device_id != -1:
        StopStreaming(_synexens_device_id)
        CloseDevice(_synexens_device_id)
        UnInitSDK()
        _synexens_device_id = -1
        _synexens_sdk_initialized = False
        print("Lidar Uninitialized successfully.")
    elif _synexens_sdk_initialized:
        print("Lidar was initialized but device might not have been fully open. Uninitializing SDK.")
        UnInitSDK()
        _synexens_sdk_initialized = False
    else:
        print("Lidar not initialized")


def normalize_u16_to_u8(frame_u16):
    if frame_u16 is None:
        return None
    arr = frame_u16.astype(np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    if not np.any(valid):
        return np.zeros_like(frame_u16, dtype=np.uint8)

    vals = arr[valid]
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo:
            return np.zeros_like(frame_u16, dtype=np.uint8)

    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def normalize_u8(frame_u8):
    if frame_u8 is None:
        return None
    arr = frame_u8.astype(np.float32)
    lo = float(np.percentile(arr, 2.0))
    hi = float(np.percentile(arr, 98.0))
    if hi <= lo:
        return frame_u8.astype(np.uint8)
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def enhance_for_markers(gray_u8, image_mode):
    if gray_u8 is None:
        return None
    g = gray_u8.astype(np.uint8)
    if image_mode == "raw":
        return g

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(g)
    if image_mode == "clahe":
        return eq

    thr = cv2.adaptiveThreshold(
        eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    return cv2.addWeighted(eq, 0.45, thr, 0.55, 0)


def apply_uvc_flip(gray_u8, mode):
    if gray_u8 is None:
        return None
    m = str(mode).strip().lower()
    if m == "none":
        return gray_u8
    if m == "h":
        return np.ascontiguousarray(np.fliplr(gray_u8))
    if m == "v":
        return np.ascontiguousarray(np.flipud(gray_u8))
    if m == "hv":
        return np.ascontiguousarray(np.flipud(np.fliplr(gray_u8)))
    return gray_u8


def extract_square_patch(gray_u8, cx, cy, size_px):
    if gray_u8 is None:
        return None
    h, w = gray_u8.shape[:2]
    half = int(max(4, size_px // 2))
    x0 = int(round(float(cx))) - half
    y0 = int(round(float(cy))) - half
    x1 = x0 + (2 * half)
    y1 = y0 + (2 * half)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return gray_u8[y0:y1, x0:x1].copy()


def patch_ncc_similarity(ref_patch, cand_patch):
    if ref_patch is None or cand_patch is None:
        return None
    if ref_patch.shape != cand_patch.shape:
        cand_patch = cv2.resize(cand_patch, (ref_patch.shape[1], ref_patch.shape[0]), interpolation=cv2.INTER_LINEAR)

    a = ref_patch.astype(np.float32)
    b = cand_patch.astype(np.float32)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return None
    return float(np.sum(a * b) / denom)


def locate_bright_blob_center(gray_u8, percentile=99.2, min_area=12.0):
    if gray_u8 is None:
        return None, None, 0.0
    pctl = float(max(90.0, min(99.99, percentile)))
    thr = int(max(120, min(254, np.percentile(gray_u8, pctl))))
    try:
        _, bw = cv2.threshold(gray_u8, thr, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        contours = []

    best_area = 0.0
    best_cx = None
    best_cy = None
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < float(min_area):
            continue
        M = cv2.moments(c)
        if M.get("m00", 0.0) <= 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        if area > best_area:
            best_area = area
            best_cx = cx
            best_cy = cy

    if best_cx is not None and best_cy is not None:
        return best_cx, best_cy, best_area

    _, _, _, max_loc = cv2.minMaxLoc(gray_u8)
    return float(max_loc[0]), float(max_loc[1]), 0.0


def load_seed_reference(seed_image_path, image_mode, orb_features, blob_patch_size, blob_percentile):
    if not seed_image_path:
        return None

    path = os.path.abspath(str(seed_image_path))
    seed_raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if seed_raw is None:
        print(f"[WARN] Seed image could not be read: {path}")
        return None

    seed_norm = normalize_u8(seed_raw)
    seed_gray = enhance_for_markers(seed_norm, image_mode)

    orb = cv2.ORB_create(nfeatures=int(max(200, orb_features)))
    kp_seed, des_seed = orb.detectAndCompute(seed_gray, None)
    kp_seed = kp_seed if kp_seed is not None else []

    cx, cy, area = locate_bright_blob_center(seed_gray, percentile=blob_percentile, min_area=12.0)
    blob_patch = None
    if cx is not None and cy is not None:
        blob_patch = extract_square_patch(seed_gray, cx, cy, int(max(8, blob_patch_size)))

    print(
        f"[INFO] Seed loaded: {path} | kp={len(kp_seed)} | "
        f"blob_center=({cx:.1f},{cy:.1f}) area={area:.1f} patch={'yes' if blob_patch is not None else 'no'}"
    )

    return {
        "path": path,
        "gray": seed_gray,
        "orb": orb,
        "bf": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
        "kp": kp_seed,
        "kp_count": int(len(kp_seed)),
        "center": (float(seed_gray.shape[1] * 0.5), float(seed_gray.shape[0] * 0.5)),
        "des": des_seed,
        "blob_patch": blob_patch,
        "last_state": {
            "orb_good": 0,
            "orb_norm": 0.0,
            "blob_sim": None,
            "inliers": 0,
            "alignment_valid": False,
            "error_dx_px": 0.0,
            "error_dy_px": 0.0,
            "error_norm_px": 0.0,
            "correction_command": "hold",
            "centered": False,
            "yaw_error_deg": 0.0,
            "yaw_valid": False,
            "yaw_correction_command": "hold",
            "yaw_centered": False,
            "score": 0,
            "is_match": False,
        },
    }


def evaluate_seed_match(
    gray_u8,
    seed_ref,
    lock_cx,
    lock_cy,
    patch_size,
    frame_idx,
    stride,
    orb_ratio_test,
    orb_min_good,
    orb_min_norm,
    require_blob,
    blob_sim_threshold,
    align_min_inliers,
    align_deadband_px,
    align_ransac_reproj,
    yaw_sign,
    yaw_deadband_deg,
):
    if gray_u8 is None or seed_ref is None:
        return None

    state = dict(seed_ref.get("last_state", {}))
    stride_i = int(max(1, stride))
    if int(frame_idx) % stride_i == 0:
        kp_frame, des_frame = seed_ref["orb"].detectAndCompute(gray_u8, None)
        kp_frame = kp_frame if kp_frame is not None else []
        good = 0
        seed_pts = []
        frame_pts = []

        des_seed = seed_ref.get("des", None)
        kp_seed = seed_ref.get("kp", [])
        if des_seed is not None and des_frame is not None and len(des_seed) > 1 and len(des_frame) > 1 and len(kp_seed) > 1:
            try:
                pairs = seed_ref["bf"].knnMatch(des_seed, des_frame, k=2)
                for pair in pairs:
                    if len(pair) < 2:
                        continue
                    a, b = pair
                    if a.distance < float(orb_ratio_test) * b.distance:
                        good += 1
                        if 0 <= int(a.queryIdx) < len(kp_seed) and 0 <= int(a.trainIdx) < len(kp_frame):
                            seed_pts.append(kp_seed[int(a.queryIdx)].pt)
                            frame_pts.append(kp_frame[int(a.trainIdx)].pt)
            except Exception:
                good = 0
                seed_pts = []
                frame_pts = []

        kp_seed_count = int(max(1, seed_ref.get("kp_count", 1)))
        state["orb_good"] = int(good)
        state["orb_norm"] = float(good) / float(kp_seed_count)

        frame_cx = float(gray_u8.shape[1] * 0.5)
        frame_cy = float(gray_u8.shape[0] * 0.5)
        seed_cx, seed_cy = seed_ref.get("center", (frame_cx, frame_cy))

        pred_cx = None
        pred_cy = None
        inliers = 0
        yaw_error_deg = 0.0
        yaw_valid = False
        yaw_cmd = "hold"
        yaw_centered = False

        if len(seed_pts) >= 4 and len(frame_pts) >= 4:
            seed_np = np.array(seed_pts, dtype=np.float32).reshape(-1, 2)
            frame_np = np.array(frame_pts, dtype=np.float32).reshape(-1, 2)
            try:
                M, inlier_mask = cv2.estimateAffinePartial2D(
                    seed_np,
                    frame_np,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=float(max(1.0, align_ransac_reproj)),
                    maxIters=1500,
                    confidence=0.99,
                    refineIters=8,
                )
            except Exception:
                M = None
                inlier_mask = None

            if M is not None and np.shape(M) == (2, 3):
                pt = np.array([seed_cx, seed_cy, 1.0], dtype=np.float32)
                pred = M.dot(pt)
                pred_cx = float(pred[0])
                pred_cy = float(pred[1])
                if inlier_mask is not None:
                    try:
                        inliers = int(np.sum(inlier_mask))
                    except Exception:
                        inliers = int(len(seed_pts))
                else:
                    inliers = int(len(seed_pts))

                # Affine (seed->frame) rotation gives minor yaw mismatch estimate.
                theta_deg = float(math.degrees(math.atan2(float(M[1, 0]), float(M[0, 0]))))
                yaw_error_deg = float(yaw_sign) * theta_deg
                yaw_dead = float(max(0.5, yaw_deadband_deg))
                yaw_valid = bool(inliers >= int(max(4, align_min_inliers)))
                if yaw_error_deg > yaw_dead:
                    yaw_cmd = "turn_right"
                elif yaw_error_deg < -yaw_dead:
                    yaw_cmd = "turn_left"
                else:
                    yaw_cmd = "hold"
                yaw_centered = bool(abs(yaw_error_deg) <= yaw_dead)
            else:
                dx = float(np.median(frame_np[:, 0] - seed_np[:, 0]))
                dy = float(np.median(frame_np[:, 1] - seed_np[:, 1]))
                pred_cx = float(seed_cx + dx)
                pred_cy = float(seed_cy + dy)
                inliers = int(len(seed_pts))

        if pred_cx is not None and pred_cy is not None:
            err_dx = float(pred_cx - frame_cx)
            err_dy = float(pred_cy - frame_cy)
            err_norm = float(np.hypot(err_dx, err_dy))

            cmd_x = ""
            cmd_y = ""
            dead_px = float(max(2.0, align_deadband_px))
            if err_dx > dead_px:
                cmd_x = "move_right"
            elif err_dx < -dead_px:
                cmd_x = "move_left"
            if err_dy > dead_px:
                cmd_y = "move_down"
            elif err_dy < -dead_px:
                cmd_y = "move_up"

            if cmd_x and cmd_y:
                cmd = cmd_x + "+" + cmd_y
            elif cmd_x:
                cmd = cmd_x
            elif cmd_y:
                cmd = cmd_y
            else:
                cmd = "hold"

            state["inliers"] = int(inliers)
            state["alignment_valid"] = bool(inliers >= int(max(4, align_min_inliers)))
            state["error_dx_px"] = float(err_dx)
            state["error_dy_px"] = float(err_dy)
            state["error_norm_px"] = float(err_norm)
            state["correction_command"] = cmd
            state["centered"] = bool(err_norm <= dead_px)
            state["yaw_error_deg"] = float(yaw_error_deg)
            state["yaw_valid"] = bool(yaw_valid)
            state["yaw_correction_command"] = str(yaw_cmd)
            state["yaw_centered"] = bool(yaw_centered)
        else:
            state["inliers"] = 0
            state["alignment_valid"] = False
            state["error_dx_px"] = 0.0
            state["error_dy_px"] = 0.0
            state["error_norm_px"] = 0.0
            state["correction_command"] = "hold"
            state["centered"] = False
            state["yaw_error_deg"] = 0.0
            state["yaw_valid"] = False
            state["yaw_correction_command"] = "hold"
            state["yaw_centered"] = False

    blob_sim = None
    if lock_cx is not None and lock_cy is not None and seed_ref.get("blob_patch") is not None:
        cand_patch = extract_square_patch(gray_u8, lock_cx, lock_cy, int(max(8, patch_size)))
        blob_sim = patch_ncc_similarity(seed_ref.get("blob_patch"), cand_patch)
    state["blob_sim"] = blob_sim

    orb_good = int(state.get("orb_good", 0))
    orb_norm = float(state.get("orb_norm", 0.0))
    orb_ok = orb_good >= int(max(1, orb_min_good)) and orb_norm >= float(max(0.0, orb_min_norm))

    blob_ok = (blob_sim is not None) and (float(blob_sim) >= float(blob_sim_threshold))
    is_match = bool(orb_ok and (blob_ok if require_blob else True))

    orb_score = min(1.0, float(orb_good) / float(max(1, orb_min_good)))
    norm_score = min(1.0, float(orb_norm) / float(max(1e-6, orb_min_norm)))
    scene_score = 0.5 * orb_score + 0.5 * norm_score
    blob_score = 0.0 if blob_sim is None else max(0.0, min(1.0, (float(blob_sim) + 1.0) * 0.5))
    combo = (0.7 * scene_score) + (0.3 * blob_score)
    state["score"] = int(max(0, min(100, round(100.0 * combo))))
    state["is_match"] = bool(is_match)

    seed_ref["last_state"] = state
    return state


def wrap_angle_360(angle_deg):
    return float(angle_deg) % 360.0


def angle_diff_deg(target_deg, current_deg):
    return ((float(target_deg) - float(current_deg) + 540.0) % 360.0) - 180.0


def blend_angle_deg(base_deg, meas_deg, alpha):
    alpha_c = float(max(0.0, min(1.0, alpha)))
    return wrap_angle_360(float(base_deg) + alpha_c * angle_diff_deg(meas_deg, base_deg))


class AngleKalman1D:
    """1D angle+bias Kalman filter (gyro rate + angle measurement)."""

    def __init__(self, q_angle=0.02, q_bias=0.003, r_measure=0.6, wrap=False):
        self.q_angle = float(max(1e-8, q_angle))
        self.q_bias = float(max(1e-8, q_bias))
        self.r_measure = float(max(1e-8, r_measure))
        self.wrap = bool(wrap)

        self.angle = 0.0
        self.bias = 0.0
        self.P = np.zeros((2, 2), dtype=np.float64)
        self.initialized = False

    def set_angle(self, angle_deg):
        a = float(angle_deg)
        if self.wrap:
            a = wrap_angle_360(a)
        self.angle = a
        self.initialized = True

    def update(self, rate_deg_s, dt_s, meas_angle_deg=None):
        dt = float(max(1e-4, dt_s))
        rate = float(rate_deg_s)

        if (not self.initialized) and (meas_angle_deg is not None):
            self.set_angle(meas_angle_deg)

        # Predict: angle from gyro, bias random walk.
        self.angle += dt * (rate - self.bias)
        if self.wrap:
            self.angle = wrap_angle_360(self.angle)

        self.P[0, 0] += dt * (dt * self.P[1, 1] - self.P[0, 1] - self.P[1, 0] + self.q_angle)
        self.P[0, 1] -= dt * self.P[1, 1]
        self.P[1, 0] -= dt * self.P[1, 1]
        self.P[1, 1] += self.q_bias * dt

        # Correct with angle measurement when available.
        if meas_angle_deg is not None:
            z = float(meas_angle_deg)
            if self.wrap:
                innovation = angle_diff_deg(z, self.angle)
            else:
                innovation = z - self.angle

            S = self.P[0, 0] + self.r_measure
            K0 = self.P[0, 0] / S
            K1 = self.P[1, 0] / S

            self.angle += K0 * innovation
            self.bias += K1 * innovation
            if self.wrap:
                self.angle = wrap_angle_360(self.angle)

            P00 = float(self.P[0, 0])
            P01 = float(self.P[0, 1])
            self.P[0, 0] -= K0 * P00
            self.P[0, 1] -= K0 * P01
            self.P[1, 0] -= K1 * P00
            self.P[1, 1] -= K1 * P01

        return float(self.angle)


def read_imu_state(json_path):
    if not json_path:
        return (0.0, 0.0, 0.0), None, None
    try:
        with open(json_path, "r") as f:
            obj = json.load(f)
        gx = float(obj.get("gx", obj.get("gyro_x", obj.get("wx", 0.0))))
        gy = float(obj.get("gy", obj.get("gyro_y", obj.get("wy", 0.0))))
        gz = float(obj.get("gz", obj.get("gyro_z", obj.get("wz", 0.0))))
        yaw = obj.get("yaw_deg", obj.get("yaw", None))
        alt = obj.get("altitude_m", obj.get("alt", None))
        yaw_deg = float(yaw) if yaw is not None else None
        alt_m = float(alt) if alt is not None else None
        return (gx, gy, gz), yaw_deg, alt_m
    except Exception:
        return (0.0, 0.0, 0.0), None, None


def quaternion_to_yaw_deg(qx, qy, qz, qw, yaw_sign=1.0):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return (float(yaw_sign) * yaw_rad * 180.0 / math.pi) % 360.0


def quaternion_to_rpy_deg(qx, qy, qz, qw, yaw_sign=1.0):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll_rad = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch_rad = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch_rad = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)

    roll_deg = math.degrees(roll_rad)
    pitch_deg = math.degrees(pitch_rad)
    yaw_deg = (float(yaw_sign) * math.degrees(yaw_rad)) % 360.0
    return roll_deg, pitch_deg, yaw_deg


def _nested_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_imu_extended(json_path, yaw_sign=1.0):
    gyro, yaw_deg, alt_m = read_imu_state(json_path)
    out = {
        "gx": float(gyro[0]),
        "gy": float(gyro[1]),
        "gz": float(gyro[2]),
        "yaw_deg": yaw_deg,
        "alt_m": alt_m,
        "roll_deg": None,
        "pitch_deg": None,
        "timestamp": None,
    }
    if not json_path:
        return out
    try:
        with open(json_path, "r") as f:
            obj = json.load(f)

        # Accept sensor_msgs/IMU-like schema from MAVROS bridge JSON dumps.
        avx = _nested_get(obj, ["angular_velocity", "x"], None)
        avy = _nested_get(obj, ["angular_velocity", "y"], None)
        avz = _nested_get(obj, ["angular_velocity", "z"], None)
        if avx is not None:
            out["gx"] = float(avx)
        elif obj.get("angular_velocity_x") is not None:
            out["gx"] = float(obj.get("angular_velocity_x"))
        if avy is not None:
            out["gy"] = float(avy)
        elif obj.get("angular_velocity_y") is not None:
            out["gy"] = float(obj.get("angular_velocity_y"))
        if avz is not None:
            out["gz"] = float(avz)
        elif obj.get("angular_velocity_z") is not None:
            out["gz"] = float(obj.get("angular_velocity_z"))

        qx = _nested_get(obj, ["orientation", "x"], obj.get("orientation_x", None))
        qy = _nested_get(obj, ["orientation", "y"], obj.get("orientation_y", None))
        qz = _nested_get(obj, ["orientation", "z"], obj.get("orientation_z", None))
        qw = _nested_get(obj, ["orientation", "w"], obj.get("orientation_w", None))

        if qx is not None and qy is not None and qz is not None and qw is not None:
            try:
                out["yaw_deg"] = quaternion_to_yaw_deg(float(qx), float(qy), float(qz), float(qw), yaw_sign=yaw_sign)
            except Exception:
                pass

        out["roll_deg"] = (
            float(obj["roll_deg"]) if obj.get("roll_deg") is not None else (float(obj["roll"]) if obj.get("roll") is not None else None)
        )
        out["pitch_deg"] = (
            float(obj["pitch_deg"]) if obj.get("pitch_deg") is not None else (float(obj["pitch"]) if obj.get("pitch") is not None else None)
        )
        if out["yaw_deg"] is None:
            if obj.get("yaw_deg") is not None:
                out["yaw_deg"] = float(obj.get("yaw_deg"))
            elif obj.get("yaw") is not None:
                out["yaw_deg"] = float(obj.get("yaw"))
            elif obj.get("heading_deg") is not None:
                out["yaw_deg"] = float(obj.get("heading_deg"))
        if out["alt_m"] is None:
            if obj.get("altitude_m") is not None:
                out["alt_m"] = float(obj.get("altitude_m"))
            elif obj.get("alt") is not None:
                out["alt_m"] = float(obj.get("alt"))
            elif obj.get("range") is not None:
                out["alt_m"] = float(obj.get("range"))
        if obj.get("timestamp") is not None:
            out["timestamp"] = float(obj.get("timestamp"))
        elif _nested_get(obj, ["header", "stamp", "sec"], None) is not None:
            sec = float(_nested_get(obj, ["header", "stamp", "sec"], 0.0))
            nsec = float(_nested_get(obj, ["header", "stamp", "nanosec"], 0.0))
            out["timestamp"] = sec + nsec * 1e-9
    except Exception:
        pass
    return out


def find_python_with_flask():
    # Prefer system Python first because many Jetson setups install Flask there.
    candidates = ["/usr/bin/python3", sys.executable, "python3"]
    for py in candidates:
        try:
            subprocess.check_call([py, "-c", "import flask"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return py
        except Exception:
            continue
    return None


class MavRangefinderClient:
    SENSOR_MSG_TYPES = [
        "RANGEFINDER",
        "DISTANCE_SENSOR",
        "NAMED_VALUE_FLOAT",
        "ATTITUDE",
        "ATTITUDE_QUATERNION",
        "HIGHRES_IMU",
        "RAW_IMU",
        "SCALED_IMU",
        "SCALED_IMU2",
        "SCALED_IMU3",
    ]

    def __init__(self, connection: str, rangefinder_interval_us: int = RANGEFINDER_INTERVAL_US):
        self.connection = connection
        self.rangefinder_interval_us = int(max(10000, rangefinder_interval_us))
        self.master = None
        self.last_alt_msg_type = "none"
        self.last_imu_msg_type = "none"
        self.last_print_time = 0.0
        self.last_altitude_m = None
        self.last_altitude_ts = 0.0
        self.last_imu_state = None
        self.last_imu_ts = 0.0

    def connect(self, heartbeat_timeout: float = 8.0) -> bool:
        if mavutil is None:
            print("pymavlink not available: rangefinder MAVLink disabled")
            return False
        try:
            self.master = mavutil.mavlink_connection(self.connection, autoreconnect=True)
            self.master.wait_heartbeat(timeout=heartbeat_timeout)
            print(f"Connected MAVLink rangefinder source: {self.connection}")
            return True
        except Exception as e:
            print(f"Failed to connect MAVLink rangefinder source: {e}")
            self.master = None
            return False

    def _merge_imu_state(self, gx=None, gy=None, gz=None, roll_deg=None, pitch_deg=None, yaw_deg=None, msg_type="none"):
        base = {
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "yaw_deg": None,
            "alt_m": None,
            "roll_deg": None,
            "pitch_deg": None,
            "timestamp": None,
        }
        if isinstance(self.last_imu_state, dict):
            base.update(self.last_imu_state)

        if gx is not None:
            base["gx"] = float(gx)
        if gy is not None:
            base["gy"] = float(gy)
        if gz is not None:
            base["gz"] = float(gz)
        if roll_deg is not None:
            base["roll_deg"] = float(roll_deg)
        if pitch_deg is not None:
            base["pitch_deg"] = float(pitch_deg)
        if yaw_deg is not None:
            base["yaw_deg"] = float(yaw_deg)

        now = time.time()
        base["timestamp"] = now
        self.last_imu_state = base
        self.last_imu_ts = now
        self.last_imu_msg_type = str(msg_type)

    def _update_from_message(self, msg, yaw_sign=1.0):
        msg_type = msg.get_type()

        if msg_type == "RANGEFINDER":
            distance = getattr(msg, "distance", None)
            if distance is None:
                return
            self.last_altitude_m = float(distance)
            self.last_altitude_ts = time.time()
            self.last_alt_msg_type = "RANGEFINDER"
            return

        if msg_type == "DISTANCE_SENSOR":
            current_cm = getattr(msg, "current_distance", None)
            if current_cm is None:
                return
            self.last_altitude_m = float(current_cm) / 100.0
            self.last_altitude_ts = time.time()
            self.last_alt_msg_type = "DISTANCE_SENSOR"
            return

        if msg_type == "NAMED_VALUE_FLOAT":
            raw_name = getattr(msg, "name", "")
            name = str(raw_name).strip().lower()
            if name in {"rangefinder1", "rangefinder", "rngfnd1", "rngfnd"}:
                value = getattr(msg, "value", None)
                if value is None:
                    return
                value_f = float(value)
                if value_f > 10.0:
                    value_f = value_f / 100.0
                self.last_altitude_m = value_f
                self.last_altitude_ts = time.time()
                self.last_alt_msg_type = f"NAMED_VALUE_FLOAT:{name}"
            return

        if msg_type == "ATTITUDE":
            roll_rad = getattr(msg, "roll", None)
            pitch_rad = getattr(msg, "pitch", None)
            yaw_rad = getattr(msg, "yaw", None)
            roll_deg = math.degrees(float(roll_rad)) if roll_rad is not None else None
            pitch_deg = math.degrees(float(pitch_rad)) if pitch_rad is not None else None
            yaw_deg = (float(yaw_sign) * math.degrees(float(yaw_rad))) % 360.0 if yaw_rad is not None else None
            gx = getattr(msg, "rollspeed", None)
            gy = getattr(msg, "pitchspeed", None)
            gz = getattr(msg, "yawspeed", None)
            self._merge_imu_state(gx=gx, gy=gy, gz=gz, roll_deg=roll_deg, pitch_deg=pitch_deg, yaw_deg=yaw_deg, msg_type="ATTITUDE")
            return

        if msg_type == "ATTITUDE_QUATERNION":
            qw = getattr(msg, "q1", None)
            qx = getattr(msg, "q2", None)
            qy = getattr(msg, "q3", None)
            qz = getattr(msg, "q4", None)

            roll_deg = None
            pitch_deg = None
            yaw_deg = None
            if None not in (qx, qy, qz, qw):
                try:
                    roll_deg, pitch_deg, yaw_deg = quaternion_to_rpy_deg(
                        float(qx),
                        float(qy),
                        float(qz),
                        float(qw),
                        yaw_sign=float(yaw_sign),
                    )
                except Exception:
                    pass

            gx = getattr(msg, "rollspeed", None)
            gy = getattr(msg, "pitchspeed", None)
            gz = getattr(msg, "yawspeed", None)
            self._merge_imu_state(
                gx=gx,
                gy=gy,
                gz=gz,
                roll_deg=roll_deg,
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
                msg_type="ATTITUDE_QUATERNION",
            )
            return

        if msg_type == "HIGHRES_IMU":
            gx = getattr(msg, "xgyro", None)
            gy = getattr(msg, "ygyro", None)
            gz = getattr(msg, "zgyro", None)
            self._merge_imu_state(gx=gx, gy=gy, gz=gz, msg_type="HIGHRES_IMU")
            return

        if msg_type in {"RAW_IMU", "SCALED_IMU", "SCALED_IMU2", "SCALED_IMU3"}:
            gx = getattr(msg, "xgyro", None)
            gy = getattr(msg, "ygyro", None)
            gz = getattr(msg, "zgyro", None)
            if gx is not None:
                gx = float(gx) * 0.001
            if gy is not None:
                gy = float(gy) * 0.001
            if gz is not None:
                gz = float(gz) * 0.001
            self._merge_imu_state(gx=gx, gy=gy, gz=gz, msg_type=msg_type)

    def pump_messages(self, timeout: float = 0.0, yaw_sign: float = 1.0, max_msgs: int = 240) -> int:
        if self.master is None:
            return 0

        timeout = float(max(0.0, timeout))
        processed = 0
        deadline = time.time() + timeout

        while processed < int(max(1, max_msgs)):
            if timeout > 0.0 and time.time() >= deadline:
                break

            blocking = (processed == 0 and timeout > 0.0)
            wait_s = max(0.0, deadline - time.time()) if blocking else 0.0
            msg = self.master.recv_match(
                type=self.SENSOR_MSG_TYPES,
                blocking=blocking,
                timeout=wait_s,
            )
            if msg is None:
                break

            self._update_from_message(msg, yaw_sign=yaw_sign)
            processed += 1

        return processed

    def get_altitude_m(self, timeout: float = 0.0, max_age_s: float = 0.25, yaw_sign: float = 1.0, max_msgs: int = 240) -> Optional[float]:
        """
        Rangefinder-only altitude source.
        """
        if self.master is None:
            return None
        self.pump_messages(timeout=timeout, yaw_sign=yaw_sign, max_msgs=max_msgs)
        if self.last_altitude_m is None:
            return None
        if max_age_s is not None and (time.time() - self.last_altitude_ts) > float(max_age_s):
            return None
        return float(self.last_altitude_m)

    def get_imu_state(self, timeout: float = 0.0, max_age_s: float = 0.6, yaw_sign: float = 1.0):
        if self.master is None:
            return None
        self.pump_messages(timeout=timeout, yaw_sign=yaw_sign)
        if not isinstance(self.last_imu_state, dict):
            return None
        if max_age_s is not None and (time.time() - self.last_imu_ts) > float(max_age_s):
            return None
        return dict(self.last_imu_state)

    def _request_message_interval(self, msg_id: int, interval_us: int):
        if self.master is None or mavutil is None:
            return
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    def request_rangefinder_stream(self) -> None:
        if self.master is None or mavutil is None:
            return
        for msg_name in ("MAVLINK_MSG_ID_RANGEFINDER", "MAVLINK_MSG_ID_DISTANCE_SENSOR"):
            msg_id = getattr(mavutil.mavlink, msg_name, None)
            if msg_id is None:
                continue
            self._request_message_interval(msg_id, self.rangefinder_interval_us)

        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
            10,
            1,
        )

    def request_imu_stream(self) -> None:
        if self.master is None or mavutil is None:
            return

        for msg_name in ("MAVLINK_MSG_ID_ATTITUDE", "MAVLINK_MSG_ID_ATTITUDE_QUATERNION", "MAVLINK_MSG_ID_HIGHRES_IMU"):
            msg_id = getattr(mavutil.mavlink, msg_name, None)
            if msg_id is None:
                continue
            self._request_message_interval(msg_id, IMU_INTERVAL_US)

        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
            50,
            1,
        )
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
            50,
            1,
        )

    def wait_for_rangefinder(self, timeout_s: float = 10.0) -> None:
        print("[CHECK] Waiting for rangefinder altitude data (RANGEFINDER/DISTANCE_SENSOR/rangefinder1)...")
        self.request_rangefinder_stream()
        deadline = time.time() + timeout_s
        next_request = time.time() + 2.0
        while time.time() < deadline:
            alt = self.get_altitude_m(timeout=0.5, max_age_s=None)
            if alt is not None:
                print(f"[OK] {self.last_alt_msg_type} active. Altitude={alt:.2f} m")
                return
            if time.time() >= next_request:
                self.request_rangefinder_stream()
                next_request = time.time() + 2.0
        raise RuntimeError(
            "No RANGEFINDER/DISTANCE_SENSOR altitude data received. "
            "Check rangefinder stream rate and MAVLink message forwarding."
        )

    def wait_for_imu(self, timeout_s: float = 6.0, yaw_sign: float = 1.0) -> None:
        print("[CHECK] Waiting for FCU IMU data (ATTITUDE/ATTITUDE_QUATERNION/HIGHRES_IMU)...")
        self.request_imu_stream()
        deadline = time.time() + timeout_s
        next_request = time.time() + 2.0
        while time.time() < deadline:
            imu_state = self.get_imu_state(timeout=0.4, max_age_s=None, yaw_sign=yaw_sign)
            if isinstance(imu_state, dict):
                gx = float(imu_state.get("gx", 0.0))
                gy = float(imu_state.get("gy", 0.0))
                gz = float(imu_state.get("gz", 0.0))
                yaw_deg = imu_state.get("yaw_deg")
                yaw_txt = f" yaw={float(yaw_deg):.1f}deg" if yaw_deg is not None else ""
                print(f"[OK] {self.last_imu_msg_type} active. gyro=({gx:+.3f},{gy:+.3f},{gz:+.3f}){yaw_txt}")
                return
            if time.time() >= next_request:
                self.request_imu_stream()
                next_request = time.time() + 2.0
        raise RuntimeError(
            "No FCU IMU data received. Check MAVLink forwarding and ATTITUDE stream rates."
        )

    def log_altitude(self, prefix: str) -> None:
        now = time.time()
        if now - self.last_print_time < 1.0:
            return
        alt = self.get_altitude_m(timeout=0.2)
        if alt is not None:
            print(f"{prefix} altitude={alt:.2f} m")
            self.last_print_time = now

    def close(self):
        try:
            if self.master is not None:
                self.master.close()
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard", default="http://localhost:5000/api/upload_frame", help="Dashboard upload URL")
    p.add_argument("--interval", type=float, default=0.05, help="Seconds between uploads")
    p.add_argument("--source", choices=["sdk", "uvc"], default="sdk", help="Frame source backend")
    p.add_argument("--video-device", default="/dev/video0", help="UVC device path used when --source uvc")
    p.add_argument("--uvc-flip", choices=["none", "h", "v", "hv"], default="hv", help="Flip mode for UVC frame orientation")
    p.add_argument("--mjpeg-port", type=int, default=8080, help="Port to serve MJPEG on /video_feed (0 to disable)")
    p.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality for stream/upload")
    p.add_argument("--post-timeout", type=float, default=0.2, help="HTTP post timeout seconds")
    p.add_argument("--post-retries", type=int, default=1, help="HTTP post retries")
    p.add_argument("--camera-feed", choices=["ir", "depth", "ir+depth"], default="ir", help="Image stream selection")
    p.add_argument("--image-mode", choices=["raw", "clahe", "adaptive"], default="clahe", help="Image enhancement mode")
    p.add_argument("--phase2-fusion", action="store_true", help="Use old Phase2 MSCKF fusion path")
    p.add_argument("--imu-source", choices=["auto", "json", "mavlink"], default="auto", help="IMU source selection")
    p.add_argument("--imu-json", default="", help="Path to JSON file with imu/yaw/alt fields")
    p.add_argument("--imu-connection", default="", help="MAVLink connection string for IMU stream. If empty, reuses --rangefinder-connection")
    p.add_argument("--imu-wait-s", type=float, default=5.0, help="Seconds to wait for initial IMU message")
    p.add_argument("--start-dashboard", action="store_true", help="Auto-launch web_dashboard.py in subprocess")
    p.add_argument("--blob-threshold", type=int, default=230, help="Binary threshold for IR glare blob")
    p.add_argument("--blob-min-area", type=float, default=50.0, help="Minimum blob area for lock")
    p.add_argument("--ema-alpha", type=float, default=0.35, help="EMA alpha for centroid smoothing (0..1)")
    p.add_argument("--lock-max-jump-px", type=float, default=120.0, help="Max centroid jump to keep same target lock")
    p.add_argument("--lock-hold-frames", type=int, default=20, help="Frames to hold last lock when blob is briefly lost")
    p.add_argument("--strict-main-blob", action=argparse.BooleanOptionalAction, default=True, help="Do not switch to a different blob after first lock")
    p.add_argument("--lock-template-size", type=int, default=44, help="Template patch size for strict blob identity")
    p.add_argument("--lock-template-threshold", type=float, default=0.30, help="Min NCC similarity to keep strict blob identity")
    p.add_argument("--lock-seed-area-min-ratio", type=float, default=0.35, help="Min area ratio vs initial locked blob area")
    p.add_argument("--lock-seed-area-max-ratio", type=float, default=2.8, help="Max area ratio vs initial locked blob area")
    p.add_argument("--lock-template-update-alpha", type=float, default=0.08, help="Template update alpha for strict blob identity")
    p.add_argument("--seed-image", default="", help="Path to seed image of base station for scene matching")
    p.add_argument("--seed-label", default="base_seed", help="Label published when seed image context is matched")
    p.add_argument("--seed-orb-features", type=int, default=1200, help="ORB feature count for seed matching")
    p.add_argument("--seed-match-stride", type=int, default=2, help="Evaluate seed ORB match every N frames")
    p.add_argument("--seed-orb-ratio-test", type=float, default=0.75, help="Lowe ratio threshold for seed ORB matches")
    p.add_argument("--seed-orb-min-good", type=int, default=14, help="Minimum good ORB matches to accept seed scene")
    p.add_argument("--seed-orb-min-norm", type=float, default=0.02, help="Minimum normalized ORB match ratio vs seed keypoints")
    p.add_argument("--seed-require-blob", action="store_true", help="Require seed blob-patch similarity in addition to ORB scene match")
    p.add_argument("--seed-blob-sim-threshold", type=float, default=0.20, help="Min NCC similarity (-1..1) for seed blob confirmation")
    p.add_argument("--seed-blob-percentile", type=float, default=99.2, help="Percentile used to find bright seed blob center")
    p.add_argument("--seed-align-min-inliers", type=int, default=8, help="Minimum inlier feature matches for seed alignment vector")
    p.add_argument("--seed-align-deadband-px", type=float, default=12.0, help="Pixel deadband for seed correction command")
    p.add_argument("--seed-align-ransac-reproj", type=float, default=6.0, help="RANSAC reprojection threshold for seed affine estimate")
    p.add_argument("--seed-yaw-sign", type=float, default=1.0, help="Sign multiplier for seed yaw correction (+1 default, use -1 if reversed)")
    p.add_argument("--seed-yaw-deadband-deg", type=float, default=3.0, help="Yaw deadband in degrees for turn_left/turn_right command")
    p.add_argument("--enable-rate-limit", action="store_true", help="Rate-limit drift command transitions")
    p.add_argument("--rate-limit-ms", type=int, default=120, help="Minimum milliseconds between command changes")
    p.add_argument("--auto-exposure-threshold", action="store_true", help="Use dynamic threshold from frame percentile")
    p.add_argument("--auto-threshold-percentile", type=float, default=99.2, help="Percentile used by auto threshold")
    p.add_argument("--capture-base-on-lock", action="store_true", help="Capture first stable lock as base reference point")
    p.add_argument("--base-lock-frames", type=int, default=8, help="Consecutive lock frames before base capture")
    p.add_argument("--hover-ema-alpha", type=float, default=0.2, help="EMA alpha for hover drift estimate")
    p.add_argument("--imu-fusion", action="store_true", help="Enable IMU orientation fusion for roll/pitch/yaw")
    p.add_argument("--imu-fusion-filter", choices=["kalman", "complementary"], default="kalman", help="Orientation fusion filter type")
    p.add_argument("--imu-fusion-alpha", type=float, default=0.92, help="Complementary filter alpha (used when --imu-fusion-filter complementary)")
    p.add_argument("--imu-kf-q-angle", type=float, default=0.02, help="Kalman process noise for angle state")
    p.add_argument("--imu-kf-q-bias", type=float, default=0.003, help="Kalman process noise for gyro bias")
    p.add_argument("--imu-kf-r-measure", type=float, default=0.6, help="Kalman measurement noise for attitude angles")
    p.add_argument("--imu-yaw-sign", type=float, default=1.0, help="Yaw sign multiplier for quaternion-derived yaw")
    p.add_argument("--track-alt-min-m", type=float, default=1.5, help="Enable blob tracking only above this altitude (m)")
    p.add_argument("--sticky-lock-forever", action=argparse.BooleanOptionalAction, default=True, help="Keep same tracked object lock even if blob disappears")
    p.add_argument("--rangefinder-connection", default="", help="MAVLink connection string for rangefinder altitude (e.g. udp:127.0.0.1:14550 or /dev/ttyACM0,57600)")
    p.add_argument("--rangefinder-wait-s", type=float, default=8.0, help="Seconds to wait for initial rangefinder message")
    p.add_argument("--rangefinder-rate-hz", type=float, default=20.0, help="Requested rangefinder MAVLink rate in Hz (higher reduces takeoff lag)")
    p.add_argument("--rangefinder-max-age-s", type=float, default=0.25, help="Drop rangefinder samples older than this age in seconds")
    p.add_argument("--rangefinder-poll-timeout", type=float, default=0.0, help="Per-frame MAVLink wait in seconds for new rangefinder sample (0 for non-blocking)")
    p.add_argument("--rangefinder-max-msgs", type=int, default=240, help="Max MAVLink sensor messages drained per polling call")
    args = p.parse_args()

    seed_ref = None
    if str(args.seed_image).strip():
        seed_ref = load_seed_reference(
            seed_image_path=str(args.seed_image).strip(),
            image_mode=args.image_mode,
            orb_features=int(args.seed_orb_features),
            blob_patch_size=int(args.lock_template_size),
            blob_percentile=float(args.seed_blob_percentile),
        )
        if seed_ref is None:
            print("[WARN] Seed matching disabled because seed image could not be loaded.")
        else:
            print(f"[INFO] Seed matcher active with label='{args.seed_label}'.")

    dashboard_state = {"proc": None, "thread_started": False}

    if args.source == "sdk":
        initialize_lidar_stream(args.camera_feed)
    else:
        print(f"Using UVC source from {args.video_device}")

    rangefinder = None
    rangefinder_conn = str(args.rangefinder_connection).strip()
    if rangefinder_conn:
        rf_rate_hz = float(max(1.0, args.rangefinder_rate_hz))
        rf_interval_us = int(max(10000, round(1e6 / rf_rate_hz)))
        rangefinder = MavRangefinderClient(rangefinder_conn, rangefinder_interval_us=rf_interval_us)
        if rangefinder.connect():
            try:
                rangefinder.wait_for_rangefinder(timeout_s=float(max(0.5, args.rangefinder_wait_s)))
            except Exception as e:
                print(f"Rangefinder warmup warning: {e}")
        else:
            rangefinder = None

    imu_client = None
    imu_source_mode = str(args.imu_source).strip().lower()
    imu_conn = str(args.imu_connection).strip()
    if imu_source_mode in ("auto", "mavlink"):
        if not imu_conn and rangefinder is not None:
            imu_client = rangefinder
            print("IMU MAVLink source: reusing rangefinder connection")
        elif imu_conn:
            if rangefinder is not None and imu_conn == rangefinder_conn:
                imu_client = rangefinder
                print("IMU MAVLink source: sharing explicit rangefinder connection")
            else:
                imu_client = MavRangefinderClient(imu_conn)
                if not imu_client.connect():
                    imu_client = None
                    print("IMU MAVLink connect failed; falling back to JSON/none")
        elif imu_source_mode == "mavlink":
            print("IMU source is mavlink but no connection was provided. Set --imu-connection or --rangefinder-connection.")

        if imu_client is not None:
            try:
                imu_client.wait_for_imu(timeout_s=float(max(0.5, args.imu_wait_s)), yaw_sign=float(args.imu_yaw_sign))
            except Exception as e:
                print(f"IMU warmup warning: {e}")

    flow_mapper = None
    if args.phase2_fusion:
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if repo_root not in sys.path:
                sys.path.append(repo_root)
            from optical_flow_realsense.msckf_velocity_estimator import MSCKFVelocityEstimator

            flow_mapper = MSCKFVelocityEstimator()
            print("Phase2 fusion enabled: MSCKFVelocityEstimator")
        except Exception as e:
            print(f"Phase2 fusion unavailable: {e}")
            flow_mapper = None

    cached_caminfo = {}
    if args.source == "sdk":
        try:
            intr = SYIntrinsics()
            if GetIntric(_synexens_device_id, SYResolutionEnum.SYRESOLUTION_640_480, intr) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                cached_caminfo = {
                    "fx": float(intr.m_fltFocalDistanceX),
                    "fy": float(intr.m_fltFocalDistanceY),
                    "cx": float(intr.m_fltCenterPointX),
                    "cy": float(intr.m_fltCenterPointY),
                }
        except Exception:
            cached_caminfo = {}

    cap = None
    if args.source == "uvc":
        cap = cv2.VideoCapture(args.video_device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open UVC device: {args.video_device}")

    latest_jpeg = None
    jpeg_lock = threading.Lock()
    http_session = requests.Session()
    http_session.mount("http://", HTTPAdapter(pool_connections=2, pool_maxsize=8, max_retries=0))
    http_session.mount("https://", HTTPAdapter(pool_connections=2, pool_maxsize=8, max_retries=0))

    def post_to_dashboard(files, meta, attempts=1, timeout=0.2):
        if not args.dashboard:
            return False
        for i in range(attempts):
            try:
                resp = http_session.post(args.dashboard, files=files, data={"meta": json.dumps(meta)}, timeout=timeout)
                if 200 <= resp.status_code < 300:
                    return True
                if i == attempts - 1:
                    print(f"POST failed: {resp.status_code}")
            except Exception:
                if i == attempts - 1:
                    print("POST exception")
            time.sleep(0.02 * (i + 1))
        return False

    def dashboard_alive(url):
        if not url:
            return False
        try:
            base = url
            if "/api/upload_frame" in base:
                base = base.split("/api/upload_frame")[0] + "/api/health"
            r = requests.get(base, timeout=0.4)
            return 200 <= r.status_code < 300
        except Exception:
            return False

    def should_autostart_dashboard(url):
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").strip().lower()
            return host in ("localhost", "127.0.0.1", "0.0.0.0")
        except Exception:
            return False

    def start_dashboard_async(reason=""):
        if dashboard_state["thread_started"]:
            return
        dashboard_state["thread_started"] = True

        def _starter():
            dashboard_script = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "optical_flow_realsense", "web_dashboard.py")
            )
            if not os.path.exists(dashboard_script):
                print(f"Dashboard script not found: {dashboard_script}")
                return

            py = find_python_with_flask()
            if py is None:
                print("Flask is not installed in any known Python interpreter. Install Flask first.")
                return

            try:
                msg = f"Auto-starting dashboard ({reason})" if reason else "Auto-starting dashboard"
                print(f"{msg}: {py} {dashboard_script}")
                dashboard_state["proc"] = subprocess.Popen(
                    [py, dashboard_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=False,
                )
                for _ in range(25):
                    if dashboard_alive(args.dashboard):
                        print(f"Dashboard started, PID={dashboard_state['proc'].pid}")
                        return
                    time.sleep(0.2)
                print("Dashboard process launched but health check did not pass in time.")
            except Exception as e:
                print("Failed to auto-start dashboard:", e)

        t = threading.Thread(target=_starter, daemon=True)
        t.start()

    post_q = Queue(maxsize=1)
    post_stop = threading.Event()
    post_stats = {"ok": 0, "fail": 0}

    def post_worker():
        while not post_stop.is_set():
            try:
                item = post_q.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                break
            files, meta = item
            try:
                ok = post_to_dashboard(
                    files,
                    meta,
                    attempts=max(1, int(args.post_retries)),
                    timeout=max(0.05, float(args.post_timeout)),
                )
                if ok:
                    post_stats["ok"] += 1
                else:
                    post_stats["fail"] += 1
            except Exception:
                post_stats["fail"] += 1

    post_thread = None
    if args.dashboard:
        post_thread = threading.Thread(target=post_worker, daemon=True)
        post_thread.start()
        local_dashboard = should_autostart_dashboard(args.dashboard)
        if args.start_dashboard:
            start_dashboard_async(reason="--start-dashboard")
        elif local_dashboard and not dashboard_alive(args.dashboard):
            start_dashboard_async(reason="local dashboard autostart")
        if not dashboard_alive(args.dashboard):
            print(f"WARNING: dashboard endpoint not reachable: {args.dashboard}")
            if local_dashboard:
                start_dashboard_async(reason="endpoint unreachable")
            else:
                print("         start dashboard manually: /usr/bin/python3 optical_flow_realsense/web_dashboard.py")

    class MJPEGHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/video_feed":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with jpeg_lock:
                        data = latest_jpeg
                    if data:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                    time.sleep(max(0.02, args.interval))
            except Exception:
                return

    if args.mjpeg_port and args.mjpeg_port > 0:
        class ReuseTCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        server = None
        port = args.mjpeg_port
        for _ in range(0, 10):
            try:
                server = ReuseTCPServer(("0.0.0.0", port), MJPEGHandler)
                break
            except OSError as e:
                if getattr(e, "errno", None) == 98:
                    port += 1
                    continue
                raise

        if server is None:
            print(f"Could not bind MJPEG server starting at port {args.mjpeg_port}; disabling MJPEG")
        else:
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            print(f"MJPEG server started on /video_feed:{port}")

    print("Starting continuous streamer ->", args.dashboard)
    frame_idx = 0
    t0 = time.time()
    last_stats_print = t0
    drop_count = 0
    track_fail_count = 0
    track_motion_px = 0.0
    track_dx_px = 0.0
    track_dy_px = 0.0
    track_dyaw_deg = 0.0
    lock_is_active = False
    lock_seen_once = False
    lock_hold_count = 0
    lock_cx = None
    lock_cy = None
    lock_bbox = None
    lock_seed_area = None
    lock_template = None
    last_command = "hold"
    last_command_ts = 0.0
    base_ref_locked = False
    base_ref_cx = None
    base_ref_cy = None
    base_ref_alt_m = None
    base_ref_world_x_m = None
    base_ref_world_y_m = None
    base_ref_roll_deg = None
    base_ref_pitch_deg = None
    base_ref_yaw_deg = None
    base_lock_streak = 0
    hover_rel_ema_dx = 0.0
    hover_rel_ema_dy = 0.0
    fused_roll_deg = None
    fused_pitch_deg = None
    fused_yaw_deg = None
    last_imu_t = None
    roll_kf = None
    pitch_kf = None
    yaw_kf = None
    if bool(args.imu_fusion) and str(args.imu_fusion_filter).strip().lower() == "kalman":
        roll_kf = AngleKalman1D(
            q_angle=float(args.imu_kf_q_angle),
            q_bias=float(args.imu_kf_q_bias),
            r_measure=float(args.imu_kf_r_measure),
            wrap=False,
        )
        pitch_kf = AngleKalman1D(
            q_angle=float(args.imu_kf_q_angle),
            q_bias=float(args.imu_kf_q_bias),
            r_measure=float(args.imu_kf_r_measure),
            wrap=False,
        )
        yaw_kf = AngleKalman1D(
            q_angle=float(args.imu_kf_q_angle),
            q_bias=float(args.imu_kf_q_bias),
            r_measure=float(args.imu_kf_r_measure),
            wrap=True,
        )

    try:
        while True:
            try:
                frame_idx += 1
                frame_capture_ts = time.time()
                depth = None
                ir = None
                mono8 = None
                source_used = None

                if args.source == "sdk":
                    pFrameData = POINTER(SYFrameData)()
                    err = GetLastFrameData(_synexens_device_id, byref(pFrameData))
                    if err != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                        print(f"GetLastFrameData err={err}")
                    if err == SYErrorCodeEnum.SYERRORCODE_SUCCESS and pFrameData and pFrameData.contents:
                        obj = pFrameData.contents
                        if obj.m_nFrameCount > 0:
                            offset = 0
                            for j in range(obj.m_nFrameCount):
                                info = obj.m_pFrameInfo[j]
                                h = int(info.m_nFrameHeight)
                                w = int(info.m_nFrameWidth)
                                n = h * w
                                ftype = int(info.m_frameType)
                                if ftype == int(SYFrameTypeEnum.SYFRAMETYPE_DEPTH):
                                    ptr = cast(obj.m_pData + offset, POINTER(c_ushort))
                                    arr = np.ctypeslib.as_array(ptr, shape=(n,)).copy().reshape(h, w)
                                    depth = arr
                                elif ftype == int(SYFrameTypeEnum.SYFRAMETYPE_IR):
                                    ptr = cast(obj.m_pData + offset, POINTER(c_ushort))
                                    arr = np.ctypeslib.as_array(ptr, shape=(n,)).copy().reshape(h, w)
                                    ir = arr
                                offset += n * 2
                else:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        if frame.ndim == 3 and frame.shape[2] >= 3:
                            b = frame[:, :, 0].astype(np.float32)
                            g = frame[:, :, 1].astype(np.float32)
                            r = frame[:, :, 2].astype(np.float32)
                            mono8 = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
                        elif frame.ndim == 2:
                            mono8 = frame.astype(np.uint8)
                        mono8 = normalize_u8(mono8)
                        mono8 = apply_uvc_flip(mono8, args.uvc_flip)
                        source_used = "uvc"

                if args.source == "sdk":
                    mono8 = None
                    source_used = None
                    if args.camera_feed == "ir":
                        if ir is not None:
                            mono8 = normalize_u16_to_u8(ir)
                            source_used = "ir"
                        elif depth is not None:
                            mono8 = normalize_u16_to_u8(depth)
                            source_used = "depth_fallback"
                    elif args.camera_feed == "depth":
                        if depth is not None:
                            mono8 = normalize_u16_to_u8(depth)
                            source_used = "depth"
                        elif ir is not None:
                            mono8 = normalize_u16_to_u8(ir)
                            source_used = "ir_fallback"
                    else:
                        combined = None
                        if ir is not None:
                            combined = normalize_u16_to_u8(ir)
                        if depth is not None:
                            dmap = normalize_u16_to_u8(depth)
                            if combined is None:
                                combined = dmap
                            else:
                                combined = cv2.addWeighted(combined, 0.6, dmap, 0.4, 0)
                        if combined is not None:
                            mono8 = combined
                            source_used = "ir+depth" if ir is not None and depth is not None else ("ir" if ir is not None else "depth")

                if mono8 is None:
                    print(
                        f"No image frame available (camera_feed={args.camera_feed}). depth:{depth is not None}, ir:{ir is not None}"
                    )

                vx = 0.0
                vy = 0.0
                vo_debug = {}
                route_payload = None
                imu_source_used = "none"
                altitude_source = "none"
                imu_state = {
                    "gx": 0.0,
                    "gy": 0.0,
                    "gz": 0.0,
                    "yaw_deg": None,
                    "alt_m": None,
                    "roll_deg": None,
                    "pitch_deg": None,
                    "timestamp": None,
                }

                if imu_client is not None:
                    try:
                        mav_imu_state = imu_client.get_imu_state(timeout=0.005, yaw_sign=float(args.imu_yaw_sign))
                        if isinstance(mav_imu_state, dict):
                            imu_state.update(mav_imu_state)
                            imu_source_used = f"mavlink:{imu_client.last_imu_msg_type}"
                    except Exception:
                        pass

                if imu_source_used == "none" and imu_source_mode in ("auto", "json") and args.imu_json:
                    imu_state = read_imu_extended(args.imu_json, yaw_sign=float(args.imu_yaw_sign))
                    imu_source_used = "json"

                imu_gyro_xyz = (float(imu_state["gx"]), float(imu_state["gy"]), float(imu_state["gz"]))
                yaw_deg = imu_state.get("yaw_deg")
                imu_alt_m = imu_state.get("alt_m")
                if imu_alt_m is not None:
                    if imu_source_used.startswith("mavlink:"):
                        altitude_source = "mavlink_imu"
                    elif imu_source_used == "json":
                        altitude_source = "imu_json"
                    else:
                        altitude_source = "imu"

                if rangefinder is not None:
                    try:
                        rf_alt = rangefinder.get_altitude_m(
                            timeout=float(max(0.0, args.rangefinder_poll_timeout)),
                            max_age_s=float(max(0.02, args.rangefinder_max_age_s)),
                            yaw_sign=float(args.imu_yaw_sign),
                            max_msgs=int(max(20, args.rangefinder_max_msgs)),
                        )
                        if rf_alt is not None:
                            imu_alt_m = float(rf_alt)
                            altitude_source = rangefinder.last_alt_msg_type
                    except Exception:
                        pass

                now_t = time.time()
                if last_imu_t is None:
                    dt_imu = float(max(1e-3, args.interval))
                else:
                    dt_imu = float(max(1e-3, now_t - last_imu_t))
                last_imu_t = now_t

                fusion_filter_used = "none"
                gx_rad_s = float(imu_state.get("gx", 0.0))
                gy_rad_s = float(imu_state.get("gy", 0.0))
                gz_rad_s = float(imu_state.get("gz", 0.0))
                gx_dps = float(math.degrees(gx_rad_s))
                gy_dps = float(math.degrees(gy_rad_s))
                gz_dps = float(math.degrees(gz_rad_s))
                roll_meas = imu_state.get("roll_deg")
                pitch_meas = imu_state.get("pitch_deg")
                yaw_meas = imu_state.get("yaw_deg")

                if args.imu_fusion:
                    filter_mode = str(args.imu_fusion_filter).strip().lower()
                    if filter_mode == "kalman" and roll_kf is not None and pitch_kf is not None and yaw_kf is not None:
                        fused_roll_deg = roll_kf.update(gx_dps, dt_imu, roll_meas)
                        fused_pitch_deg = pitch_kf.update(gy_dps, dt_imu, pitch_meas)
                        fused_yaw_deg = yaw_kf.update(gz_dps, dt_imu, yaw_meas)
                        fusion_filter_used = "kalman"
                    else:
                        alpha_f = float(max(0.0, min(1.0, args.imu_fusion_alpha)))

                        if fused_roll_deg is None:
                            fused_roll_deg = float(roll_meas) if roll_meas is not None else 0.0
                        else:
                            roll_pred = float(fused_roll_deg + gx_dps * dt_imu)
                            fused_roll_deg = float(alpha_f * roll_pred + (1.0 - alpha_f) * float(roll_meas)) if roll_meas is not None else roll_pred

                        if fused_pitch_deg is None:
                            fused_pitch_deg = float(pitch_meas) if pitch_meas is not None else 0.0
                        else:
                            pitch_pred = float(fused_pitch_deg + gy_dps * dt_imu)
                            fused_pitch_deg = float(alpha_f * pitch_pred + (1.0 - alpha_f) * float(pitch_meas)) if pitch_meas is not None else pitch_pred

                        if fused_yaw_deg is None:
                            fused_yaw_deg = float(yaw_meas) if yaw_meas is not None else 0.0
                        else:
                            yaw_pred = wrap_angle_360(float(fused_yaw_deg + gz_dps * dt_imu))
                            fused_yaw_deg = blend_angle_deg(yaw_pred, float(yaw_meas), 1.0 - alpha_f) if yaw_meas is not None else yaw_pred
                        fusion_filter_used = "complementary"
                else:
                    fused_roll_deg = float(imu_state.get("roll_deg")) if imu_state.get("roll_deg") is not None else fused_roll_deg
                    fused_pitch_deg = float(imu_state.get("pitch_deg")) if imu_state.get("pitch_deg") is not None else fused_pitch_deg
                    fused_yaw_deg = float(imu_state.get("yaw_deg")) if imu_state.get("yaw_deg") is not None else fused_yaw_deg

                if fused_yaw_deg is not None:
                    yaw_deg = float(fused_yaw_deg)

                if mono8 is not None:
                    gray = enhance_for_markers(mono8, args.image_mode)
                    display_gray = gray

                    marker_locked = False
                    matched_label = None
                    match_score = 0
                    seed_scene_matched = False
                    seed_orb_good = 0
                    seed_orb_norm = 0.0
                    seed_blob_sim = None
                    seed_match_score = 0
                    seed_inliers = 0
                    seed_alignment_valid = False
                    seed_error_dx_px = 0.0
                    seed_error_dy_px = 0.0
                    seed_error_norm_px = 0.0
                    seed_correction_command = "hold"
                    seed_centered = False
                    seed_yaw_error_deg = 0.0
                    seed_yaw_valid = False
                    seed_yaw_correction_command = "hold"
                    seed_yaw_centered = False
                    marker_bbox = None
                    marker_boxes = []
                    detected_ids = []
                    detected_labels = []
                    board_center_est = None
                    board_center_conf = 0.0
                    drift_dx_px = 0.0
                    drift_dy_px = 0.0
                    drift_norm = 0.0
                    drift_command = "hold"
                    base_rel_dx_px = 0.0
                    base_rel_dy_px = 0.0
                    base_rel_norm_px = 0.0
                    base_rel_dx_m = 0.0
                    base_rel_dy_m = 0.0
                    base_rel_norm_m = 0.0
                    base_rel_metric_valid = False
                    hover_ema_norm_px = 0.0
                    hover_correction_command = "hold"
                    roll_delta_deg = 0.0
                    pitch_delta_deg = 0.0
                    yaw_delta_deg = 0.0
                    tracking_enabled_by_alt = (imu_alt_m is not None) and (float(imu_alt_m) >= float(max(0.0, args.track_alt_min_m)))

                    if flow_mapper is not None:
                        try:
                            if depth is not None:
                                depth_for_fusion = depth
                                alt_m = float(np.median(depth_for_fusion)) / 1000.0
                            else:
                                depth_for_fusion = np.full(gray.shape, 1000, dtype=np.uint16)
                                alt_m = 1.0
                            if imu_alt_m is not None:
                                alt_m = imu_alt_m

                            vx, vy = flow_mapper.update(
                                gray,
                                depth_for_fusion,
                                alt_m,
                                time.time(),
                                imu_gyro_xyz=imu_gyro_xyz,
                                yaw_deg=yaw_deg,
                            )
                            route_payload = flow_mapper.get_dashboard_payload()
                            vo_debug = route_payload.get("debug", {}) if isinstance(route_payload, dict) else {}
                            vo_debug["imu_source"] = imu_source_used
                            vo_debug["source_used"] = source_used
                        except Exception as e:
                            vo_debug = {
                                "source": "MSCKF",
                                "success": False,
                                "reject_reason": str(e),
                                "gyro_comp": False,
                            }

                    # IR blob tracker: threshold -> contours -> sticky single-target lock.
                    if args.auto_exposure_threshold:
                        pctl = float(max(90.0, min(99.95, args.auto_threshold_percentile)))
                        dyn_thr = int(np.percentile(gray, pctl))
                        thresh_val = int(max(160, min(254, dyn_thr)))
                    else:
                        thresh_val = int(max(1, min(254, args.blob_threshold)))

                    if tracking_enabled_by_alt:
                        try:
                            _, bw = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        except Exception:
                            contours = []
                    else:
                        contours = []

                    candidates = []
                    min_area = float(max(1.0, args.blob_min_area))
                    for c in contours:
                        area = float(cv2.contourArea(c))
                        if area <= min_area:
                            continue
                        M = cv2.moments(c)
                        if M.get("m00", 0.0) <= 1e-6:
                            continue
                        cx_raw = float(M["m10"] / M["m00"])
                        cy_raw = float(M["m01"] / M["m00"])
                        x, y, ww, hh = cv2.boundingRect(c)

                        # Refine bounding box and center strictly to the actual light spot core, ignoring scattering
                        roi = gray[y:y+hh, x:x+ww]
                        _, max_val, _, _ = cv2.minMaxLoc(roi)
                        if max_val > 0:
                            # Isolate hotspot at 80% peak intensity
                            _, core_bw = cv2.threshold(roi, max_val * 0.80, 255, cv2.THRESH_BINARY)
                            core_pts = cv2.findNonZero(core_bw)
                            if core_pts is not None:
                                core_x, core_y, core_w, core_h = cv2.boundingRect(core_pts)
                                M_core = cv2.moments(core_bw)
                                if M_core.get("m00", 0.0) > 1e-6:
                                    cx_raw = float(x + M_core["m10"] / M_core["m00"])
                                    cy_raw = float(y + M_core["m01"] / M_core["m00"])
                                else:
                                    cx_raw = float(x + core_x + core_w / 2.0)
                                    cy_raw = float(y + core_y + core_h / 2.0)
                                x += core_x
                                y += core_y
                                ww = core_w
                                hh = core_h

                        candidates.append(
                            {
                                "area": area,
                                "cx": cx_raw,
                                "cy": cy_raw,
                                "bbox": [int(x), int(y), int(ww), int(hh)],
                            }
                        )

                    chosen = None
                    chosen_similarity = None
                    max_jump = float(max(5.0, args.lock_max_jump_px))
                    patch_size = int(max(8, args.lock_template_size))
                    strict_main_blob = bool(args.strict_main_blob and lock_seen_once)
                    if lock_is_active and lock_cx is not None and lock_cy is not None and len(candidates) > 0:
                        best = None
                        best_rank = -1e9
                        for c in candidates:
                            d = float(np.hypot(c["cx"] - lock_cx, c["cy"] - lock_cy))
                            if d > max_jump:
                                continue

                            area_ratio = None
                            if strict_main_blob and lock_seed_area is not None:
                                area_ratio = float(c["area"] / max(1e-6, float(lock_seed_area)))
                                if area_ratio < float(args.lock_seed_area_min_ratio) or area_ratio > float(args.lock_seed_area_max_ratio):
                                    continue

                            sim = None
                            if strict_main_blob and lock_template is not None:
                                cand_patch = extract_square_patch(gray, c["cx"], c["cy"], patch_size)
                                sim = patch_ncc_similarity(lock_template, cand_patch)
                                if sim is None or sim < float(args.lock_template_threshold):
                                    continue

                            dist_score = float(1.0 - min(1.0, d / max_jump))
                            sim_score = float(sim) if sim is not None else 0.0
                            area_score = 0.0
                            if area_ratio is not None:
                                area_score = float(max(0.0, 1.0 - abs(math.log(max(1e-6, area_ratio)))))

                            rank = float(dist_score + (2.0 * sim_score) + (0.4 * area_score))
                            if rank > best_rank:
                                best_rank = rank
                                best = c
                                chosen_similarity = sim

                        if best is not None:
                            chosen = best
                        else:
                            if bool(args.sticky_lock_forever) and lock_seen_once:
                                lock_is_active = True
                            else:
                                lock_hold_count = int(max(0, lock_hold_count - 1))
                                if lock_hold_count == 0:
                                    lock_is_active = False

                    if chosen is None and not lock_is_active and len(candidates) > 0:
                        if lock_seen_once and lock_cx is not None and lock_cy is not None:
                            nearest = None
                            nearest_rank = -1e9
                            nearest_sim = None
                            for c in candidates:
                                d = float(np.hypot(c["cx"] - lock_cx, c["cy"] - lock_cy))
                                if d > (1.8 * max_jump):
                                    continue

                                area_ratio = None
                                if strict_main_blob and lock_seed_area is not None:
                                    area_ratio = float(c["area"] / max(1e-6, float(lock_seed_area)))
                                    if area_ratio < float(args.lock_seed_area_min_ratio) or area_ratio > float(args.lock_seed_area_max_ratio):
                                        continue

                                sim = None
                                if strict_main_blob and lock_template is not None:
                                    cand_patch = extract_square_patch(gray, c["cx"], c["cy"], patch_size)
                                    sim = patch_ncc_similarity(lock_template, cand_patch)
                                    if sim is None or sim < float(args.lock_template_threshold):
                                        continue

                                rank = float(1.0 - min(1.0, d / (1.8 * max_jump)))
                                if sim is not None:
                                    rank += float(2.0 * sim)

                                if rank > nearest_rank:
                                    nearest_rank = rank
                                    nearest = c
                                    nearest_sim = sim

                            if nearest is not None:
                                chosen = nearest
                                chosen_similarity = nearest_sim
                        else:
                            chosen = max(candidates, key=lambda x: x["area"])

                    if chosen is not None:
                        alpha = float(max(0.01, min(1.0, args.ema_alpha)))
                        cx_raw = chosen["cx"]
                        cy_raw = chosen["cy"]
                        if lock_cx is None or lock_cy is None:
                            lock_cx = cx_raw
                            lock_cy = cy_raw
                        else:
                            lock_cx = alpha * cx_raw + (1.0 - alpha) * float(lock_cx)
                            lock_cy = alpha * cy_raw + (1.0 - alpha) * float(lock_cy)

                        lock_bbox = chosen["bbox"]
                        lock_is_active = True
                        lock_seen_once = True
                        lock_hold_count = int(max(0, args.lock_hold_frames))

                        if lock_seed_area is None:
                            lock_seed_area = float(chosen["area"])

                        current_patch = extract_square_patch(gray, lock_cx, lock_cy, patch_size)
                        if current_patch is not None:
                            if lock_template is None:
                                lock_template = current_patch
                            else:
                                tpl_alpha = float(max(0.0, min(1.0, args.lock_template_update_alpha)))
                                if tpl_alpha > 0.0:
                                    lock_template = cv2.addWeighted(
                                        lock_template.astype(np.float32),
                                        1.0 - tpl_alpha,
                                        current_patch.astype(np.float32),
                                        tpl_alpha,
                                        0.0,
                                    ).astype(np.uint8)

                    seed_state = None
                    if seed_ref is not None:
                        try:
                            seed_state = evaluate_seed_match(
                                gray_u8=gray,
                                seed_ref=seed_ref,
                                lock_cx=(lock_cx if lock_is_active else None),
                                lock_cy=(lock_cy if lock_is_active else None),
                                patch_size=patch_size,
                                frame_idx=frame_idx,
                                stride=int(max(1, args.seed_match_stride)),
                                orb_ratio_test=float(max(0.5, min(0.95, args.seed_orb_ratio_test))),
                                orb_min_good=int(max(1, args.seed_orb_min_good)),
                                orb_min_norm=float(max(0.0, args.seed_orb_min_norm)),
                                require_blob=bool(args.seed_require_blob),
                                blob_sim_threshold=float(args.seed_blob_sim_threshold),
                                align_min_inliers=int(max(4, args.seed_align_min_inliers)),
                                align_deadband_px=float(max(2.0, args.seed_align_deadband_px)),
                                align_ransac_reproj=float(max(1.0, args.seed_align_ransac_reproj)),
                                yaw_sign=float(args.seed_yaw_sign),
                                yaw_deadband_deg=float(max(0.5, args.seed_yaw_deadband_deg)),
                            )
                        except Exception:
                            seed_state = None

                    if tracking_enabled_by_alt and lock_is_active and lock_cx is not None and lock_cy is not None:
                        marker_locked = True
                        matched_label = "blob"
                        marker_bbox = lock_bbox if lock_bbox is not None else None
                        area_score = 60.0
                        if chosen is not None:
                            area_score = float(max(1.0, min(100.0, chosen["area"] / 8.0)))
                        match_score = int(area_score)

                        if marker_bbox is not None:
                            marker_boxes = [
                                {
                                    "label": "blob",
                                    "score": int(match_score),
                                    "bbox": marker_bbox,
                                }
                            ]
                        detected_ids = ["blob"]
                        detected_labels = ["blob"]
                        board_center_est = (float(lock_cx), float(lock_cy))

                        fh, fw = gray.shape[:2]
                        fx_local = float(cached_caminfo.get("fx", gray.shape[1] * 0.9))
                        fy_local = float(cached_caminfo.get("fy", gray.shape[0] * 0.9))
                        cx_local = float(cached_caminfo.get("cx", gray.shape[1] * 0.5))
                        cy_local = float(cached_caminfo.get("cy", gray.shape[0] * 0.5))

                        roll_delta_deg = 0.0
                        pitch_delta_deg = 0.0
                        yaw_delta_deg = 0.0
                        if (
                            args.imu_fusion
                            and base_ref_roll_deg is not None
                            and base_ref_pitch_deg is not None
                            and base_ref_yaw_deg is not None
                            and fused_roll_deg is not None
                            and fused_pitch_deg is not None
                            and fused_yaw_deg is not None
                        ):
                            roll_delta_deg = float(fused_roll_deg) - float(base_ref_roll_deg)
                            pitch_delta_deg = float(fused_pitch_deg) - float(base_ref_pitch_deg)
                            yaw_delta_deg = angle_diff_deg(float(fused_yaw_deg), float(base_ref_yaw_deg))

                        att_dx_px = 0.0
                        att_dy_px = 0.0
                        if args.imu_fusion:
                            att_dx_px = float(fx_local * np.tan(np.radians(roll_delta_deg)))
                            att_dy_px = float(fy_local * np.tan(np.radians(pitch_delta_deg)))

                        lock_cx_tilt_comp = float(lock_cx - att_dx_px)
                        lock_cy_tilt_comp = float(lock_cy - att_dy_px)
                        x_rel = float(lock_cx_tilt_comp - cx_local)
                        y_rel = float(lock_cy_tilt_comp - cy_local)
                        yaw_rad = float(np.radians(-yaw_delta_deg))
                        cyaw = float(np.cos(yaw_rad))
                        syaw = float(np.sin(yaw_rad))
                        x_derot = float(cyaw * x_rel - syaw * y_rel)
                        y_derot = float(syaw * x_rel + cyaw * y_rel)
                        lock_cx_comp = float(cx_local + x_derot)
                        lock_cy_comp = float(cy_local + y_derot)

                        drift_dx_px = float(lock_cx_comp - cx_local)
                        drift_dy_px = float(lock_cy_comp - cy_local)
                        drift_norm = float(np.hypot(drift_dx_px, drift_dy_px))

                        board_center_conf = 0.70 if chosen is None else float(min(1.0, chosen["area"] / max(1.0, 0.02 * fw * fh)))

                        dead_px = 10.0
                        cmd_x = ""
                        cmd_y = ""
                        if drift_dx_px > dead_px:
                            cmd_x = "move_right"
                        elif drift_dx_px < -dead_px:
                            cmd_x = "move_left"
                        if drift_dy_px > dead_px:
                            cmd_y = "move_down"
                        elif drift_dy_px < -dead_px:
                            cmd_y = "move_up"

                        if cmd_x and cmd_y:
                            next_command = cmd_x + "+" + cmd_y
                        elif cmd_x:
                            next_command = cmd_x
                        elif cmd_y:
                            next_command = cmd_y
                        else:
                            next_command = "hold"

                        if args.enable_rate_limit and next_command != last_command:
                            dt_ms = (time.time() - float(last_command_ts)) * 1000.0
                            if dt_ms >= float(max(0, args.rate_limit_ms)):
                                last_command = next_command
                                last_command_ts = time.time()
                            drift_command = last_command
                        else:
                            drift_command = next_command
                            if next_command != last_command:
                                last_command = next_command
                                last_command_ts = time.time()

                        if bool(args.capture_base_on_lock):
                            base_lock_streak += 1
                            if (not base_ref_locked) and base_lock_streak >= int(max(1, args.base_lock_frames)):
                                base_ref_locked = True
                                base_ref_cx = float(lock_cx_comp)
                                base_ref_cy = float(lock_cy_comp)
                                base_ref_roll_deg = float(fused_roll_deg) if fused_roll_deg is not None else None
                                base_ref_pitch_deg = float(fused_pitch_deg) if fused_pitch_deg is not None else None
                                base_ref_yaw_deg = float(fused_yaw_deg) if fused_yaw_deg is not None else None
                                base_ref_alt_m = float(imu_alt_m) if (imu_alt_m is not None and float(imu_alt_m) > 0.0) else None
                                if base_ref_alt_m is not None and fx_local > 1e-6 and fy_local > 1e-6:
                                    base_ref_world_x_m = float((lock_cx_comp - cx_local) * base_ref_alt_m / fx_local)
                                    base_ref_world_y_m = float((lock_cy_comp - cy_local) * base_ref_alt_m / fy_local)
                                else:
                                    base_ref_world_x_m = None
                                    base_ref_world_y_m = None
                        else:
                            base_lock_streak = 0

                        if base_ref_locked and base_ref_cx is not None and base_ref_cy is not None:
                            base_rel_dx_px = float(lock_cx_comp - base_ref_cx)
                            base_rel_dy_px = float(lock_cy_comp - base_ref_cy)
                            base_rel_norm_px = float(np.hypot(base_rel_dx_px, base_rel_dy_px))

                            if (
                                base_ref_world_x_m is not None
                                and base_ref_world_y_m is not None
                                and imu_alt_m is not None
                                and float(imu_alt_m) > 0.0
                                and fx_local > 1e-6
                                and fy_local > 1e-6
                            ):
                                curr_world_x_m = float((lock_cx_comp - cx_local) * float(imu_alt_m) / fx_local)
                                curr_world_y_m = float((lock_cy_comp - cy_local) * float(imu_alt_m) / fy_local)
                                base_rel_dx_m = float(curr_world_x_m - float(base_ref_world_x_m))
                                base_rel_dy_m = float(curr_world_y_m - float(base_ref_world_y_m))
                                base_rel_norm_m = float(np.hypot(base_rel_dx_m, base_rel_dy_m))
                                base_rel_metric_valid = True

                            h_alpha = float(max(0.01, min(1.0, args.hover_ema_alpha)))
                            hover_rel_ema_dx = float(h_alpha * base_rel_dx_px + (1.0 - h_alpha) * hover_rel_ema_dx)
                            hover_rel_ema_dy = float(h_alpha * base_rel_dy_px + (1.0 - h_alpha) * hover_rel_ema_dy)
                            hover_ema_norm_px = float(np.hypot(hover_rel_ema_dx, hover_rel_ema_dy))

                            dead_px_hover = 10.0
                            hc_x = ""
                            hc_y = ""
                            if hover_rel_ema_dx > dead_px_hover:
                                hc_x = "move_left"
                            elif hover_rel_ema_dx < -dead_px_hover:
                                hc_x = "move_right"
                            if hover_rel_ema_dy > dead_px_hover:
                                hc_y = "move_up"
                            elif hover_rel_ema_dy < -dead_px_hover:
                                hc_y = "move_down"

                            if hc_x and hc_y:
                                hover_correction_command = hc_x + "+" + hc_y
                            elif hc_x:
                                hover_correction_command = hc_x
                            elif hc_y:
                                hover_correction_command = hc_y
                            else:
                                hover_correction_command = "hold"
                    else:
                        base_lock_streak = 0

                    if seed_state is not None:
                        seed_scene_matched = bool(seed_state.get("is_match", False))
                        seed_orb_good = int(seed_state.get("orb_good", 0))
                        seed_orb_norm = float(seed_state.get("orb_norm", 0.0))
                        seed_blob_sim = seed_state.get("blob_sim", None)
                        seed_match_score = int(seed_state.get("score", 0))
                        seed_inliers = int(seed_state.get("inliers", 0))
                        seed_alignment_valid = bool(seed_state.get("alignment_valid", False))
                        seed_error_dx_px = float(seed_state.get("error_dx_px", 0.0))
                        seed_error_dy_px = float(seed_state.get("error_dy_px", 0.0))
                        seed_error_norm_px = float(seed_state.get("error_norm_px", 0.0))
                        seed_correction_command = str(seed_state.get("correction_command", "hold"))
                        seed_centered = bool(seed_state.get("centered", False))
                        seed_yaw_error_deg = float(seed_state.get("yaw_error_deg", 0.0))
                        seed_yaw_valid = bool(seed_state.get("yaw_valid", False))
                        seed_yaw_correction_command = str(seed_state.get("yaw_correction_command", "hold"))
                        seed_yaw_centered = bool(seed_state.get("yaw_centered", False))

                        if seed_scene_matched:
                            seed_label = str(args.seed_label).strip() or "base_seed"
                            if matched_label is None or str(matched_label).lower() == "blob":
                                matched_label = seed_label
                            match_score = int(max(int(match_score), int(seed_match_score)))
                            if not marker_locked and len(detected_ids) == 0:
                                detected_ids = [seed_label]
                                detected_labels = [seed_label]

                    detected_markers = len(marker_boxes)

                    vo_debug = dict(vo_debug) if isinstance(vo_debug, dict) else {}
                    vo_debug["marker_locked"] = bool(marker_locked)
                    vo_debug["detected_markers"] = int(detected_markers)
                    vo_debug["marker_label"] = matched_label if marker_locked else None
                    vo_debug["matched_tag"] = matched_label
                    vo_debug["match_score"] = int(match_score)
                    vo_debug["seed_enabled"] = bool(seed_ref is not None)
                    vo_debug["seed_scene_matched"] = bool(seed_scene_matched)
                    vo_debug["seed_orb_good"] = int(seed_orb_good)
                    vo_debug["seed_orb_norm"] = float(seed_orb_norm)
                    vo_debug["seed_blob_similarity"] = float(seed_blob_sim) if seed_blob_sim is not None else None
                    vo_debug["seed_match_score"] = int(seed_match_score)
                    vo_debug["seed_require_blob"] = bool(args.seed_require_blob)
                    vo_debug["seed_inliers"] = int(seed_inliers)
                    vo_debug["seed_alignment_valid"] = bool(seed_alignment_valid)
                    vo_debug["seed_error_dx_px"] = float(seed_error_dx_px)
                    vo_debug["seed_error_dy_px"] = float(seed_error_dy_px)
                    vo_debug["seed_error_norm_px"] = float(seed_error_norm_px)
                    vo_debug["seed_correction_command"] = str(seed_correction_command)
                    vo_debug["seed_centered"] = bool(seed_centered)
                    vo_debug["seed_yaw_error_deg"] = float(seed_yaw_error_deg)
                    vo_debug["seed_yaw_valid"] = bool(seed_yaw_valid)
                    vo_debug["seed_yaw_correction_command"] = str(seed_yaw_correction_command)
                    vo_debug["seed_yaw_centered"] = bool(seed_yaw_centered)
                    vo_debug["tracking_active"] = False
                    vo_debug["tracking_fail_count"] = int(track_fail_count)
                    vo_debug["track_motion_px"] = float(track_motion_px)
                    vo_debug["track_dx_px"] = float(track_dx_px)
                    vo_debug["track_dy_px"] = float(track_dy_px)
                    vo_debug["track_dyaw_deg"] = float(track_dyaw_deg)
                    vo_debug["match_ran"] = True
                    vo_debug["marker_boxes"] = marker_boxes
                    vo_debug["detected_labels"] = detected_labels
                    vo_debug["tracking_enabled_by_alt"] = bool(tracking_enabled_by_alt)
                    vo_debug["track_alt_min_m"] = float(args.track_alt_min_m)
                    vo_debug["sticky_lock_forever"] = bool(args.sticky_lock_forever)
                    vo_debug["board_center_est_px"] = [float(board_center_est[0]), float(board_center_est[1])] if board_center_est is not None else None
                    vo_debug["board_center_conf"] = float(board_center_conf)
                    vo_debug["drift_dx_px"] = float(drift_dx_px)
                    vo_debug["drift_dy_px"] = float(drift_dy_px)
                    vo_debug["drift_norm_px"] = float(drift_norm)
                    vo_debug["drift_command"] = drift_command
                    vo_debug["base_ref_locked"] = bool(base_ref_locked)
                    vo_debug["base_ref_px"] = [float(base_ref_cx), float(base_ref_cy)] if (base_ref_cx is not None and base_ref_cy is not None) else None
                    vo_debug["base_ref_alt_m"] = float(base_ref_alt_m) if base_ref_alt_m is not None else None
                    vo_debug["base_ref_roll_deg"] = float(base_ref_roll_deg) if base_ref_roll_deg is not None else None
                    vo_debug["base_ref_pitch_deg"] = float(base_ref_pitch_deg) if base_ref_pitch_deg is not None else None
                    vo_debug["base_ref_yaw_deg"] = float(base_ref_yaw_deg) if base_ref_yaw_deg is not None else None
                    vo_debug["base_rel_dx_px"] = float(base_rel_dx_px)
                    vo_debug["base_rel_dy_px"] = float(base_rel_dy_px)
                    vo_debug["base_rel_norm_px"] = float(base_rel_norm_px)
                    vo_debug["base_rel_dx_m"] = float(base_rel_dx_m)
                    vo_debug["base_rel_dy_m"] = float(base_rel_dy_m)
                    vo_debug["base_rel_norm_m"] = float(base_rel_norm_m)
                    vo_debug["base_rel_metric_valid"] = bool(base_rel_metric_valid)
                    vo_debug["hover_ema_dx_px"] = float(hover_rel_ema_dx)
                    vo_debug["hover_ema_dy_px"] = float(hover_rel_ema_dy)
                    vo_debug["hover_ema_norm_px"] = float(hover_ema_norm_px)
                    vo_debug["hover_correction_command"] = hover_correction_command
                    vo_debug["imu_fusion_enabled"] = bool(args.imu_fusion)
                    vo_debug["imu_fusion_filter"] = fusion_filter_used
                    vo_debug["imu_fused_roll_deg"] = float(fused_roll_deg) if fused_roll_deg is not None else None
                    vo_debug["imu_fused_pitch_deg"] = float(fused_pitch_deg) if fused_pitch_deg is not None else None
                    vo_debug["imu_fused_yaw_deg"] = float(fused_yaw_deg) if fused_yaw_deg is not None else None
                    vo_debug["attitude_comp_roll_delta_deg"] = float(roll_delta_deg)
                    vo_debug["attitude_comp_pitch_delta_deg"] = float(pitch_delta_deg)
                    vo_debug["attitude_comp_yaw_delta_deg"] = float(yaw_delta_deg)
                    vo_debug["blob_threshold"] = int(args.blob_threshold)
                    vo_debug["blob_threshold_applied"] = int(thresh_val)
                    vo_debug["blob_min_area"] = float(args.blob_min_area)
                    vo_debug["auto_exposure_threshold"] = bool(args.auto_exposure_threshold)
                    vo_debug["auto_threshold_percentile"] = float(args.auto_threshold_percentile)
                    vo_debug["ema_alpha"] = float(args.ema_alpha)
                    vo_debug["lock_max_jump_px"] = float(args.lock_max_jump_px)
                    vo_debug["lock_hold_frames"] = int(args.lock_hold_frames)
                    vo_debug["lock_hold_count"] = int(lock_hold_count)
                    vo_debug["sticky_lock_active"] = bool(lock_is_active)
                    vo_debug["strict_main_blob"] = bool(args.strict_main_blob)
                    vo_debug["lock_template_size"] = int(args.lock_template_size)
                    vo_debug["lock_template_threshold"] = float(args.lock_template_threshold)
                    vo_debug["lock_seed_area"] = float(lock_seed_area) if lock_seed_area is not None else None
                    vo_debug["lock_identity_similarity"] = float(chosen_similarity) if chosen_similarity is not None else None
                    vo_debug["rate_limit_enabled"] = bool(args.enable_rate_limit)
                    vo_debug["rate_limit_ms"] = int(args.rate_limit_ms)
                    vo_debug["imu_source"] = imu_source_used
                    vo_debug["altitude_source"] = altitude_source
                    if marker_bbox is not None:
                        vo_debug["marker_bbox"] = [int(marker_bbox[0]), int(marker_bbox[1]), int(marker_bbox[2]), int(marker_bbox[3])]

                    try:
                        pil_img = Image.fromarray(display_gray)
                        out_buffer = io.BytesIO()
                        pil_img.save(out_buffer, format="JPEG", quality=max(30, min(95, int(args.jpeg_quality))))
                        data = out_buffer.getvalue()
                        ret = True
                    except Exception as e:
                        print(f"PIL JPEG encode failed: {e}")
                        ret = False
                        data = None

                    if ret and data is not None:
                        with jpeg_lock:
                            latest_jpeg = data

                        if args.dashboard:
                            files = {"image": ("frame.jpg", data, "image/jpeg")}
                            meta = {
                                "camera_info": cached_caminfo,
                                "matched_tag": matched_label,
                                "match_score": int(match_score),
                                "kp_count": int(0),
                                "detected_ids": detected_ids,
                                "relocalized_bin": matched_label,
                                "camera_feed": args.camera_feed,
                                "source_used": source_used,
                                "frame_ts": float(frame_capture_ts),
                                "yaw": float(yaw_deg) if yaw_deg is not None else 0.0,
                                "altitude": float(imu_alt_m) if imu_alt_m is not None else 0.0,
                                "vx": float(vx),
                                "vy": float(vy),
                                "vo_debug": vo_debug,
                                "board_localize": False,
                                "board_area_ratio": 0.0,
                                "board_quadrant_scores": [],
                                "marker_bbox": vo_debug.get("marker_bbox", None),
                                "marker_boxes": marker_boxes,
                            }
                            if route_payload is not None:
                                meta["route"] = route_payload

                            try:
                                try:
                                    post_q.put_nowait((files, meta))
                                except Full:
                                    drop_count += 1
                                    try:
                                        _ = post_q.get_nowait()
                                    except Empty:
                                        pass
                                    try:
                                        post_q.put_nowait((files, meta))
                                    except Full:
                                        drop_count += 1
                            except Exception:
                                pass

                now = time.time()
                if now - last_stats_print >= 2.0:
                    dt = max(1e-3, now - t0)
                    fps = frame_idx / dt
                    print(
                        f"status fps={fps:.1f} frames={frame_idx} post_ok={post_stats['ok']} post_fail={post_stats['fail']} dropped={drop_count}"
                    )
                    last_stats_print = now

                time.sleep(args.interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Streamer exception: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(args.interval)
                continue

    finally:
        post_stop.set()
        if post_thread is not None:
            try:
                post_q.put_nowait(None)
            except Exception:
                pass
            try:
                post_thread.join(timeout=1.0)
            except Exception:
                pass

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        stop_lidar_stream()

        if imu_client is not None and imu_client is not rangefinder:
            imu_client.close()

        if rangefinder is not None:
            rangefinder.close()

        if dashboard_state.get("proc") is not None:
            try:
                dashboard_state["proc"].terminate()
                dashboard_state["proc"].wait(timeout=5)
                print("Dashboard subprocess terminated.")
            except Exception:
                print("Failed to terminate dashboard subprocess cleanly.")

        try:
            http_session.close()
        except Exception:
            pass
