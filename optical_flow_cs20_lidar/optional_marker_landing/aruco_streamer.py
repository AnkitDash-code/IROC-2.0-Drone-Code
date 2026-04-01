#!/usr/bin/env python3
"""
Non-ROS ArUco streamer for the CS20 LiDAR.

Reads frames from the Synexens CS20 SDK, runs ArUco detection, computes a simple
pose/velocity estimate, and POSTs the image + telemetry to the web dashboard
`/api/upload_frame` endpoint.

Run with the system python that works with the SDK (e.g. /usr/bin/python3.10):
  python3.10 aruco_streamer.py --dashboard http://localhost:5000/api/upload_frame

This avoids running rclpy/ROS for ArUco detection while still streaming to the
dashboard UI.
"""

import os
import sys
import time
import argparse
import json
import math
import base64
import threading

import numpy as np
import cv2
import requests
import glob
from pathlib import Path

from ctypes import POINTER, byref, cast, c_int, c_ushort

# Insert Synexens SDK path (same folder layout used by the bridge)
SDK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SynexensPythonSDK4_4.2.4.0_202504281506"))
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

try:
    from SynexensPythonSDK import (
        InitSDK,
        UnInitSDK,
        FindDevice,
        OpenDevice,
        CloseDevice,
        SetFrameResolution,
        StartStreaming,
        StopStreaming,
        GetLastFrameData,
        GetIntric,
        SYDeviceInfo,
        SYErrorCodeEnum,
        SYStreamTypeEnum,
        SYFrameTypeEnum,
        SYResolutionEnum,
        SYFrameData,
        SYIntrinsics,
    )
except Exception as e:
    raise RuntimeError("Synexens SDK import failed: %s" % e)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard", default="http://localhost:5000/api/upload_frame")
    p.add_argument("--resolution", choices=("320x240", "640x480"), default="320x240")
    p.add_argument("--source", choices=("sdk", "mjpeg"), default="sdk", help="Frame source: 'sdk' reads Synexens SDK, 'mjpeg' reads dashboard MJPEG stream")
    p.add_argument("--mjpeg-url", default="http://localhost:5000/video_feed", help="MJPEG URL to read when --source=mjpeg")
    p.add_argument("--marker-length", type=float, default=0.06)
    p.add_argument("--marker-separation", type=float, default=0.02)
    p.add_argument("--use-imu", action="store_true", help="Attempt to read heading from FCU via pymavlink")
    p.add_argument("--fcu", default="/dev/ttyACM0", help="FCU serial device for pymavlink (if --use-imu)")
    p.add_argument("--upload-interval", type=float, default=0.08, help="Minimum seconds between uploads")
    p.add_argument("--tags-dir", default=None, help="Directory containing labeled tag images (searched in order)")
    p.add_argument("--match-threshold", type=int, default=12, help="Minimum good feature matches to consider a template matched")
    return p.parse_args()


class CS20Device:
    def __init__(self, resolution_str):
        if resolution_str == "640x480":
            self.resolution = SYResolutionEnum.SYRESOLUTION_640_480
            self.width, self.height = 640, 480
        else:
            self.resolution = SYResolutionEnum.SYRESOLUTION_320_240
            self.width, self.height = 320, 240

        self._device_id = -1
        self._streaming = False
        self._fx = 381.0
        self._fy = 381.0
        self._cx = float(self.width) * 0.5
        self._cy = float(self.height) * 0.5

    def init(self):
        if InitSDK() != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            raise RuntimeError("InitSDK failed")

        n_count = c_int(0)
        if FindDevice(byref(n_count), None) != SYErrorCodeEnum.SYERRORCODE_SUCCESS or n_count.value <= 0:
            UnInitSDK()
            raise RuntimeError("No Synexens devices found")

        devices = (SYDeviceInfo * n_count.value)()
        if FindDevice(byref(n_count), devices) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            UnInitSDK()
            raise RuntimeError("FindDevice list failed")

        dev = devices[0]
        self._device_id = int(dev.m_nDeviceID)

        if OpenDevice(dev) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            UnInitSDK()
            raise RuntimeError("OpenDevice failed")

        if SetFrameResolution(self._device_id, SYFrameTypeEnum.SYFRAMETYPE_DEPTH, self.resolution) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            CloseDevice(self._device_id)
            UnInitSDK()
            raise RuntimeError("SetFrameResolution(depth) failed")

        st = StartStreaming(self._device_id, SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR)
        if st != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            CloseDevice(self._device_id)
            UnInitSDK()
            raise RuntimeError(f"StartStreaming failed: {int(st)}")

        self._streaming = True

        intr = SYIntrinsics()
        if GetIntric(self._device_id, self.resolution, intr) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            if intr.m_fltFocalDistanceX > 1.0:
                self._fx = float(intr.m_fltFocalDistanceX)
            if intr.m_fltFocalDistanceY > 1.0:
                self._fy = float(intr.m_fltFocalDistanceY)
            self._cx = float(intr.m_fltCenterPointX)
            self._cy = float(intr.m_fltCenterPointY)

    @staticmethod
    def _frame_bytes(frame_type, n_count):
        if int(frame_type) == int(SYFrameTypeEnum.SYFRAMETYPE_RGB):
            return (n_count * 3) // 2
        return n_count * 2

    def extract_frames(self):
        p_frame = POINTER(SYFrameData)()
        err = GetLastFrameData(self._device_id, byref(p_frame))
        if err != SYErrorCodeEnum.SYERRORCODE_SUCCESS or (not p_frame) or (not p_frame.contents):
            return None, None

        obj = p_frame.contents
        if obj.m_nFrameCount <= 0:
            return None, None

        depth = None
        ir = None
        offset = 0
        for i in range(obj.m_nFrameCount):
            info = obj.m_pFrameInfo[i]
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

            offset += self._frame_bytes(ftype, n)

        return depth, ir

    def close(self):
        try:
            if self._streaming and self._device_id >= 0:
                StopStreaming(self._device_id)
        except Exception:
            pass
        try:
            if self._device_id >= 0:
                CloseDevice(self._device_id)
        except Exception:
            pass
        try:
            UnInitSDK()
        except Exception:
            pass


def build_board(marker_length, marker_separation):
    half_len = marker_length / 2.0
    offset = (marker_length + marker_separation) / 2.0
    centers = [
        [-offset, offset, 0.0],
        [offset, offset, 0.0],
        [-offset, -offset, 0.0],
        [offset, -offset, 0.0],
    ]

    obj_points = []
    for c in centers:
        cx, cy, cz = c
        corners = np.array(
            [
                [cx - half_len, cy + half_len, cz],
                [cx + half_len, cy + half_len, cz],
                [cx + half_len, cy - half_len, cz],
                [cx - half_len, cy - half_len, cz],
            ],
            dtype=np.float32,
        )
        obj_points.append(corners)

    board_ids = np.array([[0], [1], [2], [3]], dtype=np.int32)
    try:
        board = cv2.aruco.Board_create(obj_points, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), board_ids)
    except Exception:
        board = cv2.aruco.Board(obj_points, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), board_ids)
    return board


def main():
    args = parse_args()

    # Camera intrinsics fallback (will be overwritten by SDK intrinsics when available)
    if args.resolution == "640x480":
        K = np.array([[762.0, 0.0, 320.0], [0.0, 762.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        K = np.array([[381.0, 0.0, 160.0], [0.0, 381.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros((5,), dtype=np.float64)

    use_sdk = args.source == "sdk"
    if use_sdk:
        dev = CS20Device(args.resolution)
        dev.init()
        # update intrinsics from SDK if available
        try:
            K = np.array([[dev._fx, 0.0, dev._cx], [0.0, dev._fy, dev._cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        except Exception:
            pass
    else:
        dev = None

    # ArUco dictionary and detector
    aruco = cv2.aruco
    # pick dictionary
    for dname in ("DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000"):
        if hasattr(aruco, dname):
            dict_id = getattr(aruco, dname)
            break
    else:
        dict_id = aruco.DICT_4X4_50

    try:
        dictionary = aruco.getPredefinedDictionary(dict_id)
    except Exception:
        dictionary = aruco.Dictionary_get(dict_id)

    try:
        detector_params = aruco.DetectorParameters_create()
    except Exception:
        detector_params = aruco.DetectorParameters()

    # detection compatibility
    if hasattr(aruco, "detectMarkers"):
        detect_fn = lambda img: aruco.detectMarkers(img, dictionary, parameters=detector_params)
    else:
        detector_cls = getattr(aruco, "ArucoDetector", None)
        if detector_cls is not None:
            try:
                detector = detector_cls(dictionary, detector_params)
            except Exception:
                detector = detector_cls(dictionary)
            detect_fn = lambda img: detector.detectMarkers(img)
        else:
            raise RuntimeError("cv2.aruco lacks a usable detect API")

    board = build_board(args.marker_length, args.marker_separation)

    # Load labeled template/tag images (ordered)
    tags_dir = args.tags_dir
    if not tags_dir:
        # default: project-level `aruco_tags` folder (two levels above this file)
        tags_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'aruco_tags'))

    templates = []
    try:
        p = Path(tags_dir)
        if p.exists() and p.is_dir():
            files = sorted([str(x) for x in p.iterdir() if x.is_file()])
            orb = cv2.ORB_create(1000)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            for f in files:
                try:
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    kp, des = orb.detectAndCompute(img, None)
                    label = Path(f).stem
                    templates.append({'label': label, 'path': f, 'img': img, 'kp': kp, 'des': des})
                except Exception:
                    continue
    except Exception:
        templates = []

    # print loaded template order for debugging
    if len(templates) > 0:
        print(f"Loaded {len(templates)} templates from {tags_dir}: {[t['label'] for t in templates]}")

    # ensure ORB and matcher exist even if no templates
    try:
        orb
    except NameError:
        orb = cv2.ORB_create(500)
    try:
        bf
    except NameError:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    last_pose = None
    last_time = None

    # Optional IMU via pymavlink
    mav = None
    if args.use_imu:
        try:
            from pymavlink import mavutil
            print("Connecting to FCU for heading via pymavlink...")
            mav = mavutil.mavlink_connection(args.fcu)
            mav.wait_heartbeat(timeout=5)
            print("FCU connected")
        except Exception as e:
            print("Failed to connect to FCU for IMU data:", e)
            mav = None

    def mjpeg_frame_iter(url):
        # simple MJPEG reader
        resp = requests.get(url, stream=True, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to open MJPEG stream: {resp.status_code}")
        bytes_buf = b''
        for chunk in resp.iter_content(chunk_size=1024):
            if not chunk:
                continue
            bytes_buf += chunk
            start = bytes_buf.find(b'\xff\xd8')
            end = bytes_buf.find(b'\xff\xd9')
            if start != -1 and end != -1 and end > start:
                jpg = bytes_buf[start:end+2]
                bytes_buf = bytes_buf[end+2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
                yield img

    try:
        if use_sdk:
            frame_source = None
        else:
            frame_source = mjpeg_frame_iter(args.mjpeg_url)

        while True:
            t0 = time.time()
            if use_sdk:
                depth16, ir16 = dev.extract_frames()
                if depth16 is None:
                    time.sleep(0.01)
                    continue

                if ir16 is not None:
                    mono8 = cv2.convertScaleAbs(ir16, alpha=0.5)
                else:
                    norm = cv2.normalize(depth16, None, 0, 255, cv2.NORM_MINMAX)
                    mono8 = cv2.convertScaleAbs(norm)
            else:
                try:
                    mono8 = next(frame_source)
                except StopIteration:
                    time.sleep(0.05)
                    continue
                except Exception as e:
                    print("MJPEG read error:", e)
                    time.sleep(0.1)
                    continue

            gray = mono8

            # detect markers
            try:
                res = detect_fn(gray)
            except Exception as e:
                print("Aruco detect failure:", e)
                time.sleep(0.01)
                continue

            corners = None
            ids = None
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                corners, ids = res[0], res[1]

            pose = None
            if ids is not None and len(ids) > 0:
                try:
                    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K, D)
                except Exception:
                    retval, rvec, tvec = False, None, None

                if retval and rvec is not None and tvec is not None:
                    R, _ = cv2.Rodrigues(rvec)
                    cam_pos_board = -R.T.dot(tvec).reshape(3)
                    # extract yaw (board frame)
                    R_board_cam = R.T
                    yaw = math.atan2(R_board_cam[1, 0], R_board_cam[0, 0])
                    yaw_deg = math.degrees(yaw)
                    pose = {
                        'x': float(cam_pos_board[0]),
                        'y': float(cam_pos_board[1]),
                        'z': float(cam_pos_board[2]),
                        'yaw_deg': float(yaw_deg),
                    }

            now = time.time()
            vx = vy = 0.0
            if pose is not None and last_pose is not None and last_time is not None:
                dt = max(1e-3, now - last_time)
                vx = (pose['x'] - last_pose['x']) / dt
                vy = (pose['y'] - last_pose['y']) / dt

            last_pose = pose
            last_time = now

            # read heading from FCU if available
            imu_heading = None
            if mav is not None:
                try:
                    msg = mav.recv_match(type='VFR_HUD', blocking=False)
                    if msg is not None:
                        imu_heading = float(msg.heading)
                except Exception:
                    imu_heading = None

            # draw overlay for debugging
            # perform template matching (ordered) against preloaded `templates`
            matched_label = None
            match_score = 0
            match_norm = 0.0
            try:
                kp_frame, des_frame = orb.detectAndCompute(gray, None)
            except Exception:
                kp_frame, des_frame = None, None

            best_score = 0
            for t in templates:
                if t.get('des') is None or des_frame is None:
                    continue
                try:
                    m = bf.knnMatch(t['des'], des_frame, k=2)
                except Exception:
                    continue
                # ratio test
                good = []
                for a, b in m:
                    if a.distance < 0.75 * b.distance:
                        good.append(a)
                gcount = len(good)
                if gcount > best_score:
                    best_score = gcount
                # follow ordered matching: accept first template that meets threshold
                if gcount >= args.match_threshold:
                    matched_label = t['label']
                    match_score = gcount
                    match_norm = float(gcount) / max(len(t.get('kp', [])), 1)
                    break

            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if corners is not None and ids is not None:
                aruco.drawDetectedMarkers(vis, corners, ids)
            # draw matched template label if found
            if matched_label is not None:
                cv2.putText(vis, f"TAG: {matched_label} ({match_score})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            if pose is not None:
                cv2.putText(vis, f"PX: {pose['x']:.2f}m PY: {pose['y']:.2f}m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(vis, f"V: ({vx:+.2f},{vy:+.2f}) m/s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                if imu_heading is not None:
                    cv2.putText(vis, f"IMU hdg: {imu_heading:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # encode image
            ret, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.01)
                continue

            files = {'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
            vo_dbg = {'detected_markers': int(len(ids) if ids is not None else 0)}
            if pose is not None:
                # include marker pose in vo_debug for dashboard
                vo_dbg['marker_pose'] = {
                    'x': float(pose['x']),
                    'y': float(pose['y']),
                    'z': float(pose['z']),
                    'yaw_deg': float(pose['yaw_deg'])
                }
                vo_dbg['marker_locked'] = True
            else:
                vo_dbg['marker_pose'] = None
                vo_dbg['marker_locked'] = False

            if imu_heading is not None:
                vo_dbg['imu_heading'] = float(imu_heading)

            # include template matching results
            vo_dbg['matched_tag'] = matched_label if matched_label is not None else None
            vo_dbg['match_score'] = int(match_score)
            vo_dbg['match_norm'] = float(match_norm)

            meta = {
                'yaw': float(pose['yaw_deg']) if pose is not None else (float(imu_heading) if imu_heading is not None else 0.0),
                'altitude': float(pose['z']) if pose is not None else 0.0,
                'vx': float(vx),
                'vy': float(vy),
                'kp_count': int(len(corners) if corners is not None else 0),
                'vo_debug': vo_dbg,
            }

            # post to dashboard (non-blocking send with retry)
            try:
                resp = requests.post(args.dashboard, files=files, data={'meta': json.dumps(meta)}, timeout=1.0)
                # ignore response contents
            except Exception:
                # best-effort, don't crash
                pass

            # throttle uploads
            elapsed = time.time() - t0
            to_sleep = max(0.0, args.upload_interval - elapsed)
            time.sleep(to_sleep)

    finally:
        try:
            dev.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
