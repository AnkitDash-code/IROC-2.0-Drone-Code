#!/usr/bin/env python3
"""
CS20 LiDAR -> ROS bridge for Phase2 optical flow.

Publishes RealSense-compatible topic names so existing Phase2 code can run
without sensor-specific refactors:
- /camera/camera/color/image_raw (mono8, from IR or normalized depth)
- /camera/camera/aligned_depth_to_color/image_raw (16UC1 mm)
- /camera/camera/color/camera_info
- /tof_sensor/range (sensor_msgs/Range)
"""

import os
import sys
import time
from ctypes import POINTER, byref, cast, c_bool, c_int, c_short, c_ubyte, c_ushort

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, Range
import threading
import requests
import io
import json

SDK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "SynexensPythonSDK4_4.2.4.0_202504281506")
)
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)

from SynexensPythonSDK import (  # type: ignore
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


class CS20LidarBridge(Node):
    def __init__(self):
        super().__init__("cs20_lidar_bridge")

        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("resolution", "320x240")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("mono_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("ir_topic", "/camera/camera/ir/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("range_topic", "/tof_sensor/range")
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("dashboard_url", "")

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.mono_topic = str(self.get_parameter("mono_topic").value)
        self.ir_topic = str(self.get_parameter("ir_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.range_topic = str(self.get_parameter("range_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.dashboard_url = str(self.get_parameter("dashboard_url").value)

        res_str = str(self.get_parameter("resolution").value).lower()
        if res_str == "640x480":
            self.resolution = SYResolutionEnum.SYRESOLUTION_640_480
            self.width, self.height = 640, 480
        else:
            self.resolution = SYResolutionEnum.SYRESOLUTION_320_240
            self.width, self.height = 320, 240

        # avoid CvBridge (possible binary conflicts); construct Image messages manually
        self.bridge = None
        self.pub_depth = self.create_publisher(Image, self.depth_topic, 10)
        self.pub_mono = self.create_publisher(Image, self.mono_topic, 10)
        self.pub_ir = self.create_publisher(Image, self.ir_topic, 10)
        self.pub_info = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        self.pub_range = self.create_publisher(Range, self.range_topic, 10)

        self._device_id = -1
        self._streaming = False
        self._fx = 381.0
        self._fy = 381.0
        self._cx = float(self.width) * 0.5
        self._cy = float(self.height) * 0.5

        self._init_sdk()
        # We'll call _tick from the main loop to keep SDK calls single-threaded
        self.get_logger().info(
            f"CS20 bridge started | mono={self.mono_topic} depth={self.depth_topic} range={self.range_topic}"
        )

    def _init_sdk(self):
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

        # Use depth stream (matches ransac.py) to avoid interleaved IR controls
        st = StartStreaming(self._device_id, SYStreamTypeEnum.SYSTREAMTYPE_DEPTH)
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

        if self.dashboard_url:
            self.get_logger().info(f"Dashboard posting enabled -> {self.dashboard_url}")

    @staticmethod
    def _frame_bytes(frame_type, n_count):
        if int(frame_type) == int(SYFrameTypeEnum.SYFRAMETYPE_RGB):
            return (n_count * 3) // 2  # YUYV
        return n_count * 2

    def _extract_frames(self):
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

    def _publish_camera_info(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.width = int(self.width)
        msg.height = int(self.height)
        msg.k = [self._fx, 0.0, self._cx, 0.0, self._fy, self._cy, 0.0, 0.0, 1.0]
        msg.p = [self._fx, 0.0, self._cx, 0.0, 0.0, self._fy, self._cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.distortion_model = "plumb_bob"
        self.pub_info.publish(msg)

    def _publish_range(self, depth_mm, stamp):
        if depth_mm is None:
            return
        h, w = depth_mm.shape[:2]
        cy = h // 2
        cx = w // 2
        patch = depth_mm[max(0, cy - 2):min(h, cy + 3), max(0, cx - 2):min(w, cx + 3)]
        valid = patch[(patch > 50) & (patch < 10000)]
        if valid.size == 0:
            return
        rng = float(np.median(valid)) / 1000.0

        msg = Range()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.7
        msg.min_range = 0.05
        msg.max_range = 10.0
        msg.range = rng
        self.pub_range.publish(msg)
        # Note: dashboard posting will be performed from the main _tick loop

    def _tick(self):
        depth16, ir16 = self._extract_frames()
        if depth16 is None:
            return

        if ir16 is not None:
            mono8 = cv2.convertScaleAbs(ir16, alpha=0.5)
        else:
            norm = cv2.normalize(depth16, None, 0, 255, cv2.NORM_MINMAX)
            mono8 = cv2.convertScaleAbs(norm)

        now = self.get_clock().now().to_msg()

        # build depth Image message without CvBridge
        depth_msg = Image()
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = self.frame_id
        depth_msg.height = int(depth16.shape[0])
        depth_msg.width = int(depth16.shape[1])
        depth_msg.encoding = "16UC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = int(depth16.shape[1] * 2)
        depth_msg.data = depth16.astype(np.uint16).tobytes()
        self.pub_depth.publish(depth_msg)

        mono_msg = Image()
        mono_msg.header.stamp = now
        mono_msg.header.frame_id = self.frame_id
        mono_msg.height = int(mono8.shape[0])
        mono_msg.width = int(mono8.shape[1])
        mono_msg.encoding = "mono8"
        mono_msg.is_bigendian = 0
        mono_msg.step = int(mono8.shape[1])
        mono_msg.data = mono8.tobytes()
        self.pub_mono.publish(mono_msg)
        self.pub_ir.publish(mono_msg)

        self._publish_camera_info(now)
        self._publish_range(depth16, now)
        # If dashboard_url configured, encode and POST a JPEG + camera info (best-effort)
        if self.dashboard_url:
            try:
                if ir16 is not None:
                    mono8 = cv2.convertScaleAbs(ir16, alpha=0.5)
                else:
                    norm = cv2.normalize(depth16, None, 0, 255, cv2.NORM_MINMAX)
                    mono8 = cv2.convertScaleAbs(norm)

                vis = cv2.cvtColor(mono8, cv2.COLOR_GRAY2BGR)
                ret, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    files = {'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
                    meta = {
                        'yaw': 0.0,
                        'altitude': 0.0,
                        'vx': 0.0,
                        'vy': 0.0,
                        'kp_count': 0,
                        'vo_debug': {'marker_locked': False},
                        'camera_info': {
                            'width': int(self.width),
                            'height': int(self.height),
                            'fx': float(self._fx),
                            'fy': float(self._fy),
                            'cx': float(self._cx),
                            'cy': float(self._cy),
                        }
                    }
                    try:
                        requests.post(self.dashboard_url, files=files, data={'meta': json.dumps(meta)}, timeout=0.5)
                    except Exception:
                        pass
            except Exception:
                pass

    def destroy_node(self):
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
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CS20LidarBridge()
    try:
        # Run a single-threaded loop: call _tick and spin_once so SDK calls
        # happen on the same thread as ROS spinning (avoids SDK thread-safety issues).
        rate = 1.0 / max(1.0, node.publish_hz)
        while rclpy.ok():
            t0 = time.time()
            try:
                node._tick()
            except Exception:
                pass
            # allow rclpy to process callbacks without using timer threads
            rclpy.spin_once(node, timeout_sec=0.0)
            dt = time.time() - t0
            to_sleep = max(0.0, rate - dt)
            time.sleep(to_sleep)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
