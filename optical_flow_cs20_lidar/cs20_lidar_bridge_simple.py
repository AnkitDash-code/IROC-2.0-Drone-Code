#!/usr/bin/env python3
"""
Simple CS20 streamer (non-ROS) modeled after Synexens `ransac.py`.

Owns the SDK device, reads frames, and POSTs JPEG + camera intrinsics
to a dashboard endpoint. Use this when you want the bridge to be the
single owner of the device and avoid embedding the SDK inside the ROS
process.

Usage:
  python3 cs20_lidar_bridge_simple.py --dashboard http://localhost:5000/api/upload_frame
"""

import time
import argparse
import os
import json
import requests
import numpy as np
import cv2
from ctypes import POINTER, byref, cast, c_int, c_ushort

SDK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SynexensPythonSDK4_4.2.4.0_202504281506'))
if SDK_DIR not in os.sys.path:
    os.sys.path.insert(0, SDK_DIR)

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
    SYDeviceInfo,
    SYErrorCodeEnum,
    SYStreamTypeEnum,
    SYFrameTypeEnum,
    SYResolutionEnum,
    SYFrameData,
    SYIntrinsics,
    GetIntric,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard", default="http://localhost:5000/api/upload_frame")
    p.add_argument("--resolution", choices=("320x240", "640x480"), default="320x240")
    p.add_argument("--interval", type=float, default=0.08)
    return p.parse_args()


class CS20Simple:
    def __init__(self, resolution_str):
        if resolution_str == "640x480":
            self.res = SYResolutionEnum.SYRESOLUTION_640_480
            self.w, self.h = 640, 480
        else:
            self.res = SYResolutionEnum.SYRESOLUTION_320_240
            self.w, self.h = 320, 240
        self.dev_id = -1
        self.fx = 381.0
        self.fy = 381.0
        self.cx = float(self.w) * 0.5
        self.cy = float(self.h) * 0.5

    def init(self):
        if InitSDK() != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            raise RuntimeError("InitSDK failed")

        n = c_int(0)
        if FindDevice(byref(n), None) != SYErrorCodeEnum.SYERRORCODE_SUCCESS or n.value <= 0:
            UnInitSDK()
            raise RuntimeError("No Synexens devices found")

        devs = (SYDeviceInfo * n.value)()
        if FindDevice(byref(n), devs) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            UnInitSDK()
            raise RuntimeError("FindDevice list failed")

        dev = devs[0]
        self.dev_id = int(dev.m_nDeviceID)

        if OpenDevice(dev) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            UnInitSDK()
            raise RuntimeError("OpenDevice failed")

        if SetFrameResolution(self.dev_id, SYFrameTypeEnum.SYFRAMETYPE_DEPTH, self.res) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            CloseDevice(self.dev_id)
            UnInitSDK()
            raise RuntimeError("SetFrameResolution failed")

        if StartStreaming(self.dev_id, SYStreamTypeEnum.SYSTREAMTYPE_DEPTH) != SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            CloseDevice(self.dev_id)
            UnInitSDK()
            raise RuntimeError("StartStreaming failed")

        intr = SYIntrinsics()
        if GetIntric(self.dev_id, self.res, intr) == SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            if intr.m_fltFocalDistanceX > 1.0:
                self.fx = float(intr.m_fltFocalDistanceX)
            if intr.m_fltFocalDistanceY > 1.0:
                self.fy = float(intr.m_fltFocalDistanceY)
            self.cx = float(intr.m_fltCenterPointX)
            self.cy = float(intr.m_fltCenterPointY)

    def extract(self):
        p_frame = POINTER(SYFrameData)()
        err = GetLastFrameData(self.dev_id, byref(p_frame))
        if err != SYErrorCodeEnum.SYERRORCODE_SUCCESS or not p_frame or not p_frame.contents:
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
            offset += (n * 2)

        return depth, ir

    def close(self):
        try:
            StopStreaming(self.dev_id)
        except Exception:
            pass
        try:
            CloseDevice(self.dev_id)
        except Exception:
            pass
        try:
            UnInitSDK()
        except Exception:
            pass


def main():
    args = parse_args()
    s = CS20Simple(args.resolution)
    print(SDK_DIR)
    try:
        s.init()
    except Exception as e:
        print("Init failed:", e)
        return

    try:
        while True:
            depth, ir = s.extract()
            if depth is None:
                time.sleep(0.01)
                continue

            if ir is not None:
                mono8 = cv2.convertScaleAbs(ir, alpha=0.5)
            else:
                norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                mono8 = cv2.convertScaleAbs(norm)

            vis = cv2.cvtColor(mono8, cv2.COLOR_GRAY2BGR)
            ret, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(args.interval)
                continue

            files = {'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
            meta = {
                'camera_info': {'width': s.w, 'height': s.h, 'fx': s.fx, 'fy': s.fy, 'cx': s.cx, 'cy': s.cy},
            }
            try:
                requests.post(args.dashboard, files=files, data={'meta': json.dumps(meta)}, timeout=0.5)
            except Exception:
                pass

            time.sleep(args.interval)

    finally:
        s.close()


if __name__ == '__main__':
    main()
