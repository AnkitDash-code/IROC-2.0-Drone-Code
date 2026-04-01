#!/usr/bin/env python3
"""
Generate a `camera_calib.npz` for the ArUco landing node.

By default this writes a reasonable CS20 fallback intrinsics file for
320x240 or 640x480. Optionally, capture the first `sensor_msgs/CameraInfo`
published on a topic and write those intrinsics instead (requires ROS2).

Usage:
  python3 generate_camera_calib.py --resolution 320x240
  python3 generate_camera_calib.py --from-topic /camera/camera/color/camera_info --wait 5
"""

import os
import argparse
import numpy as np


def default_intrinsics(res):
    if res == "640x480":
        w, h = 640, 480
        fx = fy = 762.0
    else:
        w, h = 320, 240
        fx = fy = 381.0
    cx = float(w) / 2.0
    cy = float(h) / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros((5,), dtype=np.float64)
    return K, D


def save_npz(path, K, D):
    np.savez(path, camera_matrix=K, dist_coeffs=D)
    print("Saved:", path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", choices=("320x240", "640x480"), default="320x240")
    p.add_argument("--out", default="camera_calib.npz", help="Output file name (placed beside this script)")
    p.add_argument("--from-topic", dest="topic", default=None, help="ROS2 CameraInfo topic to read instead of defaults")
    p.add_argument("--wait", type=float, default=5.0, help="Seconds to wait for a CameraInfo message when using --from-topic")
    return p.parse_args()


def main():
    args = parse_args()
    base = os.path.dirname(__file__)
    outp = os.path.join(base, args.out)

    if args.topic:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import CameraInfo
        except Exception as e:
            print("ROS2 not available or import failed:", e)
            print("Falling back to default intrinsics.")
            K, D = default_intrinsics(args.resolution)
            save_npz(outp, K, D)
            return

        class _Listener(Node):
            def __init__(self):
                super().__init__("caminfo_dump")
                self.msg = None
                self.sub = self.create_subscription(CameraInfo, args.topic, self._cb, 10)

            def _cb(self, msg):
                self.msg = msg

        rclpy.init()
        node = _Listener()
        import time
        start = time.time()
        while (time.time() - start) < args.wait and node.msg is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        if node.msg is not None:
            ci = node.msg
            K = np.array(ci.k, dtype=np.float64).reshape((3, 3))
            D = np.array(ci.d, dtype=np.float64)
            if D.size == 0:
                D = np.zeros((5,), dtype=np.float64)
            save_npz(outp, K, D)
        else:
            print("No CameraInfo received within timeout; using defaults.")
            K, D = default_intrinsics(args.resolution)
            save_npz(outp, K, D)

        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    else:
        K, D = default_intrinsics(args.resolution)
        save_npz(outp, K, D)


if __name__ == "__main__":
    main()
