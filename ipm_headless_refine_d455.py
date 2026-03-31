#!/usr/bin/python3
"""
Headless IPM refinement tool for D455 (no GUI required).

Use case:
- Drone must remain stationary.
- No display/X forwarding available.
- Iteratively tune 4 src points from terminal and saved images.

Example:
  source /opt/ros/humble/setup.bash
  /usr/bin/python3 ipm_headless_refine_d455.py \
    --src "0,479 847,479 620,220 228,220" --save-config

Outputs:
- ipm_headless_original.png
- ipm_headless_warped.png
- ipm_headless_preview.png (side-by-side)
- ipm_config.json (optional, when --save-config)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


WIDTH = 848
HEIGHT = 480
FPS = 30
COLOR_TOPIC = "/camera/camera/color/image_raw"


def parse_src_points(src_text):
    parts = src_text.strip().split()
    if len(parts) != 4:
        raise ValueError("--src must have exactly 4 points: 'x1,y1 x2,y2 x3,y3 x4,y4'")

    pts = []
    for p in parts:
        xy = p.split(",")
        if len(xy) != 2:
            raise ValueError(f"Invalid point format: {p}")
        x = float(xy[0])
        y = float(xy[1])
        pts.append([x, y])
    return np.float32(pts)


def capture_from_ros_topic(topic=COLOR_TOPIC, timeout_ms=8000):
    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        rclpy.init(args=None)
        node = rclpy.create_node("ipm_headless_capture")
        bridge = CvBridge()
        latest = {"img": None}

        def _cb(msg):
            try:
                latest["img"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                latest["img"] = None

        sub = node.create_subscription(Image, topic, _cb, qos_profile_sensor_data)

        start = time.time()
        while (time.time() - start) * 1000.0 < timeout_ms:
            rclpy.spin_once(node, timeout_sec=0.1)
            if latest["img"] is not None:
                img = latest["img"]
                node.destroy_subscription(sub)
                node.destroy_node()
                rclpy.shutdown()
                return img

        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()
        return None
    except Exception:
        return None


def capture_from_realsense(timeout_ms=5000):
    if rs is None:
        return None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    try:
        pipeline.start(config)
        for _ in range(40):
            pipeline.wait_for_frames(timeout_ms=timeout_ms)
        for _ in range(20):
            frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            c = frames.get_color_frame()
            if c:
                return np.asanyarray(c.get_data())
    except Exception:
        return None
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Headless IPM point refinement")
    parser.add_argument(
        "--src",
        default="0,479 847,479 571,0 277,0",
        help="4 source points: 'x1,y1 x2,y2 x3,y3 x4,y4' in order BL BR TR TL",
    )
    parser.add_argument("--save-config", action="store_true", help="Write ipm_config.json")
    parser.add_argument("--topic", default=COLOR_TOPIC, help="ROS color topic")
    args = parser.parse_args()

    src_points = parse_src_points(args.src)
    dst_points = np.float32([
        [0, HEIGHT - 1],
        [WIDTH - 1, HEIGHT - 1],
        [WIDTH - 1, 0],
        [0, 0],
    ])

    print(f"Capturing frame from ROS topic: {args.topic}", flush=True)
    frame = capture_from_ros_topic(topic=args.topic, timeout_ms=8000)

    if frame is None:
        print("ROS capture failed, trying direct pyrealsense2...", flush=True)
        frame = capture_from_realsense(timeout_ms=5000)

    if frame is None:
        print("ERROR: Could not capture frame from ROS topic or pyrealsense2.")
        sys.exit(2)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(frame, matrix, (WIDTH, HEIGHT))

    original_vis = frame.copy()
    poly = src_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(original_vis, [poly], True, (0, 255, 0), 2)

    preview = np.hstack((original_vis, warped))

    cv2.imwrite("ipm_headless_original.png", original_vis)
    cv2.imwrite("ipm_headless_warped.png", warped)
    cv2.imwrite("ipm_headless_preview.png", preview)

    print("Saved:")
    print("- ipm_headless_original.png")
    print("- ipm_headless_warped.png")
    print("- ipm_headless_preview.png")
    print("Matrix:")
    print(matrix)

    if args.save_config:
        cfg = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "resolution": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
            "src_points": src_points.tolist(),
            "dst_points": dst_points.tolist(),
            "matrix": matrix.tolist(),
        }
        with open("ipm_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print("Saved: ipm_config.json")


if __name__ == "__main__":
    main()
