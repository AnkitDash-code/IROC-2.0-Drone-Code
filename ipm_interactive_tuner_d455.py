#!/usr/bin/python3
"""
Interactive IPM tuner for Intel RealSense D455.

Features:
- Live color stream: 848x480 @ 30 FPS
- Click 4 floor points in order:
  1) Bottom-Left
  2) Bottom-Right
  3) Top-Right (just below floor-wall boundary)
  4) Top-Left  (same height as Top-Right)
- Shows warped bird's-eye view immediately after 4 points
- Keys:
  - r: reset points
  - s: save config to ipm_config.json
  - q: quit

Saved file (ipm_config.json) contains:
- src_points
- dst_points
- matrix
- resolution
"""

import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as exc:
    print("Failed to import pyrealsense2.")
    print("Run with system Python and ensure librealsense Python bindings are installed.")
    print("Example:")
    print("  /usr/bin/python3 ipm_interactive_tuner_d455.py")
    print(f"Import error: {exc}")
    sys.exit(1)


WINDOW_LIVE = "IPM Tuner - Live"
WINDOW_WARP = "IPM Tuner - Warped"

WIDTH = 848
HEIGHT = 480
FPS = 30

SRC_LABELS = ["BL", "BR", "TR", "TL"]


def draw_ui(frame, points):
    vis = frame.copy()

    # Draw instructions panel.
    cv2.rectangle(vis, (8, 8), (430, 128), (0, 0, 0), -1)
    cv2.rectangle(vis, (8, 8), (430, 128), (50, 255, 50), 1)
    cv2.putText(vis, "Click 4 floor points in order:", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 1)
    cv2.putText(vis, "1) BL  2) BR  3) TR  4) TL", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 1)
    cv2.putText(vis, "Keys: r=reset, s=save, q=quit", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 1)
    cv2.putText(vis, f"Points: {len(points)}/4", (16, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 1)

    # Draw clicked points and trapezoid.
    for i, pt in enumerate(points):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(vis, SRC_LABELS[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if len(points) >= 2:
        poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], False, (0, 255, 0), 2)

    if len(points) == 4:
        poly_closed = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly_closed], True, (0, 255, 0), 2)

    return vis


def compute_ipm(points, width, height):
    src_points = np.float32(points)
    dst_points = np.float32([
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0],
        [0, 0],
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return src_points, dst_points, matrix


def save_config(path, src_points, dst_points, matrix):
    cfg = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "resolution": {"width": WIDTH, "height": HEIGHT, "fps": FPS},
        "src_points": src_points.tolist(),
        "dst_points": dst_points.tolist(),
        "matrix": matrix.tolist(),
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def main():
    # Ensure system python if launched from conda/base env.
    exe_real = os.path.realpath(sys.executable)
    if not exe_real.startswith("/usr/bin/python3"):
        print(f"Switching interpreter to /usr/bin/python3 (current: {exe_real})", flush=True)
        os.execv("/usr/bin/python3", ["/usr/bin/python3", os.path.abspath(__file__)] + sys.argv[1:])

    points = []
    latest_frame = {"img": None}
    use_ros_topic = False
    ros_node = None
    ros_sub = None

    # Interactive tuner needs a display for mouse clicks.
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No GUI display detected. Run locally on Jetson desktop or via SSH with X forwarding.")
        print("Example: ssh -X jetson123@<jetson_ip>")
        sys.exit(3)

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append([float(x), float(y)])
                print(f"Point {len(points)} ({SRC_LABELS[len(points)-1]}): ({x}, {y})", flush=True)
            else:
                print("Already have 4 points. Press 'r' to reset.", flush=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    try:
        pipeline.start(config)
    except Exception as exc:
        print(f"Direct RealSense stream unavailable: {exc}")
        print("Falling back to ROS topic /camera/camera/color/image_raw ...")
        use_ros_topic = True

    if not use_ros_topic:
        # Warm-up for stable auto-exposure.
        print("Warming up camera...", flush=True)
        for _ in range(45):
            pipeline.wait_for_frames(timeout_ms=5000)
        print("Ready. Click points in live window.", flush=True)
    else:
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge

            rclpy.init(args=None)
            ros_node = rclpy.create_node("ipm_tuner_ros_frame_source")
            bridge = CvBridge()

            def _cb(msg):
                try:
                    latest_frame["img"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception:
                    latest_frame["img"] = None

            ros_sub = ros_node.create_subscription(Image, "/camera/camera/color/image_raw", _cb, qos_profile_sensor_data)

            print("Waiting for ROS color frames...", flush=True)
            t0 = time.time()
            while time.time() - t0 < 8.0:
                rclpy.spin_once(ros_node, timeout_sec=0.1)
                if latest_frame["img"] is not None:
                    break
            if latest_frame["img"] is None:
                print("No ROS color frames received from /camera/camera/color/image_raw")
                sys.exit(4)
            print("Ready. Click points in live window.", flush=True)
        except Exception as exc:
            print(f"ROS fallback failed: {exc}")
            sys.exit(5)

    cv2.namedWindow(WINDOW_LIVE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_WARP, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_LIVE, on_mouse)

    try:
        while True:
            if use_ros_topic:
                import rclpy

                rclpy.spin_once(ros_node, timeout_sec=0.01)
                frame = latest_frame["img"]
                if frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("Quit requested.", flush=True)
                        break
                    continue
            else:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                latest_frame["img"] = frame

            live_vis = draw_ui(frame, points)
            cv2.imshow(WINDOW_LIVE, live_vis)

            if len(points) == 4:
                src_points, dst_points, matrix = compute_ipm(points, WIDTH, HEIGHT)
                warped = cv2.warpPerspective(frame, matrix, (WIDTH, HEIGHT))
                cv2.imshow(WINDOW_WARP, warped)
            else:
                # Show blank placeholder until 4 points are selected.
                placeholder = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Select 4 points to preview warp",
                    (180, HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(WINDOW_WARP, placeholder)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.", flush=True)
                break
            if key == ord("r"):
                points.clear()
                print("Points reset.", flush=True)
            if key == ord("s"):
                if len(points) != 4:
                    print("Need exactly 4 points before saving.", flush=True)
                    continue
                src_points, dst_points, matrix = compute_ipm(points, WIDTH, HEIGHT)
                save_config("ipm_config.json", src_points, dst_points, matrix)
                print("Saved ipm_config.json", flush=True)
                print("Matrix:", flush=True)
                print(matrix, flush=True)

            time.sleep(0.001)

    finally:
        if not use_ros_topic:
            pipeline.stop()
        else:
            try:
                if ros_node is not None and ros_sub is not None:
                    ros_node.destroy_subscription(ros_sub)
                if ros_node is not None:
                    ros_node.destroy_node()
                import rclpy

                rclpy.shutdown()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
