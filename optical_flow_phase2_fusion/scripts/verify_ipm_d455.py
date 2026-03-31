#!/usr/bin/python3
"""
Standalone IPM verification script for Intel RealSense D455.

What it does:
1) Starts RealSense color stream at 848x480 @ 30 FPS (BGR8).
2) Warms up the camera for a short period.
3) Captures one clear frame.
4) Applies a perspective transform (IPM) with fixed src/dst points.
5) Displays original + warped bird's-eye image.
6) Exits cleanly on 'q' and stops pipeline.
"""

import os
import subprocess
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def capture_color_frame(pipeline, warmup_frames=45, timeout_ms=5000):
    """Warm up stream and return one valid BGR frame."""
    # Warm-up lets auto-exposure stabilize for a cleaner capture.
    for _ in range(warmup_frames):
        pipeline.wait_for_frames(timeout_ms=timeout_ms)

    for _ in range(30):
        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color_frame = frames.get_color_frame()
        if color_frame:
            return np.asanyarray(color_frame.get_data())

    return None


def capture_from_ros_topic(topic="/camera/camera/color/image_raw", timeout_ms=8000):
    """Capture one frame from ROS image topic (when realsense2_camera launch is running)."""
    # Prefer native ROS subscriber path when available.
    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        rclpy.init(args=None)
        node = rclpy.create_node("ipm_frame_grabber")
        bridge = CvBridge()
        latest = {"frame": None}

        def _cb(msg):
            try:
                latest["frame"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception:
                latest["frame"] = None

        sub = node.create_subscription(Image, topic, _cb, qos_profile_sensor_data)
        start = time.time()
        while (time.time() - start) * 1000.0 < timeout_ms:
            rclpy.spin_once(node, timeout_sec=0.1)
            frame = latest["frame"]
            if frame is not None and frame.size > 0:
                try:
                    node.destroy_subscription(sub)
                    node.destroy_node()
                    rclpy.shutdown()
                except Exception:
                    pass
                return frame

        try:
            node.destroy_subscription(sub)
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
    except Exception:
        pass

    # Fallback path for environments without rclpy/cv_bridge in this shell.
    cap = cv2.VideoCapture(
        f"{topic}?topic={topic}",
        cv2.CAP_FFMPEG,
    )

    if not cap.isOpened():
        return None

    start = time.time()
    frame = None
    while (time.time() - start) * 1000.0 < timeout_ms:
        ok, img = cap.read()
        if ok and img is not None and img.size > 0:
            frame = img
            break
        time.sleep(0.02)

    cap.release()
    return frame


def ros_topic_has_publisher(topic):
    """Return (exists, publisher_count) from `ros2 topic info` if available."""
    try:
        proc = subprocess.run(
            ["ros2", "topic", "info", topic],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0:
            return False, 0

        exists = "Type:" in proc.stdout
        pub_count = 0
        for line in proc.stdout.splitlines():
            if "Publisher count:" in line:
                try:
                    pub_count = int(line.split(":", 1)[1].strip())
                except Exception:
                    pub_count = 0
                break
        return exists, pub_count
    except Exception:
        return False, 0


def main():
    # If launched from conda/base python, switch to system python automatically.
    exe_real = os.path.realpath(sys.executable)
    if not exe_real.startswith("/usr/bin/python3"):
        print(f"Switching interpreter to /usr/bin/python3 (current: {exe_real})", flush=True)
        os.execv("/usr/bin/python3", ["/usr/bin/python3", os.path.abspath(__file__)] + sys.argv[1:])

    width = 848
    height = 480
    fps = 30

    frame = None
    started_pipeline = False
    pipeline = None

    print("Trying ROS image topic capture: /camera/camera/color/image_raw", flush=True)

    # Try ROS topic first so script works even when pyrealsense2 is absent
    # and when camera is already opened by realsense2_camera node.
    frame = capture_from_ros_topic("/camera/camera/color/image_raw", timeout_ms=8000)

    if frame is None and rs is not None:
        print("ROS capture did not return a frame. Trying direct pyrealsense2 pipeline...", flush=True)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # Direct SDK fallback.
        try:
            pipeline.start(config)
            started_pipeline = True
            frame = capture_color_frame(pipeline, warmup_frames=45, timeout_ms=5000)
        except Exception as exc:
            print(f"Direct RealSense pipeline unavailable: {exc}")
            frame = None

    try:
        if frame is None:
            print("Could not capture a valid color frame from RealSense or ROS topic.")
            if rs is None:
                print("pyrealsense2 not found. This is OK if ROS topic is active.")
                print("Ensure ROS env is sourced in this terminal:")
                print("  source /opt/ros/humble/setup.bash")
            exists, pub_count = ros_topic_has_publisher("/camera/camera/color/image_raw")
            print(f"ROS topic '/camera/camera/color/image_raw' exists: {exists}, publishers: {pub_count}")
            print("If ROS camera node is running, keep it running and ensure topic is active.")
            print("Otherwise close ROS node and retry direct mode:")
            print("  pkill -f realsense2_camera")
            print("  /usr/bin/python3 verify_ipm_d455.py")
            sys.exit(2)

        # Fixed source points from the slanted perspective frame.
        src_points = np.float32([
            [0, 480],
            [848, 480],
            [571, 0],
            [277, 0],
        ])

        # Destination points for a flat 848x480 bird's-eye image.
        dst_points = np.float32([
            [0, 480],
            [848, 480],
            [848, 0],
            [0, 0],
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        birdseye = cv2.warpPerspective(frame, matrix, (width, height))

        # Visual aid: draw trapezoid on original image.
        original_vis = frame.copy()
        poly = src_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(original_vis, [poly], True, (0, 255, 0), 2)

        combined = np.hstack((original_vis, birdseye))

        out_path = os.path.abspath("ipm_verification_result.png")
        cv2.imwrite(out_path, combined)
        print(f"Saved verification image: {out_path}", flush=True)

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            print("No GUI display detected (SSH/headless). Skipping cv2.imshow.", flush=True)
            print("Open 'ipm_verification_result.png' to verify result.", flush=True)
            return

        cv2.namedWindow("D455 IPM Verification", cv2.WINDOW_NORMAL)
        cv2.imshow("D455 IPM Verification", combined)

        print("IPM matrix:", flush=True)
        print(matrix, flush=True)
        print("Press 'q' in the image window to exit.", flush=True)

        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            time.sleep(0.01)

    finally:
        if started_pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
