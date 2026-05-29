#!/usr/bin/env python3
"""
Deep-test monitor for d455_ipm_lk_velocity_node.py.

What it provides:
1) Vector overlay diagnostics on IPM-warped floor image.
2) Velocity vs pitch correlation metrics.
3) Quality heatmap-style status in logs and CSV.

This file stays in root directory and does not modify main code folders.
"""

from __future__ import annotations

import csv
import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, Range
from std_msgs.msg import UInt8


@dataclass
class FlowStats:
    tracked: int = 0
    median_dx: float = 0.0
    median_dy: float = 0.0
    mag_median: float = 0.0
    angle_std_deg: float = 180.0
    mag_cv: float = 10.0


class DeepTestMonitor(Node):
    def __init__(self) -> None:
        super().__init__("d455_ipm_lk_deep_test_monitor")

        # ---------------------------
        # Parameters
        # ---------------------------
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("range_topic", "/mavros/distance_sensor/rangefinder")
        self.declare_parameter("imu_topic", "/camera/imu")
        self.declare_parameter("twist_topic", "/mavros/vision_speed/speed_twist")
        self.declare_parameter("quality_topic", "/d455_ipm_lk_velocity_node/quality")

        self.declare_parameter("width", 848)
        self.declare_parameter("height", 480)
        self.declare_parameter("show_window", False)
        self.declare_parameter("save_video", True)
        self.declare_parameter("video_path", "d455_ipm_lk_debug.mp4")
        self.declare_parameter("csv_path", "d455_ipm_lk_deep_test.csv")
        self.declare_parameter("process_hz", 20.0)
        self.declare_parameter("log_period_s", 1.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.range_topic = str(self.get_parameter("range_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.twist_topic = str(self.get_parameter("twist_topic").value)
        self.quality_topic = str(self.get_parameter("quality_topic").value)

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.save_video = bool(self.get_parameter("save_video").value)
        self.video_path = str(self.get_parameter("video_path").value)
        self.csv_path = str(self.get_parameter("csv_path").value)
        self.process_hz = float(self.get_parameter("process_hz").value)
        self.log_period_s = float(self.get_parameter("log_period_s").value)

        # ---------------------------
        # IPM config (same as velocity node)
        # ---------------------------
        src_points = np.float32([
            [0.0, 480.0],
            [848.0, 480.0],
            [571.0, 340.0],
            [277.0, 340.0],
        ])
        dst_points = np.float32([
            [0.0, 480.0],
            [848.0, 480.0],
            [848.0, 0.0],
            [0.0, 0.0],
        ])
        self.h_ipm = cv2.getPerspectiveTransform(src_points, dst_points)

        # ---------------------------
        # State
        # ---------------------------
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        self.latest_gray: Optional[np.ndarray] = None
        self.latest_stamp: Optional[float] = None
        self.latest_alt: Optional[float] = None

        self.latest_pitch_deg: float = 0.0
        self.latest_pitch_rate_dps: float = 0.0
        self._prev_pitch_deg: Optional[float] = None
        self._prev_pitch_stamp: Optional[float] = None

        self.v_forward: float = 0.0
        self.v_right: float = 0.0
        self.quality: int = 0

        self.prev_warped: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_stamp: Optional[float] = None

        self.stats = FlowStats()

        # Rolling windows for correlation checks.
        self.roll_n = 180
        self.roll_pitch = deque(maxlen=self.roll_n)
        self.roll_pitch_rate = deque(maxlen=self.roll_n)
        self.roll_vf = deque(maxlen=self.roll_n)
        self.roll_mag = deque(maxlen=self.roll_n)

        # CSV logging.
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "t", "alt", "pitch_deg", "pitch_rate_dps", "v_forward", "v_right",
            "quality", "tracked", "med_dx", "med_dy", "med_mag", "angle_std_deg", "mag_cv"
        ])

        self.last_log_time = self.get_clock().now()

        self.video_writer = None
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 20.0, (self.width, self.height))
            if not self.video_writer.isOpened():
                self.get_logger().warn(f"Could not open video writer: {self.video_path}")
                self.video_writer = None

        # OpenCV parameters.
        self.feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # ---------------------------
        # ROS
        # ---------------------------
        cbg = ReentrantCallbackGroup()

        self.sub_image = self.create_subscription(
            Image, self.image_topic, self.image_cb, qos_profile_sensor_data, callback_group=cbg
        )
        self.sub_range = self.create_subscription(
            Range, self.range_topic, self.range_cb, qos_profile_sensor_data, callback_group=cbg
        )
        self.sub_imu = self.create_subscription(
            Imu, self.imu_topic, self.imu_cb, qos_profile_sensor_data, callback_group=cbg
        )
        self.sub_twist = self.create_subscription(
            TwistStamped, self.twist_topic, self.twist_cb, qos_profile_sensor_data, callback_group=cbg
        )
        self.sub_quality = self.create_subscription(
            UInt8, self.quality_topic, self.quality_cb, qos_profile_sensor_data, callback_group=cbg
        )

        self.pub_debug = self.create_publisher(Image, "~/debug_image", 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.process_hz), self.process_cb, callback_group=cbg)

        self.get_logger().info("Deep test monitor started")
        self.get_logger().info(
            f"subscribers: image={self.image_topic}, range={self.range_topic}, imu={self.imu_topic}, "
            f"twist={self.twist_topic}, quality={self.quality_topic}"
        )
        self.get_logger().info(f"CSV logging: {os.path.abspath(self.csv_path)}")
        if self.video_writer is not None:
            self.get_logger().info(f"Video logging: {os.path.abspath(self.video_path)}")

    # ---------------------------
    # Callbacks
    # ---------------------------
    def image_cb(self, msg: Image) -> None:
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self.lock:
            self.latest_gray = gray
            self.latest_stamp = stamp

    def range_cb(self, msg: Range) -> None:
        with self.lock:
            self.latest_alt = float(msg.range)

    def imu_cb(self, msg: Imu) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        q = msg.orientation
        pitch = self._quaternion_to_pitch_deg(q.x, q.y, q.z, q.w)

        with self.lock:
            if self._prev_pitch_deg is not None and self._prev_pitch_stamp is not None:
                dt = stamp - self._prev_pitch_stamp
                if 1e-4 < dt < 0.2:
                    self.latest_pitch_rate_dps = (pitch - self._prev_pitch_deg) / dt
            self._prev_pitch_deg = pitch
            self._prev_pitch_stamp = stamp
            self.latest_pitch_deg = pitch

    def twist_cb(self, msg: TwistStamped) -> None:
        with self.lock:
            self.v_forward = float(msg.twist.linear.x)
            self.v_right = float(msg.twist.linear.y)

    def quality_cb(self, msg: UInt8) -> None:
        with self.lock:
            self.quality = int(msg.data)

    # ---------------------------
    # Processing
    # ---------------------------
    def process_cb(self) -> None:
        with self.lock:
            gray = None if self.latest_gray is None else self.latest_gray.copy()
            stamp = self.latest_stamp
            alt = self.latest_alt
            pitch = self.latest_pitch_deg
            pitch_rate = self.latest_pitch_rate_dps
            v_forward = self.v_forward
            v_right = self.v_right
            quality = self.quality

        if gray is None or stamp is None:
            return

        warped = cv2.warpPerspective(gray, self.h_ipm, (self.width, self.height))
        warped = cv2.GaussianBlur(warped, (5, 5), 0)

        if self.prev_warped is None or self.prev_stamp is None:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            return

        dt = stamp - self.prev_stamp
        if dt <= 1e-4 or dt > 0.25:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            return

        if self.prev_pts is None or len(self.prev_pts) < 20:
            self.prev_pts = self._detect_features(self.prev_warped)

        prev_pts = self.prev_pts
        if prev_pts is None:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            return

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_warped,
            warped,
            prev_pts,
            None,
            **self.lk_params,
        )

        if next_pts is None or status is None:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            return

        n = min(len(prev_pts), len(next_pts), len(status))
        if n <= 0:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            return

        prev_pts = prev_pts[:n]
        next_pts = next_pts[:n]
        status = status[:n]
        ok = status.ravel() == 1
        tracked = int(np.sum(ok))

        overlay = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        flow_stats = FlowStats(tracked=tracked)

        if tracked >= 5:
            p0 = prev_pts[ok].reshape(-1, 2)
            p1 = next_pts[ok].reshape(-1, 2)
            d = p1 - p0

            dx = d[:, 0]
            dy = d[:, 1]
            mag = np.sqrt(dx * dx + dy * dy)
            ang = np.degrees(np.arctan2(dy, dx))

            flow_stats.median_dx = float(np.median(dx))
            flow_stats.median_dy = float(np.median(dy))
            flow_stats.mag_median = float(np.median(mag))
            flow_stats.angle_std_deg = float(np.std(ang))
            flow_stats.mag_cv = float(np.std(mag) / max(1e-6, np.mean(mag)))

            # Draw sample vectors.
            sample_step = max(1, len(p0) // 80)
            for i in range(0, len(p0), sample_step):
                x0, y0 = int(p0[i, 0]), int(p0[i, 1])
                x1, y1 = int(p1[i, 0]), int(p1[i, 1])
                cv2.arrowedLine(overlay, (x0, y0), (x1, y1), (0, 255, 0), 1, tipLength=0.25)

        # Store for status logs and CSV.
        self.stats = flow_stats
        self.roll_pitch.append(float(pitch))
        self.roll_pitch_rate.append(float(pitch_rate))
        self.roll_vf.append(float(v_forward))
        self.roll_mag.append(float(flow_stats.mag_median))

        self._draw_hud(
            overlay,
            alt,
            pitch,
            pitch_rate,
            v_forward,
            v_right,
            quality,
            flow_stats,
        )

        # Publish debug image topic.
        try:
            dbg_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            self.pub_debug.publish(dbg_msg)
        except Exception as exc:
            self.get_logger().warn(f"Debug image publish failed: {exc}")

        if self.video_writer is not None:
            self.video_writer.write(overlay)

        if self.show_window:
            cv2.imshow("d455_ipm_lk_deep_test", overlay)
            cv2.waitKey(1)

        self._write_csv(stamp, alt, pitch, pitch_rate, v_forward, v_right, quality, flow_stats)
        self._periodic_log(stamp, alt, pitch, pitch_rate, v_forward, v_right, quality, flow_stats)

        # Advance tracking state.
        if tracked < 30:
            self.prev_pts = self._detect_features(warped)
        else:
            self.prev_pts = next_pts[ok].reshape(-1, 1, 2).astype(np.float32)
        self.prev_warped = warped
        self.prev_stamp = stamp

    # ---------------------------
    # Utility
    # ---------------------------
    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        if pts is None:
            return None
        return pts.astype(np.float32)

    @staticmethod
    def _quaternion_to_pitch_deg(x: float, y: float, z: float, w: float) -> float:
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
        return math.degrees(pitch)

    @staticmethod
    def _pearson(a: deque, b: deque) -> float:
        if len(a) < 20 or len(b) < 20:
            return 0.0
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
            return 0.0
        return float(np.corrcoef(aa, bb)[0, 1])

    def _quality_label_color(self, q: int) -> Tuple[str, Tuple[int, int, int]]:
        if q >= 2:
            return "GREEN(HIGH)", (0, 220, 0)
        if q == 1:
            return "YELLOW(MED)", (0, 220, 220)
        return "RED(LOW)", (0, 0, 220)

    def _draw_hud(
        self,
        img: np.ndarray,
        alt: Optional[float],
        pitch: float,
        pitch_rate: float,
        v_forward: float,
        v_right: float,
        quality: int,
        s: FlowStats,
    ) -> None:
        q_text, q_color = self._quality_label_color(quality)

        y = 24
        line_h = 24
        lines = [
            f"ALT={alt if alt is not None else -1.0:.2f}m  PITCH={pitch:+.2f}deg  PITCH_RATE={pitch_rate:+.2f}deg/s",
            f"V_FORWARD={v_forward:+.3f}m/s  V_RIGHT={v_right:+.3f}m/s  QUALITY={q_text}",
            f"TRACKED={s.tracked}  MED_DX={s.median_dx:+.3f}  MED_DY={s.median_dy:+.3f}  MED_MAG={s.mag_median:.3f}",
            f"ANGLE_STD={s.angle_std_deg:.2f}deg  MAG_CV={s.mag_cv:.3f}",
            "EXPECTED: forward motion -> vectors mostly downward/backward in warped frame",
        ]

        for i, t in enumerate(lines):
            c = q_color if i == 1 else (200, 255, 200)
            cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
            y += line_h

    def _write_csv(
        self,
        t: float,
        alt: Optional[float],
        pitch: float,
        pitch_rate: float,
        v_forward: float,
        v_right: float,
        quality: int,
        s: FlowStats,
    ) -> None:
        self._csv.writerow([
            f"{t:.6f}",
            f"{(alt if alt is not None else -1.0):.6f}",
            f"{pitch:.6f}",
            f"{pitch_rate:.6f}",
            f"{v_forward:.6f}",
            f"{v_right:.6f}",
            quality,
            s.tracked,
            f"{s.median_dx:.6f}",
            f"{s.median_dy:.6f}",
            f"{s.mag_median:.6f}",
            f"{s.angle_std_deg:.6f}",
            f"{s.mag_cv:.6f}",
        ])
        self._csv_file.flush()

    def _periodic_log(
        self,
        t: float,
        alt: Optional[float],
        pitch: float,
        pitch_rate: float,
        v_forward: float,
        v_right: float,
        quality: int,
        s: FlowStats,
    ) -> None:
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds < int(self.log_period_s * 1e9):
            return
        self.last_log_time = now

        corr_v_pitch = self._pearson(self.roll_vf, self.roll_pitch)
        corr_v_pitch_rate = self._pearson(self.roll_vf, self.roll_pitch_rate)

        # Drift-killing / tilt-leak proxy: high pitch-rate, low image magnitude but high forward velocity is suspicious.
        suspicious = 0
        total = 0
        if len(self.roll_mag) >= 20:
            for mag_i, v_i, pr_i in zip(self.roll_mag, self.roll_vf, self.roll_pitch_rate):
                cond = (abs(pr_i) > 8.0) and (mag_i < 0.35)
                if cond:
                    total += 1
                    if abs(v_i) > 0.20:
                        suspicious += 1
        leak_ratio = (float(suspicious) / float(total)) if total > 0 else 0.0

        q_text, _ = self._quality_label_color(quality)
        self.get_logger().info(
            "deep_test "
            f"t={t:.3f} alt={(alt if alt is not None else -1.0):.2f}m "
            f"pitch={pitch:+.2f}deg pr={pitch_rate:+.2f}deg/s "
            f"vf={v_forward:+.3f} vr={v_right:+.3f} "
            f"Q={q_text} tracked={s.tracked} medMag={s.mag_median:.3f} "
            f"angleStd={s.angle_std_deg:.2f} magCV={s.mag_cv:.3f} "
            f"corr(vf,pitch)={corr_v_pitch:+.3f} corr(vf,pitchRate)={corr_v_pitch_rate:+.3f} "
            f"tiltLeakRatio={leak_ratio:.3f}"
        )

        if s.tracked >= 10 and s.mag_median > 0.8 and s.angle_std_deg > 80.0:
            self.get_logger().warn(
                "Vector field looks swirly/unstable (high angle std). "
                "Possible reflection/noise; consider stronger blur or better floor texture."
            )

    def destroy_node(self) -> bool:
        try:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self._csv_file.close()
            if self.show_window:
                cv2.destroyAllWindows()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeepTestMonitor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
