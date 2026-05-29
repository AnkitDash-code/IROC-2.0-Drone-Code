#!/usr/bin/env python3
"""
ROS2 Humble node: D455 IPM + Lucas-Kanade optical flow -> metric velocity.

Input:
- /camera/color/image_raw (sensor_msgs/Image)
- /mavros/distance_sensor/rangefinder (sensor_msgs/Range)
- /camera/color/camera_info (sensor_msgs/CameraInfo, optional)

Output:
- /mavros/vision_speed/speed_twist (geometry_msgs/TwistStamped)
- ~/quality (std_msgs/UInt8): 0=LOW, 1=MEDIUM, 2=HIGH

Design goals:
- Floor-only IPM with fixed trapezoid ROI
- Non-blocking ROS execution with ReentrantCallbackGroup + timer processing loop
- Safety-first gating (altitude, tracked features, dt validity)
- Verbose but throttled logging for flight debugging
"""

from __future__ import annotations

import math
import threading
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
from sensor_msgs.msg import CameraInfo, Image, Range
from std_msgs.msg import UInt8


class FlowQuality:
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class D455IPMVelocityNode(Node):
    def __init__(self) -> None:
        super().__init__("d455_ipm_lk_velocity_node")

        # ---------------------------
        # Parameters
        # ---------------------------
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("range_topic", "/mavros/distance_sensor/rangefinder")
        self.declare_parameter("output_topic", "/mavros/vision_speed/speed_twist")

        self.declare_parameter("width", 848)
        self.declare_parameter("height", 480)

        # Fallback intrinsics from user-provided CameraInfo (used until CameraInfo arrives).
        self.declare_parameter("fx", 386.48)
        self.declare_parameter("fy", 386.05)
        self.declare_parameter("cx", 330.83)
        self.declare_parameter("cy", 247.26)

        self.declare_parameter("tilt_deg", 59.76)
        self.declare_parameter("min_altitude_m", 0.30)
        self.declare_parameter("min_features", 50)
        self.declare_parameter("min_tracked", 10)
        self.declare_parameter("max_dt_s", 0.25)
        self.declare_parameter("process_hz", 30.0)
        self.declare_parameter("max_speed_mps", 5.0)
        self.declare_parameter("vel_lowpass_alpha", 0.35)
        self.declare_parameter("log_period_s", 1.0)
        self.declare_parameter("ipm_blur_kernel", 7)
        self.declare_parameter("feature_quality_level", 0.02)
        self.declare_parameter("feature_min_distance", 10)
        self.declare_parameter("feature_block_size", 7)
        self.declare_parameter("lk_win_size", 21)
        self.declare_parameter("stationary_flow_px", 0.35)
        self.declare_parameter("flow_mad_k", 3.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.range_topic = str(self.get_parameter("range_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)

        self.tilt_deg = float(self.get_parameter("tilt_deg").value)
        self.min_altitude_m = float(self.get_parameter("min_altitude_m").value)
        self.min_features = int(self.get_parameter("min_features").value)
        self.min_tracked = int(self.get_parameter("min_tracked").value)
        self.max_dt_s = float(self.get_parameter("max_dt_s").value)
        self.process_hz = float(self.get_parameter("process_hz").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.vel_lowpass_alpha = float(self.get_parameter("vel_lowpass_alpha").value)
        self.log_period_s = float(self.get_parameter("log_period_s").value)
        self.ipm_blur_kernel = int(self.get_parameter("ipm_blur_kernel").value)
        self.feature_quality_level = float(self.get_parameter("feature_quality_level").value)
        self.feature_min_distance = int(self.get_parameter("feature_min_distance").value)
        self.feature_block_size = int(self.get_parameter("feature_block_size").value)
        self.lk_win_size = int(self.get_parameter("lk_win_size").value)
        self.stationary_flow_px = float(self.get_parameter("stationary_flow_px").value)
        self.flow_mad_k = float(self.get_parameter("flow_mad_k").value)

        if self.ipm_blur_kernel < 3:
            self.ipm_blur_kernel = 3
        if self.ipm_blur_kernel % 2 == 0:
            self.ipm_blur_kernel += 1
        self.lk_win_size = max(9, self.lk_win_size)
        if self.lk_win_size % 2 == 0:
            self.lk_win_size += 1

        self.tilt_rad = math.radians(self.tilt_deg)
        self.sin_tilt = max(1e-3, math.sin(self.tilt_rad))

        # ---------------------------
        # IPM setup
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
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_gray: Optional[np.ndarray] = None
        self.latest_stamp: Optional[float] = None
        self.altitude_m: Optional[float] = None

        self.prev_warped: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_stamp: Optional[float] = None

        self.fx_from_camera_info = False
        now = self.get_clock().now()
        self.last_status_log_time = now
        self.last_camera_info_warn_time = now
        self.last_zero_reason = ""
        self.last_zero_reason_log_time = now

        self.v_right_filt = 0.0
        self.v_forward_filt = 0.0

        # ---------------------------
        # ROS I/O
        # ---------------------------
        cbg = ReentrantCallbackGroup()

        self.sub_image = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            qos_profile_sensor_data,
            callback_group=cbg,
        )
        self.sub_range = self.create_subscription(
            Range,
            self.range_topic,
            self.range_cb,
            qos_profile_sensor_data,
            callback_group=cbg,
        )
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_cb,
            qos_profile_sensor_data,
            callback_group=cbg,
        )

        self.pub_twist = self.create_publisher(TwistStamped, self.output_topic, 10)
        self.pub_quality = self.create_publisher(UInt8, "~/quality", 10)

        timer_period = 1.0 / max(1.0, self.process_hz)
        self.timer = self.create_timer(timer_period, self.process_frame_cb, callback_group=cbg)

        # ---------------------------
        # OpenCV configs
        # ---------------------------
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=self.feature_quality_level,
            minDistance=self.feature_min_distance,
            blockSize=self.feature_block_size,
        )
        self.lk_params = dict(
            winSize=(self.lk_win_size, self.lk_win_size),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.get_logger().info("D455 IPM LK Velocity Node initialized")
        self.get_logger().info(
            f"Topics: image={self.image_topic}, camera_info={self.camera_info_topic}, "
            f"range={self.range_topic}, output={self.output_topic}"
        )
        self.get_logger().info(
            f"Intrinsics init: fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f}"
        )
        self.get_logger().info(
            f"Filter params: blur={self.ipm_blur_kernel} feature_q={self.feature_quality_level:.3f} "
            f"min_dist={self.feature_min_distance} lk_win={self.lk_win_size} "
            f"stationary_flow_px={self.stationary_flow_px:.3f} flow_mad_k={self.flow_mad_k:.2f}"
        )
        self.get_logger().info(f"Tilt={self.tilt_deg:.2f} deg (sin={self.sin_tilt:.5f})")
        self.get_logger().info("IPM homography matrix:")
        self.get_logger().info(np.array2string(self.h_ipm, precision=6, suppress_small=True))

    # ---------------------------
    # ROS callbacks
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
        alt = float(msg.range)
        with self.lock:
            self.altitude_m = alt

    def camera_info_cb(self, msg: CameraInfo) -> None:
        if len(msg.k) < 9:
            return

        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])

        if fx <= 0.0 or fy <= 0.0:
            return

        with self.lock:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            if not self.fx_from_camera_info:
                self.fx_from_camera_info = True
                self.get_logger().info(
                    "CameraInfo received. Using hardware intrinsics: "
                    f"fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f}"
                )

    # ---------------------------
    # Processing
    # ---------------------------
    def process_frame_cb(self) -> None:
        with self.lock:
            gray = None if self.latest_gray is None else self.latest_gray.copy()
            stamp = self.latest_stamp
            alt = self.altitude_m
            fx = self.fx
            fy = self.fy

        now = self.get_clock().now()
        if gray is None or stamp is None:
            self._publish_zero(stamp, FlowQuality.LOW, "waiting_image")
            return

        if alt is None:
            self._publish_zero(stamp, FlowQuality.LOW, "waiting_altitude")
            return

        if not self.fx_from_camera_info:
            if (now - self.last_camera_info_warn_time).nanoseconds > int(5e9):
                self.last_camera_info_warn_time = now
                self.get_logger().warn(
                    "CameraInfo not received yet; using fallback intrinsics. "
                    "Velocity scale may be biased until CameraInfo arrives."
                )

        # 1) Grayscale is already provided by image_cb.
        # 2) IPM warp using fixed floor ROI.
        warped = cv2.warpPerspective(gray, self.h_ipm, (self.width, self.height))
        # 3) Noise suppression.
        warped = cv2.GaussianBlur(warped, (self.ipm_blur_kernel, self.ipm_blur_kernel), 0)

        if self.prev_warped is None or self.prev_stamp is None:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            self._publish_zero(stamp, FlowQuality.MEDIUM, "bootstrap")
            return

        dt = float(stamp - self.prev_stamp)
        # Timer can fire faster than image callbacks; skip duplicate frame timestamps
        # without forcing low-quality zero output.
        if dt <= 1e-4:
            return

        if dt > self.max_dt_s:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            self._publish_zero(stamp, FlowQuality.LOW, f"bad_dt:{dt:.4f}")
            return

        if self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.prev_pts = self._detect_features(self.prev_warped)

        if self.prev_pts is None or len(self.prev_pts) < self.min_tracked:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            self._publish_zero(stamp, FlowQuality.LOW, "insufficient_features")
            return

        prev_pts = self.prev_pts

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
            self._publish_zero(stamp, FlowQuality.LOW, "lk_failed")
            return

        n = min(len(prev_pts), len(next_pts), len(status))
        if n <= 0:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            self._publish_zero(stamp, FlowQuality.LOW, "lk_empty")
            return

        prev_pts = prev_pts[:n]
        next_pts = next_pts[:n]
        status = status[:n]
        ok = status.ravel() == 1
        n_tracked = int(np.sum(ok))

        if alt < self.min_altitude_m:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            self._publish_zero(stamp, FlowQuality.LOW, f"low_alt:{alt:.2f}")
            self._log_status(n_tracked, dt, alt, 0.0, 0.0, 0.0, 0.0, FlowQuality.LOW)
            return

        if n_tracked < self.min_tracked:
            self.prev_warped = warped
            self.prev_pts = self._detect_features(warped)
            self.prev_stamp = stamp
            self._publish_zero(stamp, FlowQuality.LOW, f"low_tracks:{n_tracked}")
            self._log_status(n_tracked, dt, alt, 0.0, 0.0, 0.0, 0.0, FlowQuality.LOW)
            return

        prev_good = prev_pts[ok].reshape(-1, 2)
        next_good = next_pts[ok].reshape(-1, 2)
        disp = next_good - prev_good

        flow_mag = np.linalg.norm(disp, axis=1)
        median_flow = float(np.median(flow_mag)) if len(flow_mag) else 0.0

        if median_flow < self.stationary_flow_px:
            # Stationary gate: strong suppression for micro-jitter on reflective floors.
            self.v_forward_filt *= 0.6
            self.v_right_filt *= 0.6
            if abs(self.v_forward_filt) < 0.01:
                self.v_forward_filt = 0.0
            if abs(self.v_right_filt) < 0.01:
                self.v_right_filt = 0.0
            self._publish_twist(stamp, self.v_forward_filt, self.v_right_filt, FlowQuality.HIGH)

            if n_tracked < self.min_features:
                self.prev_pts = self._detect_features(warped)
            else:
                self.prev_pts = next_good.reshape(-1, 1, 2).astype(np.float32)

            self.prev_warped = warped
            self.prev_stamp = stamp
            self._log_status(
                n_tracked,
                dt,
                alt,
                0.0,
                0.0,
                self.v_forward_filt,
                self.v_right_filt,
                FlowQuality.HIGH,
            )
            return

        # Robust outlier rejection: trim vectors far from median by MAD.
        dx = disp[:, 0]
        dy = disp[:, 1]
        med_dx = float(np.median(dx))
        med_dy = float(np.median(dy))
        mad_dx = float(np.median(np.abs(dx - med_dx))) + 1e-6
        mad_dy = float(np.median(np.abs(dy - med_dy))) + 1e-6

        inlier = (np.abs(dx - med_dx) < self.flow_mad_k * mad_dx) & (np.abs(dy - med_dy) < self.flow_mad_k * mad_dy)
        if int(np.sum(inlier)) >= self.min_tracked:
            disp = disp[inlier]

        dpx = float(np.median(disp[:, 0]))
        dpy = float(np.median(disp[:, 1]))

        # Pixel drift -> metric velocity in camera image axes.
        # Negative sign: features move opposite to camera/body translation.
        v_cam_right = -(dpx * alt) / (max(1e-6, fx) * dt)
        v_cam_down = -(dpy * alt) / (max(1e-6, fy) * dt)

        # Tilt compensation for forward component.
        # With downward-pitched camera, apparent forward speed projects into image Y.
        v_body_forward = -v_cam_down / self.sin_tilt
        v_body_right = v_cam_right

        # Clamp and smooth.
        v_body_forward = float(np.clip(v_body_forward, -self.max_speed_mps, self.max_speed_mps))
        v_body_right = float(np.clip(v_body_right, -self.max_speed_mps, self.max_speed_mps))

        a = self.vel_lowpass_alpha
        self.v_forward_filt = a * v_body_forward + (1.0 - a) * self.v_forward_filt
        self.v_right_filt = a * v_body_right + (1.0 - a) * self.v_right_filt

        self._publish_twist(stamp, self.v_forward_filt, self.v_right_filt, FlowQuality.HIGH)

        # Re-seed if points start to thin out.
        if n_tracked < self.min_features:
            self.prev_pts = self._detect_features(warped)
        else:
            self.prev_pts = next_good.reshape(-1, 1, 2).astype(np.float32)

        self.prev_warped = warped
        self.prev_stamp = stamp

        self._log_status(
            n_tracked,
            dt,
            alt,
            dpx,
            dpy,
            self.v_forward_filt,
            self.v_right_filt,
            FlowQuality.HIGH,
        )

    # ---------------------------
    # Helpers
    # ---------------------------
    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        if pts is None:
            return None
        return pts.astype(np.float32)

    def _publish_zero(self, stamp: Optional[float], quality: int, reason: str) -> None:
        self.v_forward_filt = 0.0
        self.v_right_filt = 0.0
        self._publish_twist(stamp, 0.0, 0.0, quality)
        now = self.get_clock().now()
        should_log = (
            reason != self.last_zero_reason
            or (now - self.last_zero_reason_log_time).nanoseconds > int(1e9)
        )
        if should_log:
            self.last_zero_reason = reason
            self.last_zero_reason_log_time = now
            q_str = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}.get(quality, "UNKNOWN")
            self.get_logger().info(f"safety_zero reason={reason} quality={q_str}")

    def _publish_twist(self, stamp: Optional[float], v_forward: float, v_right: float, quality: int) -> None:
        msg = TwistStamped()
        if stamp is not None:
            sec = int(stamp)
            nsec = int((stamp - sec) * 1e9)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
        else:
            msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = "base_link"

        # Body-frame convention used here:
        # x = forward, y = right, z = 0 for planar optical flow velocity.
        msg.twist.linear.x = float(v_forward)
        msg.twist.linear.y = float(v_right)
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.pub_twist.publish(msg)

        q = UInt8()
        q.data = int(quality)
        self.pub_quality.publish(q)

    def _log_status(
        self,
        n_tracked: int,
        dt: float,
        alt: float,
        dpx: float,
        dpy: float,
        v_forward: float,
        v_right: float,
        quality: int,
    ) -> None:
        now = self.get_clock().now()
        if (now - self.last_status_log_time).nanoseconds < int(self.log_period_s * 1e9):
            return

        self.last_status_log_time = now
        q_str = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}.get(quality, "UNKNOWN")
        self.get_logger().info(
            "flow_status "
            f"tracks={n_tracked} dt={dt:.4f}s alt={alt:.3f}m "
            f"dpx={dpx:+.3f} dpy={dpy:+.3f} "
            f"v_forward={v_forward:+.3f}m/s v_right={v_right:+.3f}m/s "
            f"quality={q_str}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = D455IPMVelocityNode()
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
