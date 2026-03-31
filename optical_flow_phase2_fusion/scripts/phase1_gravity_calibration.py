#!/usr/bin/env python3
"""
Phase 1: Hardware Truth (Gravity Calibration)

Purpose:
- Place drone on a flat surface in hover-like posture.
- Subscribe to IMU topic.
- Estimate stable ground-truth tilt theta (degrees) from gravity vector.
- Save detailed logs (CSV) + summary (JSON) for reproducibility.

Usage:
  source /opt/ros/humble/setup.bash
  python3 phase1_gravity_calibration.py --topic /camera/camera/imu

Alternative topic examples:
  --topic /mavros/imu/data
  --topic /imu/data
"""

import argparse
import csv
import json
import math
import os
import signal
import statistics
import sys
import time
from collections import deque
from datetime import datetime


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class GravityCalibrationNode:
    def __init__(
        self,
        topic,
        log_dir,
        window_size,
        settle_std_deg,
        min_settle_seconds,
        print_period_sec,
        ema_alpha,
        auto_stop,
        rclpy_mod,
        qos_profile,
        imu_msg_type,
        no_data_warn_seconds,
        no_data_timeout_seconds,
    ):
        self.rclpy = rclpy_mod
        self.node = self.rclpy.create_node("phase1_gravity_calibration")
        self.topic = topic
        self.window_size = int(window_size)
        self.settle_std_deg = float(settle_std_deg)
        self.min_settle_seconds = float(min_settle_seconds)
        self.print_period_sec = float(print_period_sec)
        self.ema_alpha = float(ema_alpha)
        self.auto_stop = bool(auto_stop)
        self.no_data_warn_seconds = float(no_data_warn_seconds)
        self.no_data_timeout_seconds = float(no_data_timeout_seconds)

        os.makedirs(log_dir, exist_ok=True)
        self.started_wall = time.time()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"phase1_gravity_{stamp}.csv")
        self.summary_path = os.path.join(log_dir, f"phase1_gravity_{stamp}_summary.json")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "wall_time_iso",
                "ros_time_sec",
                "ax",
                "ay",
                "az",
                "g_norm",
                "roll_deg",
                "pitch_deg",
                "theta_deg_raw",
                "theta_deg_ema",
                "window_mean_deg",
                "window_std_deg",
                "is_settled",
            ]
        )

        self.theta_window = deque(maxlen=self.window_size)
        self.theta_ema = None
        self.last_print = 0.0
        self.last_ros_time = 0.0
        self.sample_count = 0
        self.settled = False
        self.final_theta_deg = None
        self.last_no_data_notice = 0.0
        self.stop_requested = False

        self.sub = self.node.create_subscription(
            imu_msg_type, self.topic, self.imu_callback, qos_profile
        )
        self.watchdog_timer = self.node.create_timer(1.0, self.watchdog_callback)

        self.node.get_logger().info("=" * 68)
        self.node.get_logger().info("PHASE 1 GRAVITY CALIBRATION STARTED")
        self.node.get_logger().info(f"IMU topic        : {self.topic}")
        self.node.get_logger().info(f"CSV log          : {self.csv_path}")
        self.node.get_logger().info(f"Summary JSON     : {self.summary_path}")
        self.node.get_logger().info(
            f"Settle criterion : std(theta) <= {self.settle_std_deg:.3f} deg for >= {self.min_settle_seconds:.1f} s"
        )
        self.node.get_logger().info("Keep drone still on flat surface until settled.")
        self.node.get_logger().info("=" * 68)

    def imu_callback(self, msg):
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        ros_t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.last_ros_time = ros_t

        g_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if g_norm < 1e-6:
            return

        # Roll/Pitch from accelerometer gravity projection.
        roll_rad = math.atan2(ay, az)
        pitch_rad = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        roll_deg = math.degrees(roll_rad)
        pitch_deg = math.degrees(pitch_rad)

        # Ground truth tilt theta from vertical magnitude.
        # Using |az| makes it robust to IMU frame sign conventions.
        theta_rad = math.atan2(math.sqrt(ax * ax + ay * ay), abs(az))
        theta_deg_raw = math.degrees(theta_rad)

        if self.theta_ema is None:
            self.theta_ema = theta_deg_raw
        else:
            self.theta_ema = (
                self.ema_alpha * theta_deg_raw + (1.0 - self.ema_alpha) * self.theta_ema
            )

        self.theta_window.append(self.theta_ema)
        self.sample_count += 1

        if len(self.theta_window) >= 2:
            win_mean = statistics.fmean(self.theta_window)
            win_std = statistics.pstdev(self.theta_window)
        else:
            win_mean = self.theta_ema
            win_std = 999.0

        elapsed = time.time() - self.started_wall
        settled_now = (
            len(self.theta_window) == self.window_size
            and win_std <= self.settle_std_deg
            and elapsed >= self.min_settle_seconds
        )

        wall_iso = datetime.now().isoformat(timespec="milliseconds")
        self.csv_writer.writerow(
            [
                wall_iso,
                f"{ros_t:.9f}",
                f"{ax:.6f}",
                f"{ay:.6f}",
                f"{az:.6f}",
                f"{g_norm:.6f}",
                f"{roll_deg:.6f}",
                f"{pitch_deg:.6f}",
                f"{theta_deg_raw:.6f}",
                f"{self.theta_ema:.6f}",
                f"{win_mean:.6f}",
                f"{win_std:.6f}",
                int(settled_now),
            ]
        )

        now = time.time()
        if (now - self.last_print) >= self.print_period_sec:
            self.last_print = now
            self.node.get_logger().info(
                (
                    f"samples={self.sample_count:5d} | "
                    f"theta_ema={self.theta_ema:7.3f} deg | "
                    f"window_mean={win_mean:7.3f} deg | "
                    f"window_std={win_std:6.3f} deg | "
                    f"g={g_norm:5.2f} m/s^2"
                )
            )

        if settled_now and not self.settled:
            self.settled = True
            self.final_theta_deg = win_mean
            self.write_summary(win_mean, win_std, elapsed)
            self.node.get_logger().info("=" * 68)
            self.node.get_logger().info(
                f"SETTLED: Ground Truth Tilt (theta) = {win_mean:.3f} deg"
            )
            self.node.get_logger().info(f"CSV log saved     : {self.csv_path}")
            self.node.get_logger().info(f"Summary saved     : {self.summary_path}")
            self.node.get_logger().info("=" * 68)
            if self.auto_stop:
                self.stop_requested = True

    def watchdog_callback(self):
        if self.sample_count > 0:
            return

        elapsed = time.time() - self.started_wall

        if elapsed >= self.no_data_warn_seconds and (time.time() - self.last_no_data_notice) >= 2.0:
            self.last_no_data_notice = time.time()
            self.node.get_logger().warn(
                (
                    f"No IMU messages received on '{self.topic}' yet (elapsed {elapsed:.1f}s). "
                    "Start your IMU publisher (RealSense/MAVROS) or use another topic."
                )
            )

        if self.no_data_timeout_seconds > 0.0 and elapsed >= self.no_data_timeout_seconds:
            self.node.get_logger().error(
                (
                    f"No IMU data received for {elapsed:.1f}s on '{self.topic}'. "
                    "Stopping calibration."
                )
            )
            self.stop_requested = True

    def write_summary(self, theta_mean_deg, theta_std_deg, elapsed_sec):
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "topic": self.topic,
            "samples": self.sample_count,
            "theta_deg": round(float(theta_mean_deg), 6),
            "theta_std_deg": round(float(theta_std_deg), 6),
            "elapsed_sec": round(float(elapsed_sec), 3),
            "window_size": self.window_size,
            "settle_std_deg": self.settle_std_deg,
            "min_settle_seconds": self.min_settle_seconds,
            "csv_log": self.csv_path,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 gravity calibration logger")
    parser.add_argument("--topic", default="/camera/camera/imu", help="IMU topic")
    parser.add_argument(
        "--log-dir",
        default="log/phase1_gravity",
        help="Directory for CSV + summary logs",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=120,
        help="Samples in settle window",
    )
    parser.add_argument(
        "--settle-std-deg",
        type=float,
        default=0.15,
        help="Settle threshold for std(theta) in degrees",
    )
    parser.add_argument(
        "--min-settle-seconds",
        type=float,
        default=6.0,
        help="Minimum run time before settle accepted",
    )
    parser.add_argument(
        "--print-period-sec",
        type=float,
        default=1.0,
        help="Console print period",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.15,
        help="EMA smoothing factor for theta",
    )
    parser.add_argument(
        "--no-auto-stop",
        action="store_true",
        help="Keep running after settled value is detected",
    )
    parser.add_argument(
        "--no-data-warn-seconds",
        type=float,
        default=3.0,
        help="Seconds before warning that no IMU data is arriving",
    )
    parser.add_argument(
        "--no-data-timeout-seconds",
        type=float,
        default=20.0,
        help="Seconds to wait for first IMU sample before exiting (0 disables timeout)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    auto_stop = not args.no_auto_stop

    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu
    except Exception as exc:
        print("Failed to import ROS 2 Python modules (rclpy/sensor_msgs).")
        print("Use ROS Humble environment with system Python, e.g.")
        print("  source /opt/ros/humble/setup.bash")
        print("  /usr/bin/python3 phase1_gravity_calibration.py --topic /camera/camera/imu")
        print(f"Import error: {exc}")
        sys.exit(1)

    rclpy.init()
    cal = GravityCalibrationNode(
        topic=args.topic,
        log_dir=args.log_dir,
        window_size=args.window_size,
        settle_std_deg=args.settle_std_deg,
        min_settle_seconds=args.min_settle_seconds,
        print_period_sec=args.print_period_sec,
        ema_alpha=args.ema_alpha,
        auto_stop=auto_stop,
        rclpy_mod=rclpy,
        qos_profile=qos_profile_sensor_data,
        imu_msg_type=Imu,
        no_data_warn_seconds=args.no_data_warn_seconds,
        no_data_timeout_seconds=args.no_data_timeout_seconds,
    )

    def _sigint_handler(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while rclpy.ok() and not cal.stop_requested:
            rclpy.spin_once(cal.node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if not cal.settled:
            # Save partial summary for traceability even on manual stop.
            elapsed = time.time() - cal.started_wall
            if cal.theta_ema is not None:
                win_std = statistics.pstdev(cal.theta_window) if len(cal.theta_window) >= 2 else 999.0
                cal.write_summary(cal.theta_ema, win_std, elapsed)
                cal.node.get_logger().info(
                    f"Stopped early. Latest theta estimate = {cal.theta_ema:.3f} deg"
                )
                cal.node.get_logger().info(f"CSV log saved     : {cal.csv_path}")
                cal.node.get_logger().info(f"Summary saved     : {cal.summary_path}")
        cal.close()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
