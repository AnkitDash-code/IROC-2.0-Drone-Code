#!/usr/bin/env python3
"""
Manual live Phase1+Phase2 tester (no pre-existing protocol folder required).

This script records live velocity/IMU data while you manually perform:
  - phase1_static (hold still)
  - phase2_translate (straight translation)

It saves phase CSV files in a new manual run directory and computes the same
error/test report used by phase12_test_suite.py.
"""

import argparse
import csv
import json
import select
import threading
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

try:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    ROS_IMPORT_ERROR = ""
except Exception as exc:
    rclpy = None
    TwistStamped = Any
    Imu = Any
    SingleThreadedExecutor = None
    Node = object
    qos_profile_sensor_data = None
    ROS_IMPORT_ERROR = str(exc)

from phase12_test_suite import evaluate_protocol, print_console_report, run_gui, write_reports


CSV_COLUMNS = [
    "t",
    "alt",
    "pitch_deg",
    "pitch_rate_dps",
    "v_forward",
    "v_right",
    "quality",
    "tracked",
    "med_dx",
    "med_dy",
    "med_mag",
    "angle_std_deg",
    "mag_cv",
]


class ManualLiveCollector(Node):
    def __init__(self, vel_topic: str, imu_topic: str, sample_hz: float, dashboard_url: str):
        super().__init__("phase12_manual_live_collector")

        self.lock = threading.Lock()
        self.current_phase = None
        self.phase_rows = {
            "phase1_static": [],
            "phase2_translate": [],
        }

        self.latest_vf = 0.0
        self.latest_vr = 0.0
        self.latest_twist_wall_ts = 0.0

        self.latest_pitch_rate_dps = 0.0
        self.latest_pitch_wall_ts = 0.0

        self.latest_vo_debug = {}
        self.dashboard_url = dashboard_url.strip()

        self.sub_twist = self.create_subscription(
            TwistStamped,
            vel_topic,
            self._twist_cb,
            qos_profile_sensor_data,
        )
        self.sub_imu = self.create_subscription(
            Imu,
            imu_topic,
            self._imu_cb,
            qos_profile_sensor_data,
        )

        self.sample_period = max(0.01, 1.0 / max(1.0, sample_hz))
        self.timer_sample = self.create_timer(self.sample_period, self._sample_cb)
        self.timer_dash = self.create_timer(0.2, self._dashboard_poll_cb)

    def _twist_cb(self, msg: TwistStamped) -> None:
        with self.lock:
            self.latest_vf = float(msg.twist.linear.x)
            self.latest_vr = float(msg.twist.linear.y)
            self.latest_twist_wall_ts = time.time()

    def _imu_cb(self, msg: Imu) -> None:
        with self.lock:
            # Keep same convention as prior logs: y-axis angular velocity as pitch-rate proxy.
            self.latest_pitch_rate_dps = float(msg.angular_velocity.y) * 57.29577951308232
            self.latest_pitch_wall_ts = time.time()

    def _dashboard_poll_cb(self) -> None:
        if not self.dashboard_url:
            return

        try:
            with urlopen(self.dashboard_url, timeout=0.2) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            vo_debug = payload.get("vo_debug", {}) if isinstance(payload, dict) else {}
            if isinstance(vo_debug, dict):
                with self.lock:
                    self.latest_vo_debug = vo_debug
        except (URLError, TimeoutError, ValueError):
            pass
        except Exception:
            pass

    def _estimate_quality(self, now_wall: float, vo_debug: dict, latest_twist_wall_ts: float) -> int:
        if vo_debug:
            success = bool(vo_debug.get("success", False))
            tracked = int(vo_debug.get("tracked", 0))
            n_samples = int(vo_debug.get("n_samples", 0))
            med_flow = float(vo_debug.get("median_flow_px", 0.0))
            if success and tracked >= 90 and n_samples >= 35 and med_flow >= 0.22:
                return 2
            if success and tracked >= 50 and n_samples >= 16:
                return 1

        # Fallback when dashboard is unavailable: use message freshness.
        twist_age = now_wall - latest_twist_wall_ts if latest_twist_wall_ts > 0.0 else 1e9
        if twist_age < 0.25:
            return 2
        if twist_age < 0.60:
            return 1
        return 0

    def _sample_cb(self) -> None:
        with self.lock:
            phase = self.current_phase
            if phase is None:
                return

            now_wall = time.time()
            vo = dict(self.latest_vo_debug) if isinstance(self.latest_vo_debug, dict) else {}
            quality = self._estimate_quality(now_wall, vo, self.latest_twist_wall_ts)
            row = {
                "t": f"{now_wall:.6f}",
                "alt": f"{float(vo.get('alt', -1.0)):.6f}",
                "pitch_deg": "0.000000",
                "pitch_rate_dps": f"{self.latest_pitch_rate_dps:.6f}",
                "v_forward": f"{self.latest_vf:.6f}",
                "v_right": f"{self.latest_vr:.6f}",
                "quality": int(quality),
                "tracked": int(vo.get("tracked", 0)),
                "med_dx": "0.000000",
                "med_dy": "0.000000",
                "med_mag": f"{float(vo.get('median_flow_px', 0.0)):.6f}",
                "angle_std_deg": "0.000000",
                "mag_cv": "0.000000",
            }
            self.phase_rows[phase].append(row)

    def start_phase(self, phase: str) -> None:
        with self.lock:
            self.current_phase = phase

    def stop_phase(self) -> None:
        with self.lock:
            self.current_phase = None

    def samples_for(self, phase: str) -> int:
        with self.lock:
            return len(self.phase_rows.get(phase, []))

    def status_line(self) -> str:
        with self.lock:
            phase = self.current_phase if self.current_phase is not None else "idle"
            tracked = int(self.latest_vo_debug.get("tracked", 0)) if isinstance(self.latest_vo_debug, dict) else 0
            med_flow = float(self.latest_vo_debug.get("median_flow_px", 0.0)) if isinstance(self.latest_vo_debug, dict) else 0.0
            return (
                f"phase={phase} vf={self.latest_vf:+.3f} vr={self.latest_vr:+.3f} m/s "
                f"pitch_rate={self.latest_pitch_rate_dps:+.2f} dps tracked={tracked} med_flow={med_flow:.3f}"
            )



def _capture_phase(collector: ManualLiveCollector, phase: str, label: str) -> None:
    input(f"\nPress Enter to START {phase} ({label})...")
    collector.start_phase(phase)
    print(f"Recording {phase}. Perform action now. Press Enter to STOP.")

    last_print = 0.0
    while True:
        now = time.time()
        if now - last_print > 1.0:
            print(f"  {collector.status_line()} | samples={collector.samples_for(phase)}")
            last_print = now

        # Keep input handling on main thread so Enter-to-stop works reliably.
        ready, _, _ = select.select([sys.stdin], [], [], 0.10)
        if ready:
            sys.stdin.readline()
            break

    collector.stop_phase()
    print(f"Stopped {phase}. Captured samples={collector.samples_for(phase)}")


def _write_phase_csv(out_dir: Path, phase: str, rows: list) -> Path:
    out_path = out_dir / f"{phase}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual live tester for Phase1+Phase2")
    parser.add_argument("--vel-topic", type=str, default="/mavros/vision_speed/speed_twist", help="Velocity topic from Phase 2")
    parser.add_argument("--imu-topic", type=str, default="/camera/camera/imu", help="IMU topic for pitch-rate logging")
    parser.add_argument("--sample-hz", type=float, default=20.0, help="Sampling frequency for CSV capture")
    parser.add_argument("--dashboard-url", type=str, default="http://127.0.0.1:5000/api/state", help="Phase2 dashboard state endpoint (optional)")
    parser.add_argument("--output-root", type=str, default=".", help="Root folder where manual run directory is created")
    parser.add_argument("--gui", action="store_true", help="Open report GUI after capture")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="GUI host")
    parser.add_argument("--port", type=int, default=5060, help="GUI port")
    args = parser.parse_args()

    if rclpy is None:
        raise RuntimeError(
            "ROS 2 Python modules are not available in this shell. "
            "Source ROS first (for example: 'source /opt/ros/humble/setup.bash') "
            f"and use Python 3.10 environment. Import error: {ROS_IMPORT_ERROR}"
        )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"manual_phase12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nManual Phase1+Phase2 Live Test")
    print("- Keep Phase 2 stack running so velocity topic is active.")
    print("- Phase1: hold drone/device still.")
    print("- Phase2: move in straight translation with minimal tilt.")
    print(f"- Output run dir: {run_dir}")

    rclpy.init()
    collector = ManualLiveCollector(
        vel_topic=args.vel_topic,
        imu_topic=args.imu_topic,
        sample_hz=args.sample_hz,
        dashboard_url=args.dashboard_url,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(collector)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        print("\nWaiting 2 seconds for first messages...")
        time.sleep(2.0)

        _capture_phase(collector, "phase1_static", "hold still")
        _capture_phase(collector, "phase2_translate", "translate straight")

        p1_csv = _write_phase_csv(run_dir, "phase1_static", collector.phase_rows["phase1_static"])
        p2_csv = _write_phase_csv(run_dir, "phase2_translate", collector.phase_rows["phase2_translate"])

        print(f"Saved: {p1_csv}")
        print(f"Saved: {p2_csv}")

        result = evaluate_protocol(run_dir)
        txt_path, json_path = write_reports(result, out_prefix="phase12_report")
        print_console_report(result)
        print(f"Saved report text: {txt_path}")
        print(f"Saved report json: {json_path}")

        if args.gui:
            print("Opening report GUI. Press Ctrl+C to exit GUI.")
            run_gui(base_dir=output_root, default_protocol=run_dir, host=args.host, port=args.port)

    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            collector.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
