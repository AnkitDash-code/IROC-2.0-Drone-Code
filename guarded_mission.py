#!/usr/bin/env python3
"""
Guarded single-flight workflow:
- arm
- take off to operator-selected altitude (<= 3.0 m)
- hover for 5 minutes (default)
- land
- disarm

This script requires manual terminal confirmations before each mission phase
and keeps checking altitude safety in all loops.
"""

import argparse
import json
import math
import select
import sys
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import drone_control as dc
from pymavlink import mavutil


DEFAULT_CONNECTION = "/dev/ttyACM0"
DEFAULT_BAUD = 57600
DEFAULT_TARGET_ALT = 1.0
MAX_TARGET_ALT = 3.0
CRITICAL_ALT = 4.0
ON_GROUND_TOLERANCE = 0.20
DEFAULT_HOVER_SECONDS = 300
RC_INPUT_DEVIATION = 120
DEFAULT_NO_PROPELLER_MODE = False
DEFAULT_SKIP_LANDING_VERIFY = False
RANGEFINDER_INTERVAL_US = 100000  # 10 Hz
BATTERY_INTERVAL_US = 200000  # 5 Hz
SLOW_TAKEOFF_SPEED_UP = 25.0
SLOW_TAKEOFF_ACCEL_Z = 20.0
SLOW_LANDING_SPEED = 25.0
TAKEOFF_PREBRAKE_MARGIN_M = 0.10
TAKEOFF_REACHED_MARGIN_M = 0.05
AUTO_LAND_MARGIN_ABOVE_TARGET_M = 1.00
DEFAULT_LOW_VOLTAGE_V = 13.2
DEFAULT_LOW_VOLTAGE_CONFIRM_COUNT = 3
DEFAULT_ENABLE_BASE_LANDING = True
DEFAULT_BASE_HOLD_ALT_M = 1.5
DEFAULT_BASE_ALIGN_TIMEOUT_S = 20.0
DEFAULT_BASE_CENTER_HOLD_S = 2.0
DEFAULT_BASE_ALIGN_SPEED_MPS = 0.12
DEFAULT_BASE_METRIC_DEADBAND_M = 0.10
DEFAULT_BASE_PIXEL_DEADBAND_PX = 14.0
DEFAULT_BASE_DESCENT_SPEED_MPS = 0.25
DEFAULT_BASE_DESCENT_TIMEOUT_S = 60.0
DEFAULT_BASE_VISUAL_STATE_URL = "http://localhost:5000/api/state"
DEFAULT_BASE_FORWARD_SIGN = 1.0
DEFAULT_BASE_RIGHT_SIGN = 1.0
DEFAULT_BASE_MATCH_SCORE_MIN = 12.0
DEFAULT_BASE_YAW_CORRECTION = True
DEFAULT_BASE_YAW_DEADBAND_DEG = 3.0
DEFAULT_BASE_YAW_STEP_DEG = 2.0
DEFAULT_BASE_YAW_RATE_DPS = 15.0
DEFAULT_BASE_YAW_CMD_INTERVAL_S = 0.5
DEFAULT_BASE_KP_METRIC = 1.0
DEFAULT_BASE_KP_PIXEL = 0.0035
DEFAULT_BASE_VEL_SLEW_MPS2 = 0.60


class ManualOverride(Exception):
    """Raised when pilot RC input is detected and autonomy must stop."""


class GuardedMission:
    def __init__(
        self,
        connection: str,
        baud: int,
        hover_seconds: int,
        no_propeller_mode: bool,
        skip_landing_verify: bool = False,
        low_voltage_v: float = DEFAULT_LOW_VOLTAGE_V,
        low_voltage_confirm_count: int = DEFAULT_LOW_VOLTAGE_CONFIRM_COUNT,
        enable_base_landing: bool = DEFAULT_ENABLE_BASE_LANDING,
        base_hold_alt_m: float = DEFAULT_BASE_HOLD_ALT_M,
        base_align_timeout_s: float = DEFAULT_BASE_ALIGN_TIMEOUT_S,
        base_center_hold_s: float = DEFAULT_BASE_CENTER_HOLD_S,
        base_align_speed_mps: float = DEFAULT_BASE_ALIGN_SPEED_MPS,
        base_metric_deadband_m: float = DEFAULT_BASE_METRIC_DEADBAND_M,
        base_pixel_deadband_px: float = DEFAULT_BASE_PIXEL_DEADBAND_PX,
        base_descent_speed_mps: float = DEFAULT_BASE_DESCENT_SPEED_MPS,
        base_descent_timeout_s: float = DEFAULT_BASE_DESCENT_TIMEOUT_S,
        base_visual_state_url: str = DEFAULT_BASE_VISUAL_STATE_URL,
        base_forward_sign: float = DEFAULT_BASE_FORWARD_SIGN,
        base_right_sign: float = DEFAULT_BASE_RIGHT_SIGN,
        base_match_score_min: float = DEFAULT_BASE_MATCH_SCORE_MIN,
        base_yaw_correction: bool = DEFAULT_BASE_YAW_CORRECTION,
        base_yaw_deadband_deg: float = DEFAULT_BASE_YAW_DEADBAND_DEG,
        base_yaw_step_deg: float = DEFAULT_BASE_YAW_STEP_DEG,
        base_yaw_rate_dps: float = DEFAULT_BASE_YAW_RATE_DPS,
        base_yaw_cmd_interval_s: float = DEFAULT_BASE_YAW_CMD_INTERVAL_S,
        base_kp_metric: float = DEFAULT_BASE_KP_METRIC,
        base_kp_pixel: float = DEFAULT_BASE_KP_PIXEL,
        base_vel_slew_mps2: float = DEFAULT_BASE_VEL_SLEW_MPS2,
    ) -> None:
        self.connection = connection
        self.baud = baud
        self.hover_seconds = hover_seconds
        self.no_propeller_mode = no_propeller_mode
        self.skip_landing_verify = skip_landing_verify
        self.master = None
        self.last_print_time = 0.0
        self.landing_forced = False
        self._saved_params = {}
        self.last_alt_msg_type = None
        self.initial_ground_altitude = None
        self.target_altitude = None
        self.low_voltage_v = float(low_voltage_v)
        self.low_voltage_confirm_count = max(1, int(low_voltage_confirm_count))
        self.last_battery_voltage_v = None
        self.last_battery_source = None
        self.last_battery_read_time = 0.0
        self._battery_low_hits = 0
        self._last_battery_log_time = 0.0
        self.enable_base_landing = bool(enable_base_landing)
        self.base_hold_alt_m = float(base_hold_alt_m)
        self.base_align_timeout_s = max(5.0, float(base_align_timeout_s))
        self.base_center_hold_s = max(0.2, float(base_center_hold_s))
        self.base_align_speed_mps = max(0.02, float(base_align_speed_mps))
        self.base_metric_deadband_m = max(0.02, float(base_metric_deadband_m))
        self.base_pixel_deadband_px = max(2.0, float(base_pixel_deadband_px))
        self.base_descent_speed_mps = max(0.05, float(base_descent_speed_mps))
        self.base_descent_timeout_s = max(5.0, float(base_descent_timeout_s))
        self.base_visual_state_url = str(base_visual_state_url).strip()
        self.base_forward_sign = -1.0 if float(base_forward_sign) < 0.0 else 1.0
        self.base_right_sign = -1.0 if float(base_right_sign) < 0.0 else 1.0
        self.base_match_score_min = max(0.0, float(base_match_score_min))
        self.base_yaw_correction = bool(base_yaw_correction)
        self.base_yaw_deadband_deg = max(0.5, float(base_yaw_deadband_deg))
        self.base_yaw_step_deg = max(0.2, float(base_yaw_step_deg))
        self.base_yaw_rate_dps = max(1.0, float(base_yaw_rate_dps))
        self.base_yaw_cmd_interval_s = max(0.1, float(base_yaw_cmd_interval_s))
        self.base_kp_metric = max(0.0, float(base_kp_metric))
        self.base_kp_pixel = max(0.0, float(base_kp_pixel))
        self.base_vel_slew_mps2 = max(0.05, float(base_vel_slew_mps2))
        self._last_align_vx = 0.0
        self._last_align_vy = 0.0
        self._last_align_ts = 0.0
        self._last_yaw_nudge_ts = 0.0
        self.landing_reason = None

    def _sync_master(self) -> None:
        self.master = dc.master

    def connect(self) -> None:
        print(f"[INFO] Connecting to vehicle: {self.connection} @ {self.baud}")
        dc.CONNECTION_STRING = self.connection
        dc.BAUD_RATE = self.baud
        dc.connect_to_vehicle()
        self._sync_master()

    def confirm_step(self, prompt: str, token: str) -> None:
        prompt_text = f"{prompt} Type '{token}' to continue: "
        while True:
            self.assert_no_manual_override(f"confirmation:{token}")
            print(prompt_text, end="", flush=True)

            # Poll stdin so RC/manual override can still be checked while waiting for input.
            while True:
                self.assert_no_manual_override(f"confirmation:{token}")
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    value = sys.stdin.readline().strip()
                    break

            if value == token:
                return
            print("[WARN] Confirmation failed. Try again.")

    def get_altitude_m(self, timeout: float = 1.0) -> Optional[float]:
        """
        Rangefinder-only altitude source.
        """
        if self.master is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            msg = self.master.recv_match(
                type=["RANGEFINDER", "DISTANCE_SENSOR", "NAMED_VALUE_FLOAT"],
                blocking=True,
                timeout=remaining,
            )
            if msg is None:
                return None

            msg_type = msg.get_type()

            if msg_type == "RANGEFINDER":
                distance = getattr(msg, "distance", None)
                if distance is None:
                    continue
                self.last_alt_msg_type = "RANGEFINDER"
                return float(distance)

            if msg_type == "DISTANCE_SENSOR":
                current_cm = getattr(msg, "current_distance", None)
                if current_cm is None:
                    continue
                self.last_alt_msg_type = "DISTANCE_SENSOR"
                return float(current_cm) / 100.0

            if msg_type == "NAMED_VALUE_FLOAT":
                raw_name = getattr(msg, "name", "")
                name = str(raw_name).strip().lower()
                # Mission Planner commonly shows this as rangefinder1.
                if name in {"rangefinder1", "rangefinder", "rngfnd1", "rngfnd"}:
                    value = getattr(msg, "value", None)
                    if value is None:
                        continue
                    self.last_alt_msg_type = f"NAMED_VALUE_FLOAT:{name}"
                    value_f = float(value)
                    # Guard against cm-style values accidentally forwarded as whole numbers.
                    if value_f > 10.0:
                        return value_f / 100.0
                    return value_f

        return None

    def request_rangefinder_stream(self) -> None:
        if self.master is None:
            return
        for msg_id in (
            mavutil.mavlink.MAVLINK_MSG_ID_RANGEFINDER,
            mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,
        ):
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                RANGEFINDER_INTERVAL_US,
                0,
                0,
                0,
                0,
                0,
            )

        # Legacy stream request path for autopilots that ignore MESSAGE_INTERVAL.
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
            10,
            1,
        )

    def request_battery_stream(self) -> None:
        if self.master is None:
            return

        for msg_id in (
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,
            mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS,
        ):
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                BATTERY_INTERVAL_US,
                0,
                0,
                0,
                0,
                0,
            )

        # Fallback for stacks that still rely on legacy stream groups.
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
            5,
            1,
        )

    def _extract_pack_voltage_v(self, msg) -> Optional[float]:
        msg_type = msg.get_type()

        if msg_type == "SYS_STATUS":
            mv = getattr(msg, "voltage_battery", None)
            if mv is None:
                return None
            mv_i = int(mv)
            if mv_i <= 0:
                return None
            self.last_battery_source = "SYS_STATUS"
            return mv_i / 1000.0

        if msg_type == "BATTERY_STATUS":
            total_mv = 0
            valid_cells = 0
            cells = getattr(msg, "voltages", None)
            if cells is not None:
                for mv in cells:
                    mv_i = int(mv)
                    if 0 < mv_i < 6000:
                        total_mv += mv_i
                        valid_cells += 1
            if valid_cells > 0 and total_mv > 0:
                self.last_battery_source = f"BATTERY_STATUS:{valid_cells}cell"
                return total_mv / 1000.0

            # Fallback when only consumed_energy/current are sent and cell voltages are invalid.
            mv = getattr(msg, "voltage", None)
            if mv is not None:
                mv_i = int(mv)
                if mv_i > 0:
                    self.last_battery_source = "BATTERY_STATUS:voltage"
                    return mv_i / 1000.0
        return None

    def get_battery_voltage_v(self, timeout: float = 0.0) -> Optional[float]:
        if self.master is None:
            return self.last_battery_voltage_v

        deadline = time.time() + max(0.0, timeout)
        while True:
            remaining = max(0.0, deadline - time.time())
            blocking = timeout > 0.0 and remaining > 0.0
            msg = self.master.recv_match(
                type=["SYS_STATUS", "BATTERY_STATUS"],
                blocking=blocking,
                timeout=remaining if blocking else 0,
            )
            if msg is None:
                break

            voltage_v = self._extract_pack_voltage_v(msg)
            if voltage_v is not None:
                self.last_battery_voltage_v = float(voltage_v)
                self.last_battery_read_time = time.time()
                return self.last_battery_voltage_v

            if not blocking:
                break

        return self.last_battery_voltage_v

    def enforce_battery_failsafe(self, context: str) -> bool:
        if self.landing_forced and self.landing_reason == "low_battery":
            return True

        voltage_v = self.get_battery_voltage_v(timeout=0.05)
        now = time.time()

        if voltage_v is None:
            if now - self._last_battery_log_time >= 3.0:
                print(f"[WARN] {context}: battery voltage unavailable from MAVLink telemetry.")
                self._last_battery_log_time = now
            return False

        if now - self._last_battery_log_time >= 1.5:
            src = self.last_battery_source or "UNKNOWN"
            print(f"[BATT] {context}: {voltage_v:.2f} V via {src}")
            self._last_battery_log_time = now

        if voltage_v <= self.low_voltage_v:
            self._battery_low_hits += 1
            if self._battery_low_hits >= self.low_voltage_confirm_count:
                print(
                    f"[CRITICAL] Battery low: {voltage_v:.2f} V <= {self.low_voltage_v:.2f} V "
                    f"({self._battery_low_hits}/{self.low_voltage_confirm_count})."
                )
                if self.motors_armed() and not self.landing_forced:
                    self.landing_forced = True
                    self.landing_reason = "low_battery"
                    print("[CRITICAL] Battery failsafe triggered. Starting landing sequence immediately.")
                return True
        else:
            if self._battery_low_hits > 0:
                print(f"[INFO] Battery recovered to {voltage_v:.2f} V. Clearing low-voltage counter.")
            self._battery_low_hits = 0

        return False

    def wait_for_rangefinder(self, timeout_s: float = 10.0) -> None:
        print("[CHECK] Waiting for rangefinder altitude data (RANGEFINDER/DISTANCE_SENSOR/rangefinder1)...")
        self.request_rangefinder_stream()
        deadline = time.time() + timeout_s
        next_request = time.time() + 2.0
        while time.time() < deadline:
            self.assert_no_manual_override("rangefinder check")
            alt = self.get_altitude_m(timeout=0.5)
            if alt is not None:
                print(f"[OK] {self.last_alt_msg_type} active. Altitude={alt:.2f} m")
                return
            if time.time() >= next_request:
                self.request_rangefinder_stream()
                next_request = time.time() + 2.0
        raise RuntimeError(
            "No RANGEFINDER/DISTANCE_SENSOR altitude data received. "
            "Check rangefinder stream rate and MAVLink message forwarding."
        )

    def log_altitude(self, prefix: str) -> None:
        now = time.time()
        if now - self.last_print_time < 1.0:
            return
        alt = self.get_altitude_m(timeout=0.2)
        if alt is not None:
            print(f"{prefix} altitude={alt:.2f} m")
            self.last_print_time = now

    def motors_armed(self) -> bool:
        if self.master is None:
            return False
        return bool(self.master.motors_armed())

    def set_mode(self, mode_name: str) -> None:
        self._sync_master()
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
        dc.set_mode(mode_name)

    def _recv_param_value(self, param_name: str, timeout_s: float = 2.0) -> Optional[float]:
        if self.master is None:
            return None
        deadline = time.time() + timeout_s
        wanted = param_name.encode("utf-8")
        while time.time() < deadline:
            msg = self.master.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.3)
            if msg is None:
                continue
            raw_param_id = getattr(msg, "param_id", b"")
            if isinstance(raw_param_id, bytes):
                msg_name = raw_param_id.split(b"\x00", 1)[0]
            else:
                msg_name = str(raw_param_id).encode("utf-8").split(b"\x00", 1)[0]
            if msg_name == wanted:
                return float(msg.param_value)
        return None

    def get_param(self, param_name: str) -> Optional[float]:
        if self.master is None:
            return None
        self.master.mav.param_request_read_send(
            self.master.target_system,
            self.master.target_component,
            param_name.encode("utf-8"),
            -1,
        )
        return self._recv_param_value(param_name, timeout_s=2.5)

    def set_param(self, param_name: str, value: float) -> None:
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            param_name.encode("utf-8"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        confirmed = self._recv_param_value(param_name, timeout_s=2.5)
        if confirmed is None:
            print(f"[WARN] No confirmation received for {param_name} set.")
            return
        print(f"[INFO] {param_name} set to {confirmed:.2f}")

    def set_slow_takeoff_profile(self) -> None:
        """
        Temporarily reduce vertical speed/acceleration for a smooth slow climb.
        """
        speeds = [
            ("WPNAV_SPEED_UP", SLOW_TAKEOFF_SPEED_UP),
            ("WPNAV_ACCEL_Z", SLOW_TAKEOFF_ACCEL_Z),
            ("PILOT_SPEED_UP", SLOW_TAKEOFF_SPEED_UP),
            ("PILOT_ACCEL_Z", SLOW_TAKEOFF_ACCEL_Z)
        ]
        for name, slow_value in speeds:
            current = self.get_param(name)
            if current is not None and name not in self._saved_params:
                self._saved_params[name] = current
                print(f"[INFO] Saved {name}={current:.2f}")
            self.set_param(name, slow_value)

    def set_slow_landing_profile(self) -> None:
        """
        Temporarily reduce vertical speed for a smooth descent.
        """
        speeds = [
            ("WPNAV_SPEED_DN", SLOW_LANDING_SPEED),
            ("LAND_SPEED", SLOW_LANDING_SPEED),
            ("LAND_SPEED_HIGH", SLOW_LANDING_SPEED)
        ]
        for name, slow_value in speeds:
            current = self.get_param(name)
            if current is not None and name not in self._saved_params:
                self._saved_params[name] = current
                print(f"[INFO] Saved {name}={current:.2f}")
            self.set_param(name, slow_value)

    def stabilize_position(self) -> None:
        """Commands 0 velocity to eliminate drift before landing modes."""
        print("[INFO] Stabilizing (zeroing horizontal velocity) to prevent drift...")
        for _ in range(15):
            self.send_body_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)

    def set_no_propeller_mode(self) -> None:
        """
        Cap max motor output to 30% using ArduPilot MOT_SPIN_MAX.
        """
        name = "MOT_SPIN_MAX"
        capped_value = 0.30
        current = self.get_param(name)
        if current is not None:
            self._saved_params[name] = current
            print(f"[INFO] Saved {name}={current:.2f}")
        else:
            print(f"[WARN] Could not read {name} before setting cap.")

        self.set_param(name, capped_value)
        print("[INFO] No-propeller mode active: motor output capped to 30%.")

    def restore_takeoff_profile(self) -> None:
        for name, value in self._saved_params.items():
            self.set_param(name, value)
        if self._saved_params:
            print("[INFO] Restored original climb profile parameters.")
        self._saved_params = {}

    def restore_all_modified_params(self) -> None:
        """Best-effort restore for all parameters changed by this script."""
        if not self._saved_params:
            return
        try:
            self.restore_takeoff_profile()
        except Exception as exc:
            print(f"[WARN] Failed to restore parameters during cleanup: {exc}")

    def rc_override_detected(self, deviation: int = RC_INPUT_DEVIATION) -> bool:
        """
        If pilot moves primary sticks away from neutral, stop automation.
        """
        if self.master is None:
            return False
        msg = self.master.recv_match(type="RC_CHANNELS", blocking=False)
        if msg is None:
            return False

        channels = [
            getattr(msg, "chan1_raw", 0),
            getattr(msg, "chan2_raw", 0),
            getattr(msg, "chan3_raw", 0),
            getattr(msg, "chan4_raw", 0),
        ]
        for index, raw in enumerate(channels, start=1):
            if not (900 <= int(raw) <= 2100):
                continue
            if abs(int(raw) - 1500) > deviation:
                print(f"[MANUAL] RC input detected on channel {index}: {raw}")
                return True
        return False

    def assert_no_manual_override(self, context: str) -> None:
        if self.rc_override_detected():
            raise ManualOverride(f"Pilot RC input detected during {context}.")

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _fetch_visual_state(self, timeout_s: float = 0.25) -> Optional[dict]:
        if not self.base_visual_state_url:
            return None
        req = Request(self.base_visual_state_url, headers={"Connection": "close"})
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                status = int(resp.getcode())
                if status < 200 or status >= 300:
                    return None
                payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, dict):
                    return payload
        except (TimeoutError, URLError, HTTPError, ValueError, OSError):
            return None
        except Exception:
            return None
        return None

    def send_body_velocity(self, vx: float, vy: float, vz: float) -> None:
        self._sync_master()
        if self.master is None:
            return
        dc.send_body_ned_velocity(float(vx), float(vy), float(vz))

    def _velocity_from_hover_command(self, hover_cmd: str):
        speed = float(self.base_align_speed_mps)
        vx = 0.0
        vy = 0.0
        for token in str(hover_cmd).strip().lower().split("+"):
            t = token.strip()
            if t == "move_up":
                vx += speed * float(self.base_forward_sign)
            elif t == "move_down":
                vx -= speed * float(self.base_forward_sign)
            elif t == "move_right":
                vy += speed * float(self.base_right_sign)
            elif t == "move_left":
                vy -= speed * float(self.base_right_sign)
        return vx, vy

    def _apply_align_velocity_slew(self, vx_cmd: float, vy_cmd: float):
        now = time.time()
        if self._last_align_ts <= 0.0:
            self._last_align_ts = now
            self._last_align_vx = float(vx_cmd)
            self._last_align_vy = float(vy_cmd)
            return float(vx_cmd), float(vy_cmd)

        dt = max(0.02, min(0.5, now - self._last_align_ts))
        max_dv = float(self.base_vel_slew_mps2) * dt

        dvx = float(vx_cmd) - float(self._last_align_vx)
        dvy = float(vy_cmd) - float(self._last_align_vy)
        if abs(dvx) > max_dv:
            vx_cmd = float(self._last_align_vx) + math.copysign(max_dv, dvx)
        if abs(dvy) > max_dv:
            vy_cmd = float(self._last_align_vy) + math.copysign(max_dv, dvy)

        self._last_align_vx = float(vx_cmd)
        self._last_align_vy = float(vy_cmd)
        self._last_align_ts = now
        return float(vx_cmd), float(vy_cmd)

    @staticmethod
    def _deadbanded_error(err: float, deadband: float) -> float:
        if abs(err) <= deadband:
            return 0.0
        return math.copysign(abs(err) - deadband, err)

    def _velocity_from_error(self, err_dx: float, err_dy: float, deadband: float, kp: float):
        # err_dx > 0 means target appears to the right; err_dy > 0 means target appears down.
        # Map to body-frame commands matching existing move_* semantics.
        ex = self._deadbanded_error(float(err_dx), float(deadband))
        ey = self._deadbanded_error(float(err_dy), float(deadband))

        vy_body = float(self.base_right_sign) * float(kp) * ex
        vx_body = -float(self.base_forward_sign) * float(kp) * ey

        vmax = float(self.base_align_speed_mps)
        vx_body = max(-vmax, min(vmax, vx_body))
        vy_body = max(-vmax, min(vmax, vy_body))
        return self._apply_align_velocity_slew(vx_body, vy_body)

    def send_relative_yaw_nudge(self, yaw_cmd: str) -> None:
        if not self.base_yaw_correction:
            return

        cmd = str(yaw_cmd).strip().lower()
        if cmd not in {"turn_left", "turn_right"}:
            return

        now = time.time()
        if (now - self._last_yaw_nudge_ts) < float(self.base_yaw_cmd_interval_s):
            return

        self._sync_master()
        if self.master is None:
            return

        # ArduPilot MAV_CMD_CONDITION_YAW: param1=angle deg, param2=speed deg/s,
        # param3=direction (1 CW, -1 CCW), param4=relative=1.
        direction = -1.0 if cmd == "turn_left" else 1.0
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            float(self.base_yaw_step_deg),
            float(self.base_yaw_rate_dps),
            float(direction),
            1,
            0,
            0,
            0,
        )
        self._last_yaw_nudge_ts = now

    def arm(self, timeout: int = 20) -> None:
        self._sync_master()
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
        dc.arm_vehicle()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.motors_armed():
                print("[OK] Vehicle armed.")
                return
            time.sleep(0.2)
        raise TimeoutError("Arming timeout.")

    def disarm(self, timeout: int = 20) -> None:
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
        print("[INFO] Disarming...")
        self.master.arducopter_disarm()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.motors_armed():
                print("[OK] Vehicle disarmed.")
                return
            time.sleep(0.2)
        raise TimeoutError("Disarm timeout.")

    def wait_for_ground(self, duration_s: float = 2.0) -> None:
        """Require continuous on-ground altitude before takeoff starts."""
        print("[CHECK] Verifying vehicle is on ground before takeoff...")
        start = time.time()
        while True:
            self.assert_no_manual_override("ground check")
            alt = self.get_altitude_m(timeout=0.5)
            if alt is None:
                continue
            if alt <= ON_GROUND_TOLERANCE:
                if time.time() - start >= duration_s:
                    self.initial_ground_altitude = alt
                    print(f"[OK] Ground check passed. Initial ground altitude={alt:.2f} m")
                    return
            else:
                print(f"[WARN] Altitude not on ground yet: {alt:.2f} m")
                start = time.time()
            time.sleep(0.1)

    def command_takeoff(self, target_alt: float) -> None:
        self._sync_master()
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
        dc.takeoff(target_alt)
        print(f"[INFO] Takeoff requested via drone_control.takeoff(). Target={target_alt:.2f} m")

    def command_land(self) -> None:
        print("[INFO] Landing via LAND mode...")
        self.set_mode("LAND")

    def verify_landed_at_ground(self, timeout_s: int = 30) -> None:
        """
        After landing, verify that final altitude matches initial ground level.
        Can be skipped via --skip-landing-verify flag if rangefinder is unreliable at ground.
        """
        if self.skip_landing_verify:
            print("[INFO] Landing verification skipped (--skip-landing-verify).")
            return

        if self.initial_ground_altitude is None:
            print("[WARN] Initial ground altitude was not captured.")
            return

        altitude_tolerance = 0.2  # Allow ±0.2m deviation from initial ground
        deadline = time.time() + timeout_s
        checked_once = False

        try:
            while time.time() < deadline:
                alt = self.get_altitude_m(timeout=0.5)
                if alt is None:
                    continue

                if checked_once:
                    print(f"[VERIFY] Current altitude: {alt:.2f} m (initial ground: {self.initial_ground_altitude:.2f} m)")

                if alt <= self.initial_ground_altitude + altitude_tolerance:
                    print(f"[OK] Landed at ground level. Final altitude: {alt:.2f} m (initial: {self.initial_ground_altitude:.2f} m)")
                    return

                checked_once = True
                time.sleep(0.5)

            print(f"[WARN] Final altitude {alt:.2f} m not at initial ground {self.initial_ground_altitude:.2f} m (tolerance: ±{altitude_tolerance}m). Proceeding anyway.")
        except KeyboardInterrupt:
            print(f"\n[VERIFY-INTERRUPT] User interrupted altitude verification. Current: {alt if alt else '?':.2f} m, Initial: {self.initial_ground_altitude:.2f} m")
            raise

    def land_and_disarm(self) -> None:
        """
        Use drone_control.land_vehicle() which waits for proper auto-disarm.
        """
        self._sync_master()
        if self.master is None:
            raise RuntimeError("Vehicle not connected.")
            
        print("[INFO] Preparing safe landing profile (25 cm/s)..")
        self.set_slow_landing_profile()
        self.stabilize_position()
        
        dc.land_vehicle()

    def descend_to_base_hold_altitude(self) -> None:
        target = float(self.base_hold_alt_m)
        tolerance = 0.15

        alt = self.get_altitude_m(timeout=0.6)
        if alt is None:
            print("[WARN] Base-hold descent skipped: altitude unavailable.")
            return

        if alt <= target + tolerance:
            print(f"[INFO] Already at/under base-hold altitude: {alt:.2f} m")
            return

        print(f"[INFO] Descending to ~{target:.2f} m before base alignment...")
        self.set_mode("GUIDED")
        deadline = time.time() + float(self.base_descent_timeout_s)
        while time.time() < deadline:
            self.assert_no_manual_override("base-descent")
            alt = self.get_altitude_m(timeout=0.3)
            if alt is None:
                time.sleep(0.15)
                continue

            err = float(alt - target)
            print(f"[BASE-DESCENT] Altitude: {alt:.2f} m (target {target:.2f} m)")

            if err <= tolerance:
                self.send_body_velocity(0.0, 0.0, 0.0)
                print("[OK] Reached base-hold altitude.")
                return

            vz_down = min(float(self.base_descent_speed_mps), max(0.08, 0.35 * err))
            self.send_body_velocity(0.0, 0.0, vz_down)
            time.sleep(0.2)

        self.send_body_velocity(0.0, 0.0, 0.0)
        print("[WARN] Timed out descending to base-hold altitude; continuing with current altitude.")

    def align_over_base_station(self) -> bool:
        if not self.base_visual_state_url:
            print("[WARN] Base alignment skipped: visual state URL is empty.")
            return False

        print("[INFO] Aligning over base station using ArUco/IR tracker state...")
        self.set_mode("GUIDED")
        deadline = time.time() + float(self.base_align_timeout_s)
        centered_since = None
        search_start_time = None
        last_log = 0.0
        no_state_count = 0

        while time.time() < deadline:
            self.assert_no_manual_override("base-align")
            state = self._fetch_visual_state(timeout_s=0.25)

            if not isinstance(state, dict):
                no_state_count += 1
                self.send_body_velocity(0.0, 0.0, 0.0)
                if no_state_count % 8 == 0:
                    print("[WARN] No /api/state from tracker dashboard. Holding position.")
                time.sleep(0.15)
                continue

            no_state_count = 0
            vo_debug = state.get("vo_debug", {}) if isinstance(state.get("vo_debug", {}), dict) else {}
            marker_locked = bool(vo_debug.get("marker_locked", False))
            base_ref_locked = bool(vo_debug.get("base_ref_locked", False))
            hover_cmd = str(vo_debug.get("hover_correction_command", "hold")).strip().lower()
            drift_cmd = str(vo_debug.get("drift_command", "hold")).strip().lower()
            seed_cmd = str(vo_debug.get("seed_correction_command", "hold")).strip().lower()
            metric_valid = bool(vo_debug.get("base_rel_metric_valid", False))
            base_dx_m = self._to_float(vo_debug.get("base_rel_dx_m", 0.0), 0.0)
            base_dy_m = self._to_float(vo_debug.get("base_rel_dy_m", 0.0), 0.0)
            err_norm_m = self._to_float(vo_debug.get("base_rel_norm_m", 0.0), 0.0)
            base_dx_px = self._to_float(vo_debug.get("base_rel_dx_px", 0.0), 0.0)
            base_dy_px = self._to_float(vo_debug.get("base_rel_dy_px", 0.0), 0.0)
            err_norm_px = self._to_float(vo_debug.get("base_rel_norm_px", vo_debug.get("drift_norm_px", 0.0)), 0.0)
            seed_scene_matched = bool(vo_debug.get("seed_scene_matched", False))
            seed_alignment_valid = bool(vo_debug.get("seed_alignment_valid", False))
            seed_err_norm_px = self._to_float(vo_debug.get("seed_error_norm_px", 0.0), 0.0)
            seed_error_dx_px = self._to_float(vo_debug.get("seed_error_dx_px", 0.0), 0.0)
            seed_error_dy_px = self._to_float(vo_debug.get("seed_error_dy_px", 0.0), 0.0)
            seed_yaw_err_deg = self._to_float(vo_debug.get("seed_yaw_error_deg", 0.0), 0.0)
            seed_yaw_valid = bool(vo_debug.get("seed_yaw_valid", False))
            seed_yaw_cmd = str(vo_debug.get("seed_yaw_correction_command", "hold")).strip().lower()
            seed_yaw_centered = bool(
                vo_debug.get("seed_yaw_centered", abs(seed_yaw_err_deg) <= float(self.base_yaw_deadband_deg))
            )

            matched_tag = vo_debug.get("matched_tag", None)
            if matched_tag is None:
                candidate_label = vo_debug.get("marker_label", None)
                if isinstance(candidate_label, str) and candidate_label and candidate_label.lower() != "blob":
                    matched_tag = candidate_label
            match_score = self._to_float(vo_debug.get("match_score", state.get("match_score", 0.0)), 0.0)
            tag_match_ok = matched_tag is not None and match_score >= float(self.base_match_score_min)
            has_drift_signal = ("drift_command" in vo_debug) or ("drift_norm_px" in vo_debug)
            drift_dx_px = self._to_float(vo_debug.get("drift_dx_px", 0.0), 0.0)
            drift_dy_px = self._to_float(vo_debug.get("drift_dy_px", 0.0), 0.0)
            drift_norm_px = self._to_float(vo_debug.get("drift_norm_px", 0.0), 0.0)

            align_mode = "search"
            active_cmd = "hold"
            centered = False
            err_text = "n/a"

            if marker_locked and base_ref_locked:
                align_mode = "base_ref"
                active_cmd = hover_cmd
                if metric_valid:
                    centered = err_norm_m <= float(self.base_metric_deadband_m)
                    err_text = f"{err_norm_m:.2f} m"
                else:
                    centered = err_norm_px <= float(self.base_pixel_deadband_px)
                    err_text = f"{err_norm_px:.1f} px"
            elif seed_alignment_valid:
                align_mode = "seed" if seed_scene_matched else "seed_search"
                active_cmd = seed_cmd
                centered = seed_scene_matched and (seed_err_norm_px <= float(self.base_pixel_deadband_px))
                if self.base_yaw_correction and seed_yaw_valid:
                    centered = centered and seed_yaw_centered
                err_text = f"{seed_err_norm_px:.1f} px, yaw {seed_yaw_err_deg:+.1f} deg"
            elif marker_locked and has_drift_signal:
                align_mode = "drift"
                active_cmd = drift_cmd
                centered = drift_norm_px <= float(self.base_pixel_deadband_px)
                err_text = f"{drift_norm_px:.1f} px"
            elif tag_match_ok:
                align_mode = "template"
                active_cmd = "hold"
                centered = True
                err_text = f"tag={matched_tag} score={match_score:.1f}"

            vx = 0.0
            vy = 0.0
            if centered:
                self.send_body_velocity(0.0, 0.0, 0.0)
                self._last_align_vx = 0.0
                self._last_align_vy = 0.0
                search_start_time = None
                if centered_since is None:
                    centered_since = time.time()
                if (time.time() - centered_since) >= float(self.base_center_hold_s):
                    print("[OK] Base centered and stable. Starting LAND.")
                    return True
            else:
                centered_since = None
                if align_mode != "search":
                    search_start_time = None

                if align_mode == "base_ref":
                    if metric_valid:
                        vx, vy = self._velocity_from_error(
                            err_dx=base_dx_m,
                            err_dy=base_dy_m,
                            deadband=float(self.base_metric_deadband_m),
                            kp=float(self.base_kp_metric),
                        )
                    else:
                        vx, vy = self._velocity_from_error(
                            err_dx=base_dx_px,
                            err_dy=base_dy_px,
                            deadband=float(self.base_pixel_deadband_px),
                            kp=float(self.base_kp_pixel),
                        )
                elif align_mode in {"seed", "seed_search"}:
                    vx, vy = self._velocity_from_error(
                        err_dx=seed_error_dx_px,
                        err_dy=seed_error_dy_px,
                        deadband=float(self.base_pixel_deadband_px),
                        kp=float(self.base_kp_pixel),
                    )
                elif align_mode == "drift":
                    vx, vy = self._velocity_from_error(
                        err_dx=drift_dx_px,
                        err_dy=drift_dy_px,
                        deadband=float(self.base_pixel_deadband_px),
                        kp=float(self.base_kp_pixel),
                    )
                elif align_mode == "search":
                    if search_start_time is None:
                        search_start_time = time.time()
                    t_scan = time.time() - search_start_time
                    # 10s period spiral, slowly expanding radius (up to 15cm/s)
                    freq = 2.0 * math.pi / 10.0
                    scan_speed = min(0.15, 0.02 + 0.005 * t_scan)
                    # Forward-right pattern
                    vx = scan_speed * math.cos(freq * t_scan)
                    vy = scan_speed * math.sin(freq * t_scan)
                    vx, vy = self._apply_align_velocity_slew(vx, vy)
                elif active_cmd and active_cmd != "hold":
                    vx, vy = self._velocity_from_hover_command(active_cmd)
                    vx, vy = self._apply_align_velocity_slew(vx, vy)

                self.send_body_velocity(vx, vy, 0.0)

                if self.base_yaw_correction and align_mode in {"seed", "seed_search"} and seed_alignment_valid and seed_yaw_valid:
                    self.send_relative_yaw_nudge(seed_yaw_cmd)

            if (time.time() - last_log) >= 1.0:
                yaw_log_cmd = seed_yaw_cmd if (self.base_yaw_correction and seed_alignment_valid and seed_yaw_valid) else "hold"
                print(
                    f"[BASE-ALIGN] mode={align_mode} marker_locked={marker_locked} base_ref_locked={base_ref_locked} "
                    f"cmd={active_cmd} yaw_cmd={yaw_log_cmd} error={err_text} v=({vx:+.2f},{vy:+.2f})"
                )
                last_log = time.time()

            time.sleep(0.15)

        self.send_body_velocity(0.0, 0.0, 0.0)
        print("[WARN] Base alignment timeout. Landing at current position.")
        return False

    def manual_slow_descent(self, target_alt: float = 0.5, speed: float = 0.10) -> None:
        """Manually lower the drone to target altitude, using time-based estimation if lidar fails."""
        start_alt = self.get_altitude_m(timeout=0.6)
        if start_alt is None:
            start_alt = float(self.base_hold_alt_m)
            print(f"[WARN] Lidar unavailable at start of descent, assuming starting height of {start_alt:.2f} m")
        else:
            print(f"[INFO] Lidar confirmed starting height at {start_alt:.2f} m.")
            
        dist_to_drop = max(0.0, start_alt - target_alt)
        max_duration = (dist_to_drop / speed) + 1.5
        
        print(f"[INFO] Manually descending to {target_alt:.2f} m at {speed} m/s (Max {max_duration:.1f}s)")
        self.set_mode("GUIDED")
        
        deadline = time.time() + max_duration
        while time.time() < deadline:
            self.assert_no_manual_override("manual-slow-descent")
            self.send_body_velocity(0.0, 0.0, speed)
            
            # Continue checking lidar, but don't block/fail if it drops out
            alt = self.get_altitude_m(timeout=0.1)
            if alt is not None:
                if alt <= target_alt:
                    print(f"[OK] Lidar confirms {target_alt:.2f} m reached ({alt:.2f} m).")
                    break
                    
            time.sleep(0.1)
            
        self.send_body_velocity(0.0, 0.0, 0.0)
        print("[INFO] Manual slow descent complete.")

    def perform_landing_sequence(self, reason: str) -> None:
        print(f"[INFO] Landing sequence reason: {reason}")

        # Keep immediate LAND behavior for altitude excursion safety.
        if self.landing_reason in {"critical_altitude", "soft_altitude"}:
            print("[SAFETY] Altitude safety trigger active. Skipping visual base alignment.")
            self.land_and_disarm()
            return

        if self.enable_base_landing:
            try:
                current_alt = self.get_altitude_m(timeout=0.5)
                if current_alt is not None and current_alt < float(self.base_hold_alt_m):
                    print(
                        f"[INFO] Current altitude {current_alt:.2f} m is below base alignment threshold "
                        f"{self.base_hold_alt_m:.2f} m. Skipping visual matching and landing now."
                    )
                else:
                    self.descend_to_base_hold_altitude()
                    self.align_over_base_station()
                
                # --- NEW: Manual slow descent before starting LAND mode ---
                self.manual_slow_descent(target_alt=0.5, speed=0.10)
                
            except ManualOverride:
                raise
            except Exception as exc:
                print(f"[WARN] Base alignment/descent stage failed: {exc}. Proceeding to LAND.")
            finally:
                try:
                    self.send_body_velocity(0.0, 0.0, 0.0)
                except Exception:
                    pass

        self.land_and_disarm()

    def enforce_altitude_safety(self, context: str) -> bool:
        if self.enforce_battery_failsafe(context):
            return True

        alt = self.get_altitude_m(timeout=0.2)
        if alt is None:
            print(f"[WARN] {context}: altitude unavailable.")
            return False

        dynamic_soft_limit = MAX_TARGET_ALT
        if self.target_altitude is not None:
            dynamic_soft_limit = self.target_altitude + AUTO_LAND_MARGIN_ABOVE_TARGET_M

        if alt > CRITICAL_ALT:
            print(f"[CRITICAL] Altitude {alt:.2f} m > {CRITICAL_ALT:.2f} m. Forcing LAND mode.")
            if not self.landing_forced:
                self.landing_reason = "critical_altitude"
                self.command_land()
                self.landing_forced = True
            return True

        if alt > dynamic_soft_limit:
            print(
                f"[SAFETY] Altitude {alt:.2f} m > soft limit {dynamic_soft_limit:.2f} m "
                f"(target + {AUTO_LAND_MARGIN_ABOVE_TARGET_M:.2f} m). Switching to LAND mode for auto descent."
            )
            if not self.landing_forced:
                self.landing_reason = "soft_altitude"
                self.command_land()
                self.landing_forced = True
            return True
        return False

    def emergency_interrupt_handler(self) -> None:
        print("\n[INTERRUPT] Keyboard interrupt detected.")
        alt = self.get_altitude_m(timeout=0.7)
        if alt is None:
            print("[WARN] Could not read altitude. Attempting safe LAND.")
            if self.motors_armed():
                self.command_land()
            return

        print(f"[INTERRUPT] Current altitude: {alt:.2f} m")
        if alt > ON_GROUND_TOLERANCE:
            print("[INTERRUPT] Above ground. Landing automatically.")
            self.land_and_disarm()
            # Skip verification during emergency—user just wants immediate landing
        else:
            print("[INTERRUPT] On/near ground. Disarming.")
            if self.motors_armed():
                try:
                    self.disarm(timeout=10)
                except TimeoutError as exc:
                    # During Ctrl+C shutdown we do not want a disarm timeout to crash exit.
                    print(f"[WARN] Emergency disarm did not confirm in time: {exc}")

    def wait_for_takeoff_reached(self, target_alt: float, timeout_s: int = 45) -> None:
        deadline = time.time() + timeout_s
        reached = max(0.5, target_alt - TAKEOFF_REACHED_MARGIN_M)
        prebrake_alt = max(0.5, target_alt - TAKEOFF_PREBRAKE_MARGIN_M)
        prebrake_applied = False
        while time.time() < deadline:
            self.assert_no_manual_override("takeoff")
            if self.enforce_altitude_safety("TAKEOFF"):
                print("[SAFETY] Takeoff interrupted by altitude safety LAND.")
                return
            alt = self.get_altitude_m(timeout=0.3)
            if alt is not None:
                print(f"[TAKEOFF] Altitude: {alt:.2f} m")
                if not prebrake_applied and alt >= prebrake_alt:
                    print(f"[TAKEOFF] Pre-brake at {alt:.2f} m. Switching to LOITER to reduce overshoot.")
                    self.set_mode("LOITER")
                    prebrake_applied = True
                if alt >= reached:
                    print(f"[OK] Target takeoff altitude reached ({alt:.2f} m).")
                    return
            time.sleep(0.2)
        raise TimeoutError("Takeoff altitude not reached in time.")

    def hover_with_safety(self) -> None:
        print(f"[INFO] Hovering for {self.hover_seconds} seconds.")
        end_time = time.time() + self.hover_seconds
        last_countdown = None
        while time.time() < end_time:
            self.assert_no_manual_override("hover")
            if self.enforce_altitude_safety("HOVER"):
                print("[SAFETY] Hover ended because LAND mode was forced.")
                return
            remaining = int(max(0, end_time - time.time()))
            if remaining != last_countdown:
                mins = remaining // 60
                secs = remaining % 60
                print(f"[HOVER] Countdown: {mins:02d}:{secs:02d}")
                last_countdown = remaining
            alt = self.get_altitude_m(timeout=0.3)
            if alt is not None:
                if alt < 0.5:
                    print("[WARN] Hover altitude unexpectedly low.")
            self.log_altitude("[HOVER]")
            time.sleep(0.5)
        print("[OK] Hover complete.")

    def monitor_landing(self, timeout_s: int = 180) -> None:
        """
        Monitor landing while checking for manual RC override.
        Expects drone_control.land_vehicle() to be called separately for auto-disarm.
        """
        print("[INFO] Monitoring landing descent...")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.assert_no_manual_override("landing")
            alt = self.get_altitude_m(timeout=0.3)
            if alt is not None:
                print(f"[LAND] Altitude: {alt:.2f} m")
            if not self.motors_armed():
                print("[OK] Landed and disarmed.")
                return
            time.sleep(0.4)
        print("[OK] Landing monitoring complete.")

    def get_target_altitude_from_terminal(self) -> float:
        while True:
            raw = input(f"Enter target takeoff altitude in meters (0.5 to {MAX_TARGET_ALT:.1f}, default {DEFAULT_TARGET_ALT:.1f}): ").strip()
            if raw == "":
                return DEFAULT_TARGET_ALT
            try:
                value = float(raw)
            except ValueError:
                print("[WARN] Enter a numeric value.")
                continue
            if value < 0.5:
                print("[WARN] Minimum allowed target altitude is 0.5m.")
                continue
            if value > MAX_TARGET_ALT:
                print(f"[WARN] Maximum allowed target altitude is {MAX_TARGET_ALT:.1f}m.")
                continue
            return value

    def run(self) -> None:
        self.connect()
        self.wait_for_rangefinder(timeout_s=10.0)
        self.request_battery_stream()
        init_v = self.get_battery_voltage_v(timeout=1.0)
        if init_v is not None:
            src = self.last_battery_source or "UNKNOWN"
            print(f"[CHECK] Battery telemetry active via {src}: {init_v:.2f} V")
        else:
            print("[WARN] Battery telemetry not received yet; failsafe will use first MAVLink battery update when available.")

        if self.enable_base_landing:
            sample_state = self._fetch_visual_state(timeout_s=0.5)
            if isinstance(sample_state, dict):
                print(f"[CHECK] Visual landing state endpoint reachable: {self.base_visual_state_url}")
            else:
                print(
                    "[WARN] Visual landing state endpoint not reachable right now. "
                    "Base alignment will fall back to straight LAND if tracker state is unavailable."
                )

        if self.no_propeller_mode:
            self.confirm_step("Enable no-propeller mode (motor cap 30%)?", "NOPROP")
            self.assert_no_manual_override("no-propeller-mode setup")
            self.set_no_propeller_mode()

        target_alt = self.get_target_altitude_from_terminal()
        self.target_altitude = target_alt
        print(f"[INFO] Selected target altitude: {target_alt:.2f} m")

        self.confirm_step("Pre-flight checks complete?", "READY")
        self.wait_for_ground(duration_s=2.0)

        self.confirm_step("Set LOITER mode now (required before takeoff)?", "LOITER")
        self.assert_no_manual_override("pre-loiter")
        self.set_mode("LOITER")

        self.confirm_step("Arm now?", "ARM")
        self.assert_no_manual_override("arming")
        self.arm()

        self.confirm_step("Start takeoff now?", "TAKEOFF")
        self.assert_no_manual_override("pre-takeoff")
        self.set_slow_takeoff_profile()
        try:
            self.command_takeoff(target_alt)
            self.wait_for_takeoff_reached(target_alt=target_alt, timeout_s=90)
        finally:
            self.restore_takeoff_profile()

        if self.landing_forced:
            reason = self.landing_reason or "takeoff_safety"
            print(f"[SAFETY] Landing requested during takeoff ({reason}).")
            self.perform_landing_sequence(reason)
            self.verify_landed_at_ground(timeout_s=30)
            print("[DONE] Mission ended by safety condition.")
            return

        print("[INFO] Target altitude reached. Starting 5-minute hover countdown automatically...")
        self.hover_with_safety()

        if self.landing_forced:
            reason = self.landing_reason or "hover_safety"
            print(f"[SAFETY] Landing requested during hover ({reason}).")
            self.perform_landing_sequence(reason)
            self.verify_landed_at_ground(timeout_s=30)
            print("[DONE] Mission ended by safety condition.")
            return

        print("[INFO] Hover complete. Initiating base-aware landing automatically...")
        self.assert_no_manual_override("pre-landing")
        self.perform_landing_sequence("mission_complete")
        self.monitor_landing(timeout_s=180)
        self.verify_landed_at_ground(timeout_s=30)

        print("[DONE] Mission complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded arm/takeoff/hover/land workflow.")
    parser.add_argument("--connect", default=DEFAULT_CONNECTION, help="MAVLink connection string")
    parser.add_argument("--baud", default=DEFAULT_BAUD, type=int, help="Serial baud rate")
    parser.add_argument(
        "--hover-seconds",
        default=DEFAULT_HOVER_SECONDS,
        type=int,
        help="Hover time in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-propeller-mode",
        action="store_true",
        default=DEFAULT_NO_PROPELLER_MODE,
        help="Cap max motor output to 30 percent (bench/no-prop testing mode)",
    )
    parser.add_argument(
        "--skip-landing-verify",
        action="store_true",
        default=DEFAULT_SKIP_LANDING_VERIFY,
        help="Skip altitude verification after landing (if rangefinder unreliable at ground)",
    )
    parser.add_argument(
        "--low-voltage",
        default=DEFAULT_LOW_VOLTAGE_V,
        type=float,
        help="Battery pack low-voltage threshold in volts for immediate LAND failsafe (default: 13.2)",
    )
    parser.add_argument(
        "--low-voltage-confirm-count",
        default=DEFAULT_LOW_VOLTAGE_CONFIRM_COUNT,
        type=int,
        help="Consecutive low-voltage MAVLink reads required before forcing LAND (default: 3)",
    )
    parser.add_argument(
        "--disable-base-landing",
        action="store_true",
        default=(not DEFAULT_ENABLE_BASE_LANDING),
        help="Disable 1m hold + visual base alignment stage before final LAND",
    )
    parser.add_argument(
        "--base-hold-alt",
        default=DEFAULT_BASE_HOLD_ALT_M,
        type=float,
        help="Altitude threshold for visual alignment; below this altitude mission skips matching and lands directly (default: 1.0)",
    )
    parser.add_argument(
        "--base-align-timeout",
        default=DEFAULT_BASE_ALIGN_TIMEOUT_S,
        type=float,
        help="Max seconds to attempt base alignment using tracker state (default: 20)",
    )
    parser.add_argument(
        "--base-center-hold",
        default=DEFAULT_BASE_CENTER_HOLD_S,
        type=float,
        help="Seconds of continuous centered error required before final LAND (default: 2.0)",
    )
    parser.add_argument(
        "--base-align-speed",
        default=DEFAULT_BASE_ALIGN_SPEED_MPS,
        type=float,
        help="Body-frame XY correction speed in m/s during base alignment (default: 0.12)",
    )
    parser.add_argument(
        "--base-metric-deadband",
        default=DEFAULT_BASE_METRIC_DEADBAND_M,
        type=float,
        help="Centered threshold in meters when metric base error is available (default: 0.10)",
    )
    parser.add_argument(
        "--base-pixel-deadband",
        default=DEFAULT_BASE_PIXEL_DEADBAND_PX,
        type=float,
        help="Centered threshold in pixels when only pixel error is available (default: 14)",
    )
    parser.add_argument(
        "--base-descent-speed",
        default=DEFAULT_BASE_DESCENT_SPEED_MPS,
        type=float,
        help="Max descent speed in m/s while approaching base-hold altitude (default: 0.25)",
    )
    parser.add_argument(
        "--base-descent-timeout",
        default=DEFAULT_BASE_DESCENT_TIMEOUT_S,
        type=float,
        help="Max seconds allowed to descend to base-hold altitude (default: 60)",
    )
    parser.add_argument(
        "--base-visual-state-url",
        default=DEFAULT_BASE_VISUAL_STATE_URL,
        help="Tracker dashboard /api/state URL used for ArUco/IR landing alignment",
    )
    parser.add_argument(
        "--base-forward-sign",
        default=DEFAULT_BASE_FORWARD_SIGN,
        type=float,
        help="Set to -1 if forward/back correction is reversed on your frame mapping",
    )
    parser.add_argument(
        "--base-right-sign",
        default=DEFAULT_BASE_RIGHT_SIGN,
        type=float,
        help="Set to -1 if left/right correction is reversed on your frame mapping",
    )
    parser.add_argument(
        "--base-match-score-min",
        default=DEFAULT_BASE_MATCH_SCORE_MIN,
        type=float,
        help="Minimum template/tag match score required for template-only base confirmation (default: 12)",
    )
    parser.add_argument(
        "--base-yaw-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BASE_YAW_CORRECTION,
        help="Enable small yaw nudges during seed-based base alignment",
    )
    parser.add_argument(
        "--base-yaw-deadband-deg",
        default=DEFAULT_BASE_YAW_DEADBAND_DEG,
        type=float,
        help="Yaw error deadband in degrees for considering alignment centered (default: 3.0)",
    )
    parser.add_argument(
        "--base-yaw-step-deg",
        default=DEFAULT_BASE_YAW_STEP_DEG,
        type=float,
        help="Relative yaw step in degrees per correction nudge (default: 2.0)",
    )
    parser.add_argument(
        "--base-yaw-rate-dps",
        default=DEFAULT_BASE_YAW_RATE_DPS,
        type=float,
        help="Yaw slew rate in deg/s for correction nudges (default: 15.0)",
    )
    parser.add_argument(
        "--base-yaw-cmd-interval",
        default=DEFAULT_BASE_YAW_CMD_INTERVAL_S,
        type=float,
        help="Minimum seconds between yaw correction nudges (default: 0.5)",
    )
    parser.add_argument(
        "--base-kp-metric",
        default=DEFAULT_BASE_KP_METRIC,
        type=float,
        help="P-gain for metric XY base alignment control (m->m/s)",
    )
    parser.add_argument(
        "--base-kp-pixel",
        default=DEFAULT_BASE_KP_PIXEL,
        type=float,
        help="P-gain for pixel XY base alignment control (px->m/s)",
    )
    parser.add_argument(
        "--base-vel-slew-mps2",
        default=DEFAULT_BASE_VEL_SLEW_MPS2,
        type=float,
        help="Max XY velocity slew during base alignment in m/s^2",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    mission = GuardedMission(
        connection=args.connect,
        baud=args.baud,
        hover_seconds=args.hover_seconds,
        no_propeller_mode=args.no_propeller_mode,
        skip_landing_verify=args.skip_landing_verify,
        low_voltage_v=args.low_voltage,
        low_voltage_confirm_count=args.low_voltage_confirm_count,
        enable_base_landing=(not args.disable_base_landing),
        base_hold_alt_m=args.base_hold_alt,
        base_align_timeout_s=args.base_align_timeout,
        base_center_hold_s=args.base_center_hold,
        base_align_speed_mps=args.base_align_speed,
        base_metric_deadband_m=args.base_metric_deadband,
        base_pixel_deadband_px=args.base_pixel_deadband,
        base_descent_speed_mps=args.base_descent_speed,
        base_descent_timeout_s=args.base_descent_timeout,
        base_visual_state_url=args.base_visual_state_url,
        base_forward_sign=args.base_forward_sign,
        base_right_sign=args.base_right_sign,
        base_match_score_min=args.base_match_score_min,
        base_yaw_correction=args.base_yaw_correction,
        base_yaw_deadband_deg=args.base_yaw_deadband_deg,
        base_yaw_step_deg=args.base_yaw_step_deg,
        base_yaw_rate_dps=args.base_yaw_rate_dps,
        base_yaw_cmd_interval_s=args.base_yaw_cmd_interval,
        base_kp_metric=args.base_kp_metric,
        base_kp_pixel=args.base_kp_pixel,
        base_vel_slew_mps2=args.base_vel_slew_mps2,
    )
    try:
        mission.run()
    except KeyboardInterrupt:
        try:
            mission.emergency_interrupt_handler()
        except Exception as exc:
            print(f"[WARN] Emergency interrupt handling failed: {exc}")
        return 130
    except TimeoutError as exc:
        print(f"[TIMEOUT] {exc}")
        try:
            if mission.master and mission.motors_armed():
                print("[TIMEOUT] Attempting safe LAND due to timeout...")
                mission.land_and_disarm()
        except Exception as land_exc:
            print(f"[TIMEOUT] Safe landing after timeout failed: {land_exc}")
        return 3
    except ManualOverride as exc:
        print(f"[MANUAL] {exc}")
        print("[MANUAL] Automation stopped. Pilot has priority control.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unhandled exception occurred: {exc}")
        print("[SAFETY] Attempting to land immediately...")
        try:
            if mission.master and mission.motors_armed():
                mission.set_mode("LAND")
                print("[SAFETY] Switched to LAND mode. Waiting to disarm...")
                import drone_control as dc
                dc.master.motors_disarmed_wait()
                print("[SAFETY] Successfully landed and disarmed after error.")
        except Exception as land_exc:
            print(f"[FATAL] Failsafe landing after exception failed: {land_exc}")
        return 99
    except Exception as exc:
        print(f"[ERROR] Unhandled exception occurred: {exc}")
        print("[SAFETY] Attempting to land immediately...")
        try:
            if mission.master and mission.motors_armed():
                mission.set_mode("LAND")
                print("[SAFETY] Switched to LAND mode. Waiting to disarm...")
                import drone_control as dc
                dc.master.motors_disarmed_wait()
                print("[SAFETY] Successfully landed and disarmed after error.")
        except Exception as land_exc:
            print(f"[FATAL] Failsafe landing after exception failed: {land_exc}")
        return 99
    except Exception as exc:
        print(f"[ERROR] {exc}")
        try:
            if mission.master and mission.motors_armed():
                print("[ERROR] Trying safe landing due to exception...")
                mission.land_and_disarm()
        except Exception as land_exc:
            print(f"[ERROR] Emergency landing also failed: {land_exc}")
        return 1
    finally:
        try:
            mission.restore_all_modified_params()
        except Exception:
            pass
        try:
            if mission.master:
                mission.master.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
