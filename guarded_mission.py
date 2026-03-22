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
import select
import sys
import time
from typing import Optional

import drone_control as dc
from pymavlink import mavutil


DEFAULT_CONNECTION = "/dev/ttyACM0"
DEFAULT_BAUD = 57600
DEFAULT_TARGET_ALT = 1.0
MAX_TARGET_ALT = 3.0
CRITICAL_ALT = 4.0
ON_GROUND_TOLERANCE = 0.15
DEFAULT_HOVER_SECONDS = 300
RC_INPUT_DEVIATION = 120
DEFAULT_NO_PROPELLER_MODE = False
DEFAULT_SKIP_LANDING_VERIFY = False
RANGEFINDER_INTERVAL_US = 100000  # 10 Hz
SLOW_TAKEOFF_SPEED_UP = 15.0
SLOW_TAKEOFF_ACCEL_Z = 10.0
TAKEOFF_PREBRAKE_MARGIN_M = 0.10
TAKEOFF_REACHED_MARGIN_M = 0.05
AUTO_LAND_MARGIN_ABOVE_TARGET_M = 0.50


class ManualOverride(Exception):
    """Raised when pilot RC input is detected and autonomy must stop."""


class GuardedMission:
    def __init__(self, connection: str, baud: int, hover_seconds: int, no_propeller_mode: bool, skip_landing_verify: bool = False) -> None:
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
        for name, slow_value in (("WPNAV_SPEED_UP", SLOW_TAKEOFF_SPEED_UP), ("WPNAV_ACCEL_Z", SLOW_TAKEOFF_ACCEL_Z)):
            current = self.get_param(name)
            if current is not None:
                self._saved_params[name] = current
                print(f"[INFO] Saved {name}={current:.2f}")
            self.set_param(name, slow_value)

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
        dc.land_vehicle()

    def enforce_altitude_safety(self, context: str) -> bool:
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
                self.command_land()
                self.landing_forced = True
            return True

        if alt > dynamic_soft_limit:
            print(
                f"[SAFETY] Altitude {alt:.2f} m > soft limit {dynamic_soft_limit:.2f} m "
                f"(target + {AUTO_LAND_MARGIN_ABOVE_TARGET_M:.2f} m). Switching to LAND mode for auto descent."
            )
            if not self.landing_forced:
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
            print("[SAFETY] LAND was already forced during takeoff. Landing and disarming.")
            self.land_and_disarm()
            self.verify_landed_at_ground(timeout_s=30)
            print("[DONE] Mission ended by safety condition.")
            return

        print("[INFO] Target altitude reached. Starting 5-minute hover countdown automatically...")
        self.hover_with_safety()

        if self.landing_forced:
            print("[SAFETY] LAND was forced during hover. Landing and disarming.")
            self.land_and_disarm()
            self.verify_landed_at_ground(timeout_s=30)
            print("[DONE] Mission ended by safety condition.")
            return

        print("[INFO] Hover complete. Initiating landing automatically...")
        self.assert_no_manual_override("pre-landing")
        self.land_and_disarm()
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mission = GuardedMission(
        connection=args.connect,
        baud=args.baud,
        hover_seconds=args.hover_seconds,
        no_propeller_mode=args.no_propeller_mode,
        skip_landing_verify=args.skip_landing_verify,
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
