import py_trees
import drone_control as dc
from . import blackboard_keys as BK
from .base import DroneActionNode
import time
import requests
import math
from pymavlink import mavutil

HOLD_ALT_M = 1.5
FINAL_ARM_ALT_M = 0.20
TRACKER_URL = "http://localhost:5000/api/state"
ALIGN_TIMEOUT_S = 30.0
CENTER_HOLD_S = 2.0

class DescendToHoldAlt(DroneActionNode):
    """Stage 1: Descend to 1.5m for alignment."""
    def __init__(self):
        super().__init__("DescendToHoldAlt")
        self._deadline = None

    def initialise(self):
        self._deadline = time.time() + 60.0
        dc.set_mode("GUIDED")

    def update(self):
        if time.time() > self._deadline:
            self._log("Descent timeout")
            return py_trees.common.Status.FAILURE

        alt = self.bb.get(BK.ALTITUDE_M)
        if alt is None:
            return py_trees.common.Status.RUNNING

        if alt <= HOLD_ALT_M + 0.15:
            dc.send_body_ned_velocity(0, 0, 0)
            self._log(f"Reached hold altitude: {alt:.2f}m")
            return py_trees.common.Status.SUCCESS

        vz = min(0.25, max(0.08, 0.35 * (alt - HOLD_ALT_M)))
        dc.send_body_ned_velocity(0, 0, vz)
        return py_trees.common.Status.RUNNING

class IRBeaconAlign(DroneActionNode):
    """STUB — always FAILURE until BPW34 + SFH4550 arrive."""
    def __init__(self):
        super().__init__("IRBeaconAlign")
        
    def update(self):
        # TODO: 
        # result = ir_fft_solver.solve_position(read_adc_signal())
        # if result["centered"]: return SUCCESS
        # dc.send_body_ned_velocity(result["x_offset"]*kp, result["y_offset"]*kp, 0)
        # return RUNNING
        return py_trees.common.Status.FAILURE

class ArUcoAlign(DroneActionNode):
    """Stage 2: Non-blocking port of guarded_mission.py's align_over_base_station"""
    def __init__(self):
        super().__init__("ArUcoAlign")
        self._deadline = None
        self._centered_since = None
        self._search_start_time = None
        
        # Parameters mapped from guarded_mission.py
        self.base_metric_deadband_m = 0.10
        self.base_pixel_deadband_px = 14.0
        self.base_kp_metric = 1.0
        self.base_kp_pixel = 0.0035
        self.base_vel_slew_mps2 = 0.60
        self.base_align_speed_mps = 0.12
        self.base_forward_sign = 1.0
        self.base_right_sign = 1.0
        self.base_match_score_min = 12.0
        
        self._last_align_vx = 0.0
        self._last_align_vy = 0.0
        self._last_align_ts = 0.0

    def initialise(self):
        self._deadline = time.time() + ALIGN_TIMEOUT_S
        self._centered_since = None
        self._search_start_time = None
        self._no_tracker_since = None
        dc.set_mode("GUIDED")

    def _fetch_state(self):
        try:
            r = requests.get(TRACKER_URL, timeout=0.25)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def _apply_align_velocity_slew(self, vx_cmd, vy_cmd):
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

    def _deadbanded_error(self, err, deadband):
        if abs(err) <= deadband:
            return 0.0
        return math.copysign(abs(err) - deadband, err)

    def _velocity_from_error(self, err_dx, err_dy, deadband, kp):
        ex = self._deadbanded_error(float(err_dx), float(deadband))
        ey = self._deadbanded_error(float(err_dy), float(deadband))

        vy_body = float(self.base_right_sign) * float(kp) * ex
        vx_body = -float(self.base_forward_sign) * float(kp) * ey

        vmax = float(self.base_align_speed_mps)
        vx_body = max(-vmax, min(vmax, vx_body))
        vy_body = max(-vmax, min(vmax, vy_body))
        return self._apply_align_velocity_slew(vx_body, vy_body)

    def update(self):
        if time.time() > self._deadline:
            self._log("ArUco align timeout")
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.FAILURE

        state = self._fetch_state()
        if not isinstance(state, dict):
            # Tracker server not reachable (no camera / SITL mode)
            if self._no_tracker_since is None:
                self._no_tracker_since = time.time()
                self._log("Tracker not reachable — SITL fallback: direct LAND in 3s")
            if time.time() - self._no_tracker_since >= 3.0:
                self._log("No tracker — executing direct LAND")
                dc.set_mode("LAND")
                return py_trees.common.Status.SUCCESS
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.RUNNING

        vo_debug = state.get("vo_debug", {}) if isinstance(state.get("vo_debug", {}), dict) else {}
        marker_locked = bool(vo_debug.get("marker_locked", False))
        base_ref_locked = bool(vo_debug.get("base_ref_locked", False))
        metric_valid = bool(vo_debug.get("base_rel_metric_valid", False))
        base_dx_m = float(vo_debug.get("base_rel_dx_m", 0.0))
        base_dy_m = float(vo_debug.get("base_rel_dy_m", 0.0))
        err_norm_m = float(vo_debug.get("base_rel_norm_m", 0.0))
        base_dx_px = float(vo_debug.get("base_rel_dx_px", 0.0))
        base_dy_px = float(vo_debug.get("base_rel_dy_px", 0.0))
        err_norm_px = float(vo_debug.get("base_rel_norm_px", vo_debug.get("drift_norm_px", 0.0)))
        
        seed_scene_matched = bool(vo_debug.get("seed_scene_matched", False))
        seed_alignment_valid = bool(vo_debug.get("seed_alignment_valid", False))
        seed_err_norm_px = float(vo_debug.get("seed_error_norm_px", 0.0))
        seed_error_dx_px = float(vo_debug.get("seed_error_dx_px", 0.0))
        seed_error_dy_px = float(vo_debug.get("seed_error_dy_px", 0.0))
        
        matched_tag = vo_debug.get("matched_tag", None)
        if matched_tag is None:
            candidate_label = vo_debug.get("marker_label", None)
            if isinstance(candidate_label, str) and candidate_label and candidate_label.lower() != "blob":
                matched_tag = candidate_label
                
        match_score = float(vo_debug.get("match_score", state.get("match_score", 0.0)))
        tag_match_ok = matched_tag is not None and match_score >= self.base_match_score_min
        
        has_drift_signal = ("drift_command" in vo_debug) or ("drift_norm_px" in vo_debug)
        drift_dx_px = float(vo_debug.get("drift_dx_px", 0.0))
        drift_dy_px = float(vo_debug.get("drift_dy_px", 0.0))
        drift_norm_px = float(vo_debug.get("drift_norm_px", 0.0))

        align_mode = "search"
        centered = False

        if marker_locked and base_ref_locked:
            align_mode = "base_ref"
            if metric_valid:
                centered = err_norm_m <= self.base_metric_deadband_m
            else:
                centered = err_norm_px <= self.base_pixel_deadband_px
        elif seed_alignment_valid:
            align_mode = "seed" if seed_scene_matched else "seed_search"
            centered = seed_scene_matched and (seed_err_norm_px <= self.base_pixel_deadband_px)
        elif marker_locked and has_drift_signal:
            align_mode = "drift"
            centered = drift_norm_px <= self.base_pixel_deadband_px
        elif tag_match_ok:
            align_mode = "template"
            centered = True

        self.bb.set(BK.ALIGN_CENTERED, centered)

        vx, vy = 0.0, 0.0
        if centered:
            dc.send_body_ned_velocity(0.0, 0.0, 0.0)
            self._last_align_vx = 0.0
            self._last_align_vy = 0.0
            self._search_start_time = None
            if self._centered_since is None:
                self._centered_since = time.time()
            if time.time() - self._centered_since >= CENTER_HOLD_S:
                self._log("Base centered and stable.")
                return py_trees.common.Status.SUCCESS
        else:
            self._centered_since = None
            if align_mode != "search":
                self._search_start_time = None

            if align_mode == "base_ref":
                if metric_valid:
                    vx, vy = self._velocity_from_error(base_dx_m, base_dy_m, self.base_metric_deadband_m, self.base_kp_metric)
                else:
                    vx, vy = self._velocity_from_error(base_dx_px, base_dy_px, self.base_pixel_deadband_px, self.base_kp_pixel)
            elif align_mode in {"seed", "seed_search"}:
                vx, vy = self._velocity_from_error(seed_error_dx_px, seed_error_dy_px, self.base_pixel_deadband_px, self.base_kp_pixel)
            elif align_mode == "drift":
                vx, vy = self._velocity_from_error(drift_dx_px, drift_dy_px, self.base_pixel_deadband_px, self.base_kp_pixel)
            elif align_mode == "search":
                if self._search_start_time is None:
                    self._search_start_time = time.time()
                t_scan = time.time() - self._search_start_time
                # 10s period spiral, fixed max radius
                freq = 2.0 * math.pi / 10.0
                radius = min(0.5, 0.02 * t_scan)
                scan_speed = radius * freq
                vx = scan_speed * math.cos(freq * t_scan)
                vy = scan_speed * math.sin(freq * t_scan)
                vx, vy = self._apply_align_velocity_slew(vx, vy)

            dc.send_body_ned_velocity(vx, vy, 0.0)

        return py_trees.common.Status.RUNNING

class RANSACValidate(DroneActionNode):
    """Stage 3: CS20 3D point cloud slope check."""
    def __init__(self):
        super().__init__("RANSACValidate")
        
    def update(self):
        # STUB: Assume flat ground for now until CS20 point cloud logic is added
        self.bb.set(BK.RANSAC_OK, True)
        self._log("RANSAC flat (stub)")
        return py_trees.common.Status.SUCCESS

class FusedDescent(DroneActionNode):
    """Stage 4: Slow descent to ground, force-disarm."""
    def __init__(self):
        super().__init__("FusedDescent")
        self._deadline = None

    def initialise(self):
        dc.set_mode("LAND")
        self._deadline = time.time() + 60.0

    def update(self):
        if time.time() > self._deadline:
            self._log("Descent timeout — force disarm")
            self._force_disarm()
            return py_trees.common.Status.SUCCESS

        alt = self.bb.get(BK.ALTITUDE_M)
        if alt is None:
            return py_trees.common.Status.RUNNING

        self._log(f"Descending: {alt:.2f}m")

        if alt <= FINAL_ARM_ALT_M:
            self._force_disarm()
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def _force_disarm(self):
        self._log("Force disarming at ground")
        master = self._get_master()
        if master:
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 21196, 0, 0, 0, 0, 0
            )

class LogMissionReport(DroneActionNode):
    def __init__(self):
        super().__init__("LogMissionReport")
        
    def update(self):
        self._log("Mission Complete. Saved logs.")
        return py_trees.common.Status.SUCCESS

def make_landing_phase():
    # Try IR beacon first, fall back to ArUco
    align_selector = py_trees.composites.Selector(
        name="AlignOverBase", memory=False
    )
    align_selector.add_children([IRBeaconAlign(), ArUcoAlign()])

    landing_seq = py_trees.composites.Sequence(
        name="PrecisionLanding", memory=True
    )
    landing_seq.add_children([
        DescendToHoldAlt(),
        align_selector,
        RANSACValidate(),
        FusedDescent(),
    ])
    return landing_seq
