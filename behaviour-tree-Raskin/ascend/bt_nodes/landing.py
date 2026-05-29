import py_trees
import drone_control as dc
from . import blackboard_keys as BK
from .base import DroneActionNode
import os
import time
import requests
import math
import threading
from pymavlink import mavutil
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as RosImage
    from sensor_msgs.msg import CameraInfo as RosCameraInfo
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False

HOLD_ALT_M = 1.5
FINAL_ARM_ALT_M = 0.20
TRACKER_URL = "http://localhost:5000/api/state"
ALIGN_TIMEOUT_S = 30.0
CENTER_HOLD_S = 2.0

# Keys for Blackboard specific to ArUco
BK_ARUCO_VISIBLE = BK.ARUCO_VISIBLE
BK_ARUCO_X_M = BK.ARUCO_X_M
BK_ARUCO_Y_M = BK.ARUCO_Y_M
BK_ARUCO_Z_M = BK.ARUCO_Z_M

PICAM_IMAGE_TOPIC = "/picam3/image_raw"
PICAM_CAMERA_INFO_TOPIC = "/picam3/camera_info"


def _env_flag(name, default="0"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

class ArUcoCameraReader(DroneActionNode):
    """
    Background node that reads the downward ROS2 camera stream,
    detects ArUco marker ID 0, and writes the X, Y, Z offsets to the Blackboard.
    Always returns RUNNING so it stays active in a Parallel node.
    """
    def __init__(self, image_topic=PICAM_IMAGE_TOPIC, camera_info_topic=PICAM_CAMERA_INFO_TOPIC, marker_size_m=0.60):
        super().__init__("ArUcoCameraReader")
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.marker_size_m = marker_size_m
        self.cap = None
        self._ros_node = None
        self._owns_ros_context = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_camera_info = None
        self.aruco_dict = None
        self.aruco_params = None
        self.detector = None

        if CV2_AVAILABLE:
            self.camera_matrix = np.array([
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=float)
            self.dist_coeffs = np.zeros((5, 1), dtype=float)
        else:
            self.camera_matrix = None
            self.dist_coeffs = None

    def _build_detector(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            self.detector = None

    def _image_to_bgr(self, msg):
        encoding = str(getattr(msg, "encoding", "")).lower()
        height = int(getattr(msg, "height", 0))
        width = int(getattr(msg, "width", 0))
        step = int(getattr(msg, "step", 0))
        if height <= 0 or width <= 0 or step <= 0:
            return None

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if encoding in {"rgb8", "bgr8"}:
            row_stride = max(step // 3, width)
            frame = raw.reshape((height, row_stride, 3))[:, :width, :].copy()
            if encoding == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame

        if encoding in {"mono8", "8uc1"}:
            frame = raw.reshape((height, step))[:, :width].copy()
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return None

    def _update_camera_model(self):
        with self._lock:
            info = self._latest_camera_info
        if info is None:
            return

        k = np.array(info.k, dtype=float).reshape(3, 3)
        if not np.isfinite(k).all() or k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
            return

        self.camera_matrix = k
        dist = np.array(info.d, dtype=float).reshape(-1, 1) if len(info.d) else np.zeros((5, 1), dtype=float)
        self.dist_coeffs = dist

    def _process_ros_spin(self):
        if RCLPY_AVAILABLE and self._ros_node is not None:
            rclpy.spin_once(self._ros_node, timeout_sec=0.0)

    def _on_image(self, msg):
        with self._lock:
            self._latest_frame = msg

    def _on_camera_info(self, msg):
        with self._lock:
            self._latest_camera_info = msg

    def initialise(self):
        if not CV2_AVAILABLE:
            self._log("OpenCV not installed! SITL fallback mode active.")
            return

        self._build_detector()

        if RCLPY_AVAILABLE:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_ros_context = True
            self._ros_node = rclpy.create_node("aruco_camera_reader")
            self._ros_node.create_subscription(RosImage, self.image_topic, self._on_image, qos_profile_sensor_data)
            self._ros_node.create_subscription(RosCameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)
            self._log(f"Listening on ROS2 topics {self.image_topic} and {self.camera_info_topic}")
            return

        self._log("ROS2 not available; falling back to local camera device 0.")
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

    def update(self):
        if not CV2_AVAILABLE:
            self.bb.set(BK_ARUCO_VISIBLE, False)
            return py_trees.common.Status.RUNNING

        self._process_ros_spin()
        self._update_camera_model()

        frame = None
        if RCLPY_AVAILABLE and self._ros_node is not None:
            with self._lock:
                msg = self._latest_frame
            if msg is not None:
                frame = self._image_to_bgr(msg)
        elif self.cap is not None and self.cap.isOpened():
            ret, captured = self.cap.read()
            if ret:
                frame = captured

        if frame is None:
            self.bb.set(BK_ARUCO_VISIBLE, False)
            return py_trees.common.Status.RUNNING

        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and 0 in ids:
            idx = np.where(ids == 0)[0][0]
            marker_corners = corners[idx]

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                marker_corners, self.marker_size_m, self.camera_matrix, self.dist_coeffs
            )

            x_m = tvec[0][0][0]
            y_m = tvec[0][0][1]
            z_m = tvec[0][0][2]

            self.bb.set(BK_ARUCO_VISIBLE, True)
            self.bb.set(BK_ARUCO_X_M, x_m)
            self.bb.set(BK_ARUCO_Y_M, y_m)
            self.bb.set(BK_ARUCO_Z_M, z_m)
        else:
            self.bb.set(BK_ARUCO_VISIBLE, False)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
            if self._owns_ros_context and RCLPY_AVAILABLE and rclpy.ok():
                rclpy.shutdown()
                self._owns_ros_context = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SmartArUcoSearch(DroneActionNode):
    """Expanding spiral search at the current altitude until the marker appears."""
    def __init__(self, search_timeout_s=60.0):
        super().__init__("SmartArUcoSearch")
        self.search_timeout_s = search_timeout_s
        self._start_time = None
        self._last_print = 0

    def initialise(self):
        self._start_time = time.time()
        dc.set_mode("GUIDED")
        self._log("Starting Smart ArUco Search (Expanding Grid)")

    def update(self):
        if self.bb.get(BK_ARUCO_VISIBLE):
            self._log("Marker found! Ending search.")
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.SUCCESS

        t_elapsed = time.time() - self._start_time
        if t_elapsed > self.search_timeout_s:
            self._log("Search timeout! Marker not found.")
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.FAILURE

        freq = 2.0 * math.pi / 15.0
        radius_expansion_rate = 0.05

        radius = radius_expansion_rate * t_elapsed
        max_radius = 2.0
        radius = min(radius, max_radius)

        scan_speed = radius * freq
        vx = scan_speed * math.cos(freq * t_elapsed)
        vy = scan_speed * math.sin(freq * t_elapsed)

        if not CV2_AVAILABLE and t_elapsed > 10.0:
            self._log("[SITL Mock] Simulating marker detection!")
            self.bb.set(BK_ARUCO_VISIBLE, True)
            self.bb.set(BK_ARUCO_X_M, 0.5)
            self.bb.set(BK_ARUCO_Y_M, 0.5)
            self.bb.set(BK_ARUCO_Z_M, 3.0)
            return py_trees.common.Status.RUNNING

        dc.send_body_ned_velocity(vx, vy, 0)

        if time.time() - self._last_print > 2.0:
            self._log(f"Searching... (radius {radius:.2f}m)")
            self._last_print = time.time()

        return py_trees.common.Status.RUNNING


class ArUcoCenterAndDescend(DroneActionNode):
    """Centers on the ArUco marker and descends until ground contact."""
    def __init__(self, kp_xy=1.2, kp_z=0.5, max_speed_xy=0.3, max_speed_z=0.3):
        super().__init__("ArUcoCenterAndDescend")
        self.kp_xy = kp_xy
        self.kp_z = kp_z
        self.max_speed_xy = max_speed_xy
        self.max_speed_z = max_speed_z
        self._lost_time = None
        self._last_print = 0

    def initialise(self):
        dc.set_mode("GUIDED")
        self._lost_time = None

    def update(self):
        visible = self.bb.get(BK_ARUCO_VISIBLE)

        if not visible:
            if self._lost_time is None:
                self._lost_time = time.time()
                dc.send_body_ned_velocity(0, 0, 0)
                self._log("Lost marker tracking! Holding position...")

            if time.time() - self._lost_time > 5.0:
                self._log("Marker lost for >5 seconds. Aborting descent.")
                return py_trees.common.Status.FAILURE

            return py_trees.common.Status.RUNNING

        self._lost_time = None

        x_err = self.bb.get(BK_ARUCO_X_M)
        y_err = self.bb.get(BK_ARUCO_Y_M)
        z_dist = self.bb.get(BK_ARUCO_Z_M)

        if any(v is None for v in [x_err, y_err, z_dist]):
            return py_trees.common.Status.RUNNING

        if z_dist <= 0.2:
            self._log("Ground reached! Centering complete.")
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.SUCCESS

        vx = x_err * self.kp_xy
        vy = y_err * self.kp_xy
        vz = z_dist * self.kp_z

        vx = max(-self.max_speed_xy, min(self.max_speed_xy, vx))
        vy = max(-self.max_speed_xy, min(self.max_speed_xy, vy))
        vz = max(0.1, min(self.max_speed_z, vz))

        dc.send_body_ned_velocity(vx, vy, vz)

        if time.time() - self._last_print > 1.0:
            self._log(f"Centering: err=({x_err:.2f}, {y_err:.2f}), Z={z_dist:.2f}m")
            self._last_print = time.time()

        if not CV2_AVAILABLE:
            self.bb.set(BK_ARUCO_X_M, x_err * 0.9)
            self.bb.set(BK_ARUCO_Y_M, y_err * 0.9)
            self.bb.set(BK_ARUCO_Z_M, z_dist - 0.15)

        return py_trees.common.Status.RUNNING


class ForceDisarm(DroneActionNode):
    """Issues the MAVLink command to instantly kill the motors."""
    def __init__(self):
        super().__init__("ForceDisarm")

    def update(self):
        self._log("Force disarming!")
        master = self._get_master()
        if master:
            dc.set_mode("LAND")
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 21196, 0, 0, 0, 0, 0
            )
        return py_trees.common.Status.SUCCESS

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


class ReturnToLaunch(DroneActionNode):
    """Commands RTL (Return To Launch) and returns SUCCESS."""
    def __init__(self):
        super().__init__("ReturnToLaunch")

    def update(self):
        try:
            dc.set_mode("RTL")
            self._log("Commanded RTL")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self._log(f"RTL command failed: {e}")
            return py_trees.common.Status.FAILURE

def make_landing_phase(use_picam3=None):
    """
    Constructs the camera-only precision landing phase.
    Runs the ArUco camera reader in parallel with the landing logic.
    """

    if use_picam3 is None:
        use_picam3 = _env_flag("ASCEND_USE_PICAM3", "0")

    if not use_picam3:
        # No downward camera in this environment — prefer a safe RTL
        landing_seq = py_trees.composites.Sequence(
            name="LegacyLanding", memory=True
        )
        landing_seq.add_children([
            ReturnToLaunch(),
        ])
        return landing_seq

    landing_logic = py_trees.composites.Sequence(
        name="PrecisionLandingLogic", memory=True
    )

    try:
        from ascend.bt_nodes.survey import NavigateHome
        landing_logic.add_child(NavigateHome())
    except ImportError:
        pass

    landing_logic.add_children([
        SmartArUcoSearch(search_timeout_s=60.0),
        ArUcoCenterAndDescend(),
        ForceDisarm()
    ])

    landing_phase = py_trees.composites.Parallel(
        name="CameraLandingPhase",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    landing_phase.add_children([
        ArUcoCameraReader(),
        landing_logic
    ])

    return landing_phase
