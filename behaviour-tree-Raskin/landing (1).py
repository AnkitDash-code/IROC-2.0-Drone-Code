import py_trees
import drone_control as dc
from . import blackboard_keys as BK
from .base import DroneActionNode
import time
import math

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Keys for Blackboard specific to ArUco
BK_ARUCO_VISIBLE = "/landing/aruco_visible"
BK_ARUCO_X_M = "/landing/aruco_x_m"
BK_ARUCO_Y_M = "/landing/aruco_y_m"
BK_ARUCO_Z_M = "/landing/aruco_z_m"

class ArUcoCameraReader(DroneActionNode):
    """
    Background node (like AltitudeMonitor) that constantly reads the downward camera,
    detects ArUco marker ID 0, and writes the X, Y, Z offsets to the Blackboard.
    Always returns RUNNING so it stays active in a Parallel node.
    """
    def __init__(self, camera_index=0, marker_size_m=0.60):
        super().__init__("ArUcoCameraReader")
        self.camera_index = camera_index
        self.marker_size_m = marker_size_m
        self.cap = None
        self.aruco_dict = None
        self.aruco_params = None
        
        # Placeholder camera intrinsics (should be calibrated in real life)
        if CV2_AVAILABLE:
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=float)
            self.dist_coeffs = np.zeros((4,1))

    def initialise(self):
        if not CV2_AVAILABLE:
            self._log("OpenCV not installed! SITL fallback mode active.")
            return

        if self.cap is None:
            self._log(f"Opening camera {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            # Ensure dictionary is correct for IROC (DICT_4X4_50 is standard)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            # If using newer OpenCV 4.7+, might need: cv2.aruco.ArucoDetector
            try:
                self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            except AttributeError:
                self.detector = None

    def update(self):
        if not CV2_AVAILABLE or self.cap is None or not self.cap.isOpened():
            # In SITL without a real camera, we simulate NOT seeing it to test timeouts,
            # or we could simulate seeing it directly below. For now, simulate NOT visible.
            self.bb.set(BK_ARUCO_VISIBLE, False)
            return py_trees.common.Status.RUNNING

        ret, frame = self.cap.read()
        if not ret:
            self.bb.set(BK_ARUCO_VISIBLE, False)
            return py_trees.common.Status.RUNNING

        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            # Fallback for older OpenCV
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and 0 in ids:
            # Get the index of marker ID 0
            idx = np.where(ids == 0)[0][0]
            marker_corners = corners[idx]

            # Estimate pose
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                marker_corners, self.marker_size_m, self.camera_matrix, self.dist_coeffs
            )
            
            # tvec is [[x, y, z]]
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
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SmartArUcoSearch(DroneActionNode):
    """
    If the ArUco marker is not visible, executes an expanding square grid search 
    at the current altitude until it becomes visible.
    """
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
        # 1. Success condition: Marker found!
        if self.bb.get(BK_ARUCO_VISIBLE):
            self._log("Marker found! Ending search.")
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.SUCCESS

        # 2. Timeout condition
        t_elapsed = time.time() - self._start_time
        if t_elapsed > self.search_timeout_s:
            self._log("Search timeout! Marker not found.")
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.FAILURE

        # 3. Search Pattern: Archimedean Spiral via Velocity
        # (A smooth spiral is continuous and easier to execute via velocity than a rigid square grid)
        # v_x = A * t * cos(wt), v_y = A * t * sin(wt)
        freq = 2.0 * math.pi / 15.0  # 15 seconds per revolution
        radius_expansion_rate = 0.05 # Expands 5cm per second
        
        radius = radius_expansion_rate * t_elapsed
        max_radius = 2.0 # Don't search further than 2 meters away
        radius = min(radius, max_radius)
        
        scan_speed = radius * freq
        vx = scan_speed * math.cos(freq * t_elapsed)
        vy = scan_speed * math.sin(freq * t_elapsed)
        
        # In SITL without OpenCV, we simulate finding it after 10 seconds of searching
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
    """
    Reads X, Y, Z from the ArUco marker via the Blackboard.
    Simultaneously centers (X, Y) and descends (Z) until Z < 0.2m.
    """
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
        
        # Handle lost marker
        if not visible:
            if self._lost_time is None:
                self._lost_time = time.time()
                dc.send_body_ned_velocity(0, 0, 0) # Stop moving
                self._log("Lost marker tracking! Holding position...")
            
            if time.time() - self._lost_time > 5.0:
                self._log("Marker lost for >5 seconds. Aborting descent.")
                return py_trees.common.Status.FAILURE
                
            return py_trees.common.Status.RUNNING
        
        # Marker is visible
        self._lost_time = None
        
        x_err = self.bb.get(BK_ARUCO_X_M)
        y_err = self.bb.get(BK_ARUCO_Y_M)
        z_dist = self.bb.get(BK_ARUCO_Z_M)
        
        if any(v is None for v in [x_err, y_err, z_dist]):
            return py_trees.common.Status.RUNNING

        # 1. Success condition: Ground reached!
        if z_dist <= 0.2:
            self._log("Ground reached! Centering complete.")
            dc.send_body_ned_velocity(0, 0, 0)
            return py_trees.common.Status.SUCCESS

        # 2. Calculate velocities
        vx = x_err * self.kp_xy
        vy = y_err * self.kp_xy
        vz = z_dist * self.kp_z  # Positive Z is downward in NED
        
        # Clamp velocities
        vx = max(-self.max_speed_xy, min(self.max_speed_xy, vx))
        vy = max(-self.max_speed_xy, min(self.max_speed_xy, vy))
        vz = max(0.1, min(self.max_speed_z, vz)) # Always descend at least 0.1m/s
        
        # Send to drone
        dc.send_body_ned_velocity(vx, vy, vz)
        
        if time.time() - self._last_print > 1.0:
            self._log(f"Centering: err=({x_err:.2f}, {y_err:.2f}), Z={z_dist:.2f}m")
            self._last_print = time.time()

        # In SITL mock, artificially decrease Z and errors over time
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
            dc.set_mode("LAND") # Backup
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 21196, 0, 0, 0, 0, 0
            )
        return py_trees.common.Status.SUCCESS


class LogMissionReport(DroneActionNode):
    def __init__(self):
        super().__init__("LogMissionReport")
        
    def update(self):
        self._log("Mission Complete. Saved logs.")
        return py_trees.common.Status.SUCCESS


def make_landing_phase():
    """
    Constructs the new Camera-Only Precision Landing phase.
    Wraps the action nodes alongside the ArUcoCameraReader in a Parallel node.
    """
    
    # The actual sequential logic of landing
    landing_logic = py_trees.composites.Sequence(
        name="PrecisionLandingLogic", memory=True
    )
    
    # Import NavigateHome from survey to go to (0,0) first
    try:
        from ascend.bt_nodes.survey import NavigateHome
        landing_logic.add_child(NavigateHome())
    except ImportError:
        pass # If survey.py doesn't exist yet in testing, skip
        
    landing_logic.add_children([
        SmartArUcoSearch(search_timeout_s=60.0),
        ArUcoCenterAndDescend(),
        ForceDisarm()
    ])
    
    # We run the Camera Reader in parallel with the landing logic.
    # ParallelPolicy.SuccessOnOne() means when landing_logic finishes (success or failure),
    # the camera reader automatically shuts down and releases the webcam!
    landing_phase = py_trees.composites.Parallel(
        name="CameraLandingPhase",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    landing_phase.add_children([
        ArUcoCameraReader(camera_index=0),
        landing_logic
    ])

    return landing_phase
