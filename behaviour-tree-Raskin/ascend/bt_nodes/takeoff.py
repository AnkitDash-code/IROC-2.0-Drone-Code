import py_trees
import drone_control as dc
from pymavlink import mavutil
from . import blackboard_keys as BK
from .base import DroneActionNode
import time

class SetMode(DroneActionNode):
    def __init__(self, mode: str):
        super().__init__(f"SetMode({mode})")
        self._mode = mode

    def update(self):
        try:
            dc.set_mode(self._mode)
            self._log(f"Mode set to {self._mode}")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self._log(f"Failed to set mode: {e}")
            return py_trees.common.Status.FAILURE

class ArmVehicle(DroneActionNode):
    def __init__(self):
        super().__init__("ArmVehicle")
        self._armed = False

    def update(self):
        try:
            if not self._armed:
                self._log("Arming vehicle...")
                dc.arm_vehicle()
                self._armed = True
                self.bb.set(BK.ARMED, True)
                self._log("Armed")
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self._log(f"Arm failed: {e}")
            return py_trees.common.Status.FAILURE

class SetSlowTakeoffParams(DroneActionNode):
    def __init__(self):
        super().__init__("SetSlowTakeoffParams")
        self._saved_params = {}

    def update(self):
        master = self._get_master()
        if master is None:
            return py_trees.common.Status.FAILURE
            
        speeds = [
            ("WPNAV_SPEED_UP", 50.0),
            ("WPNAV_ACCEL_Z", 30.0),
            ("PILOT_SPEED_UP", 50.0),
            ("PILOT_ACCEL_Z", 30.0)
        ]
        
        for name, slow_val in speeds:
            # Get current
            master.mav.param_request_read_send(
                master.target_system, master.target_component,
                name.encode("utf-8"), -1
            )
            msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg:
                self._saved_params[name] = float(msg.param_value)
                
            # Set slow
            master.mav.param_set_send(
                master.target_system, master.target_component,
                name.encode("utf-8"), float(slow_val),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            
        self._log("Slow takeoff params set")
        return py_trees.common.Status.SUCCESS
        
    def terminate(self, new_status):
        # Restore when node leaves active tree
        master = self._get_master()
        if master and self._saved_params:
            for name, val in self._saved_params.items():
                master.mav.param_set_send(
                    master.target_system, master.target_component,
                    name.encode("utf-8"), float(val),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                )
            self._log("Restored original climb params")
            self._saved_params.clear()

class CommandTakeoff(DroneActionNode):
    def __init__(self, altitude_m: float = 3.0):
        super().__init__(f"Takeoff({altitude_m}m)")
        self._alt = altitude_m

    def update(self):
        master = self._get_master()
        if master is None:
            return py_trees.common.Status.FAILURE
            
        if self._alt < 0.5 or self._alt > 6.0:
            self._log(f"Alt {self._alt} out of bounds (0.5-6.0)")
            return py_trees.common.Status.FAILURE

        self._log(f"Requesting takeoff to {self._alt}m...")
        # Send MAV_CMD_NAV_TAKEOFF directly to avoid the time.sleep() inside dc.takeoff()
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, self._alt
        )
        return py_trees.common.Status.SUCCESS

class WaitAltitudeReached(DroneActionNode):
    def __init__(self, target_m: float, tolerance_m: float = 0.15, timeout_s: float = 45):
        super().__init__(f"WaitAlt({target_m}m)")
        self._target = target_m
        self._tol = tolerance_m
        self._timeout = timeout_s
        self._deadline = None

    def initialise(self):
        self._deadline = time.time() + self._timeout

    def update(self):
        if time.time() > self._deadline:
            self._log("Takeoff timeout — triggering LAND")
            dc.set_mode("LAND")
            return py_trees.common.Status.FAILURE

        alt = self.bb.get(BK.ALTITUDE_M)
        if alt is None:
            return py_trees.common.Status.RUNNING

        self._log(f"Alt: {alt:.2f}m / {self._target:.2f}m")
        if alt >= self._target - self._tol:
            self._log("Target altitude reached. Maintaining GUIDED mode hover.")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

class Hover(DroneActionNode):
    """Hovers in place for a specified duration."""
    def __init__(self, duration_s=300):
        super().__init__(f"Hover({duration_s}s)")
        self.duration_s = duration_s
        self._end_time = None
        self._last_print = 0

    def initialise(self):
        self._end_time = time.time() + self.duration_s
        self._log(f"Starting hover for {self.duration_s} seconds")

    def update(self):
        rem = self._end_time - time.time()
        if rem <= 0:
            self._log("Hover complete!")
            return py_trees.common.Status.SUCCESS
        
        # Print every 10 seconds so we know it's alive
        now = time.time()
        if now - self._last_print >= 10.0:
            self._log(f"Hovering... {int(rem)}s remaining")
            self._last_print = now
            
        return py_trees.common.Status.RUNNING
