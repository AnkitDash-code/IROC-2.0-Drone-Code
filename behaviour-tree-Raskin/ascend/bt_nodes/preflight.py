import py_trees
import drone_control as dc
from . import blackboard_keys as BK
from .base import DroneActionNode
from pymavlink import mavutil
import time
import select
import sys

class ConnectVehicle(DroneActionNode):
    def __init__(self, connection_string="/dev/ttyACM0", baud=57600):
        super().__init__("ConnectVehicle")
        self._conn = connection_string
        self._baud = baud

    def update(self):
        try:
            # Set module-level globals in drone_control
            dc.CONNECTION_STRING = self._conn
            dc.BAUD_RATE = self._baud
            dc.connect_to_vehicle()
            self._log(f"Connected to {self._conn} @ {self._baud}")
            
            # Force ArduPilot to send us telemetry data streams
            master = dc.master
            if master:
                master.mav.request_data_stream_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1
                )
            
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self._log(f"Connect failed: {e}")
            return py_trees.common.Status.FAILURE

class WaitRangefinder(DroneActionNode):
    def __init__(self, timeout_s=10.0):
        super().__init__("WaitRangefinder")
        self._timeout = timeout_s
        self._deadline = None
        self._next_req = 0.0

    def initialise(self):
        self._deadline = time.time() + self._timeout
        self._next_req = 0.0

    def _request_stream(self):
        master = self._get_master()
        if master is None:
            return
        for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_RANGEFINDER, mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR):
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, 100000, 0, 0, 0, 0, 0
            )
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA3, 10, 1
        )

    def update(self):
        if time.time() > self._deadline:
            self._log("Rangefinder timeout")
            return py_trees.common.Status.FAILURE

        if time.time() > self._next_req:
            self._request_stream()
            self._next_req = time.time() + 2.0

        # Note: AltitudeMonitor is already running in parallel and putting ALTITUDE_M on the BB.
        # So we can just check the blackboard instead of polling MAVLink directly!
        alt = self.bb.get(BK.ALTITUDE_M)
        if alt is not None:
            self._log(f"Rangefinder OK: {alt:.2f}m")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

class WaitBatteryTelemetry(DroneActionNode):
    def __init__(self, timeout_s=10.0):
        super().__init__("WaitBatteryTelemetry")
        self._timeout = timeout_s
        self._deadline = None
        self._next_req = 0.0

    def initialise(self):
        self._deadline = time.time() + self._timeout
        self._next_req = 0.0

    def _request_stream(self):
        master = self._get_master()
        if master is None:
            return
        for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS):
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, 200000, 0, 0, 0, 0, 0
            )
        master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 5, 1
        )

    def update(self):
        if time.time() > self._deadline:
            self._log("Battery telemetry timeout")
            return py_trees.common.Status.FAILURE

        if time.time() > self._next_req:
            self._request_stream()
            self._next_req = time.time() + 2.0

        v = self.bb.get(BK.BATTERY_V)
        if v is not None:
            self._log(f"Battery Telemetry OK: {v:.2f}V")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

class GroundCheck(DroneActionNode):
    """Confirms drone is on ground (alt < 0.25m) for 2 continuous seconds."""
    def __init__(self):
        super().__init__("GroundCheck")
        self._on_ground_since = None

    def initialise(self):
        self._on_ground_since = None

    def update(self):
        alt = self.bb.get(BK.ALTITUDE_M)
        if alt is None or alt > 0.25:
            self._on_ground_since = None
            return py_trees.common.Status.RUNNING

        if self._on_ground_since is None:
            self._on_ground_since = time.time()

        if time.time() - self._on_ground_since >= 2.0:
            self._log("Ground confirmed")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

class OperatorConfirm(DroneActionNode):
    """Blocks until operator types the token, non-blocking to BT tick."""
    def __init__(self, token: str):
        super().__init__(f"Confirm:{token}")
        self.expected = token
        self._prompted = False

    def initialise(self):
        self._prompted = False

    def update(self):
        import sys
        import platform

        if not self._prompted:
            print(f"\n[CONFIRM] Type '{self.expected}' to continue: ", end="", flush=True)
            self._prompted = True
            
        has_input = False
        if platform.system() == "Windows":
            import msvcrt
            if msvcrt.kbhit():
                has_input = True
        else:
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                has_input = True
                
        if has_input:
            line = sys.stdin.readline().strip()
            if line == self.expected:
                self._log(f"User confirmed '{self.expected}'.")
                return py_trees.common.Status.SUCCESS
            else:
                self._prompted = False # Prompt again

        return py_trees.common.Status.RUNNING
