import py_trees
import drone_control as dc
from . import blackboard_keys as BK
from .base import DroneActionNode
import time

BATT_LOW_V = 16.0      # 4S: 4 × 4.0V
BATT_CRIT_V = 15.2     # 4S: 4 × 3.8V
ALT_MAX_M = 6.5        # hard ceiling
ALT_MIN_M = 1.5        # during survey

class BatteryMonitor(DroneActionNode):
    """
    RUNNING → voltage OK
    FAILURE → voltage below threshold (triggers BT preemption)
    """
    def __init__(self):
        super().__init__("BatteryMonitor")
        self._low_hits = 0
        self._CONFIRM = 3  # consecutive reads before triggering

    def update(self):
        master = self._get_master()
        if master is None:
            return py_trees.common.Status.RUNNING

        latest_v = None
        while True:
            msg = master.recv_match(
                type=["SYS_STATUS", "BATTERY_STATUS"],
                blocking=False
            )
            if msg is None:
                break
            v = self._extract_voltage(msg)
            if v is not None:
                latest_v = v

        if latest_v is None:
            return py_trees.common.Status.RUNNING

        self.bb.set(BK.BATTERY_V, latest_v)

        if latest_v <= BATT_CRIT_V:
            self._log(f"CRITICAL battery {latest_v:.2f}V — LAND NOW")
            dc.set_mode("LAND")
            return py_trees.common.Status.FAILURE

        if latest_v <= BATT_LOW_V:
            self._low_hits += 1
            if self._low_hits >= self._CONFIRM:
                self._log(f"LOW battery {v:.2f}V ({self._low_hits} hits) — RTL")
                dc.set_mode("RTL")
                return py_trees.common.Status.FAILURE
        else:
            if self._low_hits > 0:
                self._log(f"Battery recovered to {v:.2f}V. Clearing low-voltage counter.")
            self._low_hits = 0

        return py_trees.common.Status.RUNNING

    def _extract_voltage(self, msg):
        t = msg.get_type()
        if t == "SYS_STATUS":
            mv = getattr(msg, "voltage_battery", None)
            return mv / 1000.0 if mv and mv > 0 else None
        if t == "BATTERY_STATUS":
            cells = getattr(msg, "voltages", [])
            valid = [mv for mv in cells if 0 < mv < 6000]
            if valid:
                return sum(valid) / 1000.0
            
            # Fallback for stacks that only send total voltage
            mv = getattr(msg, "voltage", None)
            if mv is not None and int(mv) > 0:
                return int(mv) / 1000.0
        return None

class AltitudeMonitor(DroneActionNode):
    """Preempts if altitude leaves safe band."""
    def __init__(self):
        super().__init__("AltitudeMonitor")
        
    def update(self):
        master = self._get_master()
        if master is None:
            return py_trees.common.Status.RUNNING

        latest_alt = None
        while True:
            msg = master.recv_match(
                type=["RANGEFINDER", "DISTANCE_SENSOR", "NAMED_VALUE_FLOAT", "GLOBAL_POSITION_INT"], 
                blocking=False
            )
            if msg is None:
                break
            alt = self._extract_altitude(msg)
            if alt is not None:
                latest_alt = alt

        if latest_alt is None:
            return py_trees.common.Status.RUNNING

        self.bb.set(BK.ALTITUDE_M, latest_alt)

        if latest_alt > ALT_MAX_M:
            self._log(f"OVER ceiling {latest_alt:.2f}m — LAND")
            dc.set_mode("LAND")
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING
    
    def _extract_altitude(self, msg):
        msg_type = msg.get_type()
        if msg_type == "GLOBAL_POSITION_INT":
            return float(msg.relative_alt) / 1000.0
        if msg_type == "RANGEFINDER":
            distance = getattr(msg, "distance", None)
            return float(distance) if distance is not None else None
        if msg_type == "DISTANCE_SENSOR":
            current_cm = getattr(msg, "current_distance", None)
            return float(current_cm) / 100.0 if current_cm is not None else None
        if msg_type == "NAMED_VALUE_FLOAT":
            raw_name = getattr(msg, "name", "")
            name = str(raw_name).strip().lower()
            if name in {"rangefinder1", "rangefinder", "rngfnd1", "rngfnd"}:
                value = getattr(msg, "value", None)
                if value is not None:
                    value_f = float(value)
                    if value_f > 10.0:
                        return value_f / 100.0
                    return value_f
        return None

def make_safety_guard():
    """
    Returns a Parallel node that runs BatteryMonitor + AltitudeMonitor
    alongside whatever is happening. One FAILURE = whole parallel FAILS.
    """
    guard = py_trees.composites.Parallel(
        name="SafetyGuard",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(
            synchronise=False
        )
    )
    guard.add_children([BatteryMonitor(), AltitudeMonitor()])
    return guard
