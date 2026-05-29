import py_trees
import drone_control as dc
from . import blackboard_keys as BK
from .base import DroneActionNode
import time

BATT_LOW_V = 16.0      # 4S: 4 × 4.0V
BATT_CRIT_V = 15.2     # 4S: 4 × 3.8V
ALT_MAX_M = 6.0        # hard ceiling (updated per request)
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
    # Add an occupancy-map based fence monitor if available
    try:
        fence = MapFenceMonitor()
        guard.add_children([BatteryMonitor(), AltitudeMonitor(), fence])
    except Exception:
        # If MapFenceMonitor cannot be constructed (rclpy missing etc), fall back
        guard.add_children([BatteryMonitor(), AltitudeMonitor()])
    return guard


class MapFenceMonitor(DroneActionNode):
    """Checks the nav occupancy grid (/local_costmap/costmap or /map) and
    preempts the mission (FAIL) if the current or planned position is inside
    an occupied cell (fence/obstacle). This uses `rclpy` if available; if not,
    the monitor disables itself gracefully.
    """
    def __init__(self):
        super().__init__("MapFenceMonitor")
        self._enabled = False
        self._grid = None

        # Try to create a lightweight rclpy subscriber in a background thread.
        try:
            import rclpy
            from rclpy.node import Node
            from nav_msgs.msg import OccupancyGrid
        except Exception:
            self._log("rclpy/nav_msgs not available — MapFenceMonitor disabled")
            return

        self._enabled = True

        # Create a simple rclpy node in a background thread so this BT node
        # remains synchronous. We only subscribe and store the latest grid.
        import threading

        class _MapNode(Node):
            def __init__(self, outer):
                super().__init__("map_fence_listener")
                self._outer = outer
                # try local_costmap first, then /map
                self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._cb, 1)
                self.create_subscription(OccupancyGrid, '/map', self._cb, 1)

            def _cb(self, msg: OccupancyGrid):
                # store latest grid on outer
                self._outer._grid = msg

        def _spin_thread():
            try:
                rclpy.init()
            except Exception:
                pass
            node = _MapNode(self)
            try:
                rclpy.spin(node)
            except Exception:
                pass

        t = threading.Thread(target=_spin_thread, daemon=True)
        t.start()

    def update(self):
        if not self._enabled:
            return py_trees.common.Status.RUNNING

        # Need a position to check
        pos = self.bb.get(BK.POSITION_NED)
        if pos is None:
            return py_trees.common.Status.RUNNING

        if self._grid is None:
            # No map yet
            return py_trees.common.Status.RUNNING

        try:
            x, y, _z = pos
            info = self._grid.info
            ox = info.origin.position.x
            oy = info.origin.position.y
            res = info.resolution
            w = info.width
            h = info.height

            mx = int((x - ox) / res)
            my = int((y - oy) / res)
            if mx < 0 or my < 0 or mx >= w or my >= h:
                # Outside current map window — be conservative and continue
                return py_trees.common.Status.RUNNING

            idx = my * w + mx
            cell = self._grid.data[idx]
            # Occupied cells in nav maps are usually >=50
            if cell >= 50:
                self._log(f"Map fence breach at ({x:.2f},{y:.2f}) -> occupied cell={cell}")
                try:
                    dc.set_mode("RTL")
                except Exception:
                    pass
                return py_trees.common.Status.FAILURE

        except Exception as e:
            self._log(f"MapFenceMonitor error: {e}")

        return py_trees.common.Status.RUNNING
