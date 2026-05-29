import py_trees
import drone_control as dc
from . import blackboard_keys as BK
from .base import DroneActionNode
import os
import time
import json
import datetime
from pathlib import Path
import threading

try:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String as RosString
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False

ARENA_X_M = 10.67  # 35 ft
ARENA_Y_M = 7.62   # 25 ft
LANE_SPACING = 1.5 # meters between lawnmower lanes
SURVEY_SPEED = 0.5
MAX_DETECTIONS = 3
DETECTIONS_LOG_PATH = os.environ.get(
    "ASCEND_DETECTIONS_LOG",
    str(Path.home() / "Drone" / "detections.jsonl")
)

class GenerateWaypoints(DroneActionNode):
    """Runs once. Writes lawnmower waypoints to blackboard."""
    def __init__(self):
        super().__init__("GenerateWaypoints")

    def update(self):
        wps = []
        y, direction = 0.0, 1
        while y <= ARENA_Y_M:
            wps.append((ARENA_X_M if direction == 1 else 0.0, y))
            y += LANE_SPACING
            direction *= -1
        
        self.bb.set(BK.SURVEY_WAYPOINTS, wps)
        self.bb.set(BK.CURRENT_WP_IDX, 0)
        self.bb.set(BK.DETECTIONS, [])
        self._log(f"{len(wps)} waypoints generated")
        return py_trees.common.Status.SUCCESS

class NextWaypoint(DroneActionNode):
    """
    Advances waypoint index.
    FAILURE when all waypoints exhausted → survey done.
    """
    def __init__(self):
        super().__init__("NextWaypoint")

    def update(self):
        wps = self.bb.get(BK.SURVEY_WAYPOINTS)
        idx = self.bb.get(BK.CURRENT_WP_IDX)
        detections = self.bb.get(BK.DETECTIONS) or []

        if len(detections) >= MAX_DETECTIONS:
            self._log("All seeds found — ending survey")
            return py_trees.common.Status.FAILURE

        if idx >= len(wps):
            self._log("All waypoints done")
            return py_trees.common.Status.FAILURE

        tx, ty = wps[idx]
        self._log(f"WP {idx+1}/{len(wps)}: ({tx:.1f}, {ty:.1f})")
        self.bb.set(BK.CURRENT_WP_IDX, idx + 1)
        return py_trees.common.Status.SUCCESS

class MoveToWaypoint(DroneActionNode):
    """
    Rotates toward current waypoint then moves.
    Uses rotate_towards() + move_body_ned() from drone_control.
    Splits long moves to allow BT ticks between segments.
    """
    def __init__(self):
        super().__init__("MoveToWaypoint")
        self._rotated = False
        self._remaining_dist = 0.0

    def initialise(self):
        self._rotated = False
        self._remaining_dist = 0.0
        dc.set_mode("GUIDED")

    def update(self):
        wps = self.bb.get(BK.SURVEY_WAYPOINTS)
        # Already incremented by NextWaypoint, so index is -1
        idx = self.bb.get(BK.CURRENT_WP_IDX) - 1 

        if wps is None or idx < 0:
            return py_trees.common.Status.FAILURE

        tx, ty = wps[idx]
        try:
            
            if not self._rotated:
                self._remaining_dist = dc.rotate_towards(tx, ty)
                self._rotated = True
                return py_trees.common.Status.RUNNING

            if self._remaining_dist <= 0.0:
                return py_trees.common.Status.SUCCESS

            # Chunk moves into max 3.0m segments so the sleep in move_body_ned 
            # doesn't completely block the BT for too long.
            move_chunk = min(3.0, self._remaining_dist)
            dc.move_body_ned(move_chunk, speed_mps=SURVEY_SPEED)
            
            self._remaining_dist -= move_chunk
            
            if self._remaining_dist <= 0.1:
                return py_trees.common.Status.SUCCESS
                
            return py_trees.common.Status.RUNNING
            
        except Exception as e:
            self._log(f"Move failed: {e}")
            return py_trees.common.Status.FAILURE

class YOLODetect(DroneActionNode):
    """
    STUB — returns FAILURE (no detection) until OAK-D arrives.
    TODO: call OAK-D RGB pipeline, run YOLOv8n TensorRT
    If detection found, write candidate to blackboard.
    """
    def __init__(self):
        super().__init__("YOLODetect")
        
    def update(self):
        return py_trees.common.Status.FAILURE # stub

class VLMVerify(DroneActionNode):
    """
    STUB — always FAILURE until Moondream 2.5 is wired.
    TODO: read candidate from blackboard, send to VLM.
    Return SUCCESS if confirmed, FAILURE if rejected.
    """
    def __init__(self):
        super().__init__("VLMVerify")

    def update(self):
        return py_trees.common.Status.FAILURE # stub

class CheckMissionDetections(DroneActionNode):
    """Bridges `/detection/candidate` into the BT blackboard."""
    def __init__(self, topic="/detection/candidate"):
        super().__init__("CheckMissionDetections")
        self.topic = topic
        self._ros_node = None
        self._owns_ros_context = False
        self._latest_candidate = None
        self._lock = threading.Lock()
        self._grid = None

    def initialise(self):
        if not RCLPY_AVAILABLE:
            self._log("ROS2 not available; detection bridge disabled.")
            return

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_ros_context = True

        self._ros_node = rclpy.create_node("mission_detection_bridge")
        self._ros_node.create_subscription(RosString, self.topic, self._on_candidate, qos_profile_sensor_data)
        self._log(f"Listening for detections on {self.topic}")

        # Try to also subscribe to an occupancy grid so we can sanity-check candidate positions
        try:
            from nav_msgs.msg import OccupancyGrid
            self._ros_node.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._on_grid, 1)
            self._ros_node.create_subscription(OccupancyGrid, '/map', self._on_grid, 1)
            self._log("Subscribed to occupancy grid for detection gating")
        except Exception:
            pass

    def _on_candidate(self, msg):
        try:
            candidate = json.loads(msg.data)
        except Exception:
            candidate = {"raw": msg.data}
        with self._lock:
            self._latest_candidate = candidate

    def _on_grid(self, msg):
        # store latest grid
        self._grid = msg

    def update(self):
        if RCLPY_AVAILABLE and self._ros_node is not None:
            rclpy.spin_once(self._ros_node, timeout_sec=0.0)

        with self._lock:
            candidate = self._latest_candidate

        if candidate is None:
            return py_trees.common.Status.RUNNING

        # If candidate contains a position, sanity-check against the occupancy grid
        if candidate is not None and isinstance(candidate, dict):
            pos = candidate.get("pos") or {}
            px = pos.get("x")
            py = pos.get("y")
            if px is not None and py is not None and self._grid is not None:
                try:
                    info = self._grid.info
                    ox = info.origin.position.x
                    oy = info.origin.position.y
                    res = info.resolution
                    w = info.width
                    h = info.height
                    mx = int((px - ox) / res)
                    my = int((py - oy) / res)
                    if 0 <= mx < w and 0 <= my < h:
                        idx = my * w + mx
                        cell = self._grid.data[idx]
                        if cell >= 50:
                            self._log("Ignoring detection inside occupied map cell (likely fence)")
                            # drop candidate
                            candidate = None
                except Exception:
                    pass

        if candidate is not None:
            self.bb.set("/detection/candidate", candidate)
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        if self._owns_ros_context and RCLPY_AVAILABLE and rclpy.ok():
            rclpy.shutdown()
            self._owns_ros_context = False

class LogDetection(DroneActionNode):
    """Writes confirmed detection to JSONL log and blackboard."""
    def __init__(self):
        super().__init__("LogDetection")

    def update(self):
        candidate = self.bb.get("/detection/candidate")
        if candidate is None:
            return py_trees.common.Status.FAILURE

        detections = self.bb.get(BK.DETECTIONS) or []
        entry = {
            "id": len(detections) + 1,
            "class": candidate.get("class", "unknown"),
            "pos": candidate.get("pos", {}),
            "conf": candidate.get("conf", 0.0),
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "img": candidate.get("img", ""),
        }
        detections.append(entry)
        self.bb.set(BK.DETECTIONS, detections)

        Path(DETECTIONS_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(DETECTIONS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._log(f"Detection #{entry['id']}: {entry['class']} @ {entry['pos']}")
        return py_trees.common.Status.SUCCESS

class NavigateHome(DroneActionNode):
    """Returns to (0,0)"""
    def __init__(self):
        super().__init__("NavigateHome")
        
    def initialise(self):
        self._done = False
        dc.set_mode("GUIDED")

    def update(self):
        if self._done:
            return py_trees.common.Status.SUCCESS

        try:
            dist = dc.rotate_towards(0, 0)
            dc.move_body_ned(dist, speed_mps=SURVEY_SPEED)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            self._log(f"RTB failed: {e}")
            return py_trees.common.Status.FAILURE

def make_survey_phase():
    """
    Survey phase as a py_trees Sequence.
    SurveyLoop = Sequence(NextWaypoint, Move, DetectionAttempt)
    repeated until NextWaypoint returns FAILURE.
    """
    # Detection attempt sequence: try to detect, verify, then log.
    detection_seq = py_trees.composites.Sequence(
        name="DetectionAttempt", memory=False
    )
    detection_seq.add_children([YOLODetect(), VLMVerify(), LogDetection()])

    # We wrap the detection sequence in a Selector that always succeeds if detection fails.
    # This prevents a failed detection from failing the whole waypoint sequence.
    # A dummy node returning SUCCESS is the fallback.
    class AlwaysSuccess(py_trees.behaviour.Behaviour):
        def update(self):
            return py_trees.common.Status.SUCCESS
            
    detection_selector = py_trees.composites.Selector(name="DetectionSelector", memory=False)
    detection_selector.add_children([detection_seq, AlwaysSuccess(name="NoDetection")])

    wp_seq = py_trees.composites.Sequence(
        name="WaypointExecution", memory=True
    )
    wp_seq.add_children([
        NextWaypoint(),
        MoveToWaypoint(),
        detection_selector,
    ])

    survey_loop_repeat = py_trees.decorators.Repeat(
        child=wp_seq,
        name="SurveyLoop",
        num_success=-1 # repeat forever until child FAILS (NextWaypoint fails)
    )
    
    survey_loop = py_trees.decorators.FailureIsSuccess(
        child=survey_loop_repeat,
        name="SurveyLoopDone"
    )

    survey_phase = py_trees.composites.Sequence(
        name="SurveyPhase", memory=True
    )
    survey_phase.add_children([GenerateWaypoints(), survey_loop])
    return survey_phase
