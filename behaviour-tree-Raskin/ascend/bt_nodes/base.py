import py_trees
import drone_control as dc
from pymavlink import mavutil
import time

class DroneActionNode(py_trees.behaviour.Behaviour):
    """
    Base for all drone action nodes.
    Holds shared state via the BT Blackboard — one source of truth
    for altitude, battery voltage, position, detection list, etc.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.bb = py_trees.blackboard.Client(name=name)
        from . import blackboard_keys as BK
        for key in dir(BK):
            if not key.startswith("_"):
                val = getattr(BK, key)
                if isinstance(val, str):
                    self.bb.register_key(key=val, access=py_trees.common.Access.READ)
                    self.bb.register_key(key=val, access=py_trees.common.Access.WRITE)
                    if not self.bb.exists(val):
                        self.bb.set(val, None)
        self.bb.register_key(key="/detection/candidate", access=py_trees.common.Access.READ)
        self.bb.register_key(key="/detection/candidate", access=py_trees.common.Access.WRITE)
        if not self.bb.exists("/detection/candidate"):
            self.bb.set("/detection/candidate", None)

    def setup(self, **kwargs):
        # Called once before the tree ticks. Wire up blackboard keys here.
        pass

    def _log(self, msg: str):
        print(f"[BT:{self.name}] {msg}")

    def _get_master(self):
        return dc.master
