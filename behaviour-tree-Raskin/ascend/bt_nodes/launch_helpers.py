import py_trees
import subprocess
import os
from .base import DroneActionNode


class StartNav2(DroneActionNode):
    """Starts the local Nav2 costmap stack (depthimage_to_laserscan + nav2_costmap_2d + lifecycle manager).
    Runs once and returns SUCCESS after launching processes. Logs go to /tmp/nav2.log.
    """
    def __init__(self):
        super().__init__("StartNav2")
        self._started = False

    def update(self):
        if self._started:
            return py_trees.common.Status.SUCCESS

        nav2_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            "source /home/rock/IROC-2.0-Drone-Code/ros2_ws/install/setup.bash && "
            "ros2 run depthimage_to_laserscan depthimage_to_laserscan_node \
"
            "    --ros-args -r depth:=/oakd/depth/image_raw -r depth_camera_info:=/oakd/left/camera_info \
"
            "    -p output_frame:=oakd_link -p range_min:=0.2 -p range_max:=10.0 -p use_sim_time:=true & "
            "sleep 2 && "
            "ros2 run nav2_costmap_2d nav2_costmap_2d --ros-args -r __node:=local_costmap -r __ns:=/local_costmap --params-file /home/rock/nav2_params.yaml & "
            "sleep 2 && "
            "ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -p autostart:=true -p node_names:=['local_costmap'] -p use_sim_time:=true &"
        )

        with open('/tmp/nav2.log', 'a') as out:
            subprocess.Popen(nav2_cmd, shell=True, stdout=out, stderr=out, executable='/bin/bash')

        self._started = True
        self._log("Launched Nav2 costmap (logs -> /tmp/nav2.log)")
        return py_trees.common.Status.SUCCESS


class StartPerception(DroneActionNode):
    """Starts the VLM perception node (vlm_node.py) after environment is ready.
    Runs once and returns SUCCESS after launching. Logs to /tmp/vlm.log.
    """
    def __init__(self):
        super().__init__("StartPerception")
        self._started = False

    def update(self):
        if self._started:
            return py_trees.common.Status.SUCCESS

        cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            "source /home/rock/IROC-2.0-Drone-Code/ros2_ws/install/setup.bash && "
            "source /home/rock/venv-ardupilot/bin/activate && "
            "python3 /home/rock/IROC-2.0-Drone-Code/behaviour-tree-Raskin/vlm_node.py"
        )

        with open('/tmp/vlm.log', 'a') as out:
            subprocess.Popen(cmd, shell=True, stdout=out, stderr=out, executable='/bin/bash')

        self._started = True
        self._log("Launched VLM perception (logs -> /tmp/vlm.log)")
        return py_trees.common.Status.SUCCESS
