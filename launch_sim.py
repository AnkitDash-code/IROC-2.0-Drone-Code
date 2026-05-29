#!/usr/bin/env python3
import subprocess
import signal
import sys
import os
import time
import argparse
from pathlib import Path

# --- Setup Dynamic Command Line Arguments ---
parser = argparse.ArgumentParser(description="Launch Gazebo and ArduPilot SITL.")
parser.add_argument('--gcs-port', type=int, default=14550, help='UDP port for Mission Planner (default: 14550)')
parser.add_argument('--api-port', type=int, default=14552, help='UDP port for your Drone Code (default: 14552)')
parser.add_argument('--slam-backend', choices=['auto', 'rtabmap'], default='rtabmap', help='SLAM backend to launch (default: rtabmap)')
args = parser.parse_args()

# --- Fetch Windows IP dynamically ---
try:
    windows_ip = subprocess.check_output(
        "cat /etc/resolv.conf | grep nameserver | awk '{print $2}'",
        shell=True).decode().strip()
except Exception:
    windows_ip = "127.0.0.1"

uid = os.getuid()

if args.slam_backend == 'auto':
    slam_backend = 'rtabmap'
else:
    slam_backend = args.slam_backend

def cleanup(sig=None, frame=None):
    print("\n[INFO] Ctrl+C detected. Forcing shutdown of all simulation processes...")
    targets = [
        "sim_vehicle.py",
        "arducopter",
        "mavproxy",
        "gz sim",
        "gz-server",
        "gz-gui",
        "ros_gz_bridge parameter_bridge",
        "mavros_node",
        "vio_executable",
        "ros2 run vio_node",
        "rtabmap_launch rtabmap.launch.py",
        "rtabmap",
        "rtabmap_viz",
        "slam_toolbox",
        "online_async_launch.py",
        "depthimage_to_laserscan",
        "depthimage_to_laserscan_node",
        "orbslam3",
        "stereo-inertial",
        "stereo",
        "rgbd",
        "rviz2",
        "local_costmap",
        "nav2_costmap_2d",
        "vlm_node.py",
        "ascend/mission.py",
    ]
    for target in targets:
        subprocess.run(f"pkill -9 -f '{target}'", shell=True, stderr=subprocess.DEVNULL)
    print("Done. Simulation windows should now be forced closed.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("Clearing old processes...")
subprocess.run(
    "pkill -f arducopter 2>/dev/null; "
    "pkill -f mavproxy 2>/dev/null; "
    "pkill -f 'gz sim' 2>/dev/null; "
    "pkill -f 'ros_gz_bridge parameter_bridge' 2>/dev/null; "
    "pkill -f mavros_node 2>/dev/null; "
    "pkill -f 'ros2 run vio_node' 2>/dev/null; "
    "pkill -f vio_executable 2>/dev/null; "
    "pkill -f 'rtabmap_launch rtabmap.launch.py' 2>/dev/null; "
    "pkill -f rtabmap 2>/dev/null; "
    "pkill -f slam_toolbox 2>/dev/null; "
    "pkill -f online_async_launch.py 2>/dev/null; "
    "pkill -f depthimage_to_laserscan 2>/dev/null; "
    "pkill -f depthimage_to_laserscan_node 2>/dev/null; "
    "pkill -f local_costmap 2>/dev/null; "
    "pkill -f nav2_costmap_2d 2>/dev/null; "
    "pkill -f vlm_node.py 2>/dev/null; "
    "pkill -f ascend/mission.py 2>/dev/null; "
    "pkill -f 'python3 .*vlm_node.py' 2>/dev/null; "
    "pkill -f 'python3 .*ascend/mission.py' 2>/dev/null; "
    "pkill -f rviz2 2>/dev/null",
    shell=True,
)
time.sleep(1)

print(f"Starting Simulation Controller... (Windows IP: {windows_ip})")

# ==========================================
# Terminal 1: Gazebo
# ==========================================
gazebo_cmd = f"""
export DISPLAY=:0
export WAYLAND_DISPLAY=wayland-0
export XDG_RUNTIME_DIR=/run/user/{uid}
# Use Bullet physics engine to enable mesh-based collisions for OBJ/DAE
gz sim --physics-engine /usr/lib/x86_64-linux-gnu/gz-physics-7/engine-plugins/libgz-physics7-bullet-plugin.so -v4 -r ~/ardupilot_gazebo/worlds/iroc_mars.world
"""
subprocess.run(['gnome-terminal', '--title=Gazebo', '--', 'bash', '-c', gazebo_cmd])

print("Waiting 10s for Gazebo to load...")
time.sleep(10)

# ==========================================
# Terminal 2: ArduCopter SITL
# ==========================================
# Notice the ports are now injected dynamically from the args
sitl_cmd = f"""
source /home/rock/venv-ardupilot/bin/activate
export PATH=/home/rock/venv-ardupilot/bin:$PATH
export PATH=$PATH:$HOME/ardupilot/Tools/autotest
cd ~/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --out udp:{windows_ip}:{args.gcs_port} --out udp:127.0.0.1:{args.api_port} --no-rebuild --console
"""
subprocess.run(['gnome-terminal', '--title=ArduCopter SITL', '--', 'bash', '-c', sitl_cmd])

print("Waiting 8s for SITL to start...")
time.sleep(8)

# ==========================================
# Terminal 3: ROS 2 bridge + MAVROS + VIO
# ==========================================
ros_cmd = f"""
source /opt/ros/jazzy/setup.bash
source /home/rock/IROC-2.0-Drone-Code/ros2_ws/install/setup.bash
ros2 run ros_gz_bridge parameter_bridge /oakd/left/image_raw@sensor_msgs/msg/Image@gz.msgs.Image /oakd/right/image_raw@sensor_msgs/msg/Image@gz.msgs.Image /oakd/depth/image_raw@sensor_msgs/msg/Image@gz.msgs.Image /oakd/left/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo /oakd/right/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo /oakd/imu@sensor_msgs/msg/Imu@gz.msgs.IMU /picam3/image_raw@sensor_msgs/msg/Image@gz.msgs.Image /picam3/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo &
sleep 2
ros2 run mavros mavros_node --ros-args \
    -p fcu_url:=udp://127.0.0.1:{args.api_port}@ \
    -p system_id:=1 \
    -p component_id:=191 \
    -p plugin_allowlist:=['sys_status','sys_time','command','param','odometry','imu','local_position','vision_pose','vision_speed'] &
sleep 2
ros2 run tf2_ros static_transform_publisher \
    --x 0.1 --y 0.0 --z 0.05 \
    --qx 0 --qy 0 --qz 0 --qw 1 \
    --frame-id base_link \
    --child-frame-id camera_link &
ros2 run tf2_ros static_transform_publisher \
    --x 0.1 --y 0.0375 --z 0.05 \
    --qx 0 --qy 0 --qz 0 --qw 1 \
    --frame-id base_link \
    --child-frame-id iris_with_ardupilot/oakd_link/oakd_left &
ros2 run tf2_ros static_transform_publisher \
    --x 0.1 --y -0.0375 --z 0.05 \
    --qx 0 --qy 0 --qz 0 --qw 1 \
    --frame-id base_link \
    --child-frame-id iris_with_ardupilot/oakd_link/oakd_right &
ros2 run tf2_ros static_transform_publisher \
    --x 0.1 --y 0.0375 --z 0.05 \
    --qx 0 --qy 0 --qz 0 --qw 1 \
    --frame-id base_link \
    --child-frame-id iris_with_ardupilot/oakd_link/oakd_depth &
sleep 1
ros2 run vio_node vio_executable
"""
subprocess.run(['gnome-terminal', '--title=ROS 2 VIO Stack', '--', 'bash', '-c', ros_cmd])

# ==========================================
# Terminal 4: RTAB-Map (default)
# ==========================================
slam_cmd = """
source /opt/ros/jazzy/setup.bash
source /home/rock/IROC-2.0-Drone-Code/ros2_ws/install/setup.bash
sleep 5
ros2 launch rtabmap_launch rtabmap.launch.py \
    stereo:=true \
    visual_odometry:=false \
    odom_topic:=/odom \
    imu_topic:=/oakd/imu \
    left_image_topic:=/oakd/left/image_raw \
    right_image_topic:=/oakd/right/image_raw \
    left_camera_info_topic:=/oakd/left/camera_info \
    right_camera_info_topic:=/oakd/right/camera_info \
    frame_id:=base_link \
    approx_sync:=true \
    wait_imu_to_init:=true \
    always_check_imu_tf:=false \
    rtabmap_args:="--delete_db_on_start \
    --RGBD/OptimizeStrategy 0 \
    --Grid/3D false \
    --Grid/CellSize 0.1 \
    --Kp/MaxFeatures 200 \
    --Mem/STMSize 10" \
    rviz:=true
"""
subprocess.run(['gnome-terminal', '--title=RTAB-Map', '--', 'bash', '-c', slam_cmd])

# Nav2/local_costmap will be launched by the BT after takeoff to avoid
# building the occupancy grid before the vehicle is airborne. See
# `ascend/bt_nodes/launch_helpers.py::StartNav2` which starts Nav2 when
# the BT reaches the target altitude.

# Perception and Mission Controller are launched by the Behavior Tree
# after takeoff to avoid duplicate launches. See ascend/bt_nodes/launch_helpers.py
# which starts Nav2 and vlm_node when the BT reaches the target altitude.

print("\n=============================================")
print("✓ All simulation environments launched successfully.")
print(f"✓ Mission Planner (Windows) → udp:{windows_ip}:{args.gcs_port}")
print(f"✓ VSCode / Drone Code       → udp:127.0.0.1:{args.api_port}")
print(f"✓ SLAM backend              → {slam_backend}")
print("=============================================")
print("Press Ctrl+C in THIS window to force shutdown everything.\n")

# Try to open visible terminals for Mission and Perception so they "show up" like
# the other windows. Fall back to opening a tmux attach or tailing the logs if a
# GUI terminal is not available.
def _open_visual_term(title, command):
    emulators = ["gnome-terminal", "xfce4-terminal", "konsole", "tilix", "xterm"]
    for em in emulators:
        if subprocess.run(["which", em], stdout=subprocess.DEVNULL).returncode == 0:
            try:
                if em == "gnome-terminal":
                    subprocess.Popen(["gnome-terminal", "--title=%s" % title, "--", "bash", "-lc", command])
                elif em == "xfce4-terminal":
                    subprocess.Popen(["xfce4-terminal", "--title=%s" % title, "-e", f"bash -lc {command!r}"])
                elif em == "konsole":
                    subprocess.Popen(["konsole", "--title", title, "-e", "bash", "-lc", command])
                elif em == "tilix":
                    subprocess.Popen(["tilix", "-a", "session-add-down", "-e", "bash", "-lc", command])
                elif em == "xterm":
                    subprocess.Popen(["xterm", "-T", title, "-hold", "-e", "bash", "-lc", command])
                return True
            except Exception:
                continue
    return False

# Mission: run the actual BT mission in a visible terminal so READY/ARM can be typed there.
mission_cmd = '''bash -lc 'export PYTHONUNBUFFERED=1; source /opt/ros/jazzy/setup.bash; source /home/rock/venv-ardupilot/bin/activate; cd /home/rock/IROC-2.0-Drone-Code/behaviour-tree-Raskin; exec python3 -u ascend/mission.py --enable-detections --detections-log /tmp/detections.jsonl' '''

# Perception: run the actual VLM/YOLO node in a visible terminal.
perception_cmd = '''bash -lc 'export PYTHONUNBUFFERED=1; source /opt/ros/jazzy/setup.bash; source /home/rock/IROC-2.0-Drone-Code/ros2_ws/install/setup.bash; source /home/rock/venv-ardupilot/bin/activate; cd /home/rock/IROC-2.0-Drone-Code/behaviour-tree-Raskin; exec python3 -u vlm_node.py' '''

# Attempt to open GUI terminals; if not possible, print the attach/tail commands
opened_m = _open_visual_term("Mission", mission_cmd)
opened_p = _open_visual_term("Perception", perception_cmd)
if not (opened_m and opened_p):
    print("Note: Could not open GUI terminals. Creating a detached tmux view instead.")
    # Create a detached tmux session 'ascend_view' with two panes tailing mission and vlm logs.
    try:
        if subprocess.run(["which", "tmux"], stdout=subprocess.DEVNULL).returncode == 0:
            # Kill any existing view session and recreate
            subprocess.run(["tmux", "kill-session", "-t", "ascend_view"], stderr=subprocess.DEVNULL)
            subprocess.run(["tmux", "new-session", "-d", "-s", "ascend_view", "-n", "view", "bash", "-lc",
                             "while true; do if [ -f /tmp/mission.log ]; then tail -n +1 -f /tmp/mission.log; else echo Waiting for /tmp/mission.log; sleep 1; fi; done"], check=False)
            subprocess.run(["tmux", "split-window", "-h", "-t", "ascend_view:0", "bash", "-lc",
                             "while true; do if [ -f /tmp/vlm.log ]; then tail -n +1 -f /tmp/vlm.log; else echo Waiting for /tmp/vlm.log; sleep 1; fi; done"], check=False)
            print("Detached tmux session 'ascend_view' created. Attach with: tmux attach -t ascend_view")
        else:
            print("tmux not available — use 'tmux attach -t ascend_mission' or 'tail -f /tmp/vlm.log' to monitor logs")
    except Exception as e:
        print(f"Failed to create tmux view: {e}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    cleanup()
