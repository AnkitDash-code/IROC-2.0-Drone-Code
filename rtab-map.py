import os
import time
import socket
import sys
import signal
import threading
import glob
import shlex
import math

# --- CONFIGURATION ---
ENABLE_FOXGLOVE = False      # MASSIVE CPU SAVER: Keep False for flight
ENABLE_TERMINAL_HUD = True   # Shows real-time XYZ and Direction in SSH
LOG_FILE = "drone_debug.log"
MAVROS_BAUD = 115200
ZERO_RESET_RADIUS = 0.08
ODOM_DROP_DISTANCE = 1.0
RTABMAP_RESTART_COOLDOWN = 10.0
FCU_VERTICAL_SPEED_TRIGGER = 0.60
FCU_MIN_POSITION_NORM = 0.25
# ---------------------

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def get_fcu_serial_device():
    # Prefer stable by-id links so reboot order of ACM devices does not break MAVROS.
    preferred_patterns = [
        "/dev/serial/by-id/*CubePilot*if00",
        "/dev/serial/by-id/*Pixhawk*if00",
        "/dev/serial/by-id/*if00",
    ]
    for pattern in preferred_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]

    for fallback_pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(fallback_pattern))
        if matches:
            return matches[0]

    return "/dev/ttyACM0"

def shutdown_handler(sig, frame):
    print("\n🛑 SHUTDOWN: Cleaning processes...")
    os.system("pkill -9 -f camera && pkill -9 -f rtabmap && pkill -9 -f mavros && pkill -9 -f static_transform && pkill -9 -f web_video_server > /dev/null 2>&1")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

# 0. Initial cleanup
os.system("pkill -9 -f camera && pkill -9 -f rtabmap && pkill -9 -f mavros && pkill -9 -f rosbridge && pkill -9 -f web_video_server > /dev/null 2>&1")

jetson_ip = get_ip()
fcu_device = get_fcu_serial_device()
fcu_url = f"{fcu_device}:{MAVROS_BAUD}"
video_link = f"http://{jetson_ip}:8080/stream_viewer?topic=/camera/camera/color/image_raw"

print(f"🚀 Launching LIGHTWEIGHT VIO")
print(f"🔌 MAVROS FCU device: {fcu_device}")
print(f"⚠️  STILLNESS REQUIRED: Keep drone still for 5 seconds...")

# 1. Start Camera
os.system(f"ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 align_depth.enable:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2 depth_module.decimation_filter.enable:=true >> {LOG_FILE} 2>&1 &")
time.sleep(5) 

# 2. Start IMU Filter
os.system(f"ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -r /imu/data_raw:=/camera/camera/imu -p use_mag:=false -p publish_tf:=false >> {LOG_FILE} 2>&1 &")

# 3. FIXED STATIC TRANSFORM (Camera is facing forward, but mounted upside down: Roll = 180 deg / 3.14159 rad)
os.system(f"ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom >> {LOG_FILE} 2>&1 &")
os.system(f"ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 3.14159 base_link camera_link >> {LOG_FILE} 2>&1 &")

# 4. Web Video Server (Browser Feed)
os.system(f"ros2 run web_video_server web_video_server --ros-args -p port:=8080 >> {LOG_FILE} 2>&1 &")

# 5. MAVROS 
os.system(f"ros2 run mavros mavros_node --ros-args \
    -p fcu_url:={shlex.quote(fcu_url)} \
    -p system_id:=1 \
    -p component_id:=191 \
    -p plugin_allowlist:=['sys_status','sys_time','command','param','odometry','imu','local_position','vision_pose','vision_speed'] >> {LOG_FILE} 2>&1 &")

# 6. Foxglove Toggle
if ENABLE_FOXGLOVE:
    os.system(f"ros2 launch rosbridge_server rosbridge_websocket_launch.xml >> {LOG_FILE} 2>&1 &")

# 7. Start RTAB-Map (Fixed viz arg, removed publish_tf_odom:=false, tuned VIO parameters)
def launch_rtabmap():
    os.system(f"ros2 launch rtabmap_launch rtabmap.launch.py \
        rtabmap_args:='--delete_db_on_start --Vis/MaxFeatures 700 --Vis/MinInliers 8 --Rtabmap/DetectionRate 2 --Approx/SyncMaxInterval 0.05' \
        rgb_topic:=/camera/camera/color/image_raw \
        depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
        camera_info_topic:=/camera/camera/color/camera_info \
        imu_topic:=/imu/data \
        frame_id:=camera_link \
        odom_frame_id:=odom \
        approx_sync:=true \
        rtabmap_viz:=false >> {LOG_FILE} 2>&1 &")


def restart_rtabmap():
    print("\n⚠️  SAFETY: Odom reset detected. Restarting RTAB-Map with FCU fallback...")
    os.system("pkill -9 -f rtabmap.launch.py > /dev/null 2>&1")
    time.sleep(1.0)
    launch_rtabmap()


launch_rtabmap()


# 8. TERMINAL HUD THREAD
def terminal_hud():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    if not rclpy.ok(): rclpy.init()
    node = rclpy.create_node('terminal_hud')

    last_good_pos = [0.0, 0.0, 0.0]
    last_fcu_pos = [0.0, 0.0, 0.0]
    last_fcu_vel = [0.0, 0.0, 0.0]
    last_fcu_yawrate = 0.0
    has_fcu_fix = False
    zero_streak = 0
    fallback_active = False
    fallback_anchor = [0.0, 0.0, 0.0]
    fallback_fcu_origin = [0.0, 0.0, 0.0]
    last_restart_time = 0.0
    had_nonzero_fix = False

    def motion_state(vx, vy, wz):
        if abs(vx) < 0.05 and abs(vy) < 0.05 and abs(wz) < 0.1:
            return "STATIONARY"
        if abs(wz) > 0.2:
            return "TURNING   "
        if abs(vx) > abs(vy):
            return "FORWARD   " if vx > 0 else "BACKWARD  "
        return "SIDEWAYS  "

    def fcu_odom_callback(msg):
        nonlocal last_fcu_pos, last_fcu_vel, last_fcu_yawrate, has_fcu_fix
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        last_fcu_pos = [p.x, p.y, p.z]
        last_fcu_vel = [v.x, v.y, v.z]
        last_fcu_yawrate = w.z
        has_fcu_fix = True

    def fcu_is_valid_nonzero():
        if not has_fcu_fix:
            return False
        fcu_pos_norm = math.sqrt(
            (last_fcu_pos[0] * last_fcu_pos[0]) +
            (last_fcu_pos[1] * last_fcu_pos[1]) +
            (last_fcu_pos[2] * last_fcu_pos[2])
        )
        return fcu_pos_norm > FCU_MIN_POSITION_NORM
    
    def odom_callback(msg):
        nonlocal zero_streak, fallback_active, fallback_anchor, fallback_fcu_origin
        nonlocal last_restart_time, last_good_pos, had_nonzero_fix
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        now = time.monotonic()

        pos_norm = math.sqrt((pos.x * pos.x) + (pos.y * pos.y) + (pos.z * pos.z))
        last_good_norm = math.sqrt(
            (last_good_pos[0] * last_good_pos[0]) +
            (last_good_pos[1] * last_good_pos[1]) +
            (last_good_pos[2] * last_good_pos[2])
        )
        near_zero = pos_norm < ZERO_RESET_RADIUS

        if near_zero:
            zero_streak += 1
        else:
            had_nonzero_fix = True
            zero_streak = 0
            last_good_pos = [pos.x, pos.y, pos.z]
            if fallback_active:
                print("\n✅ Odom recovered. Leaving FCU fallback.")
            fallback_active = False

        drop_to_zero = had_nonzero_fix and near_zero and (last_good_norm > ODOM_DROP_DISTANCE)
        fast_vertical = abs(last_fcu_vel[2]) > FCU_VERTICAL_SPEED_TRIGGER
        can_use_fcu = fcu_is_valid_nonzero()

        if (drop_to_zero and fast_vertical and can_use_fcu):
            if not fallback_active:
                fallback_active = True
                fallback_anchor = list(last_good_pos)
                fallback_fcu_origin = list(last_fcu_pos)

            if (now - last_restart_time) >= RTABMAP_RESTART_COOLDOWN:
                last_restart_time = now
                threading.Thread(target=restart_rtabmap, daemon=True).start()

        if fallback_active and near_zero and can_use_fcu:
            display_x = fallback_anchor[0] + (last_fcu_pos[0] - fallback_fcu_origin[0])
            display_y = fallback_anchor[1] + (last_fcu_pos[1] - fallback_fcu_origin[1])
            display_z = fallback_anchor[2] + (last_fcu_pos[2] - fallback_fcu_origin[2])
            state = motion_state(last_fcu_vel[0], last_fcu_vel[1], last_fcu_yawrate)
            hud_suffix = " |MODE: FCU_FALLBACK"
        else:
            if fallback_active and (not can_use_fcu):
                fallback_active = False
            display_x, display_y, display_z = pos.x, pos.y, pos.z
            state = motion_state(vel.x, vel.y, ang.z)
            hud_suffix = ""

        print(f"XYZ: {display_x:+.2f}, {display_y:+.2f}, {display_z:+.2f} |BELIEF: {state}{hud_suffix}")
        print(f"VIDEO: {video_link}")
        print("")

    # rgbd_odometry publishes motion on /odom even when rtabmap graph updates are skipped.
    sub_odom = node.create_subscription(Odometry, '/odom', odom_callback, 10)
    sub_fcu = node.create_subscription(Odometry, '/mavros/local_position/odom', fcu_odom_callback, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except Exception:
        pass

if ENABLE_TERMINAL_HUD:
    threading.Thread(target=terminal_hud, daemon=True).start()

print(f"\n✅ VIO RUNNING.")
while True:
    time.sleep(1)