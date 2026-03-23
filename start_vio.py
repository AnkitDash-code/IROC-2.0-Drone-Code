import os
import time
import socket
import sys
import signal
import threading

# --- CONFIGURATION ---
ENABLE_FOXGLOVE = False      # MASSIVE CPU SAVER: Keep False for flight
ENABLE_TERMINAL_HUD = True   # Shows real-time XYZ and Direction in SSH
LOG_FILE = "drone_debug.log"
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

def shutdown_handler(sig, frame):
    print("\n🛑 SHUTDOWN: Cleaning processes...")
    os.system("pkill -9 -f camera && pkill -9 -f rtabmap && pkill -9 -f mavros && pkill -9 -f static_transform && pkill -9 -f web_video_server > /dev/null 2>&1")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

# 0. Initial cleanup
os.system("pkill -9 -f camera && pkill -9 -f rtabmap && pkill -9 -f mavros && pkill -9 -f rosbridge && pkill -9 -f web_video_server > /dev/null 2>&1")

jetson_ip = get_ip()
video_link = f"http://{jetson_ip}:8080/stream_viewer?topic=/camera/camera/color/image_raw"

print(f"🚀 Launching LIGHTWEIGHT VIO")
print(f"⚠️  STILLNESS REQUIRED: Keep drone still for 5 seconds...")

# 1. Start Camera
os.system(f"ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=424x240x30 rgb_camera.color_profile:=424x240x30 enable_gyro:=true enable_accel:=true unite_imu_method:=2 depth_module.decimation_filter.enable:=true >> {LOG_FILE} 2>&1 &")
time.sleep(5) 

# 2. Start IMU Filter
os.system(f"ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -r /imu/data_raw:=/camera/camera/imu -p use_mag:=false -p publish_tf:=false >> {LOG_FILE} 2>&1 &")

# 3. FIXED STATIC TRANSFORM 
os.system(f"ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom >> {LOG_FILE} 2>&1 &")
os.system(f"ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link >> {LOG_FILE} 2>&1 &")
os.system(f"ros2 run tf2_ros static_transform_publisher 0 0 0 -1.5708 0 -1.5708 base_link camera_link >> {LOG_FILE} 2>&1 &")

# 4. Web Video Server (Browser Feed)
os.system(f"ros2 run web_video_server web_video_server --ros-args -p port:=8080 >> {LOG_FILE} 2>&1 &")

# 5. MAVROS (THE FIX: Remapped to odometry/out and added 'odometry' to allowlist)
os.system(f"ros2 run mavros mavros_node --ros-args \
    -p fcu_url:=/dev/ttyACM0:115200 \
    -p system_id:=1 \
    -p component_id:=191 \
    -p plugin_allowlist:=['sys_status','sys_time','command','param','odometry','imu'] \
    -r /mavros/odometry/out:=/rtabmap/odom >> {LOG_FILE} 2>&1 &")

# 6. Foxglove Toggle
if ENABLE_FOXGLOVE:
    os.system(f"ros2 launch rosbridge_server rosbridge_websocket_launch.xml >> {LOG_FILE} 2>&1 &")

# 7. Start RTAB-Map
os.system(f"ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:='--delete_db_on_start --Vis/MaxFeatures 200 --Rtabmap/DetectionRate 1 --Approx/SyncMaxInterval 0.05' \
    rgb_topic:=/camera/camera/color/image_raw \
    depth_topic:=/camera/camera/depth/image_rect_raw \
    camera_info_topic:=/camera/camera/color/camera_info \
    imu_topic:=/imu/data \
    frame_id:=base_link \
    odom_frame_id:=odom \
    publish_tf_odom:=false \
    approx_sync:=true \
    viz:=false >> {LOG_FILE} 2>&1 &")


# 8. TERMINAL HUD THREAD
def terminal_hud():
    import rclpy
    from nav_msgs.msg import Odometry
    if not rclpy.ok(): rclpy.init()
    node = rclpy.create_node('terminal_hud')
    
    def odom_callback(msg):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        
        if abs(vel.x) < 0.05 and abs(vel.y) < 0.05 and abs(ang.z) < 0.1: state = "STATIONARY"
        elif abs(ang.z) > 0.2: state = "TURNING   "
        elif abs(vel.x) > abs(vel.y): state = "FORWARD   " if vel.x > 0 else "BACKWARD  "
        else: state = "SIDEWAYS  "
            
        sys.stdout.write(f"\r XYZ: {pos.x:+.2f}, {pos.y:+.2f}, {pos.z:+.2f} |BELIEF: {state} |VIDEO: {video_link}   ")
        sys.stdout.flush()

    sub = node.create_subscription(Odometry, '/rtabmap/odom', odom_callback, 10)
    try:
        rclpy.spin(node)
    except Exception:
        pass

if ENABLE_TERMINAL_HUD:
    threading.Thread(target=terminal_hud, daemon=True).start()

print(f"\n✅ VIO RUNNING.")
while True:
    time.sleep(1)