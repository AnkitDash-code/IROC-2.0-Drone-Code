# Import mavutil
from pymavlink import mavutil
import time
import sys
from tqdm import tqdm
import math # For math.radians or math.pi
import numpy as np

# --- Global Constants and Connection Setup ---
CONNECTION_STRING = '/dev/ttyACM0'
BAUD_RATE = 57600

# Type mask for velocity control (ignoring position, acceleration, and yaw/yaw_rate)
# These are the correct constant names in pymavlink
# type_mask = (
#     mavutil.mavlink.POSITION_TARGET_TYPEMASK_VELOCITY_IGNORE |
#     mavutil.mavlink.POSITION_TARGET_TYPEMASK_ACCELERATION_IGNORE |
#     mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
#     mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
# )


# Global master connection object (will be initialized in main)
master = None

# --- Helper Functions ---

def connect_to_vehicle():
    """Establishes MAVLink connection and waits for heartbeat."""
    global master
    print(f"Connecting to vehicle at {CONNECTION_STRING}...")
    master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUD_RATE)
    master.wait_heartbeat()
    print("Heartbeat received. Vehicle connected!")

def set_mode(mode_name):
    """Sets the drone's flight mode."""
    if mode_name not in master.mode_mapping():
        sys.exit(f'Error: Unknown mode "{mode_name}". Available modes: {list(master.mode_mapping().keys())}')
    mode_id = master.mode_mapping()[mode_name]
    master.mav.set_mode_send(master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
    print(f"Switched to {mode_name} mode.")
    time.sleep(1) # Give a moment for mode change to register

def arm_vehicle():
    """Arms the drone and waits for confirmation."""
    print("Arming vehicle...")
    master.arducopter_arm()
    master.motors_armed_wait()
    print('Vehicle Armed!')
    time.sleep(3) # Short pause after arming

def set_speed(speed=0.25):
    """
    Sets the horizontal speed of the drone
    """
    master.mav.command_long_send(
        master.target_system,             # Target system ID
        master.target_component,          # Target component ID
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,  # Command ID
        0,                                # Confirmation
        0,                                # param1: Speed type (1 for ground speed)
        speed,                            # param2: Speed in m/s
        0,                                # param3: Throttle (-1 indicates no change)
        0, 0, 0, 0                        # param4 ~ param7: Unused
    )
    time.sleep(1)

def takeoff(altitude=1):
    """Commands the drone to take off to a specified altitude."""
    MIN_TAKEOFF_ALT = 0.5   # metres
    MAX_TAKEOFF_ALT = 6.0   # metres
    if altitude < MIN_TAKEOFF_ALT or altitude > MAX_TAKEOFF_ALT:
        raise ValueError(f"Takeoff altitude {altitude}m is outside safe bounds ({MIN_TAKEOFF_ALT}-{MAX_TAKEOFF_ALT}m)")
    print(f"Taking Off to {altitude} meters...")
    master.mav.command_long_send(master.target_system, master.target_component,
                                 mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                                 0, 0, 0, 0, 0, 0, altitude)
    # Wait for a sufficient time for the drone to reach takeoff altitude
    # This duration might need adjustment based on your simulation/drone's performance
    print(f"Waiting for takeoff to complete (approx. {altitude * 3} seconds)...") # Rough estimate
    time.sleep(altitude * 3) # Simple heuristic for takeoff time

# def move_body_ned(distance_x, speed_mps=0.3):
#     # Calculate duration based on distance and speed
#     duration_s = abs(distance_x) / speed_mps
    
#     # Determine velocity direction based on distance sign
#     velocity_x = speed_mps if distance_x > 0 else -speed_mps
    
#     print(f"//// Moving {distance_x}m at {speed_mps} m/s ////")
    
#     start_time = time.time()
#     while time.time() - start_time < duration_s:
#         master.mav.set_position_target_local_ned_send(
#             0,  # time_boot_ms (not used)
#             master.target_system,
#             master.target_component,
#             mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Relative to current position and heading
#             int(0b110111000111),  # type_mask: ignore position, yaw, and accel (use velocity)
#             0, 0, 0,  # x, y, z positions (ignored)
#             velocity_x, 0, 0,  # vx, vy, vz velocities (forward, right, down)
#             0, 0, 0,  # afx, afy, afz accelerations (ignored)
#             0, 0  # yaw, yaw_rate (ignored)
#         )
#         time.sleep(0.1)  # Send command at 10Hz
    
#     # Stop the vehicle
#     master.mav.set_position_target_local_ned_send(
#         0, master.target_system, master.target_component,
#         mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
#         int(0b110111000111),
#         0, 0, 0,  # positions (ignored)
#         0, 0, 0,  # velocities (stop)
#         0, 0, 0,  # accelerations (ignored)
#         0, 0  # yaw, yaw_rate (ignored)
#     )
    
#     print(f"moved: {distance_x}m")

def move_body_ned(distance_x, speed_mps=0.25):
        # Send the position target command
    print("//// Moving ////")
    if speed_mps <= 0 or speed_mps > 2.0:
        raise ValueError("speed_mps must be > 0 and <= 2.0")
    master.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms (not used)
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Relative to current position and heading
        int(0b110111111000),                        # int to ignore velocity, yaw, and accel
        distance_x, 0, 0,  # x, y, z positions (forward, right, down)
        0, 0, 0,                # vx, vy, vz velocities (ignored)
        0, 0, 0,                # afx, afy, afz accelerations (ignored)
        0, 0                    # yaw, yaw_rate (ignored)
    )
    wait_s = abs(distance_x) / speed_mps + 1.0
    time.sleep(wait_s)
    print(f"moved: {distance_x}")

def rotate_yaw(angle_degrees, speed_deg_per_sec=10, clockwise=True):
    """
    Rotate drone yaw by specified angle at given speed.
    
    Args:
        angle_degrees: Angle to rotate (e.g., 90)
        speed_deg_per_sec: Rotation speed in degrees per second
        clockwise: True for clockwise, False for counter-clockwise
    """
    print("///// Rotating /////")
    direction = 1 if clockwise else -1
    
    master.mav.command_long_send(master.target_system, master.target_component,
                             mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                             0, angle_degrees, speed_deg_per_sec, int(direction), 1, 0, 0, 0)
    
    time.sleep(angle_degrees/speed_deg_per_sec + 1)
    print(f"Rotated {angle_degrees}")

def land_vehicle():
    """Commands the drone to land and waits until it has disarmed."""
    print("Switching to LAND mode...")
    set_mode("LAND")
    print("Waiting for drone to land and disarm...")
    master.motors_disarmed_wait()  # This blocks until landing is complete
    print("Landed and disarmed.")


def send_body_ned_velocity(vx, vy, vz):
    """
    Sends a velocity command to the drone in the BODY_OFFSET_NED frame.
    This is the core function for the proportional controller.
    """
    msg = master.mav.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Frame relative to drone's body
        0b0000111111000111,  # Type mask (only speeds enabled)
        0,0,0,  # x, y, z positions (not used)
        vx,vy,vz,  # x, y, z velocity in m/s
        0,0,0,  # x, y, z acceleration (not used)
        0,0,
    )  # yaw, yaw_rate (not used)
    master.mav.send(msg)

def flush_data(type=None):
    """Flush any pending messages in the queue."""
    if type:
        for _ in tqdm(range(50), desc=f"Flushing {type} messages"):
            message = master.recv_match(type=type, blocking=True)
            time.sleep(0.01)
    else:
        print("Define type of message please")


def get_current_position(): 
    """
    Retrieves the current position of the drone in the NED frame.
    Returns a tuple (x, y, z) representing the position.
    """
    flush_data('LOCAL_POSITION_NED')
    positions = []

    for _ in range(20):
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        if msg:
            positions.append([msg.x, msg.y, msg.z])
        time.sleep(0.01)

    if positions:
        positions_array = np.array(positions)
        avg_position = np.mean(positions_array, axis=0)
        print(f"Average position: x={avg_position[0]:.4f}, y={avg_position[1]:.4f}, z={avg_position[2]:.4f}")
        return tuple(avg_position)
    
    return (msg.x, msg.y, msg.z)


def get_heading():
    """
    Retrieves the current heading of the drone.
    Returns the heading in degrees.
    """
    # flush_data('VFR_HUD')
    msg = master.recv_match(type='VFR_HUD', blocking=True)
    if msg:
        return msg.heading
    return None


def rotate_towards(x, y):
    """
    Rotates the drone to face a specific point in the NED frame.
    This is a simple function that calculates the required yaw angle
    and uses the rotate_yaw function to turn the drone.
    """
    current_x, current_y, _ = get_current_position()
    
    dx = x - current_x
    dy = y - current_y
    bearing_rad = math.atan2(dy, dx)
    bearing_deg = math.degrees(bearing_rad)

    current_heading = get_heading()
    print(f"Current heading: {current_heading:.1f} degrees")

    turn = bearing_deg - current_heading

    if turn > 180:
        turn -= 360
    if turn < -180:
        turn += 360

    print(f"Turn: {turn:.2f} degrees, distance: {math.sqrt(dx**2 + dy**2):.2f} meters")

    rotate_direction = 1 if turn > 0 else -1
    rotate_yaw(abs(turn), 10, clockwise=(rotate_direction == 1))
    return math.sqrt(dx**2 + dy**2)


def special_landing():
    flush_data("RANGEFINDER")
    print("Switching to LAND mode...")
    set_mode("LAND")
    while True:
        msg = master.recv_match(type='RANGEFINDER', blocking=True)
        altitude = msg.distance
        if altitude <= 0.2:
            break
        print(altitude)
        time.sleep(0.01)
    
    print("Force disarming via MAV_CMD_COMPONENT_ARM_DISARM...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 21196, 0, 0, 0, 0, 0
    )

    print("Landing completed")

# --- Main Script Execution ---
if __name__ == "__main__":
    # This block is primarily for testing the functions directly if this file is run.
    # For the rectangle/square flight, we'll use the 'rectangle-flight-program.py'
    # which imports these functions.
    try:
        connect_to_vehicle()
        
        # WARNING: The following commands are commented out for safety.
        # Running them directly will cause live flight without confirmation.
        
        # set_mode('LOITER')
        # get_current_position()
        # takeoff(altitude=1)
        # set_mode('GUIDED')

        # time.sleep(0.5)

        # set_speed(0.25)
        # # Example: Move forward 1m and rotate 90 deg
        # time.sleep(3)

        # move_body_ned(3) #moving straight x degrees, 3 here
        # time.sleep(2)
        # rotate_yaw(90, 10, clockwise=False) # Rotate 90 degrees at 10 deg/s
        # move_body_ned(3) #moving straight x degrees, 3 here
        # time.sleep(2)
        # rotate_yaw(180, 10, clockwise=False) # Rotate 90 degrees at 10 deg/s
        # rotate_yaw(180, 10, clockwise=False) # Rotate 90 degrees at 10 deg/s

        # time.sleep(5)

        # for _ in range(4):
        #     move_body_ned_velocity(2) #moving straight x degrees, 3 here
        #     rotate_yaw(90, 10, clockwise=False) # Rotate 90 degrees at 10 deg/s


    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    print("Script finished.")
