import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import math
import os
import sys
sys.path.append('/home/jetson123/Drone/SynexensPythonSDK4_4.2.4.0_202504281506')


try:
    from drone_control import (
        connect_to_vehicle,
        set_mode,
        arm_vehicle,
        takeoff,
        rotate_yaw,
        send_body_ned_velocity, # Use this for all velocity commands
        land_vehicle,
        master,
        mavutil,
        special_landing,
        rotate_towards,
        get_current_position,
        get_heading,
        flush_data,
    )
except ImportError:
    print("Error: Make sure 'drone_control.py' is in the same directory.")
    sys.exit(1)

try:
    from ransac import (
        initialize_lidar_stream,
        stop_lidar_stream,
        get_pcd_from_stream,
        get_angle_est,
        get_current_plane_angles,
    )
except ImportError:
    print("Error: Make sure your ransac.py is in the correct place with all the functions")
    sys.exit(1)

SAFE_SPOT_MODEL_PATH = "files_cv/train17/weights/best.pt"
SLANT_SAFE_SPOT_MODEL_PATH = "files_cv/train20/weights/best.pt"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
TAKEOFF_ALTITUDE = 3
ROTATION_ANGLE_STEP = 5
ROTATION_SPEED = 5
SEARCH_ROTATION_ANGLE = 20
DETECTION_CONFIDENCE_THRESHOLD = 0.40
PROPORTIONAL_GAIN = 0.2
APPROACH_MAX_SPEED = 0.7
STOPPING_DISTANCE = 4.7
MAX_NORMAL_SAFE_SPOTS = 2  # Number of normal safe spots to visit
MAX_SLANT_SAFE_SPOTS = 1   # Number of slant safe spots to visit
TOTAL_SAFE_SPOTS = MAX_NORMAL_SAFE_SPOTS + MAX_SLANT_SAFE_SPOTS
os.environ["YOLO_VERBOSE"] = "False"

def classify_centroid_position(cx, frame_width, tolerance_percent=3):
    mid_x = frame_width / 2
    tolerance_pixels = frame_width * (tolerance_percent / 100)
    if cx < mid_x - tolerance_pixels:
        return "Left"
    elif cx > mid_x + tolerance_pixels:
        return "Right"
    else:
        return "Center"


def find_and_analyze_safe_spot(pipeline, align, model, colorizer):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None, None
    color_image = np.asanyarray(color_frame.get_data())
    colorized_depth = colorizer.colorize(depth_frame)
    depth_heatmap = np.asanyarray(colorized_depth.get_data())
    blended_image = cv2.addWeighted(color_image, 0.85, depth_heatmap, 0.15, 0)
    results = model(blended_image, verbose=False)
    best_detection = None
    max_conf = 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                best_detection = box
    if best_detection and best_detection.conf[0] > DETECTION_CONFIDENCE_THRESHOLD:
        x1, y1, x2, y2 = map(int, best_detection.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        position_tag = classify_centroid_position(cx, FRAME_WIDTH)
        distance_m = depth_frame.get_distance(cx, cy)
        annotated_frame = blended_image.copy()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)
        label_text = f"Tag: {position_tag}, Dist: {distance_m:.2f}m"
        cv2.putText(
            annotated_frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return position_tag, distance_m, annotated_frame
    return None, None, blended_image

# --- Closed-Loop Control Functions ---

def proportional_approach(pipeline, align, model, colorizer):
    print("\n--- Phase 2: Executing Proportional Approach ---")
    loop_hz = 5
    while True:
        position, distance, frame = find_and_analyze_safe_spot(
            pipeline, align, model, colorizer
        )
        if frame is not None:
            cv2.imshow("Live Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        if distance is None or distance == 0:
            print("Target lost. Hovering.")
            send_body_ned_velocity(0, 0, 0) # stop the vehicle
            time.sleep(1 / loop_hz)
            continue

        if position != "Center":
            print(f"Target drifted {position}. Re-centering.")
            send_body_ned_velocity(0, 0, 0) # Stop forward movement
            is_clockwise = position == "Right"
            rotate_yaw(ROTATION_ANGLE_STEP, ROTATION_SPEED, clockwise=is_clockwise)
            continue

        error = distance * math.cos(math.radians(30)) - STOPPING_DISTANCE
        if error <= 0.2:
            print(f"Target reached. Final distance: {distance:.2f}m. Stopping.")
            send_body_ned_velocity(0, 0, 0)
            break

        speed = min(error * PROPORTIONAL_GAIN, APPROACH_MAX_SPEED)
        print(f"Dist: {distance:.2f}m, Error: {error:.2f}m, Speed: {speed:.2f} m/s")
        send_body_ned_velocity(speed, 0, 0) # Use the imported function
        time.sleep(1 / loop_hz)
    print("Approach complete.")


def search_for_safe_spot(pipeline, align, model, colorizer):
    print("\n--- Phase 1: Searching for Safe Spot ---")
    while True:
        position, distance, frame = find_and_analyze_safe_spot(
            pipeline, align, model, colorizer
        )
        if frame is not None:
            cv2.imshow("Live Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        if position == "Center" and distance * math.cos(math.radians(30)) > 1.5:
            print(f"Target Centered! Distance: {distance:.2f} meters.")
            return True
        else:
            is_clockwise = position != "Left"
            angle = (
                SEARCH_ROTATION_ANGLE if position is None else ROTATION_ANGLE_STEP
            )
            print(
                f"Target is {position or 'not visible'}. Rotating {'Clockwise' if is_clockwise else 'CCW'}."
            )
            rotate_yaw(angle, ROTATION_SPEED, clockwise=is_clockwise)


def load_model(model_path, model_type):
    """Load and return the specified YOLO model"""
    print(f"Loading {model_type} model from: {model_path}")
    model = YOLO(model_path)
    print(f"{model_type} model loaded successfully!")
    return model


def get_model_info(spot_number):
    """Return model path and type based on spot number"""
    if spot_number <= MAX_NORMAL_SAFE_SPOTS:
        return SAFE_SPOT_MODEL_PATH, "Normal Safe Spot"
    else:
        return SLANT_SAFE_SPOT_MODEL_PATH, "Slant Safe Spot"


# --- Main Flight Control Logic (Now with model switching) ---
def main():
    print("Initializing Noir...")
    initialize_lidar_stream()
    connect_to_vehicle()
    home_x, home_y, _ = get_current_position()

    # Initialize camera and pipeline
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(
        rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS
    )
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 8)
    align = rs.align(rs.stream.color)
    pipeline.start(config)
    print("Camera and Drone Connection Initialized.")

    # Initialize with the first model (normal safe spot)
    current_model = load_model(SAFE_SPOT_MODEL_PATH, "Normal Safe Spot")


    try:
        arm_vehicle()
        takeoff(TAKEOFF_ALTITUDE)
        time.sleep(3)
        set_mode("GUIDED")

        # Main loop for multiple safe spots with model switching
        for i in range(TOTAL_SAFE_SPOTS):
            spot_number = i + 1
            model_path, model_type = get_model_info(spot_number)
            
            print(f"\n=== Looking for Safe Spot #{spot_number} ({model_type}) ===")
            
            # Check if we need to switch models
            if spot_number == MAX_NORMAL_SAFE_SPOTS + 1:
                print(f"\n SWITCHING TO SLANT MODEL ")
                print("=" * 50)
                current_model = load_model(SLANT_SAFE_SPOT_MODEL_PATH, "Slant Safe Spot")
                print("Model switch completed!")
                print("=" * 50)
            
            # Search for safe spot with current model
            search_for_safe_spot(pipeline, align, current_model, colorizer)
            
            # Approach the safe spot
            proportional_approach(pipeline, align, current_model, colorizer)
            
            # Move towards the safe spot precisely
            print("Moving towards the safe spot...")
            send_body_ned_velocity(vx=0.45, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(vx=0.5, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(vx=0.20, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(0, 0, 0)  # Stop

            if get_angle_est() == (1 or 0):
                print("Angle safe enough to land")
            
            print(f"Safe spot #{spot_number} ({model_type}) completed!")
            print("----Landing----")    
            if model_type == "Slant Safe Spot":
                special_landing()
            else:
                land_vehicle()
            time.sleep(3)

            # Check if this is the last safe spot
            if i < TOTAL_SAFE_SPOTS - 1:
                print("Landed mode now changing to LOITER")
                set_mode("LOITER")
                print("Preparing TO TAKEOFF !!!")
                time.sleep(5)
                arm_vehicle()
                takeoff(TAKEOFF_ALTITUDE)
                time.sleep(2)
                
                # Give information about what's coming next
                next_spot = spot_number + 1
                next_model_path, next_model_type = get_model_info(next_spot)
                print(f"----Searching for the next safe spot #{next_spot} ({next_model_type})----")
                set_mode("GUIDED")
            
            # Small pause before looking for next safe spot
            time.sleep(2)

        # Final landing after all safe spots going to home position
        print("All safe spots visited. Returning to home position.")
        set_mode("LOITER")
        arm_vehicle()
        takeoff(TAKEOFF_ALTITUDE)
        time.sleep(3)
        print("Going to home position...")
        set_mode("GUIDED")
        rotate_towards(home_x, home_y)
        time.sleep(5)
        send_body_ned_velocity(vx=0.5, vy=0, vz=0)
        time.sleep(4)
        current_model = load_model(SAFE_SPOT_MODEL_PATH, "Normal Safe Spot")
        search_for_safe_spot(pipeline, align, current_model, colorizer)
        proportional_approach(pipeline, align, current_model, colorizer)
        # Move towards the safe spot precisely
        print("Moving towards the safe spot...")
        send_body_ned_velocity(vx=0.5, vy=0, vz=0)
        time.sleep(4)
        send_body_ned_velocity(vx=0.45, vy=0, vz=0)
        time.sleep(4)
        send_body_ned_velocity(vx=0.20, vy=0, vz=0)
        time.sleep(4)
        send_body_ned_velocity(0, 0, 0)  # Stop
        land_vehicle()
        time.sleep(3)


        print(f"\n=== Mission Complete! ===")
        print(f"Visited {MAX_NORMAL_SAFE_SPOTS} Normal Safe Spots")
        print(f"Visited {MAX_SLANT_SAFE_SPOTS} Slant Safe Spot")
        print(f"🎯 Total Safe Spots: {TOTAL_SAFE_SPOTS}")
        print("Mission accomplished.")

    except KeyboardInterrupt:
        print("\nMission interrupted by user. Landing immediately.")
        if master and master.motors_armed():
            land_vehicle()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if master and master.motors_armed():
            print("Attempting to land safely.")
            land_vehicle()
    finally:
        print("Stopping camera and closing windows.")
        pipeline.stop()
        cv2.destroyAllWindows()
        if master:
            master.close()


if __name__ == "__main__":
    main()