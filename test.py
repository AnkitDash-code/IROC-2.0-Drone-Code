import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import math
import platform
import torch
import sys
import os

# Add paths for drone control and ransac modules
sys.path.append('/home/jetson123/Drone/SynexensPythonSDK4_4.2.4.0_202504281506')
sys.path.append('/home/jetson123/Depth-Anythingv2-TensorRT-python/models')
# Import Depth Anything v2 module
from dpt import Dpt

try:
    from drone_control import (
        connect_to_vehicle,
        set_mode,
        arm_vehicle,
        takeoff,
        rotate_yaw,
        send_body_ned_velocity,
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

try:
    from events import (
        update_events,
        update_safe_spot,
    )
except ImportError:
    print("Error: Make sure you have events.py file in your directory where you use your flight program")
    sys.exit(1)

# TensorRT imports for Depth Anything v2
if platform.system() != "Darwin":
    import warnings
    import pycuda.driver as cuda
    import tensorrt as trt
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    TRT_LOGGER = trt.Logger()
    TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

SOBEL_MODEL_PATH = "/home/jetson123/Drone/files_cv/sobel.pt"
DEPTH_ANYTHING_ENGINE_PATH = "/home/jetson123/Depth-Anythingv2-TensorRT-python/checkpoints/depth_anything_v2_vitb.engine"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
TAKEOFF_ALTITUDE = 3
ROTATION_ANGLE_STEP = 5
ROTATION_SPEED = 5
SEARCH_ROTATION_ANGLE = 15
DETECTION_CONFIDENCE_THRESHOLD = 0.45
PROPORTIONAL_GAIN = 0.2
APPROACH_MAX_SPEED = 0.7
STOPPING_DISTANCE = 4.7
MAX_NORMAL_SAFE_SPOTS = 2
MAX_SLANT_SAFE_SPOTS = 0
TOTAL_SAFE_SPOTS = MAX_NORMAL_SAFE_SPOTS + MAX_SLANT_SAFE_SPOTS

YOLO_SIZE = 640

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ["YOLO_VERBOSE"] = "False"


def classify_centroid_position(cx, frame_width, tolerance_percent=3):
    """Classify the position of a centroid relative to the frame center"""
    mid_x = frame_width / 2
    tolerance_pixels = frame_width * (tolerance_percent / 100)
    if cx < mid_x - tolerance_pixels:
        return "Left"
    elif cx > mid_x + tolerance_pixels:
        return "Right"
    else:
        return "Center"
    
def find_and_analyze_safe_spot(pipeline, align, sobel_model, colorizer, dpt_model):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None, None
    
    color_image = np.asanyarray(color_frame.get_data())
    
    # Get AI depth from Depth Anything v2
    ai_depth = dpt_model.run(color_image)
    
    # Calculate Sobel gradients
    sx = cv2.Sobel(ai_depth, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(ai_depth, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.convertScaleAbs(mag)
    
    # Convert gradient to BGR for YOLO
    grad_for_yolo = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
    grad_for_yolo = cv2.resize(grad_for_yolo, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # Run sobel YOLO model on gradient
    results = sobel_model(grad_for_yolo, verbose=False, conf=0.25, iou=0.7, imgsz=YOLO_SIZE)
    annotated_frame = results[0].plot()
    
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
    return None, None, annotated_frame

# --- Closed-Loop Control Functions ---

def proportional_approach(pipeline, align, sobel_model, colorizer, dpt_model):
    print("\n--- Phase 2: Executing Proportional Approach ---")
    loop_hz = 5
    while True:
        position, distance, frame = find_and_analyze_safe_spot(
            pipeline, align, sobel_model, colorizer, dpt_model
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


def search_for_safe_spot(pipeline, align, sobel_model, colorizer, dpt_model):
    print("\n--- Phase 1: Searching for Safe Spot ---")
    no_detection_count = 0  # Counter for frames with no detection
    FRAMES_TO_ANALYZE = 5   # Number of frames to analyze before rotating
    
    while True:
        position, distance, frame = find_and_analyze_safe_spot(
            pipeline, align, sobel_model, colorizer, dpt_model
        )
        if frame is not None:
            cv2.imshow("Live Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        if position == "Center" and distance * math.cos(math.radians(30)) > 1.5:
            print(f"Target Centered! Distance: {distance:.2f} meters.")
            return
        elif position is None:
            # No detection found, increment counter
            no_detection_count += 1
            print(f"No detection found. Frame {no_detection_count}/{FRAMES_TO_ANALYZE}")
            
            if no_detection_count >= FRAMES_TO_ANALYZE:
                # Analyzed enough frames with no detection, rotate
                print(f"No target found after {FRAMES_TO_ANALYZE} frames. Rotating...")
                is_clockwise = True  # Default rotation when no detection
                rotate_yaw(SEARCH_ROTATION_ANGLE, ROTATION_SPEED, clockwise=is_clockwise)
                no_detection_count = 0  # Reset counter after rotation
            else:
                # Wait a bit before next frame analysis
                time.sleep(0.1)
        else:
            # Detection found but not centered, reset counter and rotate immediately
            no_detection_count = 0
            is_clockwise = position != "Left"
            angle = ROTATION_ANGLE_STEP
            print(f"Target is {position}. Rotating {'Clockwise' if is_clockwise else 'CCW'}.")
            rotate_yaw(angle, ROTATION_SPEED, clockwise=is_clockwise)


def load_sobel_model():
    """Load and return the sobel YOLO model"""
    print(f"Loading Sobel model from: {SOBEL_MODEL_PATH}")
    model = YOLO(SOBEL_MODEL_PATH)
    model.to(DEVICE)
    print("Sobel model loaded successfully!")
    return model


def get_model_info(spot_number):
    """Return model path and type based on spot number"""
    return SOBEL_MODEL_PATH, "Sobel Safe Spot"


# --- Main Flight Control Logic (Now with model switching) ---
def main():
    print("Initializing Noir...")
    update_events("Initializing Noir")
    connect_to_vehicle()
    initialize_lidar_stream()
    home_x, home_y, _ = get_current_position()
    update_safe_spot(home_x, home_y)

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
    
    # Initialize Depth Anything v2
    class Args:
        def __init__(self):
            self.engine = DEPTH_ANYTHING_ENGINE_PATH
    args = Args()
    dpt_model = Dpt(args)
    
    # Load sobel model
    sobel_model = load_sobel_model()
    
    # Warmup sobel model
    print("Warming up sobel model...")
    for _ in range(30):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            warmup_img = np.asanyarray(color_frame.get_data())
            _ = sobel_model(warmup_img, verbose=False, conf=0.25, iou=0.7, imgsz=YOLO_SIZE)
            break
    
    initialize_lidar_stream()
    print("Camera and Drone Connection Initialized.")
    update_events("Intitializing Done")


    try:
        arm_vehicle()
        update_events(f"Taking off to {TAKEOFF_ALTITUDE}")
        takeoff(TAKEOFF_ALTITUDE)
        time.sleep(3)
        set_mode("GUIDED")

        update_events("Finding Safe spots")

        # Main loop for multiple safe spots using sobel model
        for i in range(TOTAL_SAFE_SPOTS):
            spot_number = i + 1
            model_path, model_type = get_model_info(spot_number)
            
            print(f"\n=== Looking for Safe Spot #{spot_number} ({model_type}) ===")
            
            # Search for safe spot with sobel model
            search_for_safe_spot(pipeline, align, sobel_model, colorizer, dpt_model)
            update_events("Found a Safe spot")
            
            # Approach the safe spot
            proportional_approach(pipeline, align, sobel_model, colorizer, dpt_model)
            
            # Move towards the safe spot precisely
            print("Moving towards the safe spot...")
            update_events("Moving towards the safe spot...")
            send_body_ned_velocity(vx=0.5, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(vx=0.45, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(vx=0.20, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(0, 0, 0)  # Stop

            # pos_x, pos_y, _ = get_current_position()
            # update_safe_spot(pos_x, pos_y)
            # update_events(f"Spot{spot_number}: x: {pos_x}, y: {pos_y}")
            start = time.time()
            status = get_angle_est()
            if status in [0, 1]:
                print("Angle safe enough to land")
                if status == 0:
                    print("Flat spot")
                    update_events("Flat safe spot")
                else:
                    print("Slope spot")
                    update_events("Sloped Safe Spot")
            print(time.time() - start)
            print(f"Safe spot #{spot_number} ({model_type}) completed!")
            print("----Landing----")    
            update_events("Landing on Safe spot")
            land_vehicle()
            pos_x, pos_y, _ = get_current_position()
            update_events(f"Spot{spot_number}: x: {pos_x}, y: {pos_y}")
            update_safe_spot(pos_x, pos_y)
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
        distance = rotate_towards(home_x, home_y)
        if distance > 4:
            time.sleep(1)
            send_body_ned_velocity(vx=0.5, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(vx=0.5, vy=0, vz=0)
            time.sleep(4)
        else:
            time.sleep(1)
            send_body_ned_velocity(vx=0.5, vy=0, vz=0)
            time.sleep(4)
        
        current_model = load_sobel_model()
        search_for_safe_spot(pipeline, align, current_model, colorizer, dpt_model)
        proportional_approach(pipeline, align, current_model, colorizer, dpt_model)
        # Move towards the safe spot precisely
        print("Moving towards the safe spot...")
        send_body_ned_velocity(vx=0.5, vy=0, vz=0)
        time.sleep(4)
        send_body_ned_velocity(vx=0.5, vy=0, vz=0)
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