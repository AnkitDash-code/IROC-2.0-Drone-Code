import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import math
import os
import sys


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
    )
except ImportError:
    print("Error: Make sure 'drone_control.py' is in the same directory.")
    sys.exit(1)

SAFE_SPOT_MODEL_PATH = "files_cv/train17/weights/best.pt"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15
TAKEOFF_ALTITUDE = 3
ROTATION_ANGLE_STEP = 5
ROTATION_SPEED = 5
SEARCH_ROTATION_ANGLE = 20
DETECTION_CONFIDENCE_THRESHOLD = 0.35
PROPORTIONAL_GAIN = 0.2
APPROACH_MAX_SPEED = 0.7
STOPPING_DISTANCE = 4.7
MAX_SAFE_SPOTS = 1  # Maximum number of safe spots to visit
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


# --- Main Flight Control Logic (Now with multiple safe spots) ---
def main():
    print("Initializing...")
    connect_to_vehicle()

    safe_spot_model = YOLO(SAFE_SPOT_MODEL_PATH)
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

    try:
        arm_vehicle()
        takeoff(TAKEOFF_ALTITUDE)
        time.sleep(2)
        set_mode("GUIDED")

        # [--- MODIFIED ---] Main loop for multiple safe spots
        for i in range(MAX_SAFE_SPOTS):
            print(f"\n=== Looking for Safe Spot #{i + 1} ===")
            
            # Search for safe spot
            search_for_safe_spot(pipeline, align, safe_spot_model, colorizer)
            
            # Approach the safe spot
            proportional_approach(pipeline, align, safe_spot_model, colorizer)
            
            # Move towards the safe spot precicely
            print("Moving towards the safe spot...")
            send_body_ned_velocity(vx=0.5, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(vx=0.5, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(vx=0.20, vy=0, vz=0)
            time.sleep(4)
            send_body_ned_velocity(0, 0, 0)  # Stop
            
            print(f"Safe spot #{i + 1} completed!")
            print("----Landing----")    

            land_vehicle()
            time.sleep(2)

            if i < MAX_SAFE_SPOTS - 1:
                print("Landed mode now changing to LOITER")
                set_mode("LOITER")
                print("Preparing TO TAKOFF !!!")
                time.sleep(5)
                takeoff(TAKEOFF_ALTITUDE)
                time.sleep(2)
                print("----Searching for the next safe spot----")
                set_mode("GUIDED")
            
            # Small pause before looking for next safe spot
            time.sleep(2)

        print(f"\n=== Mission Complete! Visited {MAX_SAFE_SPOTS} safe spots ===")
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