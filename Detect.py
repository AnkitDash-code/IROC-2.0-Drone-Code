import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import sys
import os
import time

os.environ['YOLO_VERBOSE'] = 'False'
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
safe_spot_model = YOLO('yolo11n.engine')
# safe_spot_model = YOLO('files_cv/train17/weights/best.pt')
# slant_safe_spot_model = YOLO('files_cv/train20/weights/best.pt')
slant_safe_spot_model = YOLO('yolo11n.engine')

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

def safespoty1(blended_image, annotated_frame):
    results = safe_spot_model(blended_image, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = r.names[cls]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_frame

def safespotslopey2(blended_image, annotated_frame):
    results = slant_safe_spot_model(blended_image, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = r.names[cls]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return annotated_frame

def classify_centroid_position(cx, frame_width, tolerance_percent=10):
    mid_x = frame_width / 2
    tolerance_pixels = frame_width * (tolerance_percent / 100)
    
    if cx < mid_x - tolerance_pixels:
        return "Left"
    elif cx > mid_x + tolerance_pixels:
        return "Right"
    else:
        return "Center"

def get_detection_centroids(detection_function, blended_image, annotated_frame, frame_width):
    updated_annotated_frame = detection_function(blended_image, annotated_frame)
    
    results = None
    if detection_function == safespoty1:
        results = safe_spot_model(blended_image, verbose=False)
    elif detection_function == safespotslopey2:
        results = slant_safe_spot_model(blended_image, verbose=False)
    
    centroids_data = []
    if results:
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = r.names[cls]
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                position_tag = classify_centroid_position(cx, frame_width)
                
                centroids_data.append((label, conf, (cx, cy), position_tag))
                
                cv2.circle(updated_annotated_frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(updated_annotated_frame, f'C:({cx},{cy}) {position_tag}', (cx + 10, cy + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return updated_annotated_frame, centroids_data

def get_detection_areas(detection_function, blended_image, annotated_frame):
    updated_annotated_frame = detection_function(blended_image, annotated_frame)
    
    results = None
    if detection_function == safespoty1:
        results = safe_spot_model(blended_image, verbose=False)
    elif detection_function == safespotslopey2:
        results = slant_safe_spot_model(blended_image, verbose=False)
    
    areas_data = []
    if results:
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = r.names[cls]
                
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                areas_data.append((label, conf, area))
                
                cv2.putText(updated_annotated_frame, f'Area:{area}', (x1, y2 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return updated_annotated_frame, areas_data

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 8)
align = rs.align(rs.stream.color)

pipeline.start(config)

# FPS counter variables
prev_time = time.time()
fps = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        rgb_image = np.asanyarray(color_frame.get_data())
        
        colorized_depth = colorizer.colorize(depth_frame)
        depth_heatmap = np.asanyarray(colorized_depth.get_data())

        blended = depth_heatmap.copy()

        annotated_frame = blended.copy()
        
        frame_width = blended.shape[1] # Get frame width for centroid position classification

        annotated_frame, safe_spot_centroids = get_detection_centroids(safespoty1, blended, annotated_frame, frame_width)
        annotated_frame, slant_safe_spot_areas = get_detection_areas(safespotslopey2, blended, annotated_frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Display FPS on frame
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if safe_spot_centroids:
            print("Safe Spot Centroids (Label, Conf, (cx,cy), Position):", safe_spot_centroids)
        if slant_safe_spot_areas:
            print("Slant Safe Spot Areas:", slant_safe_spot_areas)
        
        cv2.imshow("Hi", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
