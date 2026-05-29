import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import sys
import os

# Suppress Ultralytics logs and warnings
os.environ['YOLO_VERBOSE'] = 'False'
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Load YOLO models
safe_spot_model = YOLO('files_cv/train17/weights/best.pt')
slant_safe_spot_model = YOLO('files_cv/train20/weights/best.pt')
angle_correction_model = YOLO('files_cv/train14/weights/best.pt')

# Restore stdout/stderr for OpenCV errors (optional)
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Setup colorizer and aligner
colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 8)
align = rs.align(rs.stream.color)

# Start pipeline
pipeline.start(config)

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
        blended = cv2.addWeighted(rgb_image, 0.85, depth_heatmap, 0.15, 0)

        # Inference (no verbose logging)
        safe_spot_results = safe_spot_model.predict(blended, verbose=False)
        slant_safe_spot_results = slant_safe_spot_model.predict(blended, verbose=False)
        angle_correction_results = angle_correction_model.predict(blended, verbose=False)

        annotated_frame = blended.copy()
        for results, color in zip(
            [safe_spot_results, slant_safe_spot_results, angle_correction_results],
            [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        ):
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = r.names[cls]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Blended RGB + Depth with Multi-YOLO', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
