import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

os.makedirs('facing_left', exist_ok=True)
os.makedirs('facing_right', exist_ok=True)
os.makedirs('parallel', exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
            
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense', color_image)
        
        key = cv2.waitKey(1) & 0xFF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if key == ord('a'):
            cv2.imwrite(f'facing_left/img_{timestamp}.jpg', color_image)
            print(f'Saved to facing_left')
        elif key == ord('d'):
            cv2.imwrite(f'facing_right/img_{timestamp}.jpg', color_image)
            print(f'Saved to facing_right')
        elif key == ord('w'):
            cv2.imwrite(f'parallel/img_{timestamp}.jpg', color_image)
            print(f'Saved to parallel')
        elif key == ord('q'):
            break
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
