import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

def get_rgb_depth_heatmap_overlay_and_save():
    """
    Initializes the RealSense camera, displays a colorized depth heat map overlay,
    and saves the current frame to the 'safe_spot' folder when 'w' is pressed.
    """
    # --- Define the output folder ---
    output_folder = 'safe_spot'
    
    # --- Create the directory if it does not exist ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    # --- 1. Setup and Configuration ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Create colorizer for depth visualization
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 8)  # Heat map color scheme
    
    align = rs.align(rs.stream.color)
    pipeline.start(config)
    
    try:
        while True:
            # --- 2. Frame Capture and Alignment ---
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # --- 3. Image Conversion ---
            rgb_image = np.asanyarray(color_frame.get_data())
            
            # Apply colorizer to depth frame to get heat map
            colorized_depth = colorizer.colorize(depth_frame)
            depth_heatmap = np.asanyarray(colorized_depth.get_data())
            
            # --- 4. Blend RGB with Heat Map ---
            blended = cv2.addWeighted(rgb_image, 0.85, depth_heatmap, 0.15, 0)
            
            # --- 5. Display the Result ---
            cv2.imshow('RGB-Depth Heat Map Blend (Press "w" to save, "q" to quit)', blended)
            
            # --- 6. Handle Keyboard Input ---
            key = cv2.waitKey(1) & 0xFF
            
            # Exit loop if 'q' is pressed
            if key == ord('q'):
                print("Exiting...")
                break
            
            # --- Save frame if 'w' is pressed ---
            if key == ord('w'):
                # Generate a unique filename using the current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                filename = os.path.join(output_folder, f"heatmap_capture_{timestamp}.png")
                
                # Save the blended image
                cv2.imwrite(filename, blended)
                print(f"Successfully saved frame to: {filename}")
    
    finally:
        # --- 7. Cleanup ---
        pipeline.stop()
        cv2.destroyAllWindows()

# Run the main function
if __name__ == '__main__':
    get_rgb_depth_heatmap_overlay_and_save()