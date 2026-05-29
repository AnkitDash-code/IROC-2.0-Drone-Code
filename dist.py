import pyrealsense2 as rs
import numpy as np
import cv2

def get_rgb_depth_side_by_side_with_center_distance():
    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Enable depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        # Get the depth scale for converting raw depth values to meters
        # This is crucial for accurate distance calculation
        depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        print(f"Depth scale for this device: {depth_scale} meters per unit")

        while True:
            # Wait for a new set of frames from the camera
            frames = pipeline.wait_for_frames()

            # Align the depth frame to the color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # --- Calculate and display center pixel distance ---
            height, width = depth_image.shape
            center_x = width // 2
            center_y = height // 2

            # Get the raw depth value at the center pixel from the ALIGNED depth image
            center_depth_raw = depth_image[center_y, center_x]
            
            # Convert raw depth to meters
            center_distance_meters = center_depth_raw * depth_scale

            # Prepare distance string for display
            distance_text = ""
            if center_depth_raw > 0: # Check if the depth value is valid (not 0)
                distance_text = f"{center_distance_meters:.2f}m"
                print(f"Distance at center pixel: {center_distance_meters:.3f} meters")
            else:
                distance_text = "N/A"
                print("Distance at center pixel: INVALID (no data)")
            
            # Overlay marker and text on the RGB image
            # Draw a circle at the center pixel
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1) # Red circle
            # Put text (distance) near the center
            cv2.putText(color_image, distance_text, (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Red text

            # --- Prepare depth image for visualization ---
            # Apply colormap on depth image (jet is a common choice)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # --- Stack both images horizontally ---
            # Now `color_image` already has the center marker and text drawn on it
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.imshow('RGB (w/ Center Distance) | Colorized Depth', images)

            # Break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()

# Run the visualization function
get_rgb_depth_side_by_side_with_center_distance()