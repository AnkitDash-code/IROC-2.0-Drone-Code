import cv2
import os
import time
import pyrealsense2 as rs # Import RealSense library
import numpy as np

# --- Configuration ---
# NOTE: CAMERA_SOURCE is no longer a direct index for RealSense,
#       but keeping it here as a placeholder for general camera config
#       The RealSense pipeline handles camera connection.
# CAMERA_SOURCE = 'realsense' # Identifier for RealSense camera

BASE_DATA_DIR = 'l_strip_dataset_full' # Base directory for your dataset
TARGET_SAVE_DIR = os.path.join(BASE_DATA_DIR, 'train') # Save directly to train folder
CLASSES = ['left', 'right', 'straight'] # All three classes
IMAGE_WIDTH = 224 # Target width for saved images (MobileNet input size)
IMAGE_HEIGHT = 224 # Target height for saved images (MobileNet input size)
RECORDING_INTERVAL_SEC = 0.2 # Save an image every 0.2 seconds (5 images/sec)
DELAY_BEFORE_START_SEC = 3 # Gives you time to get ready

def collect_data_realsense(base_data_dir, target_save_dir, classes, img_width, img_height, interval, delay):
    """
    Captures RGB images from Intel RealSense camera and saves them to class-specific directories.
    Press keys 'l', 'r', 's' to set current class for saving.
    Press 'q' to quit.
    """
    # Verify directories exist (they should if setup_project_directories.py was run)
    for cls in classes:
        os.makedirs(os.path.join(target_save_dir, cls), exist_ok=True)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    # You can adjust resolution and FPS here. Common values: (640, 480, 30) or (1280, 720, 30)
    # Ensure your USB connection can handle the bandwidth for higher resolutions/FPS.
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # BGR8 for OpenCV compatibility
    # If you also want depth stream (not directly used for classification but good to know):
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        # Start streaming
        pipeline.start(config)
        print("Intel RealSense camera started.")
    except Exception as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please ensure the camera is connected via USB 3.0 and the RealSense SDK is installed.")
        print("Try running 'realsense-viewer' to confirm camera functionality.")
        return

    print(f"\nStarting data collection in {delay} seconds. Get ready!")
    time.sleep(delay)

    current_class_idx = -1 # -1 means no class selected yet
    last_save_time = time.time()
    num_images_saved = {cls: len(os.listdir(os.path.join(target_save_dir, cls))) for cls in classes} # Count existing images

    print("\n--- Data Collection Started ---")
    print("Press 'l' for LEFT turn")
    print("Press 'r' for RIGHT turn")
    print("Press 's' for STRAIGHT/NO turn")
    print("Press 'q' to QUIT")
    print("Currently saving to: NONE")
    print(f"Initial image counts: {num_images_saved}")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame() # If you enabled depth stream

            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Display the live feed
            display_frame = color_image.copy() # Using color_image directly as it's BGR8
            text_color = (0, 255, 0) # Green
            if current_class_idx != -1:
                display_text = f"Saving to: {classes[current_class_idx].upper()} ({num_images_saved[classes[current_class_idx]]} images)"
            else:
                display_text = "Select class (l/r/s)"
                text_color = (0, 0, 255) # Red for unselected

            cv2.putText(display_frame, display_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
            cv2.imshow('L-Strip Data Collection (RealSense)', display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting data collection.")
                break
            elif key == ord('l'):
                current_class_idx = classes.index('left')
                print(f"Set class to: {classes[current_class_idx].upper()}")
            elif key == ord('r'):
                current_class_idx = classes.index('right')
                print(f"Set class to: {classes[current_class_idx].upper()}")
            elif key == ord('s'):
                current_class_idx = classes.index('straight')
                print(f"Set class to: {classes[current_class_idx].upper()}")

            if current_class_idx != -1 and (time.time() - last_save_time) > interval:
                # Resize and save the frame
                resized_frame = cv2.resize(color_image, (img_width, img_height))
                class_name = classes[current_class_idx]
                timestamp = int(time.time() * 1000) # Milliseconds timestamp
                filename = os.path.join(target_save_dir, class_name, f"{class_name}_{timestamp}.jpg")
                cv2.imwrite(filename, resized_frame)
                num_images_saved[class_name] += 1
                last_save_time = time.time()

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Data collection finished.")
        print("Total images collected in training directories:")
        for cls, count in num_images_saved.items():
            print(f"  {cls.upper()}: {count} images")
        print(f"\nRemember to manually move about 20% of these images from '{TARGET_SAVE_DIR}' to '{os.path.join(BASE_DATA_DIR, 'val')}' for validation.")


if __name__ == "__main__":
    # Ensure project directories are set up first
    # This is a good place to call the setup script for convenience
    # (or ensure you run it manually once before this)
    # import setup_project_directories # You would need to import and call its function
    # setup_project_directories.create_directories() 
    # For now, assume it's run manually.

    collect_data_realsense(BASE_DATA_DIR, TARGET_SAVE_DIR, CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT, RECORDING_INTERVAL_SEC, DELAY_BEFORE_START_SEC)

    print("\n--- IMPORTANT DATA COLLECTION TIPS ---")
    print("1. Aim for at least **100-200 images for EACH class** (LEFT, RIGHT, STRAIGHT).")
    print("2. Ensure a good mix of angles, distances, and varying lighting conditions for the L-strip on your **MARBLE** surface.")
    print("3. **Crucially**, manually ensure the L-strip's orientation truly matches the selected class ('l', 'r', 's') when pressing keys.")
    print("4. After collection, manually move about 20% of the images from the 'train' folders to the corresponding 'val' folders for proper training/validation split.")
    print("5. **RealSense Specific**: Ensure good lighting. Avoid strong reflections on the marble that might interfere with the camera's auto-exposure.")
    print("6. **RealSense Specific**: Use a **USB 3.0 port** for reliable and high-quality streaming.")
