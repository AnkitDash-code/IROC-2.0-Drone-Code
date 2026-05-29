import sys
sys.path.append('/home/jetson123/Drone/SynexensPythonSDK4_4.2.4.0_202504281506')

from ransac import * # Assuming this imports all functions from your previous code
import time

def main():
    print("Starting LiDAR stream...")
    if not initialize_lidar_stream():
        print("Failed to initialize LiDAR stream")
        return
    
    print("LiDAR stream initialized successfully!")
    print("Commands:")
    print("  's'  - Get angles without visualization (single frame)")
    print("  'sv' - Get angles with visualization (single frame)")
    print("  'a'  - Get averaged angle estimation (over multiple frames)") # New command
    print("  'q'  - Quit")
    print("-" * 50)
    try:
        while True:
            user_input = input("Command: ").strip().lower()
            
            if user_input == 's':
                print("\nGetting current plane angles (single frame)...")
                start_time = time.perf_counter()
                angles = get_current_plane_angles(visualize_frame=False)
                duration = time.perf_counter() - start_time
                
                if angles is not None:
                    if angles[1] is not None:
                        print(f"Detected angles: Plane 1: {angles[0]:.2f}°, Plane 2: {angles[1]:.2f}°")
                    else:
                        print(f"Detected angles: Plane 1: {angles[0]:.2f}° (only one plane found)")
                    print(f"Processing time: {duration:.3f} seconds")
                else:
                    print("Could not detect plane angles")
                print("-" * 40)
            
            elif user_input == 'sv':
                print("\nGetting current plane angles with visualization (single frame)...")
                start_time = time.perf_counter()
                angles = get_current_plane_angles(visualize_frame=True)
                duration = time.perf_counter() - start_time
                
                if angles is not None:
                    if angles[1] is not None:
                        print(f"Detected angles: Plane 1: {angles[0]:.2f}°, Plane 2: {angles[1]:.2f}°")
                    else:
                        print(f"Detected angles: Plane 1: {angles[0]:.2f}° (only one plane found)")
                    print(f"Processing time: {duration:.3f} seconds")
                else:
                    print("Could not detect plane angles")
                print("-" * 40)
            
            # --- New 'a' command for averaged angle estimation ---
            elif user_input == 'a':
                print("\nGetting averaged angle estimation (over 5 frames)...")
                start_time = time.perf_counter()
                # Call get_angle_est, which already handles internal averaging and printing
                result_code = get_angle_est(n_frames=5) 
                duration = time.perf_counter() - start_time
                
                print(f"Averaged angle estimation result code: {result_code}")
                print(f"Total processing time for averaging: {duration:.3f} seconds")
                print("-" * 40)

            elif user_input == 'q':
                print("Exiting...")
                break
            
            else:
                print("Invalid command. Use 's' for single frame angles, 'sv' for single frame angles with visualization, 'a' for averaged estimation, or 'q' to quit.")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("Stopping LiDAR stream...")
        stop_lidar_stream()
        print("Program ended")

if __name__ == "__main__":
    main()
