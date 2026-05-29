import open3d as o3d
import numpy as np
import random

# (Keep your ransac_plane_segmentation function as it is)
def ransac_plane_segmentation(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000, max_vertical_angle_deg=5):
    """
    Performs RANSAC plane segmentation on a 3D point cloud, focusing on "floor" planes.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        distance_threshold (float): Max distance a point can be from the plane to be an inlier.
        ransac_n (int): Number of points to sample to estimate the plane.
        num_iterations (int): Number of RANSAC iterations.
        max_vertical_angle_deg (float): Maximum angle (in degrees) from the vertical (Z) axis
                                        for a plane's normal to be considered a "floor".

    Returns:
        tuple: A tuple containing:
            - list: A list of open3d.geometry.PointCloud objects, where each represents a detected floor plane.
            - open3d.geometry.PointCloud: The remaining outlier points (including non-floor planes).
            - list: A list of the plane models (a, b, c, d) for each detected floor plane.
    """
    remaining_pcd = pcd
    floor_planes = []        # This list will store only the detected floor planes
    floor_plane_models = []  # This list will store the models for floor planes

    # Define a consistent color for detected floor planes (e.g., green)
    floor_color = [0, 0.8, 0] # Bright green

    plane_count = 0
    # Stop if remaining points are too few or if too many iterations have passed (arbitrary limit)
    # The 0.05 * len(pcd.points) means stop if fewer than 5% of original points remain.
    # The plane_count < 20 is an arbitrary limit to prevent infinite loops on very noisy data.
    while len(remaining_pcd.points) > 0.05 * len(pcd.points) and plane_count < 20:
        print(f"\n--- Running RANSAC for potential plane {plane_count + 1} ---")
        
        # Segment the largest dominant plane using RANSAC
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # If no inliers are found or too few, stop.
        # For a large point cloud (3M points), a minimum of 1000 points might be a reasonable threshold
        # to consider a plane significant enough. Adjust this based on your data and expected plane sizes.
        min_inliers_for_significant_plane = 1000
        if len(inliers) < min_inliers_for_significant_plane:
            print(f"No more significant planes found (less than {min_inliers_for_significant_plane} inliers).")
            break

        [a, b, c, d] = plane_model

        # Calculate the magnitude of the normal vector (a, b, c)
        normal_magnitude = np.sqrt(a**2 + b**2 + c**2)

        # Handle cases where the normal is degenerate (e.g., all zeros)
        if normal_magnitude == 0:
            print("Warning: Degenerate plane normal detected. Skipping this plane.")
            # Remove these points anyway to prevent infinite loop on problematic areas
            outlier_cloud = remaining_pcd.select_by_index(inliers, invert=True)
            remaining_pcd = outlier_cloud
            continue

        # Normalize the Z-component of the normal vector
        # We use np.abs(c_normalized) because a floor plane's normal could point (0,0,1) or (0,0,-1)
        # depending on its orientation in the coordinate system, and both are equally "vertical".
        c_normalized = c / normal_magnitude
        
        # Calculate the angle with the Z-axis (vertical)
        # np.clip is used to prevent floating point errors that can result in values slightly outside [-1, 1] for arccos
        angle_rad = np.arccos(np.clip(np.abs(c_normalized), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        print(f"Plane {plane_count + 1} equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        print(f"Normal angle with Z-axis: {angle_deg:.2f} degrees")
        print(f"Number of inliers: {len(inliers)}")

        # Check if the plane's normal is within the allowed vertical angle tolerance
        if angle_deg <= max_vertical_angle_deg:
            print(f"-> This is identified as a FLOOR plane (within {max_vertical_angle_deg} degree tolerance).")
            inlier_cloud = remaining_pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color(floor_color) # Assign the special floor color
            floor_planes.append(inlier_cloud)
            floor_plane_models.append(plane_model)
        else:
            print("-> This is NOT a floor plane (its normal is too far from vertical).")
            # These points are still part of a dominant plane, so we remove them
            # from the remaining_pcd to find the *next* dominant plane, but we don't
            # add them to our list of 'floor_planes'.
            pass

        # Always remove the detected plane's inliers from the remaining point cloud
        # to ensure the next RANSAC iteration works on fresh, unsegmented data.
        outlier_cloud = remaining_pcd.select_by_index(inliers, invert=True)
        remaining_pcd = outlier_cloud
        plane_count += 1

    # Color any truly unsegmented points grey (these are often general noise or complex surfaces)
    remaining_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    return floor_planes, remaining_pcd, floor_plane_models


if __name__ == "__main__":
    # Define the path to your .ply file
    ply_file_path = "cloud1.ply"# <--- IMPORTANT: Change this to your actual file path!

    print(f"Loading point cloud from: {ply_file_path}...")
    try:
        point_cloud = o3d.io.read_point_cloud(ply_file_path)
        if not point_cloud.has_points():
            raise ValueError("The loaded point cloud has no points.")
        print(f"Successfully loaded {len(point_cloud.points)} points.")
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        print("Please ensure the file path is correct and the file is a valid .ply file.")
        exit() # Exit if the file cannot be loaded

    # Optional: Downsample for faster processing, especially with 3M points.
    # Adjust voxel_size based on your data's scale and desired detail.
    # For a typical room, 0.05 meters (5cm) might be a good starting point.
    # print("Original number of points:", len(point_cloud.points))
    # voxel_size = 0.05
    # print(f"Downsampling point cloud with voxel size: {voxel_size}...")
    # point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    # print("Number of points after downsampling:", len(point_cloud.points))


    # Perform RANSAC segmentation to find floor planes
    print("\nStarting RANSAC floor plane segmentation...")
    # Adjust parameters based on your data's characteristics.
    # `distance_threshold`: How close points must be to the plane to be an inlier (noise tolerance).
    # `num_iterations`: More iterations increase robustness for large, noisy data, but take longer.
    # `max_vertical_angle_deg`: This is your specified tolerance for floor inclination.
    detected_floor_planes, remaining_outliers, floor_models = ransac_plane_segmentation(
        point_cloud,
        distance_threshold=0.03,      # Adjust based on your data's noise (e.g., 0.02 to 0.05 meters)
        ransac_n=3,
        num_iterations=4000,          # Increased iterations for larger datasets
        max_vertical_angle_deg=15     # Floors should be within +- 15 degrees of vertical
    )

    # Visualize the results
    print("\nVisualizing segmented floor planes and remaining outliers...")
    geometries_to_visualize = detected_floor_planes + [remaining_outliers]
    
    if not geometries_to_visualize:
        print("No floor planes detected or point cloud is empty after processing.")
    else:
        o3d.visualization.draw_geometries(
            geometries_to_visualize,
            window_name="RANSAC Floor Plane Segmentation (Loaded PLY)",
            width=1024,
            height=768
        )

    print("\nDetected Floor Plane Models:")
    if floor_models:
        for i, model in enumerate(floor_models):
            a, b, c, d = model
            print(f"Floor Plane {i+1}: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    else:
        print("No floor planes were detected.")