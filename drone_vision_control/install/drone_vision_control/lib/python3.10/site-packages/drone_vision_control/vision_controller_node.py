#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo # Added CameraInfo import
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque

# --- Core Camera & Rectification Parameters ---
# Base height for scaling the rectified world dimensions
BASE_HEIGHT = 2.0  # meters
# Rectified world dimensions at the base height
BASE_RECTIFIED_WORLD_WIDTH = 1.0  # meters
BASE_RECTIFIED_WORLD_HEIGHT = 1.5 # meters

CAMERA_TILT_DEG = 30.0         # degrees: Camera's downward tilt angle from horizontal
RECTIFIED_PIXEL_WIDTH = 400
RECTIFIED_PIXEL_HEIGHT = 600

# --- Vision Processing Parameters ---
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)
ANGLE_TOLERANCE_DEG = 5.0
MIN_CONTOUR_AREA = 1000
LINE_HISTORY_LENGTH = 5

class DynamicPlankAlignment(Node):
    def __init__(self):
        super().__init__('dynamic_plank_alignment')
        self.bridge = CvBridge()
        
        # Define a QoS profile for subscriptions
        # Changed depth from 1 to 10 for better compatibility with RealSense
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10 
        )
        
        # --- Camera Info & Homography State ---
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.image_width = None
        self.image_height = None
        self.camera_info_received = False
        self.H_rectification = None
        
        # Current estimated camera height (can be updated dynamically if sensor input available)
        self.estimated_camera_height = BASE_HEIGHT 
        
        # --- Subscribers ---
        # Image subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', 
            self.image_callback, qos_profile)
        
        # Camera Info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', # Correct topic for camera intrinsics
            self.camera_info_callback, qos_profile)
            
        # --- Publishers ---
        self.annotated_pub = self.create_publisher(Image, 'vision/annotated_frame', 10)
        self.mask_pub = self.create_publisher(Image, 'vision/mask_frame', 10)
        self.rectified_pub = self.create_publisher(Image, 'vision/rectified_frame', 10)
        
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.current_frame = None
        self.plank_mask = None
        self.line_history = deque(maxlen=LINE_HISTORY_LENGTH)
        self.last_valid_line = None

        self.get_logger().info("DynamicPlankAlignment node started. Waiting for CameraInfo...")

    def camera_info_callback(self, msg):
        # Added debug log to confirm callback entry
        self.get_logger().info("CameraInfo callback received a message!") 
        
        # Check if camera info has significantly changed before re-calculating
        # This prevents redundant calculations if info is published frequently but unchanging
        if self.camera_info_received:
            if (self.image_width == msg.width and
                self.image_height == msg.height and
                np.allclose(self.camera_matrix, np.array(msg.k).reshape(3, 3)) and
                np.allclose(self.dist_coeffs, np.array(msg.d))):
                return # No change, no need to re-calculate
        
        self.get_logger().info("Updating camera parameters from CameraInfo.")
        self.image_width = msg.width
        self.image_height = msg.height
        self.camera_matrix = np.array(msg.k).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(msg.d).astype(np.float32)
        
        # Calculate new camera matrix and ROI for undistortion
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, 
            (self.image_width, self.image_height), 1, 
            (self.image_width, self.image_height)
        )
        self.camera_info_received = True
        self.get_logger().info(f"Camera resolution: {self.image_width}x{self.image_height}")
        self.get_logger().info(f"Camera matrix:\n{self.camera_matrix}")
        self.get_logger().info(f"Distortion coeffs: {self.dist_coeffs}")
        
        # Recalculate homography with new intrinsics and current height
        self._calculate_homography_matrix()

    def image_callback(self, msg):
        # Added debug log to confirm callback entry
        self.get_logger().info("Image callback received a message!") 
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def _calculate_homography_matrix(self):
        if not self.camera_info_received:
            self.get_logger().warn("Cannot calculate homography: CameraInfo not received yet.")
            self.H_rectification = None
            return

        # Calculate current rectified world dimensions based on height scaling
        current_rectified_world_width = BASE_RECTIFIED_WORLD_WIDTH * (self.estimated_camera_height / BASE_HEIGHT)
        current_rectified_world_height = BASE_RECTIFIED_WORLD_HEIGHT * (self.estimated_camera_height / BASE_HEIGHT)

        # Calculate Y-coordinate in world plane for the center of the rectified view
        # This point (0, Y_center_world, 0) in world frame is viewed by camera at (0,0,height)
        fixed_rectified_y_center = self.estimated_camera_height / math.tan(math.radians(CAMERA_TILT_DEG))

        # Define Camera (C) to World (W) transformation (T_cw) for clarity.
        # World: X-right, Y-forward, Z-up. Ground plane is Z=0.
        # Camera: X-right, Y-down, Z-forward (into scene, positive depth).

        # Rotation from World to Camera frame (R_wc)
        # Camera is pitched down by CAMERA_TILT_DEG from horizontal.
        # Total rotation about X-axis for world's Z-axis to align with camera's forward (Z) axis.
        # If camera is horizontal, its Z is parallel to world's Y. Tilt makes it point down.
        # A 0-degree tilt means camera looks horizontally. A 90-degree tilt means camera looks straight down.
        # We model rotation such that world's Z-up axis aligns with camera's negative Y-axis when looking perfectly down.
        # For a downward tilt, the camera's Z-axis (forward) is angled relative to the world's ground plane.
        # It's a rotation of (90 + CAMERA_TILT_DEG) around X-axis.
        alpha_rad = math.radians(90 + CAMERA_TILT_DEG) 
        R_wc = np.array([
            [1, 0, 0],
            [0, math.cos(alpha_rad), -math.sin(alpha_rad)],
            [0, math.sin(alpha_rad), math.cos(alpha_rad)]
        ], dtype=np.float32)

        # Translation vector: position of the world origin (0,0,0) in camera coordinates.
        # Camera's position in world coordinates is (0, 0, self.estimated_camera_height).
        # t_wc = -R_wc @ T_world_in_cam_frame
        T_cam_in_world = np.array([0, 0, self.estimated_camera_height], dtype=np.float32)
        t_wc = -R_wc @ T_cam_in_world.reshape(3,1)

        # Projection matrix P = K * [R_wc | t_wc]
        P_matrix = self.new_camera_matrix @ np.hstack((R_wc, t_wc))

        # Define the 4 corners of the rectified region in 3D world coordinates (Z=0 on ground)
        world_points_3d = np.array([
            [-current_rectified_world_width / 2, fixed_rectified_y_center + current_rectified_world_height / 2, 0], # Top-Left
            [current_rectified_world_width / 2, fixed_rectified_y_center + current_rectified_world_height / 2, 0],  # Top-Right
            [current_rectified_world_width / 2, fixed_rectified_y_center - current_rectified_world_height / 2, 0],  # Bottom-Right
            [-current_rectified_world_width / 2, fixed_rectified_y_center - current_rectified_world_height / 2, 0] # Bottom-Left
        ], dtype=np.float32)

        # Convert 3D world points to homogeneous coordinates
        world_points_homogeneous = np.vstack((world_points_3d.T, np.ones((1, world_points_3d.shape[0]))))
        
        # Project 3D world points to 2D image plane (homogeneous coordinates)
        pts_src_homogeneous = P_matrix @ world_points_homogeneous
        
        # Check for non-positive Z-coordinates (points behind or at the camera's projection plane)
        # This would cause division by zero or negative depths, leading to invalid projections.
        if np.any(pts_src_homogeneous[2, :] <= 0):
            self.get_logger().error(f"ERROR: Projected Z-coordinates non-positive: {pts_src_homogeneous[2, :]}. Adjust camera height, tilt, or world view parameters.")
            self.H_rectification = None
            return

        # Convert 2D homogeneous points to Cartesian (pixel) coordinates
        pts_src = (pts_src_homogeneous[:2, :] / pts_src_homogeneous[2, :]).T
        
        # Explicitly ensure pts_src is float32 and has the correct shape for OpenCV
        pts_src_final = pts_src.astype(np.float32)
        
        # You can uncomment these debug lines if you hit a similar error again
        # self.get_logger().info(f"DEBUG: pts_src_final before getPerspectiveTransform:\n{pts_src_final}")
        # self.get_logger().info(f"DEBUG: pts_src_final shape: {pts_src_final.shape}, dtype: {pts_src_final.dtype}")

        # Final assertion check before OpenCV call (same as OpenCV's internal check)
        if not (pts_src_final.shape == (4, 2) and pts_src_final.dtype == np.float32):
            self.get_logger().error(f"FATAL ERROR: pts_src_final is not 4x2 float32 as expected by OpenCV. Shape: {pts_src_final.shape}, Dtype: {pts_src_final.dtype}")
            self.H_rectification = None
            return

        # Check for NaN/Inf values (should be caught by the Z-coordinate check, but good double-check)
        if np.any(np.isnan(pts_src_final)) or np.any(np.isinf(pts_src_final)):
            self.get_logger().error(f"ERROR: pts_src_final contains NaN/Inf values. Homography will fail.")
            self.H_rectification = None
            return

        # Define the 4 destination points in the rectified pixel space
        pts_dst = np.array([
            [0, 0],
            [RECTIFIED_PIXEL_WIDTH, 0],
            [RECTIFIED_PIXEL_WIDTH, RECTIFIED_PIXEL_HEIGHT],
            [0, RECTIFIED_PIXEL_HEIGHT]
        ], dtype=np.float32)

        # Calculate the perspective transform matrix (homography)
        self.H_rectification = cv2.getPerspectiveTransform(pts_src_final, pts_dst)
        self.get_logger().info(f"Homography Matrix H calculated successfully for height {self.estimated_camera_height:.2f}m.")

    def create_plank_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        return mask

    def find_dynamic_center_line(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        large_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        if not large_contours: return None
        
        height, width = mask.shape[:2]
        image_center = (width//2, height//2)
        
        def contour_centrality(c):
            M = cv2.moments(c)
            if M["m00"] == 0: return float('inf')
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return math.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
        
        # Sort by centrality first, then by area (larger areas preferred if centrality is similar)
        large_contours.sort(key=lambda c: (contour_centrality(c), -cv2.contourArea(c)))
        
        best_contour = large_contours[0]
        
        # Fit a line to the largest, most central contour
        [vx, vy, x, y] = cv2.fitLine(best_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Extend the line to the image boundaries
        height, width = mask.shape[:2]
        if vy == 0: # Horizontal line
            p1, p2 = (0, int(y)), (width - 1, int(y))
        elif vx == 0: # Vertical line
            p1, p2 = (int(x), 0), (int(x), height - 1)
        else:
            # Calculate points at top (y=0) and bottom (y=height-1) of the image
            # Line equation: (x - x0) / vx = (y - y0) / vy
            # x = x0 + vx * (y - y0) / vy
            x_at_top = int(x + vx * (0 - y) / vy)
            x_at_bottom = int(x + vx * (height - 1 - y) / vy)
            p1 = (np.clip(x_at_top, 0, width - 1), 0)
            p2 = (np.clip(x_at_bottom, 0, width - 1), height - 1)
        
        return (p1, p2)

    def update_line_history(self, new_line):
        if new_line is not None:
            self.line_history.append(new_line)
            self.last_valid_line = new_line
        elif self.last_valid_line is not None:
            # If no new line detected, repeat the last valid line to smooth out momentary losses
            self.line_history.append(self.last_valid_line)
            
        if len(self.line_history) > 0:
            avg_p1 = np.mean([line[0] for line in self.line_history], axis=0).astype(int)
            avg_p2 = np.mean([line[1] for line in self.line_history], axis=0).astype(int)
            return (tuple(avg_p1), tuple(avg_p2))
        return None

    def calculate_angle(self, line):
        if line is None: return None
            
        p1, p2 = line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Avoid division by zero for perfectly horizontal/vertical lines
        if dy == 0: return 90.0 if dx > 0 else -90.0 if dx < 0 else 0.0
            
        angle = math.degrees(math.atan2(dx, dy))
        
        # Normalize angle to -90 to 90 degrees (relative to vertical center line)
        if angle > 90: angle -= 180
        elif angle < -90: angle += 180
            
        return angle

    def control_loop(self):
        # Ensure camera info is received, an image frame is available, and homography is calculated
        if self.current_frame is None or not self.camera_info_received or self.H_rectification is None:
            self.get_logger().debug("Waiting for image, camera info, or homography calculation.")
            return

        frame = self.current_frame.copy()
        
        # Undistort the raw image using parameters from CameraInfo
        undistorted_frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        
        # Rectify the undistorted image to a top-down view
        rectified_frame = cv2.warpPerspective(undistorted_frame, self.H_rectification, 
                                              (RECTIFIED_PIXEL_WIDTH, RECTIFIED_PIXEL_HEIGHT))
        
        # Process the rectified frame for the yellow plank
        self.plank_mask = self.create_plank_mask(rectified_frame)
        current_line = self.find_dynamic_center_line(self.plank_mask)
        smoothed_line = self.update_line_history(current_line)
        
        # Prepare visualization frames
        vis_frame = rectified_frame.copy() 
        mask_vis = cv2.cvtColor(self.plank_mask, cv2.COLOR_GRAY2BGR)
        
        # Draw a vertical center line on the rectified image for reference
        cv2.line(vis_frame, (RECTIFIED_PIXEL_WIDTH//2, 0), (RECTIFIED_PIXEL_WIDTH//2, RECTIFIED_PIXEL_HEIGHT), (255, 0, 0), 2)
        cv2.line(mask_vis, (RECTIFIED_PIXEL_WIDTH//2, 0), (RECTIFIED_PIXEL_WIDTH//2, RECTIFIED_PIXEL_HEIGHT), (255, 0, 0), 2)
        
        if smoothed_line:
            p1, p2 = smoothed_line
            cv2.line(vis_frame, p1, p2, (0, 255, 0), 3) # Green line for the detected plank
            cv2.line(mask_vis, p1, p2, (0, 255, 0), 3)
            
            angle = self.calculate_angle(smoothed_line)
            if angle is not None:
                abs_angle = abs(angle)
                
                # Determine alignment status and action
                if abs_angle <= ANGLE_TOLERANCE_DEG:
                    status_text = f"ALIGNED: {angle:.1f}°"
                    status_color = (0, 255, 0) # Green
                    action_text = "No rotation needed"
                else:
                    direction = "RIGHT" if angle > 0 else "LEFT"
                    turn_amount = abs(angle)
                    status_text = f"MISALIGNED: {angle:.1f}°"
                    status_color = (0, 0, 255) # Red
                    action_text = f"ROTATE {direction} by {turn_amount:.1f}°"
                
                # Display status on both visualization frames
                for f in [vis_frame, mask_vis]:
                    cv2.putText(f, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.putText(f, action_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2) # Orange
        else:
            status_text = "NO LINE DETECTED"
            status_color = (0, 0, 255) # Red
            for f in [vis_frame, mask_vis]:
                cv2.putText(f, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Publish the visualization frames
        try:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(vis_frame, 'bgr8'))
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask_vis, 'bgr8'))
            self.rectified_pub.publish(self.bridge.cv2_to_imgmsg(rectified_frame, 'bgr8'))
        except Exception as e:
            self.get_logger().error(f"Publish error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DynamicPlankAlignment()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
