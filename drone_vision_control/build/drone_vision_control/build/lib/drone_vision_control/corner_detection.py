#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
import math

class YellowBorderDetector(Node):
    def __init__(self):
        super().__init__('yellow_border_detector')
        self.bridge = CvBridge()
        
        # Subscribe to the RealSense D455 color image topic
        self.image_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10
        )
        
        # Publishers for processed image and detected yellow L-corner
        self.debug_image_pub = self.create_publisher(Image, "/yellow_border_detector/debug_image", 10)
        self.yellow_corner_pub = self.create_publisher(Point, "/yellow_border_detector/yellow_l_corner", 10)

        self.get_logger().info("Yellow L-Corner Detector - Looking for THE bright yellow L-corner!")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        processed_image = cv_image.copy()
        height, width = cv_image.shape[:2]
        
        # --- STEP 1: DETECT BRIGHT YELLOW (like your tape) ---
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Broader HSV range for bright yellow tape/floor marking
        lower_yellow = np.array([10, 100, 100])   # Broader range for bright yellow
        upper_yellow = np.array([40, 255, 255])   # Includes more yellow variations
        
        # Create yellow mask
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        
        # Aggressive cleanup for floor tape detection
        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((15, 15), np.uint8)  # Larger kernel to connect tape segments
        
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_open)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # --- STEP 2: FIND THE LARGEST YELLOW REGION ---
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            self.get_logger().warn("No yellow regions detected!")
            try:
                ros_image = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
                self.debug_image_pub.publish(ros_image)
            except:
                pass
            return
        
        # Find the largest yellow contour (should be your L-shaped tape)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        
        # Must be significantly large (floor tape should be big)
        if largest_area < 5000:
            self.get_logger().warn(f"Yellow region too small: {largest_area} pixels")
            try:
                ros_image = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
                self.debug_image_pub.publish(ros_image)
            except:
                pass
            return
        
        # Draw the detected yellow region
        cv2.drawContours(processed_image, [largest_contour], -1, (0, 255, 255), 3)
        
        # --- STEP 3: FIND THE L-CORNER ---
        # Approximate the contour to find vertices
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.02 * perimeter  # More precise approximation
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        vertices = approx.reshape(-1, 2)
        self.get_logger().info(f"Found {len(vertices)} vertices in yellow shape")
        
        # Find the inner corner (L-corner) by looking for ~90 degree angles
        l_corner = None
        best_angle_diff = float('inf')
        
        if len(vertices) >= 3:
            for i in range(len(vertices)):
                p1 = vertices[i]
                p2 = vertices[(i + 1) % len(vertices)]
                p3 = vertices[(i + 2) % len(vertices)]
                
                # Calculate vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate angle
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norms == 0:
                    continue
                
                cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                
                # Look for angles close to 90 degrees (inner corner of L)
                angle_diff = abs(angle_deg - 90)
                
                if angle_diff < 25 and angle_diff < best_angle_diff:  # Within 25 degrees of 90
                    best_angle_diff = angle_diff
                    l_corner = p2
        
        # --- STEP 4: ALTERNATIVE METHOD - Use convexity defects for L-corner ---
        if l_corner is None:
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(largest_contour, hull)
                
                if defects is not None:
                    max_defect = 0
                    defect_point = None
                    
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        far = tuple(largest_contour[f][0])
                        
                        if d > max_defect:
                            max_defect = d
                            defect_point = far
                    
                    if defect_point and max_defect > 1000:  # Significant defect
                        l_corner = np.array(defect_point)
        
        # --- STEP 5: FALLBACK - Use moments to find approximate center ---
        if l_corner is None:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                l_corner = np.array([cx, cy])
                self.get_logger().info("Using centroid as L-corner approximation")
        
        # --- STEP 6: VISUALIZE AND PUBLISH ---
        if l_corner is not None:
            x, y = int(l_corner[0]), int(l_corner[1])
            
            # Draw BIG marker for the L-corner
            cv2.circle(processed_image, (x, y), 20, (0, 0, 255), -1)      # Red filled circle
            cv2.circle(processed_image, (x, y), 25, (255, 255, 255), 4)   # White outline
            cv2.circle(processed_image, (x, y), 30, (0, 0, 0), 2)         # Black outer outline
            
            # Add text label
            cv2.putText(processed_image, 'L-CORNER', (x-40, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(processed_image, 'L-CORNER', (x-40, y-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Publish the L-corner point
            corner_msg = Point()
            corner_msg.x = float(x)
            corner_msg.y = float(y)
            corner_msg.z = 0.0
            self.yellow_corner_pub.publish(corner_msg)
            
            self.get_logger().info(f"🎯 YELLOW L-CORNER DETECTED at: ({x}, {y})")
        else:
            self.get_logger().warn("Could not determine L-corner location")
        
        # Add debug info
        cv2.putText(processed_image, f"Yellow Area: {int(largest_area)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed_image, f"Vertices: {len(vertices)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw all vertices for debugging
        for i, vertex in enumerate(vertices):
            cv2.circle(processed_image, tuple(vertex), 8, (255, 0, 255), -1)
            cv2.putText(processed_image, str(i), tuple(vertex + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Publish debug image
        try:
            ros_image = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            self.debug_image_pub.publish(ros_image)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YellowBorderDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
