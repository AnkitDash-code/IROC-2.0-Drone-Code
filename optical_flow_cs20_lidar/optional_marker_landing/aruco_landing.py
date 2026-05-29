#!/usr/bin/env python3
"""
Aruco landing node

Detects a 4-marker board (reading order 0,1,2,3) and publishes a PoseStamped
to MAVROS so the flight controller can accept VISION position estimates.

- Subscribes to the CS20 bridge topics by default:
  - /camera/camera/ir/image_raw  (mono8)
  - /tof_sensor/range             (sensor_msgs/Range)

- Publishes to (default): /mavros/vision_pose/pose (geometry_msgs/PoseStamped)

Place a camera calibration file as `camera_calib.npz` in this folder (contains
`camera_matrix` and `dist_coeffs`). Use the bundled `calibrate_checkerboard.py`
to generate it from chessboard photos.
"""

import os
import math
import time
import numpy as np
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, Range
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
except Exception as e:
    raise RuntimeError("This node requires ROS2 (rclpy) and cv_bridge: %s" % e)


class ArucoLandingNode(Node):
    def __init__(self):
        super().__init__("aruco_landing_node")

        # Parameters (can be overridden by ROS2 params or env)
        self.declare_parameter("camera_topic", "/camera/camera/ir/image_raw")
        self.declare_parameter("range_topic", "/tof_sensor/range")
        self.declare_parameter("publish_topic", "/mavros/vision_pose/pose")
        self.declare_parameter("marker_length", float(os.getenv("ARUCO_MARKER_LENGTH_M", "0.06")))
        self.declare_parameter("marker_separation", float(os.getenv("ARUCO_MARKER_SEPARATION_M", "0.02")))
        self.declare_parameter("altitude_threshold", float(os.getenv("ARUCO_ALTITUDE_THRESHOLD_M", "0.8")))
        self.declare_parameter("lock_frames_needed", int(os.getenv("ARUCO_LOCK_FRAMES", "5")))
        self.declare_parameter("publish_hz", float(os.getenv("ARUCO_PUBLISH_HZ", "15")))

        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.range_topic = str(self.get_parameter("range_topic").value)
        self.publish_topic = str(self.get_parameter("publish_topic").value)
        self.marker_length = float(self.get_parameter("marker_length").value)
        self.marker_separation = float(self.get_parameter("marker_separation").value)
        self.altitude_threshold = float(self.get_parameter("altitude_threshold").value)
        self.lock_frames_needed = int(self.get_parameter("lock_frames_needed").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        # CV/ROS helpers
        self.bridge = CvBridge()
        self.camera_matrix, self.dist_coeffs = self._load_calibration()

        # ArUco setup (robust across OpenCV versions)
        try:
            self.aruco = cv2.aruco
        except AttributeError:
            raise RuntimeError("cv2.aruco is not available in this OpenCV build")

        # pick a reasonable 4x4 dictionary available in the runtime
        for dname in ("DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000"):
            if hasattr(self.aruco, dname):
                dict_id = getattr(self.aruco, dname)
                break
        else:
            dict_id = getattr(self.aruco, "DICT_4X4_50", 0)

        try:
            self.dictionary = self.aruco.getPredefinedDictionary(dict_id)
        except Exception:
            self.dictionary = self.aruco.Dictionary_get(dict_id)

        try:
            self.detector_params = self.aruco.DetectorParameters_create()
        except Exception:
            self.detector_params = self.aruco.DetectorParameters()

        # Detection API compatibility: prefer aruco.detectMarkers, else use ArucoDetector
        if hasattr(self.aruco, "detectMarkers"):
            # legacy/top-level function available
            self._aruco_detect = lambda img: self.aruco.detectMarkers(img, self.dictionary, parameters=self.detector_params)
        else:
            detector_cls = getattr(self.aruco, "ArucoDetector", None)
            if detector_cls is not None:
                # try to construct an ArucoDetector with params, fallback to without
                try:
                    self.detector = detector_cls(self.dictionary, self.detector_params)
                except Exception:
                    try:
                        self.detector = detector_cls(self.dictionary)
                    except Exception:
                        self.detector = None
                if self.detector is None:
                    raise RuntimeError("cv2.aruco has ArucoDetector but could not instantiate it")
                self._aruco_detect = lambda img: self.detector.detectMarkers(img)
            else:
                raise RuntimeError("cv2.aruco lacks detectMarkers and ArucoDetector — install opencv-contrib-python")

        # build the 4-marker board in "reading order" (0:TL,1:TR,2:BL,3:BR)
        self.board, self.board_obj_points, self.board_ids = self._build_board(
            self.marker_length, self.marker_separation
        )

        # state
        self.current_range = None
        self.lock_count = 0
        self.state = "BLIND_CLIMB"  # BLIND_CLIMB, VISION_HOVER, BLIND_HOVER
        self.last_pose = None
        self.last_seen = 0.0

        # ROS pubs/subs
        self.pub_pose = self.create_publisher(PoseStamped, self.publish_topic, 10)
        self.sub_image = self.create_subscription(Image, self.camera_topic, self._image_cb, 10)
        self.sub_range = self.create_subscription(Range, self.range_topic, self._range_cb, 10)
        self.timer = self.create_timer(1.0 / max(0.1, self.publish_hz), self._timer_cb)

        self.get_logger().info(
            f"Aruco landing node started | camera={self.camera_topic} range={self.range_topic} publish={self.publish_topic}"
        )

    def _load_calibration(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, "camera_calib.npz")
        if os.path.exists(path):
            data = np.load(path)
            return data["camera_matrix"], data["dist_coeffs"]

        # fallback intrinsics (works but less accurate) — try to calibrate once for best results
        self.get_logger().warn("camera_calib.npz not found; using fallback intrinsics")
        fx = fy = 200.0
        cx = 160.0
        cy = 120.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64), np.zeros((5,), dtype=np.float64)

    def _build_board(self, marker_length, marker_separation):
        half_len = marker_length / 2.0
        offset = (marker_length + marker_separation) / 2.0
        centers = [
            [-offset, offset, 0.0],
            [offset, offset, 0.0],
            [-offset, -offset, 0.0],
            [offset, -offset, 0.0],
        ]

        obj_points = []
        for c in centers:
            cx, cy, cz = c
            corners = np.array(
                [
                    [cx - half_len, cy + half_len, cz],
                    [cx + half_len, cy + half_len, cz],
                    [cx + half_len, cy - half_len, cz],
                    [cx - half_len, cy - half_len, cz],
                ],
                dtype=np.float32,
            )
            obj_points.append(corners)

        board_ids = np.array([[0], [1], [2], [3]], dtype=np.int32)
        try:
            board = self.aruco.Board_create(obj_points, self.dictionary, board_ids)
        except Exception:
            # fallback for older/newer APIs
            board = self.aruco.Board(obj_points, self.dictionary, board_ids)
        return board, obj_points, board_ids

    def _range_cb(self, msg: Range):
        try:
            self.current_range = float(msg.range)
        except Exception:
            self.current_range = None

    def _image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        gray = frame
        # detect markers (supports multiple OpenCV aruco APIs)
        try:
            res = self._aruco_detect(gray)
        except Exception as e:
            self.get_logger().error(f"aruco detect failed: {e}")
            return

        corners = None
        ids = None
        rejected = None
        if isinstance(res, (tuple, list)):
            if len(res) >= 2:
                corners = res[0]
                ids = res[1]
                rejected = res[2] if len(res) > 2 else None
        else:
            # unexpected return
            self.get_logger().error("Unexpected return from ArUco detector")
            return
        now = time.time()

        if ids is not None and len(ids) > 0:
            # estimate pose of the board (works with subsets of markers)
            try:
                retval, rvec, tvec = self.aruco.estimatePoseBoard(corners, ids, self.board, self.camera_matrix, self.dist_coeffs)
            except Exception as e:
                self.get_logger().error(f"estimatePoseBoard error: {e}")
                return

            if retval and rvec is not None and tvec is not None:
                # Rotation R transforms board->camera: p_cam = R * p_board + t
                R, _ = cv2.Rodrigues(rvec)
                # camera position in board frame: p_board = -R^T * t
                cam_pos_board = -R.T.dot(tvec).reshape(3)

                # extract yaw from camera orientation in board frame
                R_board_cam = R.T
                yaw = math.atan2(R_board_cam[1, 0], R_board_cam[0, 0])
                qz = math.sin(yaw / 2.0)
                qw = math.cos(yaw / 2.0)

                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = "landing_pad"
                pose.pose.position.x = float(cam_pos_board[0])
                pose.pose.position.y = float(cam_pos_board[1])
                pose.pose.position.z = float(cam_pos_board[2])
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = float(qz)
                pose.pose.orientation.w = float(qw)

                self.last_pose = pose
                self.last_seen = now
                self.lock_count += 1

                if (
                    self.lock_count >= self.lock_frames_needed
                    and self.current_range is not None
                    and self.current_range > self.altitude_threshold
                    and self.state != "VISION_HOVER"
                ):
                    self.state = "VISION_HOVER"
                    self.get_logger().info("Vision lock acquired — engaging vision hand-off to FCU")

                self.lock_count = min(self.lock_count, self.lock_frames_needed + 10)
            else:
                # no valid pose
                self.lock_count = 0
        else:
            self.lock_count = 0
            if (now - self.last_seen) > 0.5 and self.state == "VISION_HOVER":
                self.state = "BLIND_HOVER"
                self.get_logger().warn("Lost vision lock — falling back to optical flow")

    def _timer_cb(self):
        # Only publish while in VISION_HOVER
        if self.state != "VISION_HOVER" or self.last_pose is None:
            return
        self.pub_pose.publish(self.last_pose)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
