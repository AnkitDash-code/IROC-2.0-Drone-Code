#!/usr/bin/env python3

"""ROS2 perception node that detects a target in RGB, deprojects it with depth,
and publishes/logs 3D target coordinates for the mission BT.
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
import time
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def _quat_to_rotation_matrix(x, y, z, w):
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=float)


class VLMPerceptionNode(Node):
    def __init__(self):
        super().__init__("vlm_perception_node")

        self.declare_parameter("rgb_topic", "/oakd/left/image_raw")
        self.declare_parameter("depth_topic", "/oakd/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/oakd/left/camera_info")
        self.declare_parameter("world_frame", "odom")
        default_log_path = os.environ.get(
            "ASCEND_DETECTIONS_LOG",
            str(Path.home() / "Drone" / "detections.jsonl")
        )
        self.declare_parameter("log_path", default_log_path)
        self.declare_parameter("candidate_topic", "/detection/candidate")
        self.declare_parameter("point_topic", "/detection/target_point")
        self.declare_parameter("backend", "yolo")
        self.declare_parameter("target_class", "target")
        self.declare_parameter("min_contour_area", 200.0)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("log_period_s", 0.75)
        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("yolo_device", "cuda:0")
        self.declare_parameter("fx", 460.0)
        self.declare_parameter("fy", 460.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("cy", 240.0)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.log_path = Path(str(self.get_parameter("log_path").value)).expanduser()
        self.candidate_topic = str(self.get_parameter("candidate_topic").value)
        self.point_topic = str(self.get_parameter("point_topic").value)
        self.backend = str(self.get_parameter("backend").value).strip().lower()
        self.target_class = str(self.get_parameter("target_class").value)
        self.min_contour_area = float(self.get_parameter("min_contour_area").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.log_period_s = float(self.get_parameter("log_period_s").value)
        self.yolo_model = str(self.get_parameter("yolo_model").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._camera_info = None
        self._last_emit_signature = None
        self._last_emit_time = 0.0
        self._yolo_model = None
        self._yolo_device = self._resolve_yolo_device()

        self._candidate_pub = self.create_publisher(String, self.candidate_topic, 10)
        self._point_pub = self.create_publisher(PointStamped, self.point_topic, 10)

        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            10,
        )

        self._rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic)
        self._depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=10,
            slop=0.08,
        )
        self._sync.registerCallback(self._perception_callback)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"VLM perception online | rgb={self.rgb_topic} depth={self.depth_topic} world={self.world_frame} backend={self.backend} device={self._yolo_device}"
        )

    def _resolve_yolo_device(self):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return self.yolo_device
        return "cpu"

    def _load_yolo_model(self):
        if self._yolo_model is not None:
            return self._yolo_model

        from ultralytics import YOLO

        self._yolo_model = YOLO(self.yolo_model)
        return self._yolo_model

    def _camera_info_callback(self, msg):
        self._camera_info = msg
        if len(msg.k) >= 9 and msg.k[0] > 0.0 and msg.k[4] > 0.0:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])

    def _detect_bbox(self, image_bgr):
        if self.backend == "yolo":
            try:
                model = self._load_yolo_model()
                result = model.predict(
                    image_bgr,
                    verbose=False,
                    conf=self.confidence_threshold,
                    device=self._yolo_device,
                )[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    best_box = max(result.boxes, key=lambda box: float(box.conf.item()) if box.conf is not None else 0.0)
                    xyxy = best_box.xyxy[0].cpu().numpy().tolist()
                    conf = float(best_box.conf.item()) if best_box.conf is not None else 0.0
                    cls_idx = int(best_box.cls.item()) if best_box.cls is not None else -1
                    label = model.names.get(cls_idx, self.target_class)
                    return xyxy, label, conf
            except Exception as exc:
                self.get_logger().warning(f"YOLO backend unavailable, falling back to color detection: {exc}")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        masks = [
            cv2.inRange(hsv, np.array([0, 90, 90]), np.array([15, 255, 255])),
            cv2.inRange(hsv, np.array([15, 90, 90]), np.array([30, 255, 255])),
        ]
        mask = cv2.bitwise_or(masks[0], masks[1])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < self.min_contour_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        confidence = min(0.99, area / 2500.0)
        return (float(x), float(y), float(x + w), float(y + h)), self.target_class, confidence

    def _sample_depth_m(self, depth_image, u, v):
        height, width = depth_image.shape[:2]
        u = int(max(0, min(width - 1, u)))
        v = int(max(0, min(height - 1, v)))

        window = depth_image[max(0, v - 2):min(height, v + 3), max(0, u - 2):min(width, u + 3)]
        values = window[np.isfinite(window)]
        values = values[values > 0.0]
        if values.size == 0:
            return None
        return float(np.median(values))

    def _deproject(self, u, v, depth_m):
        x = (float(u) - self.cx) * depth_m / self.fx
        y = (float(v) - self.cy) * depth_m / self.fy
        return np.array([x, y, depth_m], dtype=float)

    def _transform_to_world(self, frame_id, point_camera):
        if not frame_id:
            return self.world_frame, point_camera

        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                frame_id,
                Time(),
            )
        except Exception:
            return frame_id, point_camera

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rotation_matrix = _quat_to_rotation_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
        point_world = rotation_matrix @ point_camera + np.array([translation.x, translation.y, translation.z], dtype=float)
        return self.world_frame, point_world

    def _should_emit(self, signature):
        now = time.time()
        if self._last_emit_signature is None:
            self._last_emit_signature = signature
            self._last_emit_time = now
            return True

        if now - self._last_emit_time < self.log_period_s:
            return False

        if signature == self._last_emit_signature:
            self._last_emit_time = now
            return False

        self._last_emit_signature = signature
        self._last_emit_time = now
        return True

    def _log_candidate(self, candidate):
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate, sort_keys=True) + "\n")

    def _publish_candidate(self, candidate):
        message = String()
        message.data = json.dumps(candidate, sort_keys=True)
        self._candidate_pub.publish(message)

    def _perception_callback(self, rgb_msg, depth_msg):
        try:
            image_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Bridge conversion failed: {exc}")
            return

        detection = self._detect_bbox(image_bgr)
        if detection is None:
            return

        bbox, label, confidence = detection
        if confidence < self.confidence_threshold:
            return

        x1, y1, x2, y2 = bbox
        u = int(round((x1 + x2) * 0.5))
        v = int(round((y1 + y2) * 0.5))

        depth_array = np.array(depth_raw, dtype=float)
        if depth_array.ndim == 3:
            depth_array = depth_array[:, :, 0]

        if np.issubdtype(depth_raw.dtype, np.integer):
            depth_array = depth_array / 1000.0

        depth_m = self._sample_depth_m(depth_array, u, v)
        if depth_m is None or not math.isfinite(depth_m) or depth_m <= 0.0:
            return

        camera_frame = str(rgb_msg.header.frame_id or depth_msg.header.frame_id or "camera_link")
        camera_xyz = self._deproject(u, v, depth_m)
        world_frame, world_xyz = self._transform_to_world(camera_frame, camera_xyz)

        candidate = {
            "class": label,
            "conf": float(confidence),
            "pixel": {"u": u, "v": v},
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "camera_frame": camera_frame,
            "camera_xyz": {
                "x": float(camera_xyz[0]),
                "y": float(camera_xyz[1]),
                "z": float(camera_xyz[2]),
            },
            "world_frame": world_frame,
            "world_xyz": {
                "x": float(world_xyz[0]),
                "y": float(world_xyz[1]),
                "z": float(world_xyz[2]),
            },
            "stamp": rgb_msg.header.stamp.sec + (rgb_msg.header.stamp.nanosec * 1e-9),
            "img": self.rgb_topic,
            "depth_topic": self.depth_topic,
        }

        signature = (
            label,
            round(candidate["world_xyz"]["x"], 2),
            round(candidate["world_xyz"]["y"], 2),
            round(candidate["world_xyz"]["z"], 2),
            round(confidence, 2),
        )
        if not self._should_emit(signature):
            return

        point_msg = PointStamped()
        point_msg.header.stamp = rgb_msg.header.stamp
        point_msg.header.frame_id = world_frame
        point_msg.point.x = float(world_xyz[0])
        point_msg.point.y = float(world_xyz[1])
        point_msg.point.z = float(world_xyz[2])

        self._publish_candidate(candidate)
        self._point_pub.publish(point_msg)
        self._log_candidate(candidate)
        self.get_logger().info(
            f"Target {label} @ pixel=({u},{v}) world=({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f})"
        )


def main():
    rclpy.init()
    node = VLMPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()