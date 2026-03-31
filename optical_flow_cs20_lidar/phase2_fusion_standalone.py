#!/usr/bin/env python3
"""
CS20 LiDAR Phase2 fusion launcher.

Runs CS20 fusion with IR optical flow primary and optional depth-ICP secondary.
"""

import os
import threading
import time

from optical_flow_realsense import phase2_standalone as rs
from optical_flow_realsense import phase2_fusion_standalone as rf

from .depth_icp_odometry import CS20DepthICPEstimator
from .ir_optical_flow_estimator import CS20IROpticalFlowEstimator
from .phase2_standalone import cleanup_lidar, launch_hardware_lidar


class CS20Phase2FusionNode(rs.Phase2Node):
    def __init__(self):
        super().__init__()

        # Replace whichever mapper parent selected with explicit CS20 fusion pair.
        primary = CS20IROpticalFlowEstimator()

        secondary = None
        use_icp_secondary = os.environ.get("CS20_FUSION_USE_ICP", "1") in ("1", "true", "True")
        if use_icp_secondary:
            secondary = CS20DepthICPEstimator(
                use_gpu=os.environ.get("CS20_USE_GPU", "1") in ("1", "true", "True")
            )
            self.node.get_logger().info("CS20 fusion secondary mapper: DEPTH_ICP")
        else:
            self.node.get_logger().info("CS20 fusion running primary-only (IR_OF)")

        self.flow_mapper = rf.FusionVelocityMapper(primary_mapper=primary, secondary_mapper=secondary)
        self.node.get_logger().info("CS20 Phase2 fusion mapper enabled (IR_OF + optional DEPTH_ICP)")

    def save_route_outputs(self):
        if self.flow_mapper is None:
            return
        self.flow_mapper.save_map(rf.FUSION_ROUTE_IMG, rf.FUSION_ROUTE_CSV)


def main():
    rs.cleanup = cleanup_lidar
    rs.launch_hardware = launch_hardware_lidar
    rf.cleanup = cleanup_lidar
    rf.launch_hardware = launch_hardware_lidar

    # Keep marker path disabled in CS20 runtime.
    rs.APRILTAG_ENABLED = False

    cleanup_lidar()

    if rs.DASHBOARD_AVAILABLE:
        from optical_flow_realsense.web_dashboard import app as flask_app

        def run_flask():
            flask_app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print("Web Dashboard started on http://0.0.0.0:5000")
        time.sleep(1)

    print("Launching MAVROS...")
    fcu_device = rs.get_fcu_serial_device()
    fcu_url = f"{fcu_device}:{rs.MAVROS_BAUD}"
    os.system(f"ros2 launch mavros px4.launch fcu_url:={fcu_url} >> {rs.MAVROS_LOG_FILE} 2>&1 &")
    time.sleep(8)
    rs.request_mavros_stream_rate(rs.MAVROS_STREAM_RATE_HZ, rs.MAVROS_STREAM_RETRIES)
    time.sleep(2)

    launch_hardware_lidar()
    time.sleep(4)

    rs.rclpy.init()
    node = CS20Phase2FusionNode()

    jetson_ip = rs.get_ip()
    print(f"CS20 PHASE2 FUSION running on {jetson_ip}")

    try:
        rs.rclpy.spin(node.node)
    except KeyboardInterrupt:
        print("Shutting down CS20 Phase2 Fusion...")
    except rs.ExternalShutdownException:
        print("ROS shutdown requested.")
    finally:
        try:
            node.save_route_outputs()
        except Exception as e:
            print(f"Fusion route map save failed: {e}")
        try:
            node.node.destroy_node()
        except Exception:
            pass
        try:
            rs.rclpy.shutdown()
        except Exception:
            pass
        cleanup_lidar()


if __name__ == "__main__":
    main()
