#!/usr/bin/env python3
"""
Phase 2: Metric Scaling & Flight Controller Fusion
Extends Phase 1 with depth-based metric scaling and MAVROS velocity publishing.

Subscribes to:
  - /camera/camera/imu (200Hz) -> yaw preintegration
  - /camera/camera/color/image_raw (30Hz) -> feature extraction
  - /camera/camera/aligned_depth_to_color/image_raw (30Hz) -> Z-distance lookup
    - /mavros/rangefinder/* (FCU rangefinder via MAVROS) -> absolute altitude override
  
Publishes:
  - /mavros/vision_speed/speed_twist (30Hz) -> velocity for MAVROS/EKF3

Test: Take off in Loiter mode indoors. Drone should hold position via visual frontend + ToF.
"""

import os
import time
import signal
import sys
import threading
import glob
import shlex
import math
import subprocess
import csv
from collections import deque
from dataclasses import dataclass
import numpy as np

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Image, Range, CameraInfo
from geometry_msgs.msg import TwistStamped, PoseStamped
from cv_bridge import CvBridge
import cv2

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Web dashboard integration
try:
    from .web_dashboard import dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    try:
        from web_dashboard import dashboard
        DASHBOARD_AVAILABLE = True
    except ImportError:
        DASHBOARD_AVAILABLE = False

try:
    from .msckf_velocity_estimator import MSCKFVelocityEstimator
except ImportError:
    from msckf_velocity_estimator import MSCKFVelocityEstimator
try:
    from .xfeat_lightglue_trt_backend import XFeatLightGlueTRTBackend
    XFEAT_BACKEND_IMPORT_ERROR = ""
except Exception as _xfeat_import_err:
    try:
        from xfeat_lightglue_trt_backend import XFeatLightGlueTRTBackend
        XFEAT_BACKEND_IMPORT_ERROR = ""
    except Exception as _xfeat_import_err2:
        XFeatLightGlueTRTBackend = None
        XFEAT_BACKEND_IMPORT_ERROR = str(_xfeat_import_err2)

# ============ CONFIGURATION ============
NUM_COMPASS_BINS = 8
USE_AI_FRONTEND = os.environ.get("USE_AI_FRONTEND", "0") == "1"
FEATURE_DETECTOR = "ALIKED" if USE_AI_FRONTEND else "ORB"
# Phase 3 AI frontend (opt-in for Jetson performance; set USE_AI_FRONTEND=1 to enable)
AI_FEATURES_ENABLED = USE_AI_FRONTEND
AI_FEATURE_MODEL = "ALIKED"  # ALIKED or XFEAT (if available in environment)
AI_MATCHER_MODEL = "LIGHTGLUE"
AI_MIN_TARGET_FPS = 10.0
AI_FPS_LOG_WINDOW = 60
AI_USE_TENSORRT = False  # enable only after verifying torch2trt/lightglue export path on target
AI_RELOCALIZE_STRIDE = 12  # run expensive AI relocalization every N frames
KEYPOINTS_PER_IMAGE = 140 if not USE_AI_FRONTEND else 80
VO_BACKEND = os.environ.get("VO_BACKEND", "AUTO").upper()
XFEAT_TRT_MODULE = os.environ.get("XFEAT_TRT_MODULE", "xfeat_lightglue_trt")
XFEAT_TRT_NATIVE_MODULE_PATH = os.environ.get("XFEAT_TRT_NATIVE_MODULE_PATH", "XFeat-Lightglue-TRT/build")
XFEAT_TRT_CONFIG_PATH = os.environ.get("XFEAT_TRT_CONFIG_PATH", "XFeat-Lightglue-TRT/config/xfeat_lightglue.yaml")
XFEAT_TRT_XFEAT_ENGINE_PATH = os.environ.get("XFEAT_TRT_XFEAT_ENGINE_PATH", "XFeat-Lightglue-TRT/weights/xfeat_1_800_800.engine")
XFEAT_TRT_LIGHTGLUE_ENGINE_PATH = os.environ.get("XFEAT_TRT_LIGHTGLUE_ENGINE_PATH", "XFeat-Lightglue-TRT/weights/lightglue_L6_1_800_800.engine")
XFEAT_TRT_VEL_SCALE = float(os.environ.get("XFEAT_TRT_VEL_SCALE", "1.0"))
XFEAT_TRT_REPO_PATH = os.environ.get("XFEAT_TRT_REPO_PATH", "XFeat-Lightglue-TRT")
XFEAT_TRT_FOCAL_PX = float(os.environ.get("XFEAT_TRT_FOCAL_PX", "420.0"))
XFEAT_TRT_MIN_MATCHES = int(os.environ.get("XFEAT_TRT_MIN_MATCHES", "24"))
XFEAT_TRT_FRAME_STRIDE = int(os.environ.get("XFEAT_TRT_FRAME_STRIDE", "2"))
XFEAT_TRT_TOP_K = int(os.environ.get("XFEAT_TRT_TOP_K", "512"))
MIN_MATCH_SCORE = 0.7
RELOCALIZATION_MATCH_THRESHOLD = 12  # Reduced from 15 to account for fewer features
BIN_UPDATE_COOLDOWN = 0.25  # Faster per-bin updates while rotating through headings
LOG_FILE = "phase2_debug.log"
FEATURE_EXTRACTION_SKIP = 1  # Extract features every frame (essential for all angles during rotation)
YAW_SIGN = 1.0  # Match physical spin direction observed on this setup
YAW_OFFSET_DEG = 0.0  # Optional manual offset if needed after mounting changes
LOCAL_QUERY_RAD = 2
GLOBAL_RELOCALIZE_FALLBACK = True
HEADING_CORRECTION_ALPHA = 0.25  # Relocalization-based yaw correction gain
HEADING_CORRECTION_MIN_MATCH = 20
HEADING_CORRECTION_COOLDOWN_S = 1.0
HEADING_CORRECTION_MAX_GYRO_RAD_S = 0.06
HEADING_CORRECTION_MAX_SPEED_MPS = 0.12  # only correct heading while nearly stationary
HEADING_CORRECTION_MAX_ERROR_DEG = 90.0
HEADING_SMOOTH_ALPHA = 0.2  # Circular low-pass smoothing for displayed/used heading
DIRECTION_PRINT_EVERY_N_FRAMES = 15  # Side-by-side camera vs IMU direction print cadence

# Translation mapping (Visual-Inertial style route trace)
ROUTE_MAP_ENABLED = True
ROUTE_MAP_IMAGE_PATH = "my_drone_route.png"
ROUTE_MAP_CSV_PATH = "my_drone_route.csv"
VO_MIN_TRACKED_POINTS = 60
VO_REINIT_FEATURES = 1200
VO_DEPTH_MIN_M = 0.1
VO_DEPTH_MAX_M = 10.0
VO_MIN_FLOW_PX = 0.12  # below this median optical flow, treat as stationary/noise
VO_FLOW_REF_PX = 4.0  # flow level considered strong motion
VO_MIN_INLIER_RATIO = 0.15  # lowered to tolerate hand-held jitter during walking tests
VO_MIN_STEP_M = 0.0025  # deadband to suppress tiny jitter in map trajectory
VO_MAX_STEP_M = 0.35  # clamp per-frame step to avoid spikes/outliers
VO_ROT_REJECT_DEG = 6.0  # allow normal hand turns without freezing map updates
VO_ROT_FLOW_RELAX_PX = 2.0
VO_PNP_REPROJ_ERR_PX = 4.0
VO_PNP_MIN_INLIERS = 14
VO_PNP_STRONG_INLIERS = 30  # strong PnP support allows accepting low-flow hand motion
VO_LOW_FLOW_ACCEPT_PX = 0.01  # minimum flow when strong PnP geometry is available
VO_PNP_STEP_GAIN = 1.0  # metric gain for PnP translation; tune if route scale drifts consistently
VO_IMU_PRIOR_MAX_S = 0.20
VO_IMU_ROT_PRIOR_WEIGHT = 0.85
VO_MOTION_CONF_MIN = 0.20
VO_VERTICAL_DOMINANCE_RATIO = 1.8  # reject if vertical component dominates planar motion
VO_REJECT_LOG_PERIOD_S = 0.8

# Pixhawk IMU fusion (camera IMU correction)
PIXHAWK_IMU_TOPIC = '/mavros/imu/data'
PIXHAWK_IMU_TOPIC_ALT = '/uas1/mavros/imu/data'  # Some MAVROS setups use UAS prefix namespace
USE_PIXHAWK_IMU_CORRECTION = True
PIXHAWK_FUSE_ALPHA = 0.7  # 0=camera only, 1=pixhawk only
PIXHAWK_TIMEOUT_S = 0.2  # max age for Pixhawk sample to be considered fresh
PIXHAWK_YAW_SIGN = -1.0  # invert to align GUI compass direction with physical left/right spin
PIXHAWK_YAW_BLEND_ALPHA = 0.85  # blend toward Pixhawk yaw (0=no yaw fusion, 1=Pixhawk yaw only)
PIXHAWK_MAX_DIFF_DEG = 120.0  # reject Pixhawk yaw if angular difference is extreme
USE_PIXHAWK_YAW_PRIMARY = True
USE_PIXHAWK_RATE_CORRECTION = True
USE_PIXHAWK_RATE_PRIMARY = True  # If True, use FCU gyro-z directly when fresh

# Madgwick fused yaw (camera IMU gyro+accel fusion path)
MADGWICK_IMU_TOPIC = '/imu/data'
USE_MADGWICK_YAW = True  # ✅ ENABLED: Madgwick is working, Pixhawk IMU currently non-functional
MADGWICK_TIMEOUT_S = 0.1
MADGWICK_BLEND_ALPHA = 0.25  # conservative blend if enabled
MADGWICK_YAW_SIGN = 1.0
MADGWICK_MAX_DIFF_DEG = 35.0  # reject madgwick yaw if it differs too much from gyro estimate

# MAVROS and Pixhawk FCU settings
MAVROS_BAUD = 57600  # User-validated working baud for this FCU link
MAVROS_LOG_FILE = "mavros_debug.log"
MAVROS_STREAM_RATE_HZ = 10
MAVROS_STREAM_RETRIES = 5

# IMU bias calibration (must keep drone still during startup)
IMU_CALIBRATION_SAMPLES = 250
IMU_GYRO_DEADBAND_RAD_S = 0.005
ENABLE_SPIN_CALIBRATION = False  # disabled by default when Pixhawk correction is active
SPIN_CALIB_TARGET_DEG = 330.0
SPIN_CALIB_SCALE_MIN = 0.7
SPIN_CALIB_SCALE_MAX = 1.3
SPIN_CALIB_MIN_ACCEPT_DEG = 220.0
SPIN_CALIB_STALL_TIMEOUT_S = 2.0
SPIN_PROGRESS_EPS_DEG = 0.02

# Depth/calibration parameters (D455)
REALSENSE_FX = 381.0  # Focal length X (pixels) - adjust per camera calibration
REALSENSE_FY = 381.0  # Focal length Y (pixels)
REALSENSE_CX = 320.0  # Principal point X (image center)
REALSENSE_CY = 240.0  # Principal point Y (image center)

# Velocity scaling parameters
MAX_PIXEL_VELOCITY = 50.0  # pixels/sec at 1m distance
MAX_VELOCITY_XY = 1.0  # m/s max commanded velocity
VELOCITY_LOWPASS_ALPHA = 0.3  # Exponential smoothing

# ToF/Altitude parameters
TOF_MAX_RANGE = 4.0  # meters, max range for ToF
TOF_MIN_RANGE = 0.1  # meters, min range for ToF
DEFAULT_ALTITUDE = 1.0  # meters, fallback if no ToF
FLOW_STATIONARY_PX = 0.35  # median flow below this is treated as stationary
GYRO_STATIONARY_RAD_S = 0.08  # low yaw-rate gate for stationary detection
FLOW_BIAS_ALPHA = 0.02  # adaptation rate for drift bias cancellation during stationary windows
FLOW_MIN_SPEED_MPS = 0.03  # deadband in m/s after compensation
FLOW_VERTICAL_DOMINANT_RATIO = 1.8  # if |vy| dominates |vx| this is likely lift/height change, not planar motion
FLOW_AXIS_DOMINANCE_RATIO = 1.6  # classify dominant motion axis for cross-axis leak suppression
FLOW_LEAK_ATTENUATION = 0.12  # retained fraction on the non-dominant axis

# AprilTag absolute pose correction (hybrid with MSCKF velocity)
APRILTAG_ENABLED = True
APRILTAG_DICT_NAME = "DICT_APRILTAG_36h11"
APRILTAG_SIZE_M = 0.16
APRILTAG_MAX_REPROJ_ERR_PX = 4.0
APRILTAG_POSE_ALPHA = 0.35
APRILTAG_LOG_PERIOD_S = 1.0
MAVROS_RANGE_TOPICS = (
    '/mavros/rangefinder/rangefinder',
    '/uas1/mavros/rangefinder/rangefinder',
    '/mavros/distance_sensor/rangefinder_pub',
    '/mavros/distance_sensor/rangefinder_sub',
    '/tof_sensor/range',  # fallback if local ToF publisher is still used
)
# ======================================

def get_ip():
    """Get local IP for debug purposes."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def get_fcu_serial_device():
    """Detect Pixhawk/CubePilot serial device, prefer stable by-id symlinks."""
    preferred_patterns = [
        "/dev/serial/by-id/*CubePilot*if00",
        "/dev/serial/by-id/*Pixhawk*if00",
        "/dev/serial/by-id/*if00",
    ]
    for pattern in preferred_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            print(f"✅ FCU device (by-id): {matches[0]}")
            return matches[0]

    for fallback_pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(fallback_pattern))
        if matches:
            print(f"⚠️  FCU device (fallback): {matches[0]}")
            return matches[0]

    default = "/dev/ttyACM0"
    print(f"⚠️  No FCU device found, using default: {default}")
    return default

def cleanup():
    """Kill all spawned processes."""
    os.system("pkill -9 -f 'realsense2_camera' && pkill -9 -f 'imu_filter' && pkill -9 -f 'static_transform' && pkill -9 -f 'mavros' > /dev/null 2>&1")


def request_mavros_stream_rate(rate_hz=MAVROS_STREAM_RATE_HZ, retries=MAVROS_STREAM_RETRIES):
    """Request MAVROS to enable FCU stream rates so /mavros/imu/data publishes consistently."""
    req = "{stream_id: 0, message_rate: %d, on_off: true}" % int(rate_hz)
    cmd = [
        "ros2", "service", "call",
        "/mavros/set_stream_rate",
        "mavros_msgs/srv/StreamRate",
        req,
    ]

    for attempt in range(1, retries + 1):
        try:
            print(f"📡 Requesting MAVROS stream rate ({rate_hz} Hz), attempt {attempt}/{retries}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ MAVROS stream rate request succeeded")
                return True
            stderr = (result.stderr or "").strip()
            if stderr:
                print(f"⚠️  Stream rate request failed: {stderr}")
        except Exception as e:
            print(f"⚠️  Stream rate request error: {e}")
        time.sleep(1.0)

    print("⚠️  Could not set MAVROS stream rate automatically; IMU may remain silent")
    return False

def shutdown_handler(sig, frame):
    print("\n🛑 SHUTDOWN: Cleaning up...")
    cleanup()
    raise KeyboardInterrupt

# Let rclpy handle Ctrl+C cleanly; custom signal handlers can cause double-shutdown races.

def launch_hardware():
    """Launch camera, IMU filter, and transforms."""
    print("📷 Launching RealSense camera driver...")
    os.system(f"ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 align_depth.enable:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2 depth_module.decimation_filter.enable:=true >> {LOG_FILE} 2>&1 &")
    time.sleep(3)
    
    print("📊 Launching IMU filter...")
    os.system(f"ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -r /imu/data_raw:=/camera/camera/imu -p use_mag:=false -p publish_tf:=false >> {LOG_FILE} 2>&1 &")
    time.sleep(1)
    
    print("🔗 Setting up static transforms...")
    os.system(f"ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom >> {LOG_FILE} 2>&1 &")
    time.sleep(0.5)
    os.system(f"ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 3.14159 base_link camera_link >> {LOG_FILE} 2>&1 &")
    time.sleep(1)

# ============ PHASE 1 CLASSES (RETAINED) ============

@dataclass
class CompassBin:
    bin_id: int
    heading_range: tuple
    descriptors: list = None
    keypoints: list = None
    timestamps: deque = None
    
    def __post_init__(self):
        if self.descriptors is None:
            self.descriptors = []
        if self.keypoints is None:
            self.keypoints = []
        if self.timestamps is None:
            self.timestamps = deque(maxlen=500)

class IMUPreintegrator:
    def __init__(self, window_size_samples=200):
        self.gyr_z_buffer = deque(maxlen=window_size_samples)
        self.imu_timestamps = deque(maxlen=window_size_samples)
        self.calibration_samples = []
        self.gyro_bias_z = 0.0
        self.bias_calibrated = False
        self.spin_calibrated = not ENABLE_SPIN_CALIBRATION
        self.calib_state = "bias"
        self.spin_accum_abs_deg = 0.0
        self.spin_accum_signed_deg = 0.0
        self.spin_last_progress_ts = None
        self.yaw_scale = 1.0
        self.yaw_deg = 0.0
        self.last_timestamp = None
        self.lock = threading.Lock()
    
    def update(self, gyr_z_rad_s, timestamp_sec):
        with self.lock:
            if not self.bias_calibrated:
                self.calibration_samples.append(gyr_z_rad_s)
                self.last_timestamp = timestamp_sec
                if len(self.calibration_samples) >= IMU_CALIBRATION_SAMPLES:
                    self.gyro_bias_z = float(np.median(self.calibration_samples))
                    self.bias_calibrated = True
                    self.calib_state = "spin" if ENABLE_SPIN_CALIBRATION else "ready"
                return

            # Integrate incrementally so yaw is cumulative across long rotations.
            if self.last_timestamp is not None:
                dt = timestamp_sec - self.last_timestamp
                if 0.0 < dt < 0.1:
                    corrected_gyr_z = gyr_z_rad_s - self.gyro_bias_z
                    if abs(corrected_gyr_z) < IMU_GYRO_DEADBAND_RAD_S:
                        corrected_gyr_z = 0.0

                    raw_delta_deg = YAW_SIGN * (corrected_gyr_z * dt * 180.0 / math.pi)

                    # Stage 2 calibration: user spins ~360 degrees to estimate scale.
                    if ENABLE_SPIN_CALIBRATION and not self.spin_calibrated:
                        self.spin_accum_abs_deg += abs(raw_delta_deg)
                        self.spin_accum_signed_deg += raw_delta_deg

                        if abs(raw_delta_deg) > SPIN_PROGRESS_EPS_DEG:
                            self.spin_last_progress_ts = timestamp_sec

                        spin_target_reached = self.spin_accum_abs_deg >= SPIN_CALIB_TARGET_DEG
                        spin_stalled = (
                            self.spin_last_progress_ts is not None
                            and (timestamp_sec - self.spin_last_progress_ts) > SPIN_CALIB_STALL_TIMEOUT_S
                            and self.spin_accum_abs_deg >= SPIN_CALIB_MIN_ACCEPT_DEG
                        )

                        if spin_target_reached or spin_stalled:
                            if self.spin_accum_abs_deg > 1e-6:
                                scale = 360.0 / self.spin_accum_abs_deg
                                self.yaw_scale = float(np.clip(scale, SPIN_CALIB_SCALE_MIN, SPIN_CALIB_SCALE_MAX))
                            self.spin_calibrated = True
                            self.calib_state = "ready"
                            self.yaw_deg = 0.0
                            self.spin_accum_abs_deg = 0.0
                            self.spin_accum_signed_deg = 0.0
                            self.spin_last_progress_ts = None
                        self.last_timestamp = timestamp_sec
                        self.gyr_z_buffer.append(gyr_z_rad_s)
                        self.imu_timestamps.append(timestamp_sec)
                        return

                    delta_deg = self.yaw_scale * raw_delta_deg
                    self.yaw_deg = (self.yaw_deg + delta_deg) % 360.0
            self.last_timestamp = timestamp_sec
            self.gyr_z_buffer.append(gyr_z_rad_s)
            self.imu_timestamps.append(timestamp_sec)

    def is_calibrated(self):
        with self.lock:
            return self.bias_calibrated and self.spin_calibrated

    def get_calibration_state(self):
        with self.lock:
            return self.calib_state

    def get_spin_progress_deg(self):
        with self.lock:
            return self.spin_accum_abs_deg

    def get_bias(self):
        with self.lock:
            return self.gyro_bias_z

    def get_scale(self):
        with self.lock:
            return self.yaw_scale

    def apply_heading_correction(self, target_heading_deg, alpha=0.25):
        """Blend yaw toward a target heading using shortest angular distance."""
        with self.lock:
            current = (self.yaw_deg + YAW_OFFSET_DEG) % 360.0
            diff = ((target_heading_deg - current + 540.0) % 360.0) - 180.0
            corrected = (current + alpha * diff) % 360.0
            self.yaw_deg = (corrected - YAW_OFFSET_DEG) % 360.0
    
    def get_yaw(self):
        with self.lock:
            return (self.yaw_deg + YAW_OFFSET_DEG) % 360.0

class FeatureExtractor:
    def __init__(self, method="ORB", n_features=200):
        self.method = method
        self.n_features = n_features
        self._ai_ready = False
        self._ai_last_warn_ts = 0.0
        self._extract_times = deque(maxlen=AI_FPS_LOG_WINDOW)
        self._match_times = deque(maxlen=AI_FPS_LOG_WINDOW)
        self._orb_aux = None

        if AI_FEATURES_ENABLED and method.upper() in ("ALIKED", "XFEAT"):
            self._init_ai_backend(method.upper(), n_features)
            if self._ai_ready:
                print(f"✅ AI frontend active: {method.upper()} + {AI_MATCHER_MODEL}")
                return
            print("⚠️  AI frontend unavailable, falling back to ORB")
            self.method = "ORB"
        
        if method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def _init_ai_backend(self, method, n_features):
        try:
            import torch
            from lightglue import LightGlue, ALIKED
            self._torch = torch
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # ALIKED is broadly available in LightGlue package and robust to blur.
            self._extractor = ALIKED(max_num_keypoints=n_features).eval().to(self._device)
            self._matcher = LightGlue(features='aliked').eval().to(self._device)
            self._orb_aux = cv2.ORB_create(nfeatures=n_features)

            # Optional TensorRT hook point (kept guarded to avoid runtime breakage).
            if AI_USE_TENSORRT:
                try:
                    import torch2trt  # noqa: F401
                    print("⚠️  AI_USE_TENSORRT requested: manual model export path required for LightGlue stack")
                except Exception:
                    print("⚠️  torch2trt not available; running AI models in PyTorch mode")

            self._ai_ready = True
        except Exception as e:
            print(f"⚠️  AI backend init failed: {e}")
            self._ai_ready = False

    def _log_ai_fps(self):
        if len(self._extract_times) < 10:
            return
        mean_extract = float(np.mean(self._extract_times))
        mean_match = float(np.mean(self._match_times)) if self._match_times else 0.0
        total = mean_extract + mean_match
        if total <= 1e-6:
            return
        fps = 1.0 / total
        now = time.time()
        if fps < AI_MIN_TARGET_FPS and (now - self._ai_last_warn_ts) > 2.0:
            print(f"⚠️  AI pipeline FPS low: {fps:.1f} (target {AI_MIN_TARGET_FPS:.1f}+) ")
            self._ai_last_warn_ts = now

    def _extract_ai(self, image):
        if image is None or image.size == 0:
            return [], None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        t0 = time.time()
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(gray).float().to(self._device) / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            feats = self._extractor.extract(tensor)

        # Keep ORB keypoints/descriptors for legacy motion tracking pipeline.
        orb_kp, orb_desc = self._orb_aux.detectAndCompute(gray, None) if self._orb_aux is not None else ([], None)
        if orb_kp is None:
            orb_kp = []

        self._extract_times.append(time.time() - t0)
        self._log_ai_fps()
        return orb_kp, {'ai': feats, 'orb': orb_desc}

    def _match_ai(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return 0
        t0 = time.time()
        try:
            with self._torch.no_grad():
                out = self._matcher({'image0': desc1, 'image1': desc2})
                if isinstance(out, dict) and 'matches' in out:
                    m = out['matches']
                    if hasattr(m, 'shape'):
                        count = int(m.shape[0])
                    else:
                        count = len(m)
                else:
                    count = 0
        except Exception:
            count = 0

        self._match_times.append(time.time() - t0)
        self._log_ai_fps()
        return count

    def _match_orb(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return 0

        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < MIN_MATCH_SCORE * n.distance:
                        good.append(m)
            return len(good)
        except Exception:
            return 0
    
    def extract(self, image):
        if self._ai_ready:
            return self._extract_ai(image)

        if image is None or image.size == 0:
            return [], None
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        kp, desc = self.detector.detectAndCompute(gray, None)
        return kp, desc
    
    def match(self, desc1, desc2):
        if self._ai_ready:
            ai_desc1 = desc1.get('ai') if isinstance(desc1, dict) else desc1
            ai_desc2 = desc2.get('ai') if isinstance(desc2, dict) else desc2
            ai_matches = self._match_ai(ai_desc1, ai_desc2)
            if ai_matches > 0:
                return ai_matches

            # Fallback for robustness if AI match fails unexpectedly.
            orb_desc1 = desc1.get('orb') if isinstance(desc1, dict) else None
            orb_desc2 = desc2.get('orb') if isinstance(desc2, dict) else None
            return self._match_orb(orb_desc1, orb_desc2)

        if desc1 is None or desc2 is None:
            return 0
        
        return self._match_orb(desc1, desc2)

    def get_runtime_status(self):
        if self._ai_ready:
            mean_extract = float(np.mean(self._extract_times)) if len(self._extract_times) > 0 else 0.0
            mean_match = float(np.mean(self._match_times)) if len(self._match_times) > 0 else 0.0
            total = mean_extract + mean_match
            fps = (1.0 / total) if total > 1e-6 else 0.0
            return {
                'backend': f"{AI_FEATURE_MODEL}+{AI_MATCHER_MODEL}",
                'ai_enabled': True,
                'ai_fps': fps,
                'extract_ms': 1000.0 * mean_extract,
                'match_ms': 1000.0 * mean_match,
            }
        return {
            'backend': self.method,
            'ai_enabled': False,
            'ai_fps': 0.0,
            'extract_ms': 0.0,
            'match_ms': 0.0,
        }

class CompassBinManager:
    def __init__(self, n_bins=8):
        self.n_bins = n_bins
        self.bin_width = 360.0 / n_bins
        self.bins = []
        self.lock = threading.Lock()
        self.last_bin_update_time = {}
        self.ready_bins_logged = set()
        self.last_heading_deg = None
        
        for i in range(n_bins):
            min_heading = (i * self.bin_width) % 360.0
            max_heading = ((i + 1) * self.bin_width) % 360.0
            bin_obj = CompassBin(bin_id=i, heading_range=(min_heading, max_heading))
            self.bins.append(bin_obj)
            self.last_bin_update_time[i] = 0.0

    def _bins_crossed(self, prev_heading_deg, curr_heading_deg):
        """Return bins crossed from previous to current heading (including current bin)."""
        curr_bin = self.heading_to_bin(curr_heading_deg)
        if prev_heading_deg is None:
            return [curr_bin]

        prev_bin = self.heading_to_bin(prev_heading_deg)
        if prev_bin == curr_bin:
            return [curr_bin]

        cw_steps = (curr_bin - prev_bin) % self.n_bins
        ccw_steps = (prev_bin - curr_bin) % self.n_bins

        if cw_steps <= ccw_steps:
            step = 1
            steps = cw_steps
        else:
            step = -1
            steps = ccw_steps

        crossed = []
        for i in range(1, steps + 1):
            crossed.append((prev_bin + step * i) % self.n_bins)
        return crossed
    
    def heading_to_bin(self, heading_deg):
        heading_deg = heading_deg % 360.0
        bin_idx = int(heading_deg / self.bin_width) % self.n_bins
        return bin_idx
    
    def store_features(self, heading_deg, keypoints, descriptors, current_time):
        if descriptors is None or len(descriptors) == 0:
            return

        with self.lock:
            target_bins = self._bins_crossed(self.last_heading_deg, heading_deg)
            self.last_heading_deg = heading_deg

            for bin_idx in target_bins:
                if current_time - self.last_bin_update_time[bin_idx] < BIN_UPDATE_COOLDOWN:
                    continue

                bin_obj = self.bins[bin_idx]
                bin_obj.descriptors.append(descriptors)
                bin_obj.keypoints.append(keypoints)
                bin_obj.timestamps.append(current_time)
                self.last_bin_update_time[bin_idx] = current_time

                if len(bin_obj.descriptors) >= 3 and bin_idx not in self.ready_bins_logged:
                    self.ready_bins_logged.add(bin_idx)
                    bin_range = bin_obj.heading_range
                    print(f"\n✅ BIN {bin_idx} READY ({bin_range[0]:.0f}°-{bin_range[1]:.0f}°) | {len([b for b in self.bins if len(b.descriptors) >= 3])}/{self.n_bins} bins ready")

                    if len(self.ready_bins_logged) == self.n_bins:
                        print(f"\n🎯 ALL BINS READY! YOU CAN NOW START THE FLIGHT TEST.\n")
    
    def query_bin(self, bin_idx, query_descriptors, feature_extractor):
        if query_descriptors is None or len(query_descriptors) == 0:
            return 0
        
        with self.lock:
            bin_obj = self.bins[bin_idx]
            total_matches = 0
            
            # Limit to most recent 5 descriptors per bin for faster matching
            stored_descriptors = bin_obj.descriptors[-5:] if len(bin_obj.descriptors) > 5 else bin_obj.descriptors
            
            for stored_desc in stored_descriptors:
                matches = feature_extractor.match(query_descriptors, stored_desc)
                total_matches += matches
            
            return total_matches
    
    def relocalize(self, heading_deg, query_descriptors, feature_extractor, query_rad_search=3):
        if query_descriptors is None or len(query_descriptors) == 0:
            return None, 0
        
        current_bin = self.heading_to_bin(heading_deg)
        best_bin = None
        best_score = 0
        
        for offset in range(-query_rad_search, query_rad_search + 1):
            search_bin = (current_bin + offset) % self.n_bins
            score = self.query_bin(search_bin, query_descriptors, feature_extractor)
            
            if score > best_score:
                best_score = score
                best_bin = search_bin
        
        if best_score >= RELOCALIZATION_MATCH_THRESHOLD:
            return best_bin, best_score

        # If heading is off, try global relocalization fallback across all bins.
        if GLOBAL_RELOCALIZE_FALLBACK:
            for search_bin in range(self.n_bins):
                score = self.query_bin(search_bin, query_descriptors, feature_extractor)
                if score > best_score:
                    best_score = score
                    best_bin = search_bin

            if best_score >= RELOCALIZATION_MATCH_THRESHOLD:
                return best_bin, best_score
        
        return None, best_score

# ============ PHASE 2 ADDITIONS ============

class OpticalFlowCalculator:
    """Converts tracked pixel movement to metric velocity."""
    def __init__(self, fx=REALSENSE_FX, fy=REALSENSE_FY, cx=REALSENSE_CX, cy=REALSENSE_CY):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.prev_kp = None
        self.prev_desc = None
        self.prev_timestamp = None
        self.velocity_x_filt = 0.0  # m/s, filtered
        self.velocity_y_filt = 0.0  # m/s, filtered
        self.lock = threading.Lock()
    
    def depth_to_metric(self, pixel_x, pixel_y, depth_mm):
        """
        Convert pixel + depth to 3D point.
        Returns (X_m, Y_m, Z_m) in meters relative to camera.
        """
        if depth_mm <= 0 or depth_mm > 4000:
            return None, None, None
        
        z_m = depth_mm / 1000.0
        x_m = (pixel_x - self.cx) * z_m / self.fx
        y_m = (pixel_y - self.cy) * z_m / self.fy
        
        return x_m, y_m, z_m
    
    def calculate_velocity(self, curr_kp, curr_desc, depth_image, current_time):
        """
        Track keypoints across frames and compute metric velocity.
        Returns (vx, vy) in m/s.
        """
        if isinstance(curr_desc, dict):
            curr_desc = curr_desc.get('orb')

        if self.prev_kp is None or curr_kp is None or curr_desc is None or self.prev_desc is None:
            with self.lock:
                self.prev_kp = curr_kp
                self.prev_desc = curr_desc
                self.prev_timestamp = current_time
            return 0.0, 0.0
        
        # Match features between frames
        norm = cv2.NORM_HAMMING if curr_desc.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm, crossCheck=True)
        try:
            matches = bf.match(self.prev_desc, curr_desc)
            matches = sorted(matches, key=lambda x: x.distance)
        except Exception:
            matches = []
        
        if len(matches) < 5:
            with self.lock:
                self.prev_kp = curr_kp
                self.prev_desc = curr_desc
                self.prev_timestamp = current_time
            return self.velocity_x_filt, self.velocity_y_filt
        
        # Calculate metric velocity from matches
        velocities_x = []
        velocities_y = []
        
        for match in matches[:10]:  # Use best 10 matches (faster, still accurate)
            prev_pt = self.prev_kp[match.queryIdx].pt
            curr_pt = curr_kp[match.trainIdx].pt
            
            # Get depth at current point
            x_int = int(curr_pt[0])
            y_int = int(curr_pt[1])
            
            if 0 <= x_int < depth_image.shape[1] and 0 <= y_int < depth_image.shape[0]:
                depth_mm = depth_image[y_int, x_int]
                
                if depth_mm > 0:
                    # Get 3D position
                    x_m, y_m, z_m = self.depth_to_metric(curr_pt[0], curr_pt[1], depth_mm)
                    
                    if x_m is not None:
                        # Pixel delta
                        dx_pix = curr_pt[0] - prev_pt[0]
                        dy_pix = curr_pt[1] - prev_pt[1]
                        dt = current_time - self.prev_timestamp
                        
                        if dt > 0:
                            # Metric velocity (simplified: scale by depth)
                            vx = (dx_pix * z_m / self.fx) / dt
                            vy = (dy_pix * z_m / self.fy) / dt
                            
                            # Clamp to max velocity
                            vx = np.clip(vx, -MAX_VELOCITY_XY, MAX_VELOCITY_XY)
                            vy = np.clip(vy, -MAX_VELOCITY_XY, MAX_VELOCITY_XY)
                            
                            velocities_x.append(vx)
                            velocities_y.append(vy)
        
        # Average velocities
        if velocities_x and velocities_y:
            vx_raw = np.median(velocities_x)
            vy_raw = np.median(velocities_y)
        else:
            vx_raw = 0.0
            vy_raw = 0.0
        
        # Apply lowpass filter
        with self.lock:
            self.velocity_x_filt = VELOCITY_LOWPASS_ALPHA * vx_raw + (1 - VELOCITY_LOWPASS_ALPHA) * self.velocity_x_filt
            self.velocity_y_filt = VELOCITY_LOWPASS_ALPHA * vy_raw + (1 - VELOCITY_LOWPASS_ALPHA) * self.velocity_y_filt
            self.prev_kp = curr_kp
            self.prev_desc = curr_desc
            self.prev_timestamp = current_time
        
        return self.velocity_x_filt, self.velocity_y_filt
    
    def get_velocity(self):
        with self.lock:
            return self.velocity_x_filt, self.velocity_y_filt


class VisualRouteMapper:
    """2D route mapper using essential matrix + depth-based metric scaling."""

    def __init__(self, fx=REALSENSE_FX, fy=REALSENSE_FY, cx=REALSENSE_CX, cy=REALSENSE_CY):
        self.prev_image = None
        self.prev_points = None
        self.prev_depth = None
        self.prev_timestamp = None
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))
        self.trajectory_x = [0.0]
        self.trajectory_z = [0.0]
        self.current_t_pnp = np.zeros((3, 1))
        self.current_t_fallback = np.zeros((3, 1))
        self.trajectory_pnp_x = [0.0]
        self.trajectory_pnp_z = [0.0]
        self.trajectory_fallback_x = [0.0]
        self.trajectory_fallback_z = [0.0]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.focal = (fx + fy) / 2.0
        self.camera_matrix = np.array([
            [self.focal, 0.0, cx],
            [0.0, self.focal, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.last_print_time = 0.0
        self.last_reject_print_time = 0.0
        self.last_debug = {
            'source': 'INIT',
            'flow_px': 0.0,
            'inlier_ratio': 0.0,
            'rot_deg': 0.0,
            'step_m': 0.0,
            'motion_conf': 0.0,
            'rejected': False,
            'reject_reason': 'INIT',
            'pnp_points': 0,
            'pnp_inliers': 0,
        }
        self.lock = threading.Lock()

    def _reinitialize(self, gray):
        self.prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=VO_REINIT_FEATURES,
            qualityLevel=0.01,
            minDistance=10
        )
        self.prev_image = gray

    def _integrate_imu_rotation(self, imu_samples):
        """Integrate gyroscope samples to a relative rotation matrix."""
        if imu_samples is None or len(imu_samples) < 2:
            return np.eye(3), 0.0

        R_delta = np.eye(3)
        used_dt = 0.0
        for i in range(1, len(imu_samples)):
            t0, wx0, wy0, wz0 = imu_samples[i - 1]
            t1, wx1, wy1, wz1 = imu_samples[i]
            dt = t1 - t0
            if dt <= 0.0 or dt > 0.05:
                continue

            wx = 0.5 * (wx0 + wx1)
            wy = 0.5 * (wy0 + wy1)
            wz = 0.5 * (wz0 + wz1)
            rvec = np.array([wx * dt, wy * dt, wz * dt], dtype=np.float64)
            dR, _ = cv2.Rodrigues(rvec)
            R_delta = dR.dot(R_delta)
            used_dt += dt

        return R_delta, used_dt

    def _compose_relative_rotation(self, R_visual, R_imu, imu_dt):
        if imu_dt <= 0.0 or imu_dt > VO_IMU_PRIOR_MAX_S:
            return R_visual

        # Weighted fusion in axis-angle domain.
        r_vis, _ = cv2.Rodrigues(R_visual)
        r_imu, _ = cv2.Rodrigues(R_imu)
        r_fused = (1.0 - VO_IMU_ROT_PRIOR_WEIGHT) * r_vis + VO_IMU_ROT_PRIOR_WEIGHT * r_imu
        R_fused, _ = cv2.Rodrigues(r_fused)
        return R_fused

    def _build_pnp_correspondences(self, good_old, good_new):
        if self.prev_depth is None:
            return None, None

        h, w = self.prev_depth.shape[:2]
        obj_pts = []
        img_pts = []
        for pt_old, pt_new in zip(good_old, good_new):
            u0, v0 = int(pt_old[0]), int(pt_old[1])
            if not (0 <= u0 < w and 0 <= v0 < h):
                continue
            d = float(self.prev_depth[v0, u0])
            if not (VO_DEPTH_MIN_M < d < VO_DEPTH_MAX_M):
                continue

            X = (pt_old[0] - self.cx) * d / self.fx
            Y = (pt_old[1] - self.cy) * d / self.fy
            Z = d
            obj_pts.append([X, Y, Z])
            img_pts.append([float(pt_new[0]), float(pt_new[1])])

        if len(obj_pts) < VO_PNP_MIN_INLIERS:
            return None, None

        return np.asarray(obj_pts, dtype=np.float32), np.asarray(img_pts, dtype=np.float32)

    def update(self, gray, depth_m, timestamp_sec, imu_samples=None):
        if gray is None or depth_m is None:
            return None

        with self.lock:
            if self.prev_image is None or self.prev_points is None or len(self.prev_points) < 10:
                self._reinitialize(gray)
                self.prev_depth = depth_m.copy()
                self.prev_timestamp = timestamp_sec
                return None

            current_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_image, gray, self.prev_points, None)
            if current_points is None or status is None:
                self._reinitialize(gray)
                return None

            good_new = current_points[status == 1]
            good_old = self.prev_points[status == 1]
            if len(good_new) < VO_MIN_TRACKED_POINTS:
                self._reinitialize(gray)
                self.prev_depth = depth_m.copy()
                self.prev_timestamp = timestamp_sec
                return None

            flow = good_new - good_old
            flow_mag = np.linalg.norm(flow, axis=1)
            median_flow_px = float(np.median(flow_mag)) if len(flow_mag) > 0 else 0.0

            obj_pts, img_pts = self._build_pnp_correspondences(good_old, good_new)
            pnp_points = int(len(obj_pts)) if obj_pts is not None else 0
            pnp_inliers = 0

            R_imu_delta, imu_dt = self._integrate_imu_rotation(imu_samples)
            solve_src = "PNP"
            inlier_ratio = 0.0
            R_rel = None
            t_rel = None

            if obj_pts is not None and img_pts is not None:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=None,
                    reprojectionError=VO_PNP_REPROJ_ERR_PX,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_EPNP
                )

                if ok and inliers is not None and len(inliers) >= VO_PNP_MIN_INLIERS:
                    pnp_inliers = int(len(inliers))
                    inlier_ratio = float(len(inliers)) / float(len(obj_pts))
                    R_vis, _ = cv2.Rodrigues(rvec)

                    # PnP returns world motion in camera frame; invert for camera motion integration.
                    t_cam = -R_vis.T.dot(tvec)
                    R_rel = self._compose_relative_rotation(R_vis, R_imu_delta, imu_dt)
                    t_rel = t_cam.astype(np.float64)

            # Fallback when PnP underconstrained: recover relative motion direction + depth scale.
            if R_rel is None or t_rel is None:
                E, emask = cv2.findEssentialMat(good_new, good_old, self.camera_matrix, cv2.RANSAC, 0.999, 1.0)
                if E is not None:
                    _, R_vis, t_dir, pmask = cv2.recoverPose(E, good_new, good_old, self.camera_matrix)
                    inlier_count = int(np.count_nonzero(pmask)) if pmask is not None else 0
                    inlier_ratio = float(inlier_count) / float(max(len(good_new), 1))

                    depths = []
                    if self.prev_depth is not None:
                        h, w = self.prev_depth.shape[:2]
                        for pt in good_old:
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= x < w and 0 <= y < h:
                                d = float(self.prev_depth[y, x])
                                if VO_DEPTH_MIN_M < d < VO_DEPTH_MAX_M:
                                    depths.append(d)

                    depth_scale = float(np.median(depths)) if depths else 0.0
                    if depth_scale > VO_DEPTH_MIN_M:
                        solve_src = "E_FALLBACK"
                        R_rel = self._compose_relative_rotation(R_vis, R_imu_delta, imu_dt)
                        t_rel = depth_scale * t_dir.astype(np.float64)

            if R_rel is None or t_rel is None:
                if (timestamp_sec - self.last_reject_print_time) >= VO_REJECT_LOG_PERIOD_S:
                    print("📍 ROUTE HOLD | insufficient geometry (PnP/E fallback unavailable)")
                    self.last_reject_print_time = timestamp_sec
                self.last_debug = {
                    'source': 'NONE',
                    'flow_px': median_flow_px,
                    'inlier_ratio': inlier_ratio,
                    'rot_deg': 0.0,
                    'step_m': 0.0,
                    'motion_conf': 0.0,
                    'rejected': True,
                    'reject_reason': 'INSUFFICIENT_GEOMETRY',
                    'pnp_points': pnp_points,
                    'pnp_inliers': pnp_inliers,
                }
                self.prev_image = gray
                self.prev_depth = depth_m.copy()
                self.prev_points = good_new.reshape(-1, 1, 2)
                self.prev_timestamp = timestamp_sec
                return None

            rvec_rel, _ = cv2.Rodrigues(R_rel)
            rot_deg = float(np.linalg.norm(rvec_rel) * 180.0 / math.pi)
            scale = float(np.linalg.norm(t_rel))
            tx = float(t_rel[0][0])
            ty = float(t_rel[1][0])
            tz = float(t_rel[2][0])
            planar_mag = math.hypot(tx, tz)
            vertical_ratio = abs(ty) / max(planar_mag, 1e-6)

            low_flow_gate = median_flow_px < VO_MIN_FLOW_PX
            strong_pnp_low_flow_ok = (
                solve_src == "PNP"
                and pnp_inliers >= VO_PNP_STRONG_INLIERS
                and inlier_ratio >= 0.70
                and median_flow_px >= VO_LOW_FLOW_ACCEPT_PX
            )
            if strong_pnp_low_flow_ok:
                low_flow_gate = False

            reject_translation = (
                low_flow_gate
                or inlier_ratio < VO_MIN_INLIER_RATIO
                or (rot_deg > VO_ROT_REJECT_DEG and median_flow_px < VO_ROT_FLOW_RELAX_PX)
                or (vertical_ratio > VO_VERTICAL_DOMINANCE_RATIO and planar_mag < 0.05)
            )

            if reject_translation:
                if (timestamp_sec - self.last_reject_print_time) >= VO_REJECT_LOG_PERIOD_S:
                    print(
                        f"📍 ROUTE HOLD | src={solve_src} flow={median_flow_px:.2f}px inliers={inlier_ratio:.2f} "
                        f"rot={rot_deg:.2f}° (stationary/rotation-dominant)"
                    )
                    self.last_reject_print_time = timestamp_sec
                self.last_debug = {
                    'source': solve_src,
                    'flow_px': median_flow_px,
                    'inlier_ratio': inlier_ratio,
                    'rot_deg': rot_deg,
                    'step_m': 0.0,
                    'motion_conf': 0.0,
                    'vertical_ratio': vertical_ratio,
                    'rejected': True,
                    'reject_reason': 'LOW_FLOW_OR_ROTATION',
                    'pnp_points': pnp_points,
                    'pnp_inliers': pnp_inliers,
                }
                self.prev_image = gray
                self.prev_depth = depth_m.copy()
                self.prev_points = good_new.reshape(-1, 1, 2)
                self.prev_timestamp = timestamp_sec
                return 0.0

            if scale > VO_DEPTH_MIN_M:
                if solve_src == "PNP":
                    # PnP already gives metric translation; do not shrink it by low-flow confidence.
                    motion_conf = 1.0
                    step_vec = VO_PNP_STEP_GAIN * t_rel
                else:
                    motion_conf = float(np.clip((median_flow_px - VO_MIN_FLOW_PX) / max(VO_FLOW_REF_PX - VO_MIN_FLOW_PX, 1e-6), VO_MOTION_CONF_MIN, 1.0))
                    step_vec = motion_conf * t_rel
                # Route map is planar (X/Z); ignore vertical component to avoid lift -> X/Z leakage.
                step_vec[1][0] = 0.0
                step_x = float(step_vec[0][0])
                step_z = float(step_vec[2][0])
                step_norm = math.hypot(step_x, step_z)

                if step_norm < VO_MIN_STEP_M:
                    self.prev_image = gray
                    self.prev_depth = depth_m.copy()
                    self.prev_points = good_new.reshape(-1, 1, 2)
                    self.prev_timestamp = timestamp_sec
                    return 0.0

                if step_norm > VO_MAX_STEP_M:
                    clamp = VO_MAX_STEP_M / max(step_norm, 1e-6)
                    step_vec = step_vec * clamp

                # Pose composition: world<-curr = world<-prev composed with prev<-curr delta
                world_step = self.current_R.dot(step_vec)
                self.current_t = self.current_t + world_step
                self.current_R = self.current_R.dot(R_rel)
                x_pos = float(self.current_t[0][0])
                z_pos = float(self.current_t[2][0])
                self.trajectory_x.append(x_pos)
                self.trajectory_z.append(z_pos)

                if solve_src == "PNP":
                    self.current_t_pnp = self.current_t_pnp + world_step
                    self.trajectory_pnp_x.append(float(self.current_t_pnp[0][0]))
                    self.trajectory_pnp_z.append(float(self.current_t_pnp[2][0]))
                else:
                    self.current_t_fallback = self.current_t_fallback + world_step
                    self.trajectory_fallback_x.append(float(self.current_t_fallback[0][0]))
                    self.trajectory_fallback_z.append(float(self.current_t_fallback[2][0]))

                self.last_debug = {
                    'source': solve_src,
                    'flow_px': median_flow_px,
                    'inlier_ratio': inlier_ratio,
                    'rot_deg': rot_deg,
                    'step_m': step_norm,
                    'motion_conf': motion_conf,
                    'vertical_ratio': vertical_ratio,
                    'rejected': False,
                    'reject_reason': 'NONE',
                    'pnp_points': pnp_points,
                    'pnp_inliers': pnp_inliers,
                }

                if (timestamp_sec - self.last_print_time) >= 0.5:
                    print(
                        f"📍 ROUTE | X: {x_pos:+.2f}m | Forward(Z): {z_pos:+.2f}m | "
                        f"src={solve_src} Step: {step_norm:.3f}m | flow={median_flow_px:.2f}px | inliers={inlier_ratio:.2f}"
                    )
                    self.last_print_time = timestamp_sec

            self.prev_image = gray
            self.prev_depth = depth_m.copy()
            self.prev_points = good_new.reshape(-1, 1, 2)
            self.prev_timestamp = timestamp_sec
            return scale

    def get_dashboard_payload(self):
        with self.lock:
            return {
                'tracks': {
                    'fused': list(zip(self.trajectory_x, self.trajectory_z)),
                    'pnp': list(zip(self.trajectory_pnp_x, self.trajectory_pnp_z)),
                    'fallback': list(zip(self.trajectory_fallback_x, self.trajectory_fallback_z)),
                },
                'debug': dict(self.last_debug),
            }

    def save_map(self, image_path=ROUTE_MAP_IMAGE_PATH, csv_path=ROUTE_MAP_CSV_PATH):
        with self.lock:
            if len(self.trajectory_x) < 2:
                print("⚠️  Route map skipped: not enough trajectory points")
                return False

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x_m", "z_m"])
                for x, z in zip(self.trajectory_x, self.trajectory_z):
                    writer.writerow([f"{x:.6f}", f"{z:.6f}"])
            print(f"✅ Saved route points CSV: {csv_path}")

            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(8, 8))
                plt.plot(self.trajectory_x, self.trajectory_z, marker='o', color='b', markersize=2, linestyle='-', linewidth=1)
                plt.plot(0.0, 0.0, marker='*', color='g', markersize=15, label="Start")
                plt.plot(self.trajectory_x[-1], self.trajectory_z[-1], marker='X', color='r', markersize=10, label="End")
                plt.title("Visual Odometry 2D Route Map")
                plt.xlabel("Left / Right Translation (m)")
                plt.ylabel("Forward / Backward Translation (m)")
                plt.grid(True)
                plt.legend()
                plt.axis('equal')
                plt.savefig(image_path, dpi=200)
                plt.close()
                print(f"✅ Saved route map image: {image_path}")
            else:
                print("⚠️  matplotlib unavailable; PNG route map not generated")

            return True


class FastKLTTranslator:
    """Metric velocity estimator: FAST corners + KLT pyramid + depth-weighted median.
    
    Lightweight, pure OpenCV approach targeting 25–35 FPS on Jetson Orin Nano.
    No ICP, no Open3D — just corner tracking + depth lookup + median aggregation.
    """

    def __init__(self, fx=REALSENSE_FX, fy=REALSENSE_FY, cx=REALSENSE_CX, cy=REALSENSE_CY):
        # Halve intrinsics for 320×240 downsampling
        self.fx = fx * 0.5
        self.fy = fy * 0.5
        self.cx = cx * 0.5
        self.cy = cy * 0.5

        self._prev_gray = None
        self._prev_pts = None
        self._prev_depth = None
        self._prev_ts = None

        self.vx_filt = 0.0
        self.vy_filt = 0.0

        # Route map
        self._pos_x = 0.0
        self._pos_z = 0.0
        self._traj_x = [0.0]
        self._traj_z = [0.0]

        self._lock = threading.Lock()
        self._last_debug = {
            'source': 'FAST_KLT',
            'success': False,
            'flow_x': 0.0,
            'flow_y': 0.0,
            'alt': DEFAULT_ALTITUDE,
        }

    def update(self, gray_image, depth_image_16uc1, tof_altitude_m, timestamp_sec, imu_gyro_z_rads=0.0, yaw_deg=None):
        if gray_image is None or depth_image_16uc1 is None:
            return self.vx_filt, self.vy_filt

        with self._lock:
            # Downsample to 320×240
            small_gray = cv2.resize(gray_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            small_depth = cv2.resize(depth_image_16uc1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            # Bootstrap
            if self._prev_gray is None or self._prev_pts is None:
                self._seed(small_gray, small_depth, timestamp_sec)
                return self.vx_filt, self.vy_filt

            dt = timestamp_sec - self._prev_ts
            if dt <= 0.0 or dt > 0.5:
                self._seed(small_gray, small_depth, timestamp_sec)
                return self.vx_filt, self.vy_filt

            # KLT tracking with pyramid
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
            )
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, small_gray, self._prev_pts, None, **lk_params
            )

            if next_pts is None or status is None:
                self._seed(small_gray, small_depth, timestamp_sec)
                return self.vx_filt, self.vy_filt

            ok = status.ravel() == 1
            n_tracked = int(ok.sum())

            if n_tracked < 60:
                self._seed(small_gray, small_depth, timestamp_sec)
                return self.vx_filt, self.vy_filt

            prev_good = self._prev_pts[ok].reshape(-1, 2)
            next_good = next_pts[ok].reshape(-1, 2)

            # Flow magnitude gate
            flow_vecs = next_good - prev_good
            flow_mag = np.linalg.norm(flow_vecs, axis=1)
            median_flow = float(np.median(flow_mag))

            if median_flow < 0.10:
                # Stationary — decay velocity
                self.vx_filt *= 0.8
                self.vy_filt *= 0.8
                self._update_state(small_gray, small_depth, next_good, timestamp_sec)
                return self.vx_filt, self.vy_filt

            # Depth lookup at tracked points
            h, w = small_depth.shape
            vx_samples, vy_samples = [], []

            for (px, py), (nx, ny) in zip(prev_good, next_good):
                xi, yi = int(round(nx)), int(round(ny))
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                d_mm = float(small_depth[yi, xi])
                d_m = d_mm / 1000.0
                if not (0.15 < d_m < 8.0):
                    continue

                dpx = nx - px
                dpy = ny - py
                vx_samples.append((dpx * d_m / self.fx) / dt)
                vy_samples.append((dpy * d_m / self.fy) / dt)

            if len(vx_samples) < 5:
                self._update_state(small_gray, small_depth, next_good, timestamp_sec)
                return self.vx_filt, self.vy_filt

            # Median with spike clipping
            vx_arr = np.clip(np.array(vx_samples), -3.0, 3.0)
            vy_arr = np.clip(np.array(vy_samples), -3.0, 3.0)

            vx_raw = float(np.median(vx_arr))
            vy_raw = float(np.median(vy_arr))

            # Suppress suspected Z-lift artefacts
            planar = math.hypot(vx_raw, vy_raw)
            if planar > 1e-4 and abs(vy_raw) / planar > 1.8:
                vy_raw *= 0.15

            # Deadband
            if abs(vx_raw) < 0.03:
                vx_raw = 0.0
            if abs(vy_raw) < 0.03:
                vy_raw = 0.0

            # Clamp per-frame step
            step = math.hypot(vx_raw * dt, vy_raw * dt)
            if step > 0.30:
                k = 0.30 / max(step, 1e-6)
                vx_raw *= k
                vy_raw *= k

            # EMA low-pass
            self.vx_filt = 0.35 * vx_raw + 0.65 * self.vx_filt
            self.vy_filt = 0.35 * vy_raw + 0.65 * self.vy_filt

            # Route map with yaw rotation
            if yaw_deg is not None:
                yr = math.radians(float(yaw_deg))
                world_x = math.cos(yr) * self.vx_filt - math.sin(yr) * self.vy_filt
                world_z = math.sin(yr) * self.vx_filt + math.cos(yr) * self.vy_filt
            else:
                world_x, world_z = self.vx_filt, self.vy_filt

            self._pos_x += world_x * dt
            self._pos_z += world_z * dt
            self._traj_x.append(self._pos_x)
            self._traj_z.append(self._pos_z)

            self._update_state(small_gray, small_depth, next_good, timestamp_sec)
            self._last_debug = {
                'source': 'FAST_KLT',
                'success': True,
                'flow_x': self.vx_filt,
                'flow_y': self.vy_filt,
                'alt': tof_altitude_m,
            }

        return self.vx_filt, self.vy_filt

    def get_dashboard_payload(self):
        with self._lock:
            pts = list(zip(self._traj_x, self._traj_z))
            return {
                'tracks': {'fused': pts, 'pnp': [], 'fallback': []},
                'debug': dict(self._last_debug),
            }

    def save_map(self, image_path=ROUTE_MAP_IMAGE_PATH, csv_path=ROUTE_MAP_CSV_PATH):
        with self._lock:
            if len(self._traj_x) < 2:
                print("⚠️  Route map skipped: not enough trajectory points")
                return False

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x_m", "z_m"])
                for x, z in zip(self._traj_x, self._traj_z):
                    writer.writerow([f"{x:.6f}", f"{z:.6f}"])
            print(f"✅ Saved route CSV: {csv_path}")

            if MATPLOTLIB_AVAILABLE:
                plt.figure(figsize=(8, 8))
                plt.plot(self._traj_x, self._traj_z, color='steelblue', linewidth=1.2, marker='o', markersize=1.5)
                plt.plot(0, 0, '*g', markersize=14, label="Start")
                plt.plot(self._traj_x[-1], self._traj_z[-1], 'Xr', markersize=10, label="End")
                plt.title("FastKLT 2-D Route Map")
                plt.xlabel("Left / Right (m)")
                plt.ylabel("Forward / Back (m)")
                plt.axis('equal')
                plt.grid(True)
                plt.legend()
                plt.savefig(image_path, dpi=200)
                plt.close()
                print(f"✅ Saved route image: {image_path}")
            else:
                print("⚠️  matplotlib unavailable; PNG not generated")

            return True

    def _seed(self, gray, depth, ts):
        """Detect corners and store first frame."""
        corner_params = dict(
            maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=5,
        )
        pts = cv2.goodFeaturesToTrack(gray, **corner_params)
        if pts is None:
            pts = np.zeros((0, 1, 2), dtype=np.float32)
        self._prev_gray = gray
        self._prev_pts = pts
        self._prev_depth = depth
        self._prev_ts = ts

    def _update_state(self, gray, depth, good_next, ts):
        """Advance to next frame."""
        pts = good_next.reshape(-1, 1, 2).astype(np.float32)
        if len(pts) < 60:
            new_pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=5)
            if new_pts is not None:
                pts = new_pts
        self._prev_gray = gray
        self._prev_pts = pts
        self._prev_depth = depth
        self._prev_ts = ts

class Phase2Node:
    """Phase 2: Metric scaling + MAVROS velocity publishing."""
    def __init__(self):
        self.node = rclpy.create_node('phase2_mvp')
        self.bridge = CvBridge()
        
        # Phase 1 systems
        self.imu_integrator = IMUPreintegrator()
        self.feature_extractor = FeatureExtractor(method=FEATURE_DETECTOR, n_features=KEYPOINTS_PER_IMAGE)
        self.bin_manager = CompassBinManager(n_bins=NUM_COMPASS_BINS)
        
        # Phase 2 systems
        use_xfeat_backend = VO_BACKEND in ("AUTO", "XFEAT_LIGHTGLUE_TRT", "XFEAT_TRT")
        if use_xfeat_backend:
            if XFeatLightGlueTRTBackend is None:
                self.flow_mapper = MSCKFVelocityEstimator()
                self.node.get_logger().warn(
                    f"⚠️ XFEAT backend import failed: {XFEAT_BACKEND_IMPORT_ERROR}"
                )
                if VO_BACKEND == "AUTO":
                    self.node.get_logger().warn("⚠️ AUTO mode fallback to MSCKF backend")
                else:
                    self.node.get_logger().warn("⚠️ Falling back to MSCKF backend")
            else:
                candidate = XFeatLightGlueTRTBackend(
                    module_name=XFEAT_TRT_MODULE,
                    native_module_path=XFEAT_TRT_NATIVE_MODULE_PATH,
                    config_path=XFEAT_TRT_CONFIG_PATH,
                    xfeat_engine_path=XFEAT_TRT_XFEAT_ENGINE_PATH,
                    lightglue_engine_path=XFEAT_TRT_LIGHTGLUE_ENGINE_PATH,
                    velocity_scale=XFEAT_TRT_VEL_SCALE,
                    focal_px=XFEAT_TRT_FOCAL_PX,
                    min_matches=XFEAT_TRT_MIN_MATCHES,
                    frame_stride=XFEAT_TRT_FRAME_STRIDE,
                    repo_path=XFEAT_TRT_REPO_PATH,
                    top_k=XFEAT_TRT_TOP_K,
                )
                if getattr(candidate, "available", False):
                    self.flow_mapper = candidate
                    self.node.get_logger().info(
                        f"✅ VO backend: XFEAT_LIGHTGLUE_TRT ({XFEAT_TRT_MODULE}, mode={getattr(candidate, 'backend_mode', 'unknown')}, repo={XFEAT_TRT_REPO_PATH})"
                    )
                else:
                    self.flow_mapper = MSCKFVelocityEstimator()
                    self.node.get_logger().warn(
                        f"⚠️ VO backend XFEAT_LIGHTGLUE_TRT unavailable: {candidate.get_dashboard_payload().get('debug', {}).get('backend_error', 'unknown error')}"
                    )
                    if VO_BACKEND == "AUTO":
                        self.node.get_logger().warn("⚠️ AUTO mode fallback to MSCKF backend")
                    else:
                        self.node.get_logger().warn("⚠️ Falling back to MSCKF backend")
        else:
            self.flow_mapper = MSCKFVelocityEstimator()
            self.node.get_logger().info("✅ VO backend: MSCKF")
        self.current_altitude = DEFAULT_ALTITUDE  # meters
        self.altitude_lock = threading.Lock()
        self.current_altitude_source = "DEFAULT"
        
        # Data storage
        self.current_image = None
        self.current_depth = None
        self.current_keypoints = None
        self.current_descriptors = None
        self.lock = threading.Lock()
        self.last_relocalization_report = 0.0
        self.last_relocalized_bin = None
        self.last_match_score = 0
        self.last_heading_correction_time = 0.0
        self.frame_count = 0  # For frame skipping optimization
        self.heading_zero_ref_deg = None
        self.heading_filtered_deg = None
        self.was_calibrated = False
        self.latest_pixhawk_gyr_z = None
        self.latest_pixhawk_gyr_x = None
        self.latest_pixhawk_gyr_y = None
        self.latest_pixhawk_yaw_deg = None
        self.latest_pixhawk_ts = 0.0
        self.latest_pixhawk_recv_ts = 0.0
        self.pixhawk_lock = threading.Lock()
        self.latest_fused_gyr_z = 0.0
        self.latest_fused_gyro_xyz = (0.0, 0.0, 0.0)
        self.camera_imu_history = deque(maxlen=2000)  # (ts, wx, wy, wz) for visual-inertial coupling
        self.latest_madgwick_yaw_deg = None
        self.latest_madgwick_ts = 0.0
        self.latest_madgwick_recv_ts = 0.0
        self.madgwick_lock = threading.Lock()
        self._camera_info_intrinsics_applied = False
        self._camera_info_logged = False
        
        # Subscribers
        self.sub_imu = self.node.create_subscription(
            Imu,
            '/camera/camera/imu',
            self.imu_callback,
            qos_profile_sensor_data
        )
        
        self.sub_image = self.node.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.sub_camera_info = self.node.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data
        )
        
        self.sub_depth = self.node.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.sub_madgwick_imu = self.node.create_subscription(
            Imu,
            MADGWICK_IMU_TOPIC,
            self.madgwick_imu_callback,
            qos_profile_sensor_data
        )

        self.sub_pixhawk_imu = self.node.create_subscription(
            Imu,
            PIXHAWK_IMU_TOPIC,
            self.pixhawk_imu_callback,
            qos_profile_sensor_data
        )
        self.sub_pixhawk_imu_alt = self.node.create_subscription(
            Imu,
            PIXHAWK_IMU_TOPIC_ALT,
            self.pixhawk_imu_callback,
            qos_profile_sensor_data
        )
        self.node.get_logger().info(
            f"✅ Subscribed to Pixhawk IMU topics: {PIXHAWK_IMU_TOPIC}, {PIXHAWK_IMU_TOPIC_ALT}"
        )
        
        self.sub_altitude = []
        for topic in MAVROS_RANGE_TOPICS:
            sub = self.node.create_subscription(
                Range,
                topic,
                (lambda m, t=topic: self.tof_callback(m, t)),
                qos_profile_sensor_data
            )
            self.sub_altitude.append(sub)
        self.node.get_logger().info(
            f"✅ Subscribed to rangefinder topics: {', '.join(MAVROS_RANGE_TOPICS)}"
        )
        
        # Publisher: MAVROS velocity
        self.pub_velocity = self.node.create_publisher(
            TwistStamped,
            '/mavros/vision_speed/speed_twist',
            10
        )
        self.pub_pose = self.node.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        # AprilTag detector + filtered pose state
        self.apriltag_enabled = False
        self.apriltag_detector = None
        self.apriltag_dict = None
        self.apriltag_params = None
        self.apriltag_last_pub_ts = 0.0
        self.apriltag_last_log_ts = 0.0
        self.apriltag_tvec_filt = None
        self.apriltag_rvec_filt = None
        self.latest_april_debug = {
            'april_enabled': False,
            'april_detected': False,
            'april_id': -1,
            'april_reproj_px': 0.0,
            'april_x': 0.0,
            'april_y': 0.0,
            'april_z': 0.0,
        }
        self.cam_matrix = np.array([
            [REALSENSE_FX, 0.0, REALSENSE_CX],
            [0.0, REALSENSE_FY, REALSENSE_CY],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self._init_apriltag_detector()
        
        self._image_count = 0
        self._imu_count = 0
        
        self.node.get_logger().info("✅ Phase 2 Node initialized (Metric Scaling + MAVROS)")
        self.node.get_logger().info(f"   Camera Calibration: fx={REALSENSE_FX}, fy={REALSENSE_FY}, cx={REALSENSE_CX}, cy={REALSENSE_CY}")
        self.node.get_logger().info(f"   Madgwick yaw fusion: {'ENABLED' if USE_MADGWICK_YAW else 'DISABLED'} | Topic: {MADGWICK_IMU_TOPIC}")
        self.node.get_logger().info(f"   Pixhawk IMU fusion: {'ENABLED' if USE_PIXHAWK_IMU_CORRECTION else 'DISABLED'} | Topic: {PIXHAWK_IMU_TOPIC}")
        self.node.get_logger().info(f"   Pixhawk yaw primary: {'ENABLED' if USE_PIXHAWK_YAW_PRIMARY else 'DISABLED'} | Rate correction: {'ENABLED' if USE_PIXHAWK_RATE_CORRECTION else 'DISABLED'}")

    def _init_apriltag_detector(self):
        if not APRILTAG_ENABLED:
            return
        if not hasattr(cv2, 'aruco'):
            self.node.get_logger().warn("⚠️ cv2.aruco unavailable; AprilTag pose publishing disabled")
            return

        dict_id = getattr(cv2.aruco, APRILTAG_DICT_NAME, None)
        if dict_id is None:
            self.node.get_logger().warn(f"⚠️ {APRILTAG_DICT_NAME} not found in cv2.aruco; AprilTag pose disabled")
            return

        self.apriltag_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.apriltag_params = cv2.aruco.DetectorParameters()
        else:
            self.apriltag_params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.apriltag_detector = cv2.aruco.ArucoDetector(self.apriltag_dict, self.apriltag_params)

        self.apriltag_enabled = True
        self.latest_april_debug['april_enabled'] = True
        self.node.get_logger().info(
            f"✅ AprilTag detector enabled ({APRILTAG_DICT_NAME}, size={APRILTAG_SIZE_M:.3f}m)"
        )

    @staticmethod
    def _rotation_matrix_to_quaternion(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0.0:
            S = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return qx, qy, qz, qw

    def _detect_and_publish_apriltag_pose(self, gray_image, stamp, timestamp_sec):
        if not self.apriltag_enabled or gray_image is None:
            return False

        if self.apriltag_detector is not None:
            corners, ids, _ = self.apriltag_detector.detectMarkers(gray_image)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_image,
                self.apriltag_dict,
                parameters=self.apriltag_params,
            )

        if ids is None or len(ids) == 0:
            self.latest_april_debug.update({
                'april_detected': False,
                'april_id': -1,
                'april_reproj_px': 0.0,
            })
            return False

        # Pick largest detected tag by image area for best stability.
        best_idx = 0
        best_area = -1.0
        for i, c in enumerate(corners):
            pts = c.reshape(4, 2)
            area = abs(cv2.contourArea(pts.astype(np.float32)))
            if area > best_area:
                best_area = area
                best_idx = i

        img_pts = corners[best_idx].reshape(4, 2).astype(np.float32)
        tag_id = int(ids[best_idx][0])
        half = APRILTAG_SIZE_M * 0.5
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            self.cam_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                self.cam_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return False

        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.cam_matrix, self.dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(proj_pts.reshape(-1, 2) - img_pts, axis=1)))
        if reproj_err > APRILTAG_MAX_REPROJ_ERR_PX:
            self.latest_april_debug.update({
                'april_detected': False,
                'april_id': tag_id,
                'april_reproj_px': reproj_err,
            })
            return False

        if self.apriltag_tvec_filt is None:
            self.apriltag_tvec_filt = tvec.astype(np.float64)
            self.apriltag_rvec_filt = rvec.astype(np.float64)
        else:
            a = APRILTAG_POSE_ALPHA
            self.apriltag_tvec_filt = (1.0 - a) * self.apriltag_tvec_filt + a * tvec
            self.apriltag_rvec_filt = (1.0 - a) * self.apriltag_rvec_filt + a * rvec

        R, _ = cv2.Rodrigues(self.apriltag_rvec_filt)
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(R)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "camera_link"
        pose_msg.pose.position.x = float(self.apriltag_tvec_filt[0][0])
        pose_msg.pose.position.y = float(self.apriltag_tvec_filt[1][0])
        pose_msg.pose.position.z = float(self.apriltag_tvec_filt[2][0])
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        try:
            self.pub_pose.publish(pose_msg)
        except RCLError:
            return False

        self.latest_april_debug.update({
            'april_enabled': True,
            'april_detected': True,
            'april_id': tag_id,
            'april_reproj_px': reproj_err,
            'april_x': pose_msg.pose.position.x,
            'april_y': pose_msg.pose.position.y,
            'april_z': pose_msg.pose.position.z,
        })

        if (timestamp_sec - self.apriltag_last_log_ts) > APRILTAG_LOG_PERIOD_S:
            self.node.get_logger().info(
                f"🏷️ AprilTag {tag_id} | pose(m)=({pose_msg.pose.position.x:+.2f}, {pose_msg.pose.position.y:+.2f}, {pose_msg.pose.position.z:+.2f}) | reproj={reproj_err:.2f}px"
            )
            self.apriltag_last_log_ts = timestamp_sec

        return True

    def quaternion_to_yaw_deg(self, qx, qy, qz, qw, yaw_sign=1.0):
        # Standard yaw extraction from quaternion (Z-axis heading)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        return (yaw_sign * yaw_rad * 180.0 / math.pi) % 360.0

    def _blend_angles_deg(self, base_deg, target_deg, alpha):
        diff = ((target_deg - base_deg + 540.0) % 360.0) - 180.0
        return (base_deg + alpha * diff) % 360.0

    def madgwick_imu_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        recv_ts = time.time()
        q = msg.orientation
        yaw_deg = self.quaternion_to_yaw_deg(q.x, q.y, q.z, q.w, yaw_sign=MADGWICK_YAW_SIGN)
        with self.madgwick_lock:
            self.latest_madgwick_yaw_deg = yaw_deg
            self.latest_madgwick_ts = timestamp
            self.latest_madgwick_recv_ts = recv_ts

    def pixhawk_imu_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        recv_ts = time.time()
        q = msg.orientation
        
        # DETAILED DEBUG: Log raw quaternion and extracted yaw
        if not hasattr(self, 'px_msg_count'):
            self.px_msg_count = 0
        self.px_msg_count += 1
        
        yaw_deg = self.quaternion_to_yaw_deg(q.x, q.y, q.z, q.w, yaw_sign=PIXHAWK_YAW_SIGN)
        
        # Log with extreme detail for first 10 messages
        if self.px_msg_count <= 10:
            self.node.get_logger().info(
                f"[PIXHAWK-{self.px_msg_count}] RAW QUAT: x={q.x:.4f}, y={q.y:.4f}, z={q.z:.4f}, w={q.w:.4f} | "
                f"EXTRACTED YAW: {yaw_deg:.1f}° | GYR_Z: {msg.angular_velocity.z:.3f}")
        
        with self.pixhawk_lock:
            self.latest_pixhawk_gyr_x = float(msg.angular_velocity.x)
            self.latest_pixhawk_gyr_y = float(msg.angular_velocity.y)
            self.latest_pixhawk_gyr_z = PIXHAWK_YAW_SIGN * msg.angular_velocity.z
            self.latest_pixhawk_yaw_deg = yaw_deg
            self.latest_pixhawk_ts = timestamp
            self.latest_pixhawk_recv_ts = recv_ts
            if self.frame_count < 5:
                self.node.get_logger().info(
                    f"[PIXHAWK-FRAME] Received: yaw={yaw_deg:.1f}° msg_ts={timestamp:.3f} recv_ts={recv_ts:.3f}"
                )

    def get_fused_yaw(self, timestamp_sec):
        gyro_yaw = self.imu_integrator.get_yaw()
        fused_yaw = gyro_yaw
        source = "GYRO"

        if USE_MADGWICK_YAW:
            with self.madgwick_lock:
                mad_yaw = self.latest_madgwick_yaw_deg
                mad_age = timestamp_sec - self.latest_madgwick_ts
                mad_recv_age = time.time() - self.latest_madgwick_recv_ts if self.latest_madgwick_recv_ts > 0.0 else 1e9
            if mad_yaw is not None and (0.0 <= mad_age <= MADGWICK_TIMEOUT_S or mad_recv_age <= MADGWICK_TIMEOUT_S):
                mad_diff = ((mad_yaw - fused_yaw + 540.0) % 360.0) - 180.0
                if abs(mad_diff) <= MADGWICK_MAX_DIFF_DEG:
                    fused_yaw = self._blend_angles_deg(fused_yaw, mad_yaw, MADGWICK_BLEND_ALPHA)
                    source += "+MAD"

        if USE_PIXHAWK_IMU_CORRECTION:
            with self.pixhawk_lock:
                px_yaw = self.latest_pixhawk_yaw_deg
                px_ts = self.latest_pixhawk_ts
                px_age = timestamp_sec - px_ts
                px_recv_age = time.time() - self.latest_pixhawk_recv_ts if self.latest_pixhawk_recv_ts > 0.0 else 1e9
            
            # Debug: first 5 frames print Pixhawk status
            if self.frame_count < 5:
                self.node.get_logger().info(
                    f"[DEBUG] Frame {self.frame_count}: px_yaw={px_yaw}, px_ts={px_ts:.3f}, "
                    f"px_age={px_age:.4f}, px_recv_age={px_recv_age:.4f}, timeout={PIXHAWK_TIMEOUT_S}"
                )
            
            if px_yaw is not None and (0.0 <= px_age <= PIXHAWK_TIMEOUT_S or px_recv_age <= PIXHAWK_TIMEOUT_S):
                if USE_PIXHAWK_YAW_PRIMARY:
                    return px_yaw, "PX4_YAW"
                px_diff = ((px_yaw - fused_yaw + 540.0) % 360.0) - 180.0
                if abs(px_diff) <= PIXHAWK_MAX_DIFF_DEG:
                    fused_yaw = self._blend_angles_deg(fused_yaw, px_yaw, PIXHAWK_YAW_BLEND_ALPHA)
                    source += "+PX4"

        return fused_yaw, source

    def get_zeroed_heading(self, raw_yaw_deg):
        """Return heading relative to first calibrated sample, with circular smoothing."""
        if self.heading_zero_ref_deg is None:
            self.heading_zero_ref_deg = raw_yaw_deg
            self.heading_filtered_deg = 0.0
            return 0.0

        heading = (raw_yaw_deg - self.heading_zero_ref_deg) % 360.0
        if self.heading_filtered_deg is None:
            self.heading_filtered_deg = heading
            return heading

        self.heading_filtered_deg = self._blend_angles_deg(self.heading_filtered_deg, heading, HEADING_SMOOTH_ALPHA)
        return self.heading_filtered_deg
    
    def imu_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        camera_gyr_z = msg.angular_velocity.z
        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        with self.lock:
            self.camera_imu_history.append((timestamp, wx, wy, wz))

        # Fuse camera IMU with Pixhawk IMU if Pixhawk sample is fresh.
        fused_gyr_z = camera_gyr_z
        pixhawk_fused = False
        if USE_PIXHAWK_IMU_CORRECTION and USE_PIXHAWK_RATE_CORRECTION:
            with self.pixhawk_lock:
                if self.latest_pixhawk_gyr_z is not None:
                    age_msg = timestamp - self.latest_pixhawk_ts
                    age_recv = time.time() - self.latest_pixhawk_recv_ts if self.latest_pixhawk_recv_ts > 0.0 else 1e9
                    if (0.0 <= age_msg <= PIXHAWK_TIMEOUT_S) or (age_recv <= PIXHAWK_TIMEOUT_S):
                        if USE_PIXHAWK_RATE_PRIMARY:
                            fused_gyr_z = self.latest_pixhawk_gyr_z
                        else:
                            fused_gyr_z = ((1.0 - PIXHAWK_FUSE_ALPHA) * camera_gyr_z) + (PIXHAWK_FUSE_ALPHA * self.latest_pixhawk_gyr_z)
                        pixhawk_fused = True

        self.imu_integrator.update(fused_gyr_z, timestamp)
        self.latest_fused_gyr_z = fused_gyr_z
        fused_wx = float(msg.angular_velocity.x)
        fused_wy = float(msg.angular_velocity.y)
        if USE_PIXHAWK_IMU_CORRECTION and USE_PIXHAWK_RATE_CORRECTION:
            with self.pixhawk_lock:
                if self.latest_pixhawk_gyr_x is not None and self.latest_pixhawk_gyr_y is not None:
                    age_msg = timestamp - self.latest_pixhawk_ts
                    age_recv = time.time() - self.latest_pixhawk_recv_ts if self.latest_pixhawk_recv_ts > 0.0 else 1e9
                    if (0.0 <= age_msg <= PIXHAWK_TIMEOUT_S) or (age_recv <= PIXHAWK_TIMEOUT_S):
                        if USE_PIXHAWK_RATE_PRIMARY:
                            fused_wx = self.latest_pixhawk_gyr_x
                            fused_wy = self.latest_pixhawk_gyr_y
                        else:
                            fused_wx = ((1.0 - PIXHAWK_FUSE_ALPHA) * fused_wx) + (PIXHAWK_FUSE_ALPHA * self.latest_pixhawk_gyr_x)
                            fused_wy = ((1.0 - PIXHAWK_FUSE_ALPHA) * fused_wy) + (PIXHAWK_FUSE_ALPHA * self.latest_pixhawk_gyr_y)
        self.latest_fused_gyro_xyz = (
            float(fused_wx),
            float(fused_wy),
            fused_gyr_z,
        )
        
        self._imu_count += 1
        if self._imu_count % 100 == 0:
            if not self.imu_integrator.is_calibrated():
                self.was_calibrated = False
                state = self.imu_integrator.get_calibration_state()
                if state == "bias":
                    progress = min(100.0, 100.0 * self._imu_count / max(IMU_CALIBRATION_SAMPLES, 1))
                    print(f"📊 IMU: stage 1/2 bias calibration {progress:.0f}% (keep drone still)")
                elif state == "spin":
                    spin_deg = self.imu_integrator.get_spin_progress_deg()
                    print(f"📊 IMU: stage 2/2 spin calibration {spin_deg:.0f}/{SPIN_CALIB_TARGET_DEG:.0f} deg (rotate drone now)")
                return
            if not self.was_calibrated:
                self.heading_zero_ref_deg = None
                self.heading_filtered_deg = None
                self.was_calibrated = True
                print("✅ Heading zeroed at calibration complete")

            raw_yaw, yaw_src = self.get_fused_yaw(timestamp)
            yaw = self.get_zeroed_heading(raw_yaw)
            bias = self.imu_integrator.get_bias()
            scale = self.imu_integrator.get_scale()
            src = "CAM+PX4" if pixhawk_fused else "CAM"
            print(f"📊 IMU[{src}|{yaw_src}]: {self._imu_count} | Yaw: {yaw:.1f}° | BiasZ: {bias:+.5f} rad/s | Scale: {scale:.3f} | Alt: {self.current_altitude:.2f}m")
    
    def depth_callback(self, msg):
        try:
            self.current_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as e:
            self.node.get_logger().warn(f"Depth conversion failed: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        if len(msg.k) < 9:
            return

        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])

        if (not self._camera_info_intrinsics_applied) and hasattr(self.flow_mapper, 'set_intrinsics'):
            self.flow_mapper.set_intrinsics(fx, fy, cx, cy, source='ros_camera_info')
            self._camera_info_intrinsics_applied = True
            self.node.get_logger().info(
                f"✅ Applied camera intrinsics from /camera_info: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}"
            )

        if not self._camera_info_logged:
            self._camera_info_logged = True
            max_dist = max((abs(float(d)) for d in msg.d), default=0.0)
            if (not bool(msg.roi.do_rectify)) and max_dist > 1e-6:
                self.node.get_logger().warn(
                    "⚠️ color/image_raw appears distorted (non-zero D, do_rectify=false). "
                    "Prefer a rectified color topic or undistort before VO if floor lines bow in IPM."
                )
    
    def tof_callback(self, msg, source_topic='unknown'):
        """Update altitude from MAVROS/local rangefinder topic."""
        range_m = msg.range
        if TOF_MIN_RANGE <= range_m <= TOF_MAX_RANGE:
            with self.altitude_lock:
                self.current_altitude = range_m
                if self.current_altitude_source != source_topic:
                    self.current_altitude_source = source_topic
                    self.node.get_logger().info(
                        f"📏 Altitude source active: {source_topic} ({range_m:.2f} m)"
                    )
    
    def image_callback(self, msg):
        self._image_count += 1
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            if cv_image is None:
                return
        except Exception as e:
            self.node.get_logger().warn(f"Image conversion failed: {e}")
            return
        
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if not self.imu_integrator.is_calibrated():
            self.was_calibrated = False
        elif not self.was_calibrated:
            self.heading_zero_ref_deg = None
            self.heading_filtered_deg = None
            self.was_calibrated = True

        raw_yaw, yaw_src = self.get_fused_yaw(timestamp)
        yaw = self.get_zeroed_heading(raw_yaw)

        if not self.imu_integrator.is_calibrated():
            if self._image_count % 30 == 0:
                state = self.imu_integrator.get_calibration_state()
                if state == "bias":
                    print("⏳ Waiting for IMU calibration stage 1/2... hold drone still")
                elif state == "spin":
                    print("⏳ Waiting for IMU calibration stage 2/2... rotate drone ~360 deg")
            return

        vx, vy = self.flow_mapper.update(
            cv_image,
            self.current_depth,
            self.current_altitude,
            timestamp,
            imu_gyro_xyz=self.latest_fused_gyro_xyz,
            yaw_deg=yaw,
        )

        self._detect_and_publish_apriltag_pose(cv_image, msg.header.stamp, timestamp)

        if DASHBOARD_AVAILABLE:
            dashboard.update_route(self.flow_mapper.get_dashboard_payload())
        
        # Extract features every frame (essential for compass bins during rotation)
        kp, desc = self.feature_extractor.extract(cv_image)
        with self.lock:
            self.current_image = cv_image
            self.current_keypoints = kp
            self.current_descriptors = desc
        
        current_bin = self.bin_manager.heading_to_bin(yaw)
        
        # Phase 1: Store features and relocalize
        runtime = self.feature_extractor.get_runtime_status()
        ai_enabled = bool(runtime.get('ai_enabled', False))
        do_relocalize = (not ai_enabled) or (self.frame_count % AI_RELOCALIZE_STRIDE == 0)

        if do_relocalize:
            self.bin_manager.store_features(yaw, kp, desc, timestamp)
            relocated_bin, match_score = self.bin_manager.relocalize(
                yaw,
                desc,
                self.feature_extractor,
                query_rad_search=LOCAL_QUERY_RAD
            )
            self.last_relocalized_bin = relocated_bin
            self.last_match_score = match_score
        else:
            relocated_bin = self.last_relocalized_bin
            match_score = self.last_match_score

        # Camera-derived direction from visual relocalization (bin center heading).
        camera_heading_deg = None
        if relocated_bin is not None and match_score >= RELOCALIZATION_MATCH_THRESHOLD:
            camera_heading_deg = ((relocated_bin + 0.5) * (360.0 / NUM_COMPASS_BINS)) % 360.0

        # If loop closure is confident and vehicle is not rotating, gently pull heading toward matched bin center.
        if relocated_bin is not None and match_score >= HEADING_CORRECTION_MIN_MATCH:
            bin_center = ((relocated_bin + 0.5) * (360.0 / NUM_COMPASS_BINS)) % 360.0
            yaw_err = ((bin_center - yaw + 540.0) % 360.0) - 180.0
            not_rotating = abs(self.latest_fused_gyr_z) < HEADING_CORRECTION_MAX_GYRO_RAD_S
            nearly_stationary = math.hypot(vx, vy) < HEADING_CORRECTION_MAX_SPEED_MPS
            cooldown_ok = (timestamp - self.last_heading_correction_time) > HEADING_CORRECTION_COOLDOWN_S
            if not_rotating and nearly_stationary and cooldown_ok and abs(yaw_err) <= HEADING_CORRECTION_MAX_ERROR_DEG:
                self.imu_integrator.apply_heading_correction(bin_center, alpha=HEADING_CORRECTION_ALPHA)
                self.last_heading_correction_time = timestamp
                raw_yaw, yaw_src = self.get_fused_yaw(timestamp)
                yaw = self.get_zeroed_heading(raw_yaw)
        
        if relocated_bin is not None and (timestamp - self.last_relocalization_report) > 1.0:
            self.node.get_logger().info(f"✅ Relocalized in Bin {relocated_bin} | Heading: {yaw:.1f}° ({yaw_src}) | Matches: {match_score}")
            self.last_relocalization_report = timestamp

        if self._image_count % DIRECTION_PRINT_EVERY_N_FRAMES == 0:
            if camera_heading_deg is None:
                print(
                    f"🧭 DIR | CAMERA(VISION): unknown (no lock) | "
                    f"IMU({yaw_src}): {yaw:6.1f}°"
                )
            else:
                heading_err = ((camera_heading_deg - yaw + 540.0) % 360.0) - 180.0
                print(
                    f"🧭 DIR | CAMERA(VISION): {camera_heading_deg:6.1f}° "
                    f"(bin {relocated_bin}, m={match_score}) | "
                    f"IMU({yaw_src}): {yaw:6.1f}° | Δ={heading_err:+.1f}°"
                )
        
        # Publish velocity to MAVROS
        twist_msg = TwistStamped()
        twist_msg.header.stamp = msg.header.stamp
        twist_msg.header.frame_id = "camera_link"
        twist_msg.twist.linear.x = vx
        twist_msg.twist.linear.y = vy
        twist_msg.twist.linear.z = 0.0  # No Z control from vision
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = 0.0

        try:
            self.pub_velocity.publish(twist_msg)
        except RCLError:
            # ROS context can become invalid during shutdown while callbacks are still draining.
            return

        # Update web dashboard
        # Rate-gate dashboard updates to ~10Hz (every 3rd frame at 30Hz) to prevent blocking
        if DASHBOARD_AVAILABLE and kp is not None and (self._image_count % 3 == 0):
            vo_debug = self.flow_mapper.get_dashboard_payload().get('debug', {})
            vo_debug.update(self.latest_april_debug)
            dashboard.update_frame(
                cv_image,
                yaw,
                self.current_altitude,
                vx,
                vy,
                len(kp),
                ai_debug=self.feature_extractor.get_runtime_status(),
                vo_debug=vo_debug
            )
            for bin_id, bin_obj in enumerate(self.bin_manager.bins):
                is_ready = len(bin_obj.descriptors) >= 3
                dashboard.update_bin(bin_id, len(bin_obj.descriptors), is_ready)
            if relocated_bin is not None:
                dashboard.update_relocalization(relocated_bin, match_score)

        kp_count = len(kp) if kp is not None else 0
        print(f"📸 IMG #{self._image_count:4d} | KP: {kp_count:3d} | VEL: ({vx:+.2f}, {vy:+.2f}) m/s | ALT: {self.current_altitude:.2f}m")

    def save_route_outputs(self):
        if self.flow_mapper is None:
            return
        print("\n🗺️ Generating 2D Route Map...")
        self.flow_mapper.save_map(ROUTE_MAP_IMAGE_PATH, ROUTE_MAP_CSV_PATH)

def main():
    cleanup()

    # Start web dashboard early so UI is available while back-end systems initialize.
    if DASHBOARD_AVAILABLE:
        from web_dashboard import app as flask_app

        def run_flask():
            flask_app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print("\n🌐 Web Dashboard started on http://0.0.0.0:5000")
        time.sleep(1)
    
    # Launch MAVROS (connects to Pixhawk FCU and provides IMU/odometry data)
    print("\n📡 Launching MAVROS (Pixhawk flight controller interface)...")
    fcu_device = get_fcu_serial_device()
    fcu_url = f"{fcu_device}:{MAVROS_BAUD}"
    os.system(f"ros2 launch mavros px4.launch fcu_url:={shlex.quote(fcu_url)} >> {MAVROS_LOG_FILE} 2>&1 &")
    time.sleep(8)  # Wait for MAVROS node/service startup
    request_mavros_stream_rate(MAVROS_STREAM_RATE_HZ, MAVROS_STREAM_RETRIES)
    time.sleep(2)
    
    launch_hardware()
    
    print("\n⏳ Waiting for hardware to stabilize (5 seconds)...")
    time.sleep(5)
    
    rclpy.init()
    node = Phase2Node()
    
    jetson_ip = get_ip()
    print("\n" + "="*70)
    print("🚀 PHASE 2: METRIC SCALING & FLIGHT CONTROLLER FUSION")
    print("="*70)
    print(f"   IP: {jetson_ip}")
    print(f"   Camera Calibration: fx={REALSENSE_FX}, fy={REALSENSE_FY}")
    if DASHBOARD_AVAILABLE:
        print(f"\n🌐 OPEN WEB GUI:")
        print(f"   http://{jetson_ip}:5000")
        print(f"   (or http://localhost:5000 if on local machine)")
    print("\nEKF3 Configuration Required:")
    print("   Mission Planner → CONFIG/TUNING → Full Parameter List")
    print("   - EK3_SRC1_VELXY = 6 (External Vision velocity)")
    print("   - EK3_SRC1_POSZ = 2 (External altitude, use ToF)")
    print("   - EK3_SRC1_POSX = 0 (Don't use external position)")
    print("="*70 + "\n")
    
    print("🎯 Flight Test Procedure:")
    print("   1. Configure EKF3 in Mission Planner (see above)")
    print("   2. Let Phase 2 run 20-30 seconds to build feature memory")
    print("   3. Open web dashboard: http://<jetson_ip>:5000")
    print("   4. Monitor bin initialization in web GUI")
    print("   5. Arm drone in Loiter mode")
    print("   6. Take off indoors (2-3 meters)")
    print("   7. Release sticks—drone should hold position via vision + ToF")
    print("="*70 + "\n")
    
    try:
        rclpy.spin(node.node)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Phase 2...")
    except ExternalShutdownException:
        print("\n🛑 ROS shutdown requested.")
    finally:
        try:
            node.save_route_outputs()
        except Exception as e:
            print(f"⚠️  Route map save failed: {e}")
        try:
            node.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        cleanup()
        print("✅ Cleanup complete.")

if __name__ == '__main__':
    main()
