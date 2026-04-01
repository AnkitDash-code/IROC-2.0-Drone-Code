#!/usr/bin/env python3
"""
Web Dashboard for Phase 2: Real-time visualization of compass bins, video stream, and drone state.

Provides:
  - 3D compass bin visualization (which bins have features)
  - Live video stream with bin overlay
  - Yaw heading compass indicator
  - Real-time metrics (FPS, feature count, altitude, velocity)
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
import json
import math
import os
from collections import deque
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_THIS_DIR, "templates"))
CORS(app)

# Global state shared with Phase 2 node
class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.new_frame_cv = threading.Condition(self.lock)
        self.current_image = None
        self.current_jpeg = None
        self.current_frame_ts = 0.0
        self.current_ingest_ts = 0.0
        self.frame_id = 0
        self.current_yaw = 0.0
        self.current_altitude = 1.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.keypoint_count = 0
        self.bin_states = {}  # {bin_id: (heading_range, num_features, is_ready)}
        self.fps = 30.0
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.relocalized_bin = None
        self.match_score = 0
        self.num_bins = 8
        self.route_points = [(0.0, 0.0)]  # list of (x_m, z_m)
        self.route_tracks = {
            'fused': [(0.0, 0.0)],
            'pnp': [(0.0, 0.0)],
            'fallback': [(0.0, 0.0)],
        }
        self.route_extent_m = 1.0
        self.vo_debug = {}
        self.ai_debug = {}
        
        # Initialize bin states
        for i in range(self.num_bins):
            bin_width = 360.0 / self.num_bins
            min_h = (i * bin_width) % 360.0
            max_h = ((i + 1) * bin_width) % 360.0
            self.bin_states[i] = {
                'heading_range': (min_h, max_h),
                'num_features': 0,
                'is_ready': False
            }
    
    def update_frame(self, image, yaw, altitude, vx, vy, kp_count, ai_debug=None, vo_debug=None, jpeg_bytes=None, frame_ts=None):
        with self.lock:
            if image is not None:
                self.current_image = image
            if jpeg_bytes is not None:
                self.current_jpeg = jpeg_bytes
                self.current_ingest_ts = time.time()
                self.current_frame_ts = float(frame_ts) if frame_ts is not None else self.current_ingest_ts
                self.frame_id += 1
                self.new_frame_cv.notify_all()
            self.current_yaw = yaw
            self.current_altitude = altitude
            self.velocity_x = vx
            self.velocity_y = vy
            self.keypoint_count = kp_count
            if ai_debug is not None:
                self.ai_debug = ai_debug
            if vo_debug is not None:
                self.vo_debug = vo_debug
            
            # Update FPS
            now = time.time()
            self.frame_times.append(now - self.last_frame_time)
            self.last_frame_time = now
            if len(self.frame_times) > 0:
                avg_dt = np.mean(self.frame_times)
                if avg_dt > 0:
                    self.fps = 1.0 / avg_dt
    
    def update_bin(self, bin_id, num_features, is_ready):
        with self.lock:
            if bin_id in self.bin_states:
                self.bin_states[bin_id]['num_features'] = num_features
                self.bin_states[bin_id]['is_ready'] = is_ready
    
    def update_relocalization(self, bin_id, match_score):
        with self.lock:
            self.relocalized_bin = bin_id
            self.match_score = match_score

    def update_route(self, x_m, z_m=None):
        with self.lock:
            if isinstance(x_m, dict):
                payload = x_m
                tracks = payload.get('tracks', {})
                if isinstance(tracks, dict):
                    for k in ('fused', 'pnp', 'fallback'):
                        if k in tracks and isinstance(tracks[k], list) and len(tracks[k]) > 0:
                            self.route_tracks[k] = [(float(p[0]), float(p[1])) for p in tracks[k][-3000:]]

                    if 'fused' in self.route_tracks and len(self.route_tracks['fused']) > 0:
                        self.route_points = self.route_tracks['fused']

                dbg = payload.get('debug')
                if isinstance(dbg, dict):
                    self.vo_debug = dbg
            else:
                self.route_points.append((float(x_m), float(z_m)))
                if len(self.route_points) > 3000:
                    self.route_points = self.route_points[-3000:]
                self.route_tracks['fused'] = list(self.route_points)

            all_pts = []
            for pts in self.route_tracks.values():
                all_pts.extend(pts)
            if len(all_pts) == 0:
                all_pts = [(0.0, 0.0)]
            max_abs = max(max(abs(p[0]), abs(p[1])) for p in all_pts)
            self.route_extent_m = max(1.0, max_abs * 1.1)
    
    def get_state(self):
        with self.lock:
            now = time.time()
            source_age_ms = 0.0
            ingest_age_ms = 0.0
            if self.current_frame_ts > 0.0:
                source_age_ms = max(0.0, (now - self.current_frame_ts) * 1000.0)
            if self.current_ingest_ts > 0.0:
                ingest_age_ms = max(0.0, (now - self.current_ingest_ts) * 1000.0)
            return {
                'yaw': self.current_yaw,
                'altitude': self.current_altitude,
                'velocity_x': self.velocity_x,
                'velocity_y': self.velocity_y,
                'keypoint_count': self.keypoint_count,
                'fps': self.fps,
                'bin_states': self.bin_states,
                'relocalized_bin': self.relocalized_bin,
                'match_score': self.match_score,
                'route_points': self.route_points,
                'route_tracks': self.route_tracks,
                'route_extent_m': self.route_extent_m,
                'vo_debug': self.vo_debug,
                'ai_debug': self.ai_debug,
                'frame_id': self.frame_id,
                'frame_source_age_ms': source_age_ms,
                'frame_ingest_age_ms': ingest_age_ms,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_image(self):
        with self.lock:
            return self.current_image

    def get_jpeg(self):
        with self.lock:
            return self.current_jpeg

    def wait_for_new_jpeg(self, last_frame_id, timeout_s=0.25):
        with self.new_frame_cv:
            if self.frame_id <= int(last_frame_id):
                self.new_frame_cv.wait(timeout=float(max(0.0, timeout_s)))
            return self.frame_id, self.current_jpeg, self.current_frame_ts, self.current_ingest_ts

# Global dashboard state
dashboard = DashboardState()


def _safe_float(value, default=0.0):
    """Convert numeric values to finite float for JSON responses."""
    try:
        f = float(value)
        return f if math.isfinite(f) else float(default)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    """Convert numeric values to int for JSON responses."""
    try:
        return int(value)
    except Exception:
        return int(default)


def _compact_tracks(route_tracks, max_points=500):
    """Return route tracks with bounded size to keep API payload small."""
    compact = {}
    tracks = route_tracks if isinstance(route_tracks, dict) else {}
    for key in ('fused', 'pnp', 'fallback'):
        pts = tracks.get(key, [])
        if not isinstance(pts, list) or not pts:
            compact[key] = []
            continue
        sampled = pts[-max_points:]
        compact[key] = [[_safe_float(p[0]), _safe_float(p[1])] for p in sampled if isinstance(p, (list, tuple)) and len(p) >= 2]
    return compact


def _sanitize_bin_states(bin_states):
    """Normalize bin state structure to JSON-safe primitive values."""
    out = {}
    if not isinstance(bin_states, dict):
        return out

    for k, v in bin_states.items():
        if not isinstance(v, dict):
            continue
        hr = v.get('heading_range', (0.0, 0.0))
        if isinstance(hr, (list, tuple)) and len(hr) >= 2:
            heading_range = [_safe_float(hr[0]), _safe_float(hr[1])]
        else:
            heading_range = [0.0, 0.0]

        out[str(k)] = {
            'heading_range': heading_range,
            'num_features': _safe_int(v.get('num_features', 0)),
            'is_ready': bool(v.get('is_ready', False)),
        }
    return out

@app.route('/')
def index():
    """Serve the main dashboard HTML."""
    return render_template('dashboard.html')

@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/api/state')
def api_state():
    """Return current state as JSON (lightweight)."""
    try:
        state = dashboard.get_state()
        vo_debug = state.get('vo_debug', {}) if isinstance(state.get('vo_debug', {}), dict) else {}
        ai_debug = state.get('ai_debug', {}) if isinstance(state.get('ai_debug', {}), dict) else {}

        # Keep payload compact and JSON-safe so the frontend can poll reliably.
        safe_state = {
            'yaw': _safe_float(state.get('yaw', 0)),
            'altitude': _safe_float(state.get('altitude', 0)),
            'velocity_x': _safe_float(state.get('velocity_x', 0)),
            'velocity_y': _safe_float(state.get('velocity_y', 0)),
            'keypoint_count': _safe_int(state.get('keypoint_count', 0)),
            'fps': _safe_float(state.get('fps', 0)),
            'bin_states': _sanitize_bin_states(state.get('bin_states', {})),
            'relocalized_bin': state.get('relocalized_bin'),
            'match_score': _safe_int(state.get('match_score', 0)),
            'route_extent_m': _safe_float(state.get('route_extent_m', 1.0), 1.0),
            'route_tracks': _compact_tracks(state.get('route_tracks', {}), max_points=500),
            'frame_id': _safe_int(state.get('frame_id', 0)),
            'frame_source_age_ms': _safe_float(state.get('frame_source_age_ms', 0.0)),
            'frame_ingest_age_ms': _safe_float(state.get('frame_ingest_age_ms', 0.0)),
            'vo_debug': {
                'source': str(vo_debug.get('source', '')),
                'success': bool(vo_debug.get('success', False)),
                'flow_x': _safe_float(vo_debug.get('flow_x', 0.0)),
                'flow_y': _safe_float(vo_debug.get('flow_y', 0.0)),
                'alt': _safe_float(vo_debug.get('alt', 0.0)),
                'tracked': _safe_int(vo_debug.get('tracked', 0)),
                'n_samples': _safe_int(vo_debug.get('n_samples', 0)),
                'median_flow_px': _safe_float(vo_debug.get('median_flow_px', 0.0)),
                'gyro_comp': bool(vo_debug.get('gyro_comp', False)),
                # Backward-compatible fields used by existing frontend widgets.
                'step_m': _safe_float(vo_debug.get('step_m', 0.0)),
                'flow_px': _safe_float(vo_debug.get('flow_px', vo_debug.get('median_flow_px', 0.0))),
                'inlier_ratio': _safe_float(vo_debug.get('inlier_ratio', 0.0)),
                'rejected': bool(vo_debug.get('rejected', False)),
                'reject_reason': str(vo_debug.get('reject_reason', '')),
                'marker_locked': bool(vo_debug.get('marker_locked', False)),
                'detected_markers': _safe_int(vo_debug.get('detected_markers', 0)),
                'marker_label': vo_debug.get('marker_label', None),
                'marker_bbox': vo_debug.get('marker_bbox', None),
                'marker_boxes': vo_debug.get('marker_boxes', []),
                'tracking_active': bool(vo_debug.get('tracking_active', False)),
                'track_motion_px': _safe_float(vo_debug.get('track_motion_px', 0.0)),
                'drift_dx_px': _safe_float(vo_debug.get('drift_dx_px', 0.0)),
                'drift_dy_px': _safe_float(vo_debug.get('drift_dy_px', 0.0)),
                'drift_norm_px': _safe_float(vo_debug.get('drift_norm_px', 0.0)),
                'drift_command': str(vo_debug.get('drift_command', 'hold')),
                'board_center_est_px': vo_debug.get('board_center_est_px', None),
                'board_center_conf': _safe_float(vo_debug.get('board_center_conf', 0.0)),
                'blob_threshold': _safe_int(vo_debug.get('blob_threshold', 0)),
                'blob_threshold_applied': _safe_int(vo_debug.get('blob_threshold_applied', 0)),
                'blob_min_area': _safe_float(vo_debug.get('blob_min_area', 0.0)),
                'auto_exposure_threshold': bool(vo_debug.get('auto_exposure_threshold', False)),
                'auto_threshold_percentile': _safe_float(vo_debug.get('auto_threshold_percentile', 0.0)),
                'sticky_lock_active': bool(vo_debug.get('sticky_lock_active', False)),
                'lock_hold_count': _safe_int(vo_debug.get('lock_hold_count', 0)),
                'lock_hold_frames': _safe_int(vo_debug.get('lock_hold_frames', 0)),
                'lock_max_jump_px': _safe_float(vo_debug.get('lock_max_jump_px', 0.0)),
                'ema_alpha': _safe_float(vo_debug.get('ema_alpha', 0.0)),
                'base_ref_locked': bool(vo_debug.get('base_ref_locked', False)),
                'base_ref_px': vo_debug.get('base_ref_px', None),
                'base_ref_alt_m': _safe_float(vo_debug.get('base_ref_alt_m', 0.0)),
                'base_rel_dx_px': _safe_float(vo_debug.get('base_rel_dx_px', 0.0)),
                'base_rel_dy_px': _safe_float(vo_debug.get('base_rel_dy_px', 0.0)),
                'base_rel_norm_px': _safe_float(vo_debug.get('base_rel_norm_px', 0.0)),
                'base_rel_dx_m': _safe_float(vo_debug.get('base_rel_dx_m', 0.0)),
                'base_rel_dy_m': _safe_float(vo_debug.get('base_rel_dy_m', 0.0)),
                'base_rel_norm_m': _safe_float(vo_debug.get('base_rel_norm_m', 0.0)),
                'base_rel_metric_valid': bool(vo_debug.get('base_rel_metric_valid', False)),
                'hover_ema_dx_px': _safe_float(vo_debug.get('hover_ema_dx_px', 0.0)),
                'hover_ema_dy_px': _safe_float(vo_debug.get('hover_ema_dy_px', 0.0)),
                'hover_ema_norm_px': _safe_float(vo_debug.get('hover_ema_norm_px', 0.0)),
                'hover_correction_command': str(vo_debug.get('hover_correction_command', 'hold')),
                'imu_fusion_enabled': bool(vo_debug.get('imu_fusion_enabled', False)),
                'imu_fusion_filter': str(vo_debug.get('imu_fusion_filter', 'none')),
                'imu_fused_roll_deg': _safe_float(vo_debug.get('imu_fused_roll_deg', 0.0)),
                'imu_fused_pitch_deg': _safe_float(vo_debug.get('imu_fused_pitch_deg', 0.0)),
                'imu_fused_yaw_deg': _safe_float(vo_debug.get('imu_fused_yaw_deg', 0.0)),
                'attitude_comp_roll_delta_deg': _safe_float(vo_debug.get('attitude_comp_roll_delta_deg', 0.0)),
                'attitude_comp_pitch_delta_deg': _safe_float(vo_debug.get('attitude_comp_pitch_delta_deg', 0.0)),
                'attitude_comp_yaw_delta_deg': _safe_float(vo_debug.get('attitude_comp_yaw_delta_deg', 0.0)),
                'base_ref_roll_deg': _safe_float(vo_debug.get('base_ref_roll_deg', 0.0)),
                'base_ref_pitch_deg': _safe_float(vo_debug.get('base_ref_pitch_deg', 0.0)),
                'base_ref_yaw_deg': _safe_float(vo_debug.get('base_ref_yaw_deg', 0.0)),
            },
            'ai_debug': {
                'backend': str(ai_debug.get('backend', 'ORB')),
                'ai_fps': _safe_float(ai_debug.get('ai_fps', 0.0)),
            },
            'timestamp': state.get('timestamp', ''),
        }

        # Ensure all values are JSON-serializable
        return jsonify(safe_state)
    except Exception as e:
        print(f"⚠️  Error in /api/state: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_frame', methods=['POST'])
def api_upload_frame():
    """Accept an image + telemetry POST from an external process (e.g. non-ROS ArUco streamer).

    Supports multipart form (`image` file + `meta` JSON string) or JSON with
    `image_b64` and `meta` object.
    """
    try:
        img = None
        raw_jpeg = None
        meta = {}

        if request.files and 'image' in request.files:
            file = request.files['image']
            data = file.read()
            raw_jpeg = data
            # meta may be in form field
            meta_raw = request.form.get('meta')
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {}
        else:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                return jsonify({'error': 'invalid payload'}), 400
            meta = payload.get('meta', {}) if isinstance(payload.get('meta', {}), dict) else {}
            image_b64 = payload.get('image_b64')
            if image_b64:
                import base64
                raw_jpeg = base64.b64decode(image_b64)

        print("[web_dashboard] /api/upload_frame received meta:", {k: meta.get(k) for k in ['matched_tag', 'match_score', 'kp_count', 'detected_ids', 'camera_feed', 'source_used']})
        # Telemetry fields (with safe defaults)
        yaw = float(meta.get('yaw', 0.0))
        altitude = float(meta.get('altitude', 0.0))
        vx = float(meta.get('vx', 0.0))
        vy = float(meta.get('vy', 0.0))
        kp_count = int(meta.get('kp_count', 0))
        frame_ts = _safe_float(meta.get('frame_ts', time.time()))
        ai_debug = meta.get('ai_debug', None)
        vo_debug = meta.get('vo_debug', None)

        # Backward-compatible marker bridge: infer lock/count from matched_tag path.
        matched_tag = meta.get('matched_tag', None)
        match_score = int(meta.get('match_score', 0))
        detected_ids = meta.get('detected_ids', [])
        marker_bbox = meta.get('marker_bbox', None)
        marker_boxes = meta.get('marker_boxes', None)

        vo = dict(vo_debug) if isinstance(vo_debug, dict) else {}
        if matched_tag is not None:
            vo['marker_locked'] = True
            vo['marker_label'] = str(matched_tag)
        else:
            vo.setdefault('marker_locked', False)
            vo.setdefault('marker_label', None)

        if isinstance(detected_ids, list) and len(detected_ids) > 0:
            vo['detected_markers'] = len(detected_ids)
        else:
            vo['detected_markers'] = 1 if vo.get('marker_locked', False) else 0

        if isinstance(marker_bbox, (list, tuple)) and len(marker_bbox) >= 4:
            vo['marker_bbox'] = [int(marker_bbox[0]), int(marker_bbox[1]), int(marker_bbox[2]), int(marker_bbox[3])]

        if isinstance(marker_boxes, list):
            norm_boxes = []
            for mb in marker_boxes:
                if not isinstance(mb, dict):
                    continue
                bb = mb.get('bbox', None)
                if not isinstance(bb, (list, tuple)) or len(bb) < 4:
                    continue
                norm_boxes.append({
                    'label': str(mb.get('label', '?')),
                    'score': _safe_int(mb.get('score', 0)),
                    'bbox': [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                })
            vo['marker_boxes'] = norm_boxes
            if len(norm_boxes) > 0:
                vo['detected_markers'] = len(norm_boxes)

        if raw_jpeg is not None:
            dashboard.update_frame(None, yaw, altitude, vx, vy, kp_count, ai_debug=ai_debug, vo_debug=vo, jpeg_bytes=raw_jpeg, frame_ts=frame_ts)
        elif img is not None:
            dashboard.update_frame(img, yaw, altitude, vx, vy, kp_count, ai_debug=ai_debug, vo_debug=vo, frame_ts=frame_ts)

        # Optional bin states
        bin_states = meta.get('bin_states', {})
        if isinstance(bin_states, dict):
            for k, v in bin_states.items():
                try:
                    bid = int(k)
                    if isinstance(v, dict):
                        num = int(v.get('num_features', 0))
                        ready = bool(v.get('is_ready', False))
                    else:
                        num = int(v)
                        ready = num >= 3
                    dashboard.update_bin(bid, num, ready)
                except Exception:
                    continue

        # Optional route/tracks
        if 'route' in meta:
            dashboard.update_route(meta.get('route'))

        # Optional relocalization event
        if 'relocalized_bin' in meta:
            dashboard.update_relocalization(meta.get('relocalized_bin'), match_score)
        elif matched_tag is not None:
            dashboard.update_relocalization(matched_tag, match_score)

        return jsonify({'status': 'ok'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def draw_bin_overlay(image, yaw_deg, bin_states):
    """Simplified overlay: show marker lock, marker pose, IMU heading, and predicted velocity."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # marker info box (top-right)
    box_w, box_h = 260, 110
    box_x, box_y = w - box_w - 12, 12
    cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 0), 1)

    # We expect marker info to be in global dashboard state via vo_debug; try to access it
    try:
        state = dashboard.get_state()
        vo = state.get('vo_debug', {}) if isinstance(state.get('vo_debug', {}), dict) else {}
    except Exception:
        vo = {}

    detected = int(vo.get('detected_markers', 0))
    locked = bool(vo.get('marker_locked', False))
    pose = vo.get('marker_pose', None)
    imu_heading = vo.get('imu_heading', None)
    marker_bbox = vo.get('marker_bbox', None)
    marker_boxes = vo.get('marker_boxes', []) if isinstance(vo.get('marker_boxes', []), list) else []

    # Marker lock status
    status_text = 'LOCKED' if locked else 'NO LOCK'
    status_color = (0, 200, 0) if locked else (0, 100, 255)
    cv2.putText(image, f"Marker: {status_text}", (box_x + 8, box_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(image, f"Detected: {detected}", (box_x + 8, box_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw marker bounding box when available.
    if isinstance(marker_bbox, (list, tuple)) and len(marker_bbox) >= 4:
        try:
            x, y, bw, bh = int(marker_bbox[0]), int(marker_bbox[1]), int(marker_bbox[2]), int(marker_bbox[3])
            if bw > 4 and bh > 4:
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(w - 1, x + bw)
                y1 = min(h - 1, y + bh)
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 220, 0) if locked else (0, 170, 255), 2)
        except Exception:
            pass

    # Draw multiple marker boxes when provided.
    for mb in marker_boxes:
        if not isinstance(mb, dict):
            continue
        bb = mb.get('bbox', None)
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            continue
        try:
            x, y, bw, bh = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            if bw <= 4 or bh <= 4:
                continue
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(w - 1, x + bw)
            y1 = min(h - 1, y + bh)
            label = str(mb.get('label', '?'))
            score = int(mb.get('score', 0))
            color = (0, 220, 0) if (locked and label == str(vo.get('marker_label', ''))) else (0, 200, 255)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            cv2.putText(image, f"{label}:{score}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        except Exception:
            continue

    if pose:
        try:
            mx = float(pose.get('x', 0.0))
            my = float(pose.get('y', 0.0))
            mz = float(pose.get('z', 0.0))
            myaw = float(pose.get('yaw_deg', 0.0))
            cv2.putText(image, f"PX:{mx:.2f}m PY:{my:.2f}m PZ:{mz:.2f}m", (box_x + 8, box_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.putText(image, f"Yaw:{myaw:.1f}°", (box_x + 8, box_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        except Exception:
            pass
    else:
        cv2.putText(image, "Pose: —", (box_x + 8, box_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # Draw IMU heading near bottom-left
    try:
        imu_h = float(imu_heading) if imu_heading is not None else yaw_deg
    except Exception:
        imu_h = yaw_deg
    imu_txt = f"IMU Yaw: {imu_h:.1f}°"
    cv2.putText(image, imu_txt, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    # Draw predicted velocity vector from center (scale pixels per m/s)
    try:
        state2 = dashboard.get_state()
        vx = float(state2.get('velocity_x', 0.0))
        vy = float(state2.get('velocity_y', 0.0))
    except Exception:
        vx = vy = 0.0

    scale = max(20.0, min(w, h) * 0.12)  # scale factor for visualization (px per m/s approx)
    end_x = int(center[0] + vx * scale)
    end_y = int(center[1] - vy * scale)
    cv2.arrowedLine(image, center, (end_x, end_y), (255, 0, 0), 3, tipLength=0.2)
    cv2.circle(image, center, 4, (255, 0, 0), -1)
    cv2.putText(image, f"V: ({vx:+.2f},{vy:+.2f}) m/s", (center[0] - 80, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    return image

def generate_video_frames(jpeg_quality=62, scale=1.0, max_fps=15.0):
    """Generate low-latency stream from latest uploaded JPEG bytes."""
    frame_dt = 0.0 if float(max_fps) <= 0.0 else (1.0 / max(1.0, float(max_fps)))
    last_emit = 0.0
    last_frame_id = -1
    while True:
        try:
            frame_id, frame_bytes, _, _ = dashboard.wait_for_new_jpeg(last_frame_id, timeout_s=0.25)
            if frame_bytes is None:
                time.sleep(0.01)
                continue

            if frame_id == last_frame_id:
                continue
            last_frame_id = frame_id

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                   + frame_bytes + b'\r\n')

            if frame_dt > 0.0:
                now = time.time()
                elapsed = now - last_emit
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
                last_emit = time.time()
        except Exception as e:
            print(f"Error in generate_video_frames: {e}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Stream video with overlay."""
    try:
        q = float(request.args.get('quality', 62))
    except Exception:
        q = 62
    try:
        s = float(request.args.get('scale', 1.0))
    except Exception:
        s = 1.0
    try:
        fps = float(request.args.get('fps', 0))
    except Exception:
        fps = 0

    resp = Response(
        generate_video_frames(jpeg_quality=q, scale=s, max_fps=fps),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp

if __name__ == '__main__':
    print("🌐 Web Dashboard starting...")
    print("   Open browser: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
