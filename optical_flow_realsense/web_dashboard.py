#!/usr/bin/env python3
"""
Web Dashboard for Phase 2: Real-time visualization of compass bins, video stream, and drone state.

Provides:
  - 3D compass bin visualization (which bins have features)
  - Live video stream with bin overlay
  - Yaw heading compass indicator
  - Real-time metrics (FPS, feature count, altitude, velocity)
"""

from flask import Flask, render_template, Response, jsonify
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
        self.current_image = None
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
    
    def update_frame(self, image, yaw, altitude, vx, vy, kp_count, ai_debug=None, vo_debug=None):
        with self.lock:
            self.current_image = image
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
                'timestamp': datetime.now().isoformat()
            }
    
    def get_image(self):
        with self.lock:
            return self.current_image

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

def draw_bin_overlay(image, yaw_deg, bin_states):
    """Draw compass bin overlay on video frame."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = 80
    
    # Draw compass circle
    cv2.circle(image, center, radius, (100, 100, 100), 2)
    
    # Draw bins
    num_bins = len(bin_states)
    for bin_id, state in bin_states.items():
        bin_width = 360.0 / num_bins
        min_h = state['heading_range'][0]
        max_h = state['heading_range'][1]
        mid_h = (min_h + max_h) / 2.0
        
        # Color based on ready status
        if state['is_ready']:
            color = (0, 255, 0)  # Green = ready
        else:
            color = (100, 100, 100)  # Gray = not ready
        
        # Draw bin wedge (simplified with lines)
        angle_rad = np.radians(mid_h)
        x = int(center[0] + radius * np.sin(angle_rad))
        y = int(center[1] - radius * np.cos(angle_rad))
        
        cv2.line(image, center, (x, y), color, 2)
        
        # Draw bin label
        label_r = radius + 20
        label_x = int(center[0] + label_r * np.sin(angle_rad))
        label_y = int(center[1] - label_r * np.cos(angle_rad))
        cv2.putText(image, f"B{bin_id}", (label_x - 10, label_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw yaw indicator (red line pointing current heading)
    yaw_rad = np.radians(yaw_deg)
    yaw_x = int(center[0] + radius * np.sin(yaw_rad))
    yaw_y = int(center[1] - radius * np.cos(yaw_rad))
    cv2.line(image, center, (yaw_x, yaw_y), (0, 0, 255), 3)  # Red
    cv2.circle(image, center, 5, (0, 0, 255), -1)  # Red circle at center
    
    # Draw cardinal directions
    for direction, angle in [('N', 0), ('E', 90), ('S', 180), ('W', 270)]:
        angle_rad = np.radians(angle)
        x = int(center[0] + (radius + 30) * np.sin(angle_rad))
        y = int(center[1] - (radius + 30) * np.cos(angle_rad))
        cv2.putText(image, direction, (x - 10, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

def generate_video_frames():
    """Generate video stream with overlay."""
    while True:
        try:
            # Get image and state with minimal lock contention
            image = dashboard.get_image()
            if image is None:
                time.sleep(0.01)
                continue

            # Get state snapshot (locks briefly)
            state = dashboard.get_state()

            # Make a copy for overlay (outside lock)
            frame = image.copy()
            if frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # All heavy work is done WITHOUT holding the lock
            # Draw bin overlay
            frame = draw_bin_overlay(frame, state['yaw'], state['bin_states'])

            # Draw metrics text
            h, w = frame.shape[:2]
            text_y = 30

            cv2.putText(frame, f"YAW: {state['yaw']:.1f}°", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 30

            cv2.putText(frame, f"ALT: {state['altitude']:.2f}m", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 30

            cv2.putText(frame, f"VEL: ({state['velocity_x']:+.2f}, {state['velocity_y']:+.2f}) m/s", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            text_y += 30

            cv2.putText(frame, f"FPS: {state['fps']:.1f} | KP: {state['keypoint_count']}", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if state['relocalized_bin'] is not None:
                cv2.putText(frame, f"RELOCALIZED BIN {state['relocalized_bin']} | Matches: {state['match_score']}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Encode frame (no lock)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.01)
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                   + frame_bytes + b'\r\n')

            time.sleep(0.01)
        except Exception as e:
            print(f"Error in generate_video_frames: {e}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Stream video with overlay."""
    return Response(generate_video_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 Web Dashboard starting...")
    print("   Open browser: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
