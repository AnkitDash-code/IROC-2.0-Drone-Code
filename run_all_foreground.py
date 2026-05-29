#!/usr/bin/env python3
"""
Foreground launcher that runs the dashboard, CS20 bridge, and ArUco streamer
in subprocesses and forwards their output. Ctrl+C will stop all processes cleanly.

Usage: python3 run_all_foreground.py
"""

import os
import signal
import subprocess
import sys
import threading
import time
import argparse

ROOT = os.path.abspath(os.path.dirname(__file__))


def build_commands(mode="full"):
    # mode: 'full' -> dashboard + bridge + aruco(mjpeg)
    # mode: 'sdk'  -> dashboard + aruco(sdk)
    # resolve dashboard script location; prefer optical_flow_realsense/web_dashboard.py
    dashboard_path = os.path.join(ROOT, "optical_flow_realsense", "web_dashboard.py")
    if not os.path.exists(dashboard_path):
        # fallback to top-level web_dashboard.py if present
        dashboard_path = os.path.join(ROOT, "web_dashboard.py")

    # if file exists, call directly, otherwise run via flask module
    if os.path.exists(dashboard_path):
        dashboard_cmd = [sys.executable, dashboard_path]
    else:
        dashboard_cmd = [sys.executable, "-m", "flask", "--app", "optical_flow_realsense.web_dashboard", "run", "--port", "5000"]

    # Use a stable system Python interpreter for SDK-bound processes to avoid
    # ABI mismatches (rclpy / Synexens SDK). Allow override via SYSTEM_PYTHON.
    system_python = os.environ.get("SYSTEM_PYTHON", "/usr/bin/python3.10")

    if mode == "sdk":
        return [
            ("dashboard", dashboard_cmd),
            ("aruco", [system_python, os.path.join(ROOT, "optical_flow_cs20_lidar/optional_marker_landing/aruco_streamer.py"), "--dashboard", "http://localhost:5000/api/upload_frame", "--source", "sdk"]),
        ]
    else:
        return [
            ("dashboard", dashboard_cmd),
            ("bridge", [os.path.join(ROOT, "run_cs20_bridge.sh")]),
            ("aruco", [system_python, os.path.join(ROOT, "optical_flow_cs20_lidar/optional_marker_landing/aruco_streamer.py"), "--dashboard", "http://localhost:5000/api/upload_frame", "--source", "mjpeg", "--mjpeg-url", "http://localhost:5000/video_feed"]),
        ]


def forward_output(proc, name):
    try:
        for line in iter(proc.stdout.readline, b""):
            sys.stdout.buffer.write(f"[{name}] ".encode('utf-8') + line)
            sys.stdout.flush()
    except Exception:
        pass


def start_proc(name, args):
    # Start process with stdout/stderr piped
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
    t = threading.Thread(target=forward_output, args=(p, name), daemon=True)
    t.start()
    return p


def wait_for_bridge_ready(timeout=8.0, log_match="CS20 bridge started"):
    # Wait for the bridge process to print a ready message
    start = time.time()
    while time.time() - start < timeout:
        # check running procs stdout is being forwarded; best-effort sleep
        time.sleep(0.1)
        # no reliable IPC here; assume bridge will be ready soon
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "sdk"), default="full", help="Launch mode: 'full' uses bridge + mjpeg streamer, 'sdk' uses SDK streamer and no bridge")
    args = parser.parse_args()

    COMMANDS = build_commands(mode=args.mode)

    procs = {}

    try:
        # Start dashboard first
        name, cmd = COMMANDS[0]
        print(f"Starting {name}: {' '.join(cmd)}")
        procs[name] = start_proc(name, cmd)

        # If bridge present, start it and wait briefly
        if len(COMMANDS) > 2 and COMMANDS[1][0] == 'bridge':
            name, cmd = COMMANDS[1]
            print(f"Starting {name}: {' '.join(cmd)}")
            procs[name] = start_proc(name, cmd)
            wait_for_bridge_ready()

        # Start remaining commands (aruco etc)
        for entry in COMMANDS[1 if len(COMMANDS) == 2 else 2:]:
            name, cmd = entry
            # Skip bridge which is already started
            if name == 'bridge' and name in procs:
                continue
            print(f"Starting {name}: {' '.join(cmd)}")
            procs[name] = start_proc(name, cmd)

        # Wait until any process exits or user hits Ctrl+C
        while True:
            for n, p in list(procs.items()):
                ret = p.poll()
                if ret is not None:
                    print(f"Process {n} exited with {ret}")
                    raise SystemExit(1)
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — shutting down child processes...")
    finally:
        # Terminate children
        for n, p in procs.items():
            try:
                print(f"Terminating {n} (pid {p.pid})")
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                try:
                    p.terminate()
                except Exception:
                    pass

        # Give them a moment, then kill if necessary
        time.sleep(1.0)
        for n, p in procs.items():
            if p.poll() is None:
                try:
                    print(f"Killing {n} (pid {p.pid})")
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

    print("All processes stopped.")


if __name__ == '__main__':
    main()
