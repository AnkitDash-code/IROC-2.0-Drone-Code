#!/usr/bin/python3
"""Print Intel RealSense D455 factory intrinsics from firmware calibration."""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    import pyrealsense2 as rs
except Exception as exc:
    print(f"ERROR: pyrealsense2 import failed: {exc}")
    sys.exit(2)


def get_d455_intrinsics(width: int, height: int, fps: int):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))

    profile = None
    try:
        profile = pipe.start(cfg)
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "model": str(intr.model),
            "coeffs": [float(c) for c in intr.coeffs],
        }
    finally:
        if profile is not None:
            try:
                pipe.stop()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Dump D455 color intrinsics")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="d455_intrinsics.json")
    args = parser.parse_args()

    intr = get_d455_intrinsics(args.width, args.height, args.fps)

    print(f"Hardware Focal Length: fx={intr['fx']:.6f}, fy={intr['fy']:.6f}")
    print(f"Principal Point: ppx={intr['ppx']:.6f}, ppy={intr['ppy']:.6f}")
    print(f"Distortion Model: {intr['model']}")
    print(f"Distortion Coeffs: {intr['coeffs']}")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stream": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        },
        "intrinsics": intr,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
