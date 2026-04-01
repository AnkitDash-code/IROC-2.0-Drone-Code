#!/usr/bin/env python3
"""
Simple checkerboard camera calibrator

Collect a set of checkerboard images (use the CS20 IR/mono images or a USB camera),
place them in `calib_images/` under this folder, and run this script to produce
`camera_calib.npz` which is read by the ArUco landing node.

Example:
  python3 calibrate_checkerboard.py --images calib_images --rows 6 --cols 9 --square 0.024
"""

import os
import glob
import argparse
import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="calib_images", help="Directory with calibration images")
    p.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners per column (rows)")
    p.add_argument("--cols", type=int, default=9, help="Checkerboard inner corners per row (cols)")
    p.add_argument("--square", type=float, default=0.024, help="Square size in meters")
    p.add_argument("--out", default="camera_calib.npz", help="Output npz calibration file")
    return p.parse_args()


def main():
    args = parse_args()
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        imgs.extend(glob.glob(os.path.join(args.images, ext)))
    imgs = sorted(imgs)
    if len(imgs) == 0:
        print("No images found in", args.images)
        return

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

    objpoints = []
    imgpoints = []
    h, w = None, None

    for fn in imgs:
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if w is None:
            h, w = img.shape
        found, corners = cv2.findChessboardCorners(img, (args.cols, args.rows), None)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), term)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print("Found corners:", fn)
        else:
            print("No corners:", fn)

    if len(objpoints) < 3:
        print("Not enough good images for calibration (need >=3)")
        return

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    print("Calibration RMS:", ret)
    print("Camera matrix:\n", camera_matrix)
    print("Dist coeffs:", dist_coeffs.ravel())

    out = os.path.join(os.path.dirname(__file__), args.out)
    np.savez(out, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("Saved:", out)


if __name__ == "__main__":
    main()
