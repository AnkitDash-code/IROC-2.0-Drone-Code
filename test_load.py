import cv2
import numpy as np

img1 = cv2.imread('/home/jetson123/Drone/aruco_tags/copper_1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/jetson123/Drone/aruco_tags/copper_2.png', cv2.IMREAD_GRAYSCALE)
print(f"img1: {img1.shape if img1 is not None else None}")
print(f"img2: {img2.shape if img2 is not None else None}")
