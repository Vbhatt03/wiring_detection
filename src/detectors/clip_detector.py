"""Blue circular clip detection module for wiring diagrams."""

import cv2
import numpy as np


def detect_blue_clips(img, gray):
    """
    Detect blue circular clips in the wiring diagram.
    
    Blue circles with an X inside appear as small (~12-25 px radius)
    blue-filled or blue-outlined circles.
    
    Strategy: HSV mask for blue, then HoughCircles.
    
    Args:
        img: Color image
        gray: Grayscale image
    
    Returns:
        List of detected clips with centers and radii
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue hue range (OpenCV hue 0-179)
    blue_lo = np.array([95, 80, 80])
    blue_hi = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, blue_lo, blue_hi)
    # Dilate to connect nearby pixels
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    # Find circles in the blue mask
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                                minDist=15, param1=50, param2=15,
                                minRadius=5, maxRadius=25)
    clips = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (cx, cy, r) in circles:
            clips.append({'center': (cx, cy), 'radius': r})

    # Fallback: contour-based if HoughCircles misses tiny circles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > 2000:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
        if circularity > 0.5:
            clips.append({'center': (int(cx), int(cy)), 'radius': int(r)})

    # Deduplicate
    deduped = []
    for clip in clips:
        cx, cy = clip['center']
        dup = any(abs(cx - p['center'][0]) < 20 and abs(cy - p['center'][1]) < 20
                  for p in deduped)
        if not dup:
            deduped.append(clip)

    return deduped
