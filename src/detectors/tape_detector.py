"""Tape label detection module for wiring diagrams."""

import re
import cv2
import numpy as np

TAPE_PATTERNS = re.compile(
    r'\b(VT-WH|VT-BK|VT-PK|AT-BK|MLC\w+)\b', re.IGNORECASE)

TAPE_COLOR_BGR = {
    'VT-WH':  (  0,   0, 180),   # dark red
    'VT-BK':  (  0,   0, 180),   # dark red
    'VT-PK':  (  0,   0, 180),   # dark red
    'AT-BK':  (  0,   0, 180),   # dark red
    'COT-BK': (  0,   0, 180),   # dark red
    'DEFAULT':(  0,   0, 180),   # dark red
}


def detect_tape_labels(img, gray, ocr_data):
    """
    Detect tape labels in the wiring diagram.
    
    Strategy:
      1. Find all tape labels first (VT-PK, VT-WH, AT-BK, MLC).
      2. Search region around each label for rectangular tape box.
      3. Ensures 1 label = 1 tape (no duplicates).
    
    Args:
        img: Color image
        gray: Grayscale image
        ocr_data: List of OCR results from ocr_full()
    
    Returns:
        List of detected tapes with labels and bounding boxes
    """
    found = []

    # — Step 1: Find all potential tape labels in OCR data —
    label_list = []
    for item in ocr_data:
        # Handle both old format (txt, x, y, w, h) and new format (txt, x, y, w, h, angle, conf)
        if len(item) >= 5:
            txt, tx, ty, tw, th = item[0], item[1], item[2], item[3], item[4]
        else:
            continue
        m = TAPE_PATTERNS.search(txt.upper())
        if m:
            label_list.append({
                'text': m.group(0).upper(),
                'x': tx, 'y': ty, 'w': tw, 'h': th,
                'cx': tx + tw//2, 'cy': ty + th//2
            })

    if not label_list:
        return found
    
    # — Step 2: Detect small tape boxes using morphology —
    # Invert gray image and threshold to find dark areas (black rectangles)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # SMALL rectangles only: actual tape boxes are tiny (8-30px wide)
        if w < 8 or h < 2 or w > 30 or h > 15:
            continue
        
        # Aspect ratio: width should be 3-15x height
        aspect_ratio = w / max(h, 1)
        if not (3 <= aspect_ratio <= 15):
            continue
        
        rectangles.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': x + w//2, 'cy': y + h//2,
            'area': w * h
        })
    
    # — Step 3: Match each label to closest rectangle —
    assigned_rects = set()
    
    for label in label_list:
        best_dist = float('inf')
        best_rect_idx = -1
        
        for rect_idx, rect in enumerate(rectangles):
            if rect_idx in assigned_rects:
                continue
            
            dist = abs(label['cx'] - rect['cx']) + abs(label['cy'] - rect['cy'])
            
            # Overlap check
            label_left, label_right = label['x'], label['x'] + label['w']
            label_top, label_bottom = label['y'], label['y'] + label['h']
            rect_left, rect_right = rect['x'], rect['x'] + rect['w']
            rect_top, rect_bottom = rect['y'], rect['y'] + rect['h']
            overlaps = not (rect_right < label_left or rect_left > label_right or 
                           rect_bottom < label_top or rect_top > label_bottom)
            
            if overlaps:
                continue
            
            # Pick the closest non-overlapping rectangle
            if dist < 300 and dist < best_dist:
                best_dist = dist
                best_rect_idx = rect_idx
        
        # Assign the best matching rectangle to this label
        if best_rect_idx >= 0:
            rect = rectangles[best_rect_idx]
            assigned_rects.add(best_rect_idx)
            found.append({
                'label': label['text'],
                'bbox': (rect['x'], rect['y'], rect['w'], rect['h']),
                'source': 'edge_rectangle'
            })
    
    return found
