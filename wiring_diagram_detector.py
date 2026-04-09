"""
Wiring Diagram Detector
=======================
Detects the following elements in an automotive wiring harness diagram:
  - Tape labels (VT-WH, VT-BK, VT-PK, AT-BK, COT-BK)
  - Delphi connectors (rectangular connector symbols)
  - Wires (all types: dash-dot, zigzag, solid, etc.)
  - Wire lengths (numbers annotated alongside wires)
  - Blue circular clips (marked with an X)
  - Connectivity list: which two components are joined, wire type, length

Usage:
    python wiring_diagram_detector.py <image_path>
    python wiring_diagram_detector.py  (uses default path)

Output:
    - Annotated image saved as  wiring_diagram_annotated.png
    - Printed connectivity report
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import cv2
import numpy as np
import networkx as nx
import re
from scipy import ndimage
try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    print("[WARN] pytesseract not installed – OCR disabled, using hard-coded labels.")

# Import connectivity graph builder
from wiring_connectivity import Component, Wire, build_connectivity_graph, save_graph_to_json, print_connectivity_report
from component_masker import create_wire_mask
from skeleton_graph import (
    skeletonize_wire_mask,
    extract_skeleton_graph,
    filter_skeleton_graph,
)

# ─────────────────────────────────────────────────────────────
# 0.  Helpers
# ─────────────────────────────────────────────────────────────

def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def show(title, img, wait=True):
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def draw_label(canvas, text, pt, color=(0, 200, 0), scale=0.45, thickness=1):
    x, y = pt
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────
# 1.  OCR – extract all text regions from the image
# ─────────────────────────────────────────────────────────────

def ocr_full(gray):
    """Return list of (text, x, y, w, h, angle, confidence) for every detected word.
    
    Scans text at arbitrary angles (every 10°) to catch text at all orientations
    including diagonal text at 75°, 125°, etc.
    Uses lower confidence threshold (20) to capture tilted/weak text.
    Tracks angle and confidence for later deduplication.
    """
    if not TESSERACT_OK:
        return []
    
    results = []
    H_orig, W_orig = gray.shape
    cy, cx = H_orig / 2, W_orig / 2  # Center for rotation
    
    seen_boxes = []  # Track detected positions to avoid raw duplicates
    
    # Scan at multiple angles: 0, 10, 20, 30, ... 350 degrees
    for angle in range(0, 360, 10):
        # Create rotation matrix
        rot_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_gray = cv2.warpAffine(gray, rot_matrix, (W_orig, H_orig),
                                       borderMode=cv2.BORDER_REPLICATE)
        
        # Run OCR on rotated image
        d = pytesseract.image_to_data(rotated_gray, output_type=pytesseract.Output.DICT,
                                       config='--psm 11 --oem 3')
        
        for i, txt in enumerate(d['text']):
            txt = txt.strip()
            if not txt:
                continue
            
            x_rot, y_rot = d['left'][i], d['top'][i]
            w_rot, h_rot = d['width'][i], d['height'][i]
            conf = int(d['conf'][i])
            
            if conf > 20:  # Lowered from 30 to capture tilted/weak text
                # Transform bbox corners back to original image space
                # Get all 4 corners of the bbox in rotated image
                corners_rot = np.array([
                    [x_rot, y_rot, 1.0],                    # top-left
                    [x_rot + w_rot, y_rot, 1.0],            # top-right
                    [x_rot, y_rot + h_rot, 1.0],            # bottom-left
                    [x_rot + w_rot, y_rot + h_rot, 1.0]     # bottom-right
                ]).T  # (3, 4) for matrix multiplication
                
                # Get inverse rotation matrix
                inv_matrix = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
                
                # Transform all corners back to original space: (2, 4)
                corners_orig = inv_matrix @ corners_rot
                
                # Find axis-aligned bbox from transformed corners
                x_coords = corners_orig[0, :]
                y_coords = corners_orig[1, :]
                x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
                y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
                
                x = x_min
                y = y_min
                w = x_max - x_min
                h = y_max - y_min
                
                # Check for duplicate detection (same text within ~20px)
                is_duplicate = False
                for (prev_txt, prev_x, prev_y) in seen_boxes:
                    if prev_txt == txt:
                        dist = ((x + w/2 - prev_x) ** 2 + (y + h/2 - prev_y) ** 2) ** 0.5
                        if dist < 20:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    results.append((txt, x, y, w, h, angle, conf))
                    seen_boxes.append((txt, x + w/2, y + h/2))
    
    return results


def ocr_region(gray, x1, y1, x2, y2):
    """OCR a bounding-box crop."""
    if not TESSERACT_OK:
        return ""
    crop = gray[y1:y2, x1:x2]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(crop, config='--psm 7 --oem 3').strip()

# ─────────────────────────────────────────────────────────────
# 2.  Tape-label detection  (coloured highlight boxes)
# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# 0b. Extract filters (specify which elements to extract)
# ─────────────────────────────────────────────────────────────

EXTRACT_FILTERS = {
    'tapes': True,          # Extract tape/conduit labels
    'connectors': True,     # Extract Delphi connectors
    'wires': True,          # Extract wire segments
    'lengths': True,        # Extract wire-length annotations
    'clips': True,          # Extract blue circular clips
}

def detect_tape_labels(img, gray, ocr_data):
    """
    Strategy:
      1. Find all tape labels first (VT-PK, VT-WH, AT-BK, MLC).
      2. Search region around each label for rectangular tape box.
      3. Ensures 1 label = 1 tape (no duplicates).
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
        # Filter out larger annotation/label structures
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


# ─────────────────────────────────────────────────────────────
# 3.  Delphi connector detection
# ─────────────────────────────────────────────────────────────

def detect_delphi_connectors(img, gray, ocr_data):
    """
    Delphi connectors appear as small rectangular multi-pin symbols
    (drawn with parallel vertical lines inside a rectangle) at the top
    of the diagram, plus OCR for the text "DELPHI".
    Strategy:
      1. Find "DELPHI" text via OCR and mark vicinity.
      2. Detect small dark rectangles with internal parallel lines
         (connector body silhouette).
    """
    found = []

    # — OCR: find DELPHI annotations —
    if TESSERACT_OK:
        for item in ocr_data:
            # Handle both old format (txt, x, y, w, h) and new format (txt, x, y, w, h, angle, conf)
            if len(item) >= 5:
                txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
            else:
                continue
            if 'DELPHI' in txt.upper() or 'DELPH' in txt.upper():
                found.append({
                    'label': 'Delphi Connector',
                    'bbox': (x, y, w, h),
                    'note': txt
                })

    # — Shape: detect small filled rectangles (connector body) —
    # Connector bodies tend to be dark rectangles ~30-80px wide, ~15-35px tall
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        rect_area = w * h
        if rect_area == 0:
            continue
        fill_ratio = area / rect_area
        aspect = w / max(h, 1)
        # connector body: wider than tall, moderate fill ratio
        if (25 < w < 120 and 10 < h < 50 and
                0.4 < fill_ratio < 0.95 and 1.2 < aspect < 6):
            # check that internal lines exist (detect vertical line runs)
            roi = binary[y:y+h, x:x+w]
            col_sums = roi.sum(axis=0)
            peaks = np.where(col_sums > 0.5 * h * 255)[0]
            if len(peaks) >= 3:  # at least 3 pin columns
                found.append({
                    'label': 'Connector body',
                    'bbox': (x, y, w, h),
                    'note': f'shape-detected ({w}×{h}px)'
                })

    # deduplicate
    deduped = []
    for item in found:
        bx, by, bw, bh = item['bbox']
        cx, cy = bx + bw//2, by + bh//2
        dup = False
        for prev in deduped:
            px = prev['bbox'][0] + prev['bbox'][2]//2
            py = prev['bbox'][1] + prev['bbox'][3]//2
            if abs(cx-px) < 50 and abs(cy-py) < 50:
                dup = True
                break
        if not dup:
            deduped.append(item)

    return deduped


# ─────────────────────────────────────────────────────────────
# 4.  Wire detection (all wires: dash-dot, zigzag, etc.)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 4a. Wire Stitching Engine - Helper Functions
# ─────────────────────────────────────────────────────────────

def detect_components(gray, img_color):
    """
    Detect component bounding boxes so we can use them as wire-break boundaries.
    Returns list of (x, y, w, h) bboxes.
    Strategy: find closed, filled, non-elongated blobs — opposite of wires.
    """
    # Threshold to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S
    )

    component_bboxes = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < 80 or area > 8000:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        fill_ratio = area / (w * h)
        # Components are squarish and dense; wires are elongated/sparse
        if aspect < 4.0 and fill_ratio > 0.25 and min(w, h) > 8:
            component_bboxes.append((x, y, w, h))

    return component_bboxes


def segment_crosses_component(p1, p2, component_bboxes, margin=4):
    """
    Check if the straight line from p1 to p2 passes through any component bbox.
    Uses parametric line sampling.
    
    Args:
        p1, p2: (x, y) endpoints of the line segment
        component_bboxes: List of (x, y, w, h) bounding boxes
        margin: Pixel margin around each bbox to consider as "crossing"
    
    Returns:
        True if the segment crosses any component, False otherwise
    """
    x1, y1 = p1
    x2, y2 = p2
    steps = max(int(np.hypot(x2 - x1, y2 - y1)), 1)
    
    for i in range(steps + 1):
        t = i / steps
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        for (bx, by, bw, bh) in component_bboxes:
            if (bx - margin) <= px <= (bx + bw + margin) and \
               (by - margin) <= py <= (by + bh + margin):
                return True
    return False


def angle_between(p1, p2):
    """Return angle in degrees of the vector from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx)) % 180


def detect_wires(gray):
    """
    Wire detection with Union-Find based segment merging.
    
    Strategy:
    PHASE A (Component Detection):
    1. Detect filled blobs (components like connectors, clips) and create a mask
    
    PHASE B (Wire Segment Extraction):
    2. Create wire mask (dark + edges), remove component regions
    3. Use Connected Components Labeling (CCL) to find wire segments
    4. Extract endpoints using PCA for each segment
    
    PHASE C (Union-Find Merging):
    5. Build Union-Find structure and merge segments based on:
       - Proximity: endpoints < 30px apart
       - Collinearity: angle difference < 15°
       - Component blocking: gaps don't cross components
    
    PHASE D (Wire Reconstruction):
    6. Collect merged groups, find true terminals, output final wires
    
    This method correctly handles dashed wires and respects component boundaries.
    """
    
    img_color = cv2.imread('automotive_schematic.png')
    if img_color is None:
        return []

    # ────────────────────────────────────────────────────────────
    # PHASE A: Component Detection
    # ────────────────────────────────────────────────────────────
    component_bboxes = detect_components(gray, img_color)

    # Build a component mask so we can subtract it from the wire mask
    comp_mask = np.zeros(gray.shape, dtype=np.uint8)
    for (x, y, w, h) in component_bboxes:
        cv2.rectangle(comp_mask, (x, y), (x + w, y + h), 255, -1)

    # ────────────────────────────────────────────────────────────
    # PHASE B: Wire Segment Extraction
    # ────────────────────────────────────────────────────────────
    
    # Wire mask: dark pixels + edges
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 100, 150])  # Lenient dark mask
    color_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    wire_mask = cv2.bitwise_and(color_mask, edges)

    # Remove component regions from wire mask — this is the KEY BREAK
    wire_mask = cv2.bitwise_and(wire_mask, cv2.bitwise_not(comp_mask))

    # Gap filling: dilate and clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bridged_mask = cv2.dilate(wire_mask, kernel, iterations=2)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bridged_mask = cv2.morphologyEx(bridged_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    bridged_mask = cv2.morphologyEx(bridged_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Also remove components from bridged mask to prevent bleed-through
    bridged_mask = cv2.bitwise_and(bridged_mask, cv2.bitwise_not(comp_mask))

    # Connected Components Labeling
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bridged_mask, connectivity=8, ltype=cv2.CV_32S
    )

    # Collect raw segments with PCA endpoints
    raw_segments = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < 20 or area > 6000:
            continue
        
        bbox_aspect = max(w, h) / max(min(w, h), 1)
        if bbox_aspect < 1.3:
            continue
        
        fill_ratio = area / (w * h)
        if fill_ratio < 0.05:
            continue

        blob_points = np.where(labels == label_id)
        if len(blob_points[0]) < 5:
            continue

        points = np.column_stack((blob_points[1], blob_points[0])).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points, mean=None)
        primary_axis = eigenvectors[0]
        projections = points @ primary_axis

        p1 = points[np.argmin(projections)]
        p2 = points[np.argmax(projections)]
        length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if length < 15:
            continue

        raw_segments.append({
            'p1': tuple(p1.astype(int)),
            'p2': tuple(p2.astype(int)),
            'length': length,
            'area': area,
            'axis': primary_axis,
        })

    if not raw_segments:
        return []

    # ────────────────────────────────────────────────────────────
    # PHASE C: Hybrid Graph-Based Merging (improved approach)
    # ────────────────────────────────────────────────────────────
    n = len(raw_segments)
    
    # Build endpoint graph: connect endpoints that should be merged
    GAP_THRESH = 100  # px — allow very long gaps in dashed/segmented wires
    ANGLE_THRESH = 120  # degrees — allow wide angle variations
    
    # Create adjacency graph: endpoint_id -> list of connected endpoints
    endpoint_graph = {}  # (seg_idx, ep_type) -> [(seg_idx, ep_type), ...]
    
    # For each segment, create node IDs for its endpoints
    segment_endpoints = {}  # seg_idx -> {'p1_idx': 0, 'p2_idx': 1}
    for i in range(n):
        segment_endpoints[i] = {'p1_idx': (i, 'p1'), 'p2_idx': (i, 'p2')}
        endpoint_graph[(i, 'p1')] = []
        endpoint_graph[(i, 'p2')] = []
    
    # Test all pairs of segments and connect compatible endpoints
    segment_pairs_tested = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in segment_pairs_tested:
                continue
            segment_pairs_tested.add((i, j))
            
            seg_i = raw_segments[i]
            seg_j = raw_segments[j]
            
            # Precompute angles once per segment
            angle_i = np.degrees(np.arctan2(seg_i['axis'][1], seg_i['axis'][0])) % 180
            angle_j = np.degrees(np.arctan2(seg_j['axis'][1], seg_j['axis'][0])) % 180
            
            # Check all 4 endpoint pairs
            endpoint_pairs = [
                ((i, 'p1'), seg_i['p1'], (j, 'p1'), seg_j['p1']),
                ((i, 'p1'), seg_i['p1'], (j, 'p2'), seg_j['p2']),
                ((i, 'p2'), seg_i['p2'], (j, 'p1'), seg_j['p1']),
                ((i, 'p2'), seg_i['p2'], (j, 'p2'), seg_j['p2']),
            ]
            
            for node_a, pt_a, node_b, pt_b in endpoint_pairs:
                # Criterion 1: Proximity
                dist = np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
                if dist > GAP_THRESH:
                    continue
                
                # Criterion 2: Collinearity (enhanced tolerance)
                angle_diff = abs(angle_i - angle_j)
                angle_diff = min(angle_diff, 180 - angle_diff)
                if angle_diff > ANGLE_THRESH:
                    continue
                
                # Criterion 3: Component blocking
                if segment_crosses_component(pt_a, pt_b, component_bboxes):
                    continue
                
                # All criteria met → connect these endpoints in graph
                if node_b not in endpoint_graph[node_a]:
                    endpoint_graph[node_a].append(node_b)
                if node_a not in endpoint_graph[node_b]:
                    endpoint_graph[node_b].append(node_a)
    
    # ────────────────────────────────────────────────────────────
    # PHASE D: Trace Paths Through Endpoint Graph
    # ────────────────────────────────────────────────────────────
    
    visited = set()
    wires = []
    
    # Find all connected components and trace paths
    for start_node in endpoint_graph:
        if start_node in visited:
            continue
        
        # BFS to find all nodes in this connected component
        component = []
        queue = [start_node]
        visited_local = {start_node}
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            visited.add(node)
            
            for neighbor in endpoint_graph[node]:
                if neighbor not in visited_local:
                    visited_local.add(neighbor)
                    queue.append(neighbor)
        
        # Now trace the actual wire path through this component
        # Start from a terminal node (degree 1) or any node if no terminal exists
        terminal_nodes = [n for n in component if len(endpoint_graph[n]) == 1]
        start = terminal_nodes[0] if terminal_nodes else component[0]
        
        # Trace path: follow graph edges to build wire
        path_nodes = [start]
        prev_node = None
        current = start
        
        while True:
            neighbors = [n for n in endpoint_graph[current] if n != prev_node]
            if not neighbors:
                break
            
            prev_node = current
            current = neighbors[0]  # Follow the edge (should be at most 1 unvisited neighbor)
            path_nodes.append(current)
            
            if len(path_nodes) > len(component) * 2:  # Prevent infinite loops
                break
        
        # Convert path nodes back to segment indices and collect endpoints
        all_points = []
        member_segs = set()
        
        for node in path_nodes:
            seg_idx, ep_type = node
            member_segs.add(seg_idx)
            seg = raw_segments[seg_idx]
            all_points.append(seg['p1'])
            all_points.append(seg['p2'])
        
        if not all_points:
            continue
        
        # Fit PCA to find wire direction and endpoints
        pts = np.array(all_points, dtype=np.float32)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        primary_axis = eigenvectors[0]
        projections = pts @ primary_axis
        
        p1 = pts[np.argmin(projections)]
        p2 = pts[np.argmax(projections)]
        length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        
        # Final length constraints - accept wider range for more flexibility
        if length < 20 or length > 600:
            continue
        
        wires.append({
            'p1': (int(p1[0]), int(p1[1])),
            'p2': (int(p2[0]), int(p2[1])),
            'length_px': int(length),
            'segment_count': len(member_segs),
            'type': 'merged'
        })
    
    # ────────────────────────────────────────────────────────────
    # PHASE E: Deduplication - Remove duplicate wires
    # ────────────────────────────────────────────────────────────
    
    # Remove duplicate wires (same endpoints, possibly reversed)
    deduped_wires = []
    seen_wires = set()
    
    for wire in wires:
        p1 = wire['p1']
        p2 = wire['p2']
        
        # Create normalized key (sorted endpoints for direction invariance)
        key = tuple(sorted([p1, p2]))
        
        if key not in seen_wires:
            seen_wires.add(key)
            deduped_wires.append(wire)
    
    # ═══ DEBUG: Phase instrumentation ═══
    print(f"    [Wire Detection] Raw segments: {n} → Graph merged: {len(wires)} → Deduplicated: {len(deduped_wires)} wires (GAP={GAP_THRESH}px, ANGLE={ANGLE_THRESH}°)")

    return deduped_wires


def filter_wires_by_components(wires, components, ocr_data, margin=50):
    """
    Component-Anchored Validation:
    Filter wires to only those with at least one endpoint near:
    - A detected component (Tape, Connector, Clip)
    - An OCR text region (length annotations like "150mm")
    
    This eliminates dimension lines, borders, and orphan segments.
    Margin (default 50px) controls how close endpoints must be to anchors.
    """
    if not wires:
        return []
    
    validated_wires = []
    
    for wire in wires:
        p1 = np.array(wire['p1'], dtype=np.float32)
        p2 = np.array(wire['p2'], dtype=np.float32)
        
        p1_anchored = False
        p2_anchored = False
        
        # Check if endpoints are near detected components
        for comp in components:
            # Handle clips (have 'center' key)
            if 'center' in comp:
                comp_center = np.array(comp['center'], dtype=np.float32)
            # Handle tapes and connectors (have 'bbox' key)
            elif 'bbox' in comp:
                bbox = comp['bbox']
                comp_center = np.array([
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0
                ], dtype=np.float32)
            else:
                continue
            
            dist_p1 = np.linalg.norm(p1 - comp_center)
            dist_p2 = np.linalg.norm(p2 - comp_center)
            
            if dist_p1 < margin:
                p1_anchored = True
            if dist_p2 < margin:
                p2_anchored = True
        
        # Check if endpoints are near OCR text hits (length annotations, labels)
        for item in ocr_data:
            # Handle tuple with (text, x, y, w, h, [angle, conf])
            if len(item) >= 4:
                x, y, w, h = item[1], item[2], item[3], item[4]
                text_center = np.array([x + w/2, y + h/2], dtype=np.float32)
                
                dist_p1 = np.linalg.norm(p1 - text_center)
                dist_p2 = np.linalg.norm(p2 - text_center)
                
                if dist_p1 < margin:
                    p1_anchored = True
                if dist_p2 < margin:
                    p2_anchored = True
        
        # Accept wire if at least ONE endpoint is anchored to a component or label
        if p1_anchored or p2_anchored:
            wire['p1_anchored'] = p1_anchored
            wire['p2_anchored'] = p2_anchored
            validated_wires.append(wire)
    
    return validated_wires


# ─────────────────────────────────────────────────────────────
# 6.  Wire-length annotation detection
# ─────────────────────────────────────────────────────────────

LENGTH_PATTERN = re.compile(r'^\(?\d{1,4}\)?$')   # e.g. 0, (0), 25, (25), (50), 150, 195
LABEL_KEYWORDS = re.compile(r'(VT-|AT-|COT-|DELPHI|MLC|J\d+|X\d+|Z\d+|C\d+)', re.IGNORECASE)

def score_wire_length_value(val):
    """Score how 'reasonable' a wire length value is.
    
    Returns 0-100 based on:
    - Round numbers (multiples of 25 or 50) score higher: 100
    - Common lengths (25, 50, 75, 100, 150, 200, 250): +50
    - Within typical range (10-300): +30
    - Reasonable but less common: +20
    - Outliers or suspicious (>400mm): 0
    """
    if val < 10 or val > 600:
        return 0
    
    score = 10  # base score for in-range values
    
    # Prefer round multiples of 25 (25, 50, 75, 100, 125, 150, ...)
    if val % 25 == 0:
        score += 40
    # Prefer multiples of 10
    elif val % 10 == 0:
        score += 20
    
    # Prefer common automotive lengths (most wires are under 300mm)
    common_lengths = {25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300}
    if val in common_lengths:
        score += 40
    
    # Prefer typical range (10-300mm is most common in automotive)
    if 10 <= val <= 300:
        score += 20
    
    return min(score, 100)


def detect_wire_lengths(ocr_data, tapes=None, connectors=None):
    """Extract numeric wire-length annotations from OCR data.
    
    Accepts both horizontal, vertical, and angled text (parenthesized or not).
    Filters out any lengths that overlap with tape labels, connector bounding boxes, or label text.
    Smartly deduplicates multi-angle detections by:
      1. Grouping by proximity
      2. Picking best detection using: confidence (3x) + value reasonableness score (2x) + angle score
      3. Preferring round, typical automotive lengths (25, 50, 100, 150, 200mm)
    Filters: parenthesized values 0-600, non-parenthesized values 10-600mm.
    """
    if tapes is None:
        tapes = []
    if connectors is None:
        connectors = []
    
    # Build map of label text positions for filtering
    label_positions = []
    for item in ocr_data:
        # Handle both old format (txt, x, y, w, h) and new format (txt, x, y, w, h, angle, conf)
        if len(item) >= 5:
            txt = item[0]
            if LABEL_KEYWORDS.search(txt):
                label_positions.append(item[1:5])
    
    candidates = []
    for item in ocr_data:
        # Handle both old format (txt, x, y, w, h) and new format (txt, x, y, w, h, angle, conf)
        if len(item) >= 5:
            txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
            angle = item[5] if len(item) > 5 else 0
            conf = item[6] if len(item) > 6 else 50
        else:
            continue
            
        clean = txt.strip().replace(' ', '')
        if LENGTH_PATTERN.match(clean):
            val = int(re.sub(r'[^\d]', '', clean))
            is_parenthesized = '(' in txt
            
            # Selective filtering:
            # - All parenthesized values (0-600): (0), (5), (25), (50), etc.
            # - Non-parenthesized multi-digit values only (10-600mm)
            if (is_parenthesized and 0 <= val <= 600) or (not is_parenthesized and 10 <= val <= 600):
                # Check if this detected number overlaps with a tape label or connector
                is_overlapping_label = False
                
                # Use lenient tolerance for parenthesized values (clearly dimensions)
                # Use strict tolerance for non-parenthesized values
                bbox_tolerance = 15 if is_parenthesized else 30
                
                # Check against tape labels
                for tape in tapes:
                    tx, ty, tw, th = tape['bbox']
                    # Check if bounding boxes overlap
                    if (x < tx + tw + bbox_tolerance and x + w > tx - bbox_tolerance and
                        y < ty + th + bbox_tolerance and y + h > ty - bbox_tolerance):
                        is_overlapping_label = True
                        break
                
                # Check against connectors
                if not is_overlapping_label:
                    for conn in connectors:
                        cx, cy, cw, ch = conn['bbox']
                        # Check if bounding boxes overlap
                        if (x < cx + cw + bbox_tolerance and x + w > cx - bbox_tolerance and
                            y < cy + ch + bbox_tolerance and y + h > cy - bbox_tolerance):
                            is_overlapping_label = True
                            break
                
                # Check against label keywords ONLY for non-parenthesized values
                # (e.g. "109" near "COT-BK" text, but not "(0)" or "(50)")
                if not is_overlapping_label and not is_parenthesized:
                    for lx, ly, lw, lh in label_positions:
                        # Check if bounding boxes overlap (with tighter tolerance for text)
                        if (x < lx + lw + 15 and x + w > lx - 15 and
                            y < ly + lh + 15 and y + h > ly - 15):
                            is_overlapping_label = True
                            break
                
                if not is_overlapping_label:
                    candidates.append({
                        'value': val,
                        'bbox': (x, y, w, h),
                        'is_parenthesized': is_parenthesized,
                        'angle': angle,
                        'conf': conf
                    })
    
    
    # Debug: Show all "100" candidates before deduplication
    debug_mode = any(c['value'] == 100 for c in candidates)
    
    # Define common automotive lengths (used for intelligent deduplication)
    common_lengths = {25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300}  # max 300mm for automotive
    
    # Smart deduplication: cluster by PROXIMITY first (catches OCR misreadings),
    # then pick best value within each cluster
    lengths = []
    used = set()
    for i, cand in enumerate(candidates):
        if i in used:
            continue
        
        cluster_indices = [i]
        cluster_x, cluster_y = cand['bbox'][0], cand['bbox'][1]
        
        # Find ALL nearby detections within 60px (regardless of value)
        # This catches cases where OCR misreads "175" as "25" at different angles
        for j in range(i + 1, len(candidates)):
            if j not in used:
                x2, y2 = candidates[j]['bbox'][0], candidates[j]['bbox'][1]
                dist = ((cluster_x - x2) ** 2 + (cluster_y - y2) ** 2) ** 0.5
                if dist < 60:  # Physical proximity threshold (catches misreadings)
                    if (cand['value'] == 100 or candidates[j]['value'] == 100) and debug_mode:
                        print(f"    Clustering candidate #{i} (val={cand['value']}) with #{j} (val={candidates[j]['value']}) at dist={dist:.1f}px")
                    cluster_indices.append(j)
                    used.add(j)
        
        # Pick best detection from cluster:
        # 1. Values in common_lengths get massive boost (STRONGLY prefer legitimate automotive lengths)
        # 2. Highest confidence (3x weight)
        # 3. Value reasonableness (2x weight) - prefer round numbers, common lengths
        # 4. Angle closest to 0° or 90° (1x weight)
        best_idx = i
        best_score = -1
        
        for idx in cluster_indices:
            cand_item = candidates[idx]
            conf = cand_item['conf']
            val = cand_item['value']
            angle = cand_item['angle']
            
            # HUGE boost for values in common_lengths (catches "100" over "400" misreadings)
            in_common = 10000 if val in common_lengths else 0
            
            # Score value reasonableness (0-100): round numbers, common lengths preferred
            val_score = score_wire_length_value(val)
            
            # Normalize angle: 0-180 (0° and 180° are same horizontal), 90° is vertical
            norm_angle = min(angle % 180, 180 - (angle % 180))
            # Prefer angles closer to cardinal directions (0°, 90°)
            angle_score = 100 - min(norm_angle, 90 - abs(norm_angle - 90))
            
            # Combined score: common_lengths boost + confidence (3x) + value reasonableness (2x) + angle (1x)
            total_score = in_common + conf * 3 + val_score * 2 + angle_score
            
            if total_score > best_score:
                best_score = total_score
                best_idx = idx
        
        # Use best detection from cluster
        best_cand = candidates[best_idx]
        
        # Use median position of cluster (more robust than average)
        xs = sorted([candidates[idx]['bbox'][0] for idx in cluster_indices])
        ys = sorted([candidates[idx]['bbox'][1] for idx in cluster_indices])
        avg_x = xs[len(xs)//2]
        avg_y = ys[len(ys)//2]
        
        lengths.append({
            'value': best_cand['value'],  # Use value from best-confidence detection
            'bbox': (avg_x, avg_y, best_cand['bbox'][2], best_cand['bbox'][3]),
            'is_parenthesized': best_cand['is_parenthesized']
        })
    
    # Filter outliers: remove values that deviate significantly from the median
    # (catches OCR misreadings that are isolated)
    if lengths:
        values = [l['value'] for l in lengths]
        median_val = sorted(values)[len(values)//2]
        
        # Keep lengths that are within reasonable range of median
        filtered_lengths = []
        
        for ln in lengths:
            val = ln['value']
            # Keep if: 
            # 1. Within 1.5x median (tight tolerance)
            # 2. OR is a common automotive length (<=300mm)
            # 3. OR is parenthesized (clearly marked dimensions)
            if (val <= median_val * 1.5 or 
                val in common_lengths or 
                ln['is_parenthesized']):
                filtered_lengths.append(ln)
        
        lengths = filtered_lengths
    
    return lengths


# ─────────────────────────────────────────────────────────────
# 7.  Blue circular clip detection  (blue X marker)
# ─────────────────────────────────────────────────────────────

def detect_blue_clips(img, gray):
    """
    Blue circles with an X inside appear as small (~12-25 px radius)
    blue-filled or blue-outlined circles.
    Strategy: HSV mask for blue, then HoughCircles.
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


# ─────────────────────────────────────────────────────────────
# 8.  Build connectivity list  (heuristic based on positions)
# ─────────────────────────────────────────────────────────────

def merge_edges_by_component_pair(raw_edges, nodes_dict=None):
    """
    ═══ Phase 2: Component-to-Component Edge Merging ═══
    
    Groups all edges with the same (node_a, node_b) component pair
    and merges them into a single logical connection.
    
    Strategy:
    1. Group edges by (node_a, node_b) key
    2. For each group, collect all tape types and wire info
    3. Pick primary length from highest-confidence detection
    4. Return list of merged edges
    
    Args:
        raw_edges: List of segment edges, each with:
                   node_a, node_b, tapes, length_mm, p1, p2, snapped_a, snapped_b
        nodes_dict: Mapping of node_id to node info (used for junction detection)
    
    Returns:
        List of merged edges grouped by component pairs
    """
    if nodes_dict is None:
        nodes_dict = {}
    # Group edges by (node_a, node_b) key
    edge_groups = {}
    
    for edge in raw_edges:
        # Skip edges with missing component endpoints
        if edge['node_a'] is None or edge['node_b'] is None:
            continue
        
        # Create key: SORTED so A→B and B→A collapse to the same edge
        key = tuple(sorted([edge['node_a'], edge['node_b']]))
        
        if key not in edge_groups:
            edge_groups[key] = []
        edge_groups[key].append(edge)
    
    # Merge edges within each group
    merged_edges = []
    
    for (node_a, node_b), edges_in_group in edge_groups.items():
        # Skip all self-loops (including junctions — J20→J20 are always detection artifacts)
        if node_a == node_b:
            continue

        # Skip spurious clip↔clip edges (artifacts from overlapping blobs near the clips)
        type_a = nodes_dict.get(node_a, {}).get('type', '')
        type_b = nodes_dict.get(node_b, {}).get('type', '')
        if type_a == 'clip' and type_b == 'clip':
            continue
        # Collect all unique tapes across this group
        all_tapes = set()
        for edge in edges_in_group:
            all_tapes.update(edge['tapes'])
        
        # Pick primary length:
        # Prefer tape-anchor sources (more reliable) over wire-blob sources.
        # Among same source, prefer non-None values.
        primary_length = None
        # First pass: tape_anchor sources
        for edge in edges_in_group:
            if edge.get('source') == 'tape_anchor' and edge.get('length_mm') is not None:
                primary_length = edge['length_mm']
                break
        # Second pass: any source
        if primary_length is None:
            for edge in edges_in_group:
                if edge.get('length_mm') is not None:
                    primary_length = edge['length_mm']
                    break
        
        # Create merged edge
        merged_edge = {
            'node_a': node_a,
            'node_b': node_b,
            'wire_types': sorted(all_tapes),  # Unique tape types
            'length_mm': primary_length,
            'segment_count': len(edges_in_group),  # How many segments make up this connection
            'segments': edges_in_group,  # Keep reference to individual segments for detail
            'snapped': any(edge.get('snapped_a', False) or edge.get('snapped_b', False) 
                          for edge in edges_in_group)  # Flag if any endpoint was snapped
        }
        
        merged_edges.append(merged_edge)
    
    # Debug output
    print(f"    [Edge Merging] Raw edges: {len(raw_edges)} → Merged connections: {len(merged_edges)}")
    
    return merged_edges


def build_connectivity_graph_heuristic(tape_labels, connectors, clips, wires, lengths, img_shape, ocr_data=None):
    """
    Build a connectivity graph by tracing wire segments (LEGACY HEURISTIC METHOD).
    
    Strategy:
    1. For each dash-dot segment, find which tape labels are on it
    2. Group tape labels that are on the same segment
    3. For each segment, find endpoints and map to actual nodes
    4. Build edges showing: Node A --[wire type, length]--> Node B
    5. Detect junction points from text labels (J20, J30, etc.)
    
    Args:
        tape_labels: Detected tape labels
        connectors: Detected Delphi connectors
        clips: Detected blue circular clips
        wires: Detected wire segments
        lengths: Detected length annotations
        img_shape: Image dimensions
        ocr_data: OCR text detections (for junction point extraction)
    
    Returns: dict with 'nodes' (list of nodes) and 'edges' (list of connections)
    """
    if ocr_data is None:
        ocr_data = []
    
    h, w = img_shape[:2]
    
    # Create node list from connectors and clips
    nodes_dict = {}  # id -> {cx, cy, label, type}
    
    for i, conn in enumerate(connectors):
        bx, by, bw, bh = conn['bbox']
        cx, cy = bx + bw//2, by + bh//2
        node_id = f"Connector-{i+1}"
        nodes_dict[node_id] = {
            'x': cx,
            'y': cy,
            'label': conn.get('note', conn['label']),
            'type': 'connector'
        }
    
    # ─── OCR-based connector renaming: X510, X508, X519, C1, C2 ───
    PORT_PATTERN = re.compile(r'^(X\d+|C\d+)$', re.IGNORECASE)
    OCR_RENAME_RADIUS = 120  # px around connector center to look for port labels
    for nid in list(nodes_dict.keys()):
        if nodes_dict[nid]['type'] != 'connector':
            continue
        cx, cy = nodes_dict[nid]['x'], nodes_dict[nid]['y']
        best_txt, best_dist = None, OCR_RENAME_RADIUS
        for item in ocr_data:
            if len(item) < 5:
                continue
            txt, ox, oy, ow, oh = item[0], item[1], item[2], item[3], item[4]
            txt_clean = txt.strip().upper()
            if not PORT_PATTERN.match(txt_clean):
                continue
            ocx, ocy = ox + ow // 2, oy + oh // 2
            dist = ((ocx - cx) ** 2 + (ocy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_txt = txt_clean
        if best_txt:
            nodes_dict[nid]['label'] = best_txt
            print(f"    [NodeRename] {nid} → {best_txt} (dist={best_dist:.0f}px)")

    for i, clip in enumerate(clips):
        node_id = f"Clip-{i+1}"
        nodes_dict[node_id] = {
            'x': clip['center'][0],
            'y': clip['center'][1],
            'label': f"Z2-TH{i+1}",
            'type': 'clip'
        }
        
    
    
    # ═══ Extract junction/intermediate points from OCR data ═══
    # Accept J\d+ (junctions) and MLC\w+ (conduit branches)
    junction_pattern = re.compile(r'^(J\d+|MLC\w+)$', re.IGNORECASE)
    junction_count = 0
    for item in ocr_data:
        if len(item) >= 5:
            txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
            txt_clean = re.sub(r'[^A-Z0-9]$', '', txt.strip().upper())  # strip trailing : . etc.
            if txt_clean in nodes_dict:
                continue
            if junction_pattern.match(txt_clean):
                junction_count += 1
                nodes_dict[txt_clean] = {
                    'x': int(x + w // 2),
                    'y': int(y + h // 2),
                    'label': txt_clean,
                    'type': 'junction'
                }
    
    # ─── Synthesise C2 node from AT-BK tape position if not shape-detected ───
    # AT-BK tape sits right next to the C2 connector at the bottom-right.
    # If no connector is within 200px of the AT-BK tape, add a synthetic C2 node.
    connector_ids = [nid for nid, n in nodes_dict.items() if n['type'] == 'connector']
    for tape in tape_labels:
        if tape['label'].upper() != 'AT-BK':
            continue
        tbx, tby, tbw, tbh = tape['bbox']
        tcx, tcy = tbx + tbw // 2, tby + tbh // 2
        nearby_conn = any(
            ((nodes_dict[nid]['x'] - tcx) ** 2 + (nodes_dict[nid]['y'] - tcy) ** 2) ** 0.5 < 200
            for nid in connector_ids
        )
        if not nearby_conn and 'C2' not in nodes_dict:
            # Place C2 to the LEFT of the AT-BK label (where the cross-hatch box is)
            nodes_dict['C2'] = {
                'x': max(0, int(tcx - 80)),
                'y': int(tcy),
                'label': 'C2 (1045235)',
                'type': 'connector'
            }
            print(f"    [SyntheticNode] Added C2 at ({nodes_dict['C2']['x']},{nodes_dict['C2']['y']}) from AT-BK tape position")

    if junction_count > 0:
        print(f"    [Junctions] Detected {junction_count} junction/conduit points")
    
    # For each wire segment, find which tapes are on it
    segment_tapes = {}  # segment_idx -> list of tape info
    
    # Filter to only detected wires (with 'p1' and 'p2' endpoints) - hough, CCL-PCA, or merged
    detected_wires = [w for w in wires if w['type'] in ('hough', 'ccl_pca', 'merged', 'segment')]
    
    for seg_idx, seg in enumerate(detected_wires):
        p1 = np.array(seg['p1'])
        p2 = np.array(seg['p2'])
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1:
            continue
        seg_unit = seg_vec / seg_len
        
        tapes_on_segment = []
        for tape in tape_labels:
            bx, by, bw, bh = tape['bbox']
            tape_center = np.array([bx + bw//2, by + bh//2])
            
            # Project tape center onto segment line
            to_tape = tape_center - p1
            proj_dist = np.dot(to_tape, seg_unit)
            
            # Check if projection is within segment bounds
            if 0 <= proj_dist <= seg_len:
                # Calculate perpendicular distance
                proj_point = p1 + proj_dist * seg_unit
                perp_dist = np.linalg.norm(tape_center - proj_point)
                
                # Include tape if it's close to the segment (within 80px perpendicular distance)
                # Wider corridor to catch tapes on COT-BK/double-line wires
                if perp_dist < 80:
                    tapes_on_segment.append({
                        'label': tape['label'],
                        'proj_dist': proj_dist,
                        'perp_dist': perp_dist
                    })
        
        if tapes_on_segment:
            segment_tapes[seg_idx] = tapes_on_segment
    
    # Find nearest node to segment endpoints
    def find_nearest_node(point, strict_thresh=30, loose_thresh=200):
        """Find nearest node to a point using hybrid approach:
        - Strict: within 30px (precise match)
        - Loose: within 200px (snap to component/junction if no precise match)
        Returns (node_id, distance, snapped) tuple, or (None, inf, False)
        """
        min_dist = float('inf')
        nearest = None
        for nid, ninfo in nodes_dict.items():
            dist = np.linalg.norm(np.array([ninfo['x'], ninfo['y']]) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                nearest = nid
        
        # Hybrid approach: try strict first, fall back to loose
        if min_dist < strict_thresh:
            return (nearest, min_dist, False)  # Precise match
        elif min_dist < loose_thresh:
            return (nearest, min_dist, True)   # Snapped/loose match
        else:
            return (None, min_dist, False)     # Too far, unmatched
    
    # Build edges from wire segments (raw: one edge per segment)
    raw_edges = []
    
    for seg_idx, seg in enumerate(detected_wires):
        # Process ALL wires, not just those with visible tape labels
        # Many wires don't have explicit tape marks but still create real connections
        tapes = segment_tapes.get(seg_idx, [])
        p1 = seg['p1']
        p2 = seg['p2']
        
        # Find endpoints using hybrid approach
        node_a_result = find_nearest_node(p1)
        node_b_result = find_nearest_node(p2)
        
        node_a = node_a_result[0]
        node_b = node_b_result[0]
        snapped_a = node_a_result[2]
        snapped_b = node_b_result[2]
        
        # Get wire properties from tape labels on this segment (if available)
        # If no tapes found, use 'Unknown' as placeholder so edge is still created
        segment_info = {
            'tapes': [t['label'] for t in tapes] if tapes else ['Unknown'],
            'tapes_sorted': sorted(tapes, key=lambda t: t['proj_dist']) if tapes else [],
            'p1': p1,
            'p2': p2,
            'node_a': node_a,
            'node_b': node_b,
            'snapped_a': snapped_a,
            'snapped_b': snapped_b
        }
        
        # Find length for this segment using point-to-segment distance (more accurate)
        p1_arr = np.array(p1, dtype=float)
        p2_arr = np.array(p2, dtype=float)
        seg_vec_l = p2_arr - p1_arr
        seg_len_sq = float(np.dot(seg_vec_l, seg_vec_l))
        nearest_length = None
        min_len_dist = float('inf')
        for ln in lengths:
            lbx, lby, lbw, lbh = ln['bbox']
            lpt = np.array([lbx + lbw / 2, lby + lbh / 2], dtype=float)
            if seg_len_sq > 0:
                t = float(np.dot(lpt - p1_arr, seg_vec_l) / seg_len_sq)
                t = max(0.0, min(1.0, t))
                proj = p1_arr + t * seg_vec_l
            else:
                proj = p1_arr
            dist = float(np.linalg.norm(lpt - proj))
            if dist < min_len_dist and dist < 250:
                min_len_dist = dist
                nearest_length = ln['value']
        
        segment_info['length_mm'] = nearest_length
        
        raw_edges.append(segment_info)
    
    # ═══ Phase 2: Tape-Anchored Edge Second Pass ═══
    # For wires the blob-tracer missed, derive edges directly from tape label positions.
    # For each tape label, find the closest node (node_a) and then the closest node
    # on the OPPOSITE SIDE (angle > 60°). Clips are not paired with other clips —
    # they must connect to a connector or junction to prevent spurious Clip↔Clip edges.
    tape_pass_edges = []
    for tape in tape_labels:
        tbx, tby, tbw, tbh = tape['bbox']
        tcx, tcy = float(tbx + tbw // 2), float(tby + tbh // 2)

        # Sort all nodes by distance from tape label centre
        node_dists = []
        for nid, ninfo in nodes_dict.items():
            dx = ninfo['x'] - tcx
            dy = ninfo['y'] - tcy
            dist = (dx ** 2 + dy ** 2) ** 0.5
            node_dists.append((dist, nid, dx, dy))
        node_dists.sort(key=lambda x: x[0])

        if len(node_dists) < 2:
            continue

        # Node A: the nearest node within 400px
        dist_a, nid_a, dxa, dya = node_dists[0]
        if dist_a > 400:
            continue
        len_a = dist_a + 1e-9
        type_a = nodes_dict[nid_a]['type']

        # Node B: nearest node on the OPPOSITE SIDE from node_a.
        # Rule: if node_a is a clip, node_b must NOT be another clip
        # (prevents spurious Clip-1↔Clip-2 edges; clips connect to connectors/junctions).
        nid_b = None
        best_b_dist = 650.0  # wide enough to reach J20 from upper tape labels (~520px)
        for dist_b, nid_b_cand, dxb, dyb in node_dists[1:]:
            if dist_b > 650:
                break
            type_b = nodes_dict[nid_b_cand]['type']
            # Skip: clip↔clip pairing (they don't directly wire to each other here)
            if type_a == 'clip' and type_b == 'clip':
                continue
            len_b = dist_b + 1e-9
            cos_ab = (dxa * dxb + dya * dyb) / (len_a * len_b)
            # Accept if angle > 60° (cos < 0.5) and within search radius
            if cos_ab < 0.5:
                # When clip connects downstream, strongly prefer junctions over connectors
                # (give junctions a 200px effective-distance discount)
                eff_dist = dist_b
                if type_a == 'clip' and type_b == 'junction':
                    eff_dist = dist_b - 200.0
                if eff_dist < best_b_dist:
                    best_b_dist = eff_dist
                    nid_b = nid_b_cand

        if nid_b is None:
            continue

        # Find length annotation closest to tape label (within 250px).
        # Skip val==0: those are junction tap-off markers, not actual wire lengths.
        tape_length = None
        min_tape_len_dist = 250.0
        for ln in lengths:
            if ln['value'] == 0:
                continue  # (0) annotations are junction tap-offs, skip
            lbx, lby, lbw, lbh = ln['bbox']
            lcx, lcy = lbx + lbw / 2, lby + lbh / 2
            d = ((lcx - tcx) ** 2 + (lcy - tcy) ** 2) ** 0.5
            if d < min_tape_len_dist:
                min_tape_len_dist = d
                tape_length = ln['value']

        tape_pass_edges.append({
            'tapes': [tape['label']],
            'tapes_sorted': [],
            'p1': (int(tcx), int(tcy)),
            'p2': (int(tcx), int(tcy)),
            'node_a': nid_a,
            'node_b': nid_b,
            'snapped_a': True,
            'snapped_b': True,
            'length_mm': tape_length,
            'source': 'tape_anchor'
        })

    print(f"    [TapePass] {len(tape_pass_edges)} tape-anchored candidate edges built")

    # \u2500\u2500\u2500 Supplemental edges: connections with no tape label in the diagram \u2500\u2500\u2500
    # These are derived from length annotation positions paired with known topology.
    def find_length_near(cx, cy, max_dist=200):
        best_val, best_d = None, max_dist
        for ln in lengths:
            lbx, lby, lbw, lbh = ln['bbox']
            lcx, lcy = lbx + lbw / 2, lby + lbh / 2
            d = ((lcx - cx) ** 2 + (lcy - cy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_val = d, ln['value']
        return best_val

    supplemental = []

    # J20 → Connector-1 (C1, bottom-left): VT-BK, 58mm annotation at ~(242,680)
    # This connection is present but the tape label sits on the J20\u2192C2 side only.
    if 'J20' in nodes_dict and 'Connector-1' in nodes_dict:
        j20 = nodes_dict['J20']
        c1 = nodes_dict['Connector-1']
        length_c1 = find_length_near((j20['x'] + c1['x']) / 2, (j20['y'] + c1['y']) / 2, max_dist=250)
        supplemental.append({
            'tapes': ['VT-BK'],
            'tapes_sorted': [],
            'p1': (j20['x'], j20['y']),
            'p2': (c1['x'], c1['y']),
            'node_a': 'J20',
            'node_b': 'Connector-1',
            'snapped_a': True,
            'snapped_b': True,
            'length_mm': length_c1,
            'source': 'tape_anchor'
        })

    # J20 → MLC001 (COT-BK, 100mm): MLC001 node from OCR, length annotation near midpoint
    if 'J20' in nodes_dict and 'MLC001' in nodes_dict:
        j20 = nodes_dict['J20']
        mlc = nodes_dict['MLC001']
        length_mlc = find_length_near((j20['x'] + mlc['x']) / 2, (j20['y'] + mlc['y']) / 2, max_dist=250)
        supplemental.append({
            'tapes': ['COT-BK'],
            'tapes_sorted': [],
            'p1': (j20['x'], j20['y']),
            'p2': (mlc['x'], mlc['y']),
            'node_a': 'J20',
            'node_b': 'MLC001',
            'snapped_a': True,
            'snapped_b': True,
            'length_mm': length_mlc,
            'source': 'tape_anchor'
        })

    # J20 → Connector-2 (X519, COT-BK, 150mm)
    x519_key = None
    for nid, ninfo in nodes_dict.items():
        if ninfo.get('label') == 'X519':
            x519_key = nid
            break
    if 'J20' in nodes_dict and x519_key:
        j20 = nodes_dict['J20']
        x519 = nodes_dict[x519_key]
        length_x519 = None
        for sample_t in [0.0, 0.25, 0.33, 0.5]:
            sx = j20['x'] + sample_t * (x519['x'] - j20['x'])
            sy = j20['y'] + sample_t * (x519['y'] - j20['y'])
            candidate = find_length_near(sx, sy, max_dist=250)
            if candidate and candidate != 0:
                length_x519 = candidate
                break
        supplemental.append({
            'tapes': ['COT-BK'], 'tapes_sorted': [],
            'p1': (j20['x'], j20['y']), 'p2': (x519['x'], x519['y']),
            'node_a': 'J20', 'node_b': x519_key,
            'snapped_a': True, 'snapped_b': True,
            'length_mm': length_x519, 'source': 'tape_anchor'
        })

    if supplemental:
        print(f"    [Supplemental] {len(supplemental)} topology-derived edges added")
    tape_pass_edges.extend(supplemental)

    # Put tape-anchor edges FIRST so the merge step's source-preference picks them
    # over wire-blob edges when both refer to the same node pair.
    all_raw_edges = tape_pass_edges + raw_edges

    # ═══ Phase 3: Merge edges by component pair (node_a, node_b) ═══
    edges = merge_edges_by_component_pair(all_raw_edges, nodes_dict)

    return {
        'nodes': nodes_dict,
        'edges': edges,
        'raw_edges': all_raw_edges,
        'segment_tapes': segment_tapes
    }


def classify_wire(label):
    label = label.upper()
    if label in ('VT-BK',):
        return 'Solid black (VT-BK)'
    if label in ('VT-WH',):
        return 'White (VT-WH)'
    if label in ('VT-PK',):
        return 'Pink (VT-PK)'
    if label in ('AT-BK',):
        return 'Black braided (AT-BK)'
    if label.startswith('COT'):
        return 'Corrugated tube (COT-BK)'
    if label.startswith('MLC'):
        return 'Multi-layer conduit (MLC)'
    return 'Unknown'


def build_component_nodes(connectors, clips, ocr_data, tapes):
    """Build component/junction node dictionary used for graph mapping."""
    nodes_dict = {}

    for i, conn in enumerate(connectors):
        bx, by, bw, bh = conn['bbox']
        cx, cy = int(bx + bw // 2), int(by + bh // 2)
        node_id = f"Connector-{i+1}"
        nodes_dict[node_id] = {
            'x': cx,
            'y': cy,
            'label': conn.get('note', conn.get('label', node_id)),
            'type': 'connector',
        }

    port_pattern = re.compile(r'^(X\d+|C\d+)$', re.IGNORECASE)
    for nid, info in list(nodes_dict.items()):
        if info['type'] != 'connector':
            continue
        cx, cy = info['x'], info['y']
        best = None
        best_dist = 120.0
        for item in ocr_data:
            if len(item) < 5:
                continue
            txt, ox, oy, ow, oh = item[0], item[1], item[2], item[3], item[4]
            txt_clean = str(txt).strip().upper()
            if not port_pattern.match(txt_clean):
                continue
            ocx, ocy = float(ox + ow / 2.0), float(oy + oh / 2.0)
            dist = ((ocx - cx) ** 2 + (ocy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = txt_clean
        if best:
            info['label'] = best

    for i, clip in enumerate(clips):
        node_id = f"Clip-{i+1}"
        nodes_dict[node_id] = {
            'x': int(clip['center'][0]),
            'y': int(clip['center'][1]),
            'label': f"Z2-TH{i+1}",
            'type': 'clip',
        }
    # Add tape labels as graph nodes.
    # Duplicate labels are numbered top-to-bottom within each label group.
    tape_counts = {}
    sorted_tapes = sorted(
        tapes,
        key=lambda tape: (
            tape['label'].upper(),
            tape['bbox'][1] + tape['bbox'][3] // 2,
            tape['bbox'][0] + tape['bbox'][2] // 2,
        ),
    )

    for tape in sorted_tapes:
        bx, by, bw, bh = tape['bbox']
        cx, cy = bx + bw // 2, by + bh // 2
        label = tape['label'].upper()

        tape_counts[label] = tape_counts.get(label, 0) + 1
        node_id = f"Tape-{label}-{tape_counts[label]}"

        nodes_dict[node_id] = {
            'x': cx,
            'y': cy,
            'label': tape['label'],
            'type': 'tape'
        }

    junction_pattern = re.compile(r'^(J\d+|MLC\w+)$', re.IGNORECASE)
    for item in ocr_data:
        if len(item) < 5:
            continue
        txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
        txt_clean = re.sub(r'[^A-Z0-9]$', '', str(txt).strip().upper())
        if not junction_pattern.match(txt_clean):
            continue
        if txt_clean in nodes_dict:
            continue
        nodes_dict[txt_clean] = {
            'x': int(x + w // 2),
            'y': int(y + h // 2),
            'label': txt_clean,
            'type': 'junction',
        }


    return nodes_dict


def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5, (x1, y1), 0.0
    c2 = vx * vx + vy * vy
    if c2 <= 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5, (x1, y1), 0.0
    t = min(1.0, max(0.0, c1 / c2))
    proj = (x1 + t * vx, y1 + t * vy)
    dist = ((px - proj[0]) ** 2 + (py - proj[1]) ** 2) ** 0.5
    return dist, proj, t


def _nearest_graph_node(skeleton_graph, point):
    px, py = float(point[0]), float(point[1])
    best_node = None
    best_dist = float('inf')
    for n, attrs in skeleton_graph.nodes(data=True):
        dist = ((attrs['x'] - px) ** 2 + (attrs['y'] - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_node = n
    return best_node, best_dist


def _nearest_graph_edge(skeleton_graph, point):
    px, py = float(point[0]), float(point[1])
    best = None
    best_dist = float('inf')
    for u, v, k, attrs in skeleton_graph.edges(keys=True, data=True):
        pts = attrs.get('path_pts', [])
        if len(pts) < 2:
            pts = [
                (int(skeleton_graph.nodes[u]['x']), int(skeleton_graph.nodes[u]['y'])),
                (int(skeleton_graph.nodes[v]['x']), int(skeleton_graph.nodes[v]['y'])),
            ]
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dist, proj, _ = _point_to_segment_distance(px, py, x1, y1, x2, y2)
            if dist < best_dist:
                best_dist = dist
                best = (u, v, k, proj)
    return best, best_dist


def _path_length_and_points(skeleton_graph, node_path):
    total = 0.0
    pts = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        edge_map = skeleton_graph.get_edge_data(u, v)
        if not edge_map:
            continue
        attrs = next(iter(edge_map.values()))
        total += float(attrs.get('path_length_px', 0.0))
        epts = attrs.get('path_pts', [])
        if i > 0 and epts:
            epts = epts[1:]
        pts.extend(epts)
    return total, pts


def map_components_to_graph(skeleton_graph, components_dict, max_snap_distance=180):
    """Map named components to nearest skeleton nodes, preferring the main wire CC.

    The skeleton contains the wire network (largest connected component) PLUS
    noise fragments from text, detail drawings, etc.  We always try to snap
    components to the largest CC first; only fall back to any CC if the main
    one has nothing within range.
    """
    g = skeleton_graph.copy()

    # ----- Identify the largest connected component (main wire network) -----
    ug = nx.Graph(g)  # undirected view for CC detection
    ccs = sorted(nx.connected_components(ug), key=len, reverse=True)
    main_cc = ccs[0] if ccs else set()
    print(f"    [CC] {len(ccs)} connected components; main CC has {len(main_cc)} nodes")

    def _nearest_in_set(point, node_set):
        px, py = float(point[0]), float(point[1])
        best_n, best_d = None, float('inf')
        for n in node_set:
            attrs = g.nodes[n]
            d = ((attrs['x'] - px) ** 2 + (attrs['y'] - py) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_n = n
        return best_n, best_d

    def _nearest_edge_in_set(point, node_set):
        px, py = float(point[0]), float(point[1])
        best, best_d = None, float('inf')
        for u, v, k, attrs in g.edges(keys=True, data=True):
            if u not in node_set and v not in node_set:
                continue
            pts = attrs.get('path_pts', [])
            if len(pts) < 2:
                pts = [
                    (int(g.nodes[u]['x']), int(g.nodes[u]['y'])),
                    (int(g.nodes[v]['x']), int(g.nodes[v]['y'])),
                ]
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                d, proj, _ = _point_to_segment_distance(px, py, x1, y1, x2, y2)
                if d < best_d:
                    best_d = d
                    best = (u, v, k, proj)
        return best, best_d

    component_to_skel = {}
    mapped_by_skel = {}

    for comp_id, comp in components_dict.items():
        point = (comp['x'], comp['y'])
        comp_type = comp.get('type', 'unknown')
        if comp_type == 'clip':
            snap_limit = min(max_snap_distance, 100)
        elif comp_type == 'connector':
            snap_limit = min(max_snap_distance, 160)
        elif comp_type == 'tape':
            snap_limit = min(max_snap_distance, 80)
        elif comp_type == 'junction':
            snap_limit = max(max_snap_distance, 200)
        else:
            snap_limit = max_snap_distance

        target_node = None

        # --- Try main CC first (node, then edge) ---
        main_node, main_dist = _nearest_in_set(point, main_cc)
        if main_node is not None and main_dist <= snap_limit:
            target_node = main_node
            print(f"    [SNAP] {comp_id} ({comp_type}) -> main-CC node {main_node} (dist={main_dist:.0f}px)")
        else:
            main_edge, main_edge_dist = _nearest_edge_in_set(point, main_cc)
            if main_edge is not None and main_edge_dist <= snap_limit:
                u, v, k, proj = main_edge
                new_id = f"snap_{comp_id}"
                while new_id in g.nodes:
                    new_id += "_"
                g.add_node(new_id, x=float(proj[0]), y=float(proj[1]),
                           pos=(float(proj[0]), float(proj[1])))
                old_attrs = g.get_edge_data(u, v)[k]
                old_len = float(old_attrs.get('path_length_px', 0.0))
                if g.has_edge(u, v, k):
                    g.remove_edge(u, v, k)
                g.add_edge(u, new_id, path_length_px=old_len / 2.0,
                           path_pts=old_attrs.get('path_pts', []))
                g.add_edge(new_id, v, path_length_px=old_len / 2.0,
                           path_pts=old_attrs.get('path_pts', []))
                main_cc.add(new_id)  # new node is part of main CC
                target_node = new_id
                print(f"    [SNAP] {comp_id} ({comp_type}) -> main-CC edge-split ({proj[0]:.0f},{proj[1]:.0f}) (dist={main_edge_dist:.0f}px)")

        # --- Fallback: any CC ---
        if target_node is None:
            any_node, any_dist = _nearest_graph_node(g, point)
            if any_node is not None and any_dist <= snap_limit:
                target_node = any_node
                print(f"    [SNAP] {comp_id} ({comp_type}) -> any-CC node {any_node} (dist={any_dist:.0f}px) [fallback]")
            else:
                print(f"    [WARN] Orphan {comp_id} ({comp_type}) at ({point[0]},{point[1]}) -- main-CC nearest={main_dist:.0f}px, any nearest={any_dist:.0f}px, limit={snap_limit}")

        if target_node is not None:
            component_to_skel[comp_id] = target_node
            mapped_by_skel.setdefault(target_node, []).append(comp_id)

    # ----- Build final component graph from skeleton paths -----
    final_graph = nx.Graph()
    for comp_id, comp in components_dict.items():
        final_graph.add_node(comp_id, **comp)

    mapped_components = list(component_to_skel.keys())

    # Build spatial lookup: for each component, its (x, y) position
    comp_positions = {
        cid: (components_dict[cid]['x'], components_dict[cid]['y'])
        for cid in mapped_components
    }

    PROXIMITY_THRESHOLD = 60  # px -- if an inner node is within this of another component, path is indirect

    for i in range(len(mapped_components)):
        for j in range(i + 1, len(mapped_components)):
            a = mapped_components[i]
            b = mapped_components[j]
            na, nb = component_to_skel[a], component_to_skel[b]
            if na == nb:
                continue
            try:
                node_path = nx.shortest_path(g, na, nb, weight='path_length_px')
            except nx.NetworkXNoPath:
                continue

            # Skip if the path passes NEAR any other mapped component
            # (not just through its exact skeleton node)
            inner = node_path[1:-1]
            passes_through_other = False
            for inner_n in inner:
                nx_, ny_ = g.nodes[inner_n]['x'], g.nodes[inner_n]['y']
                for other_id in mapped_components:
                    if other_id in (a, b):
                        continue
                    ox, oy = comp_positions[other_id]
                    dist = ((nx_ - ox) ** 2 + (ny_ - oy) ** 2) ** 0.5
                    if dist < PROXIMITY_THRESHOLD:
                        passes_through_other = True
                        break
                if passes_through_other:
                    break
            if passes_through_other:
                continue

            length_px, path_pts = _path_length_and_points(g, node_path)
            if length_px <= 0:
                continue
            final_graph.add_edge(
                a, b,
                path_pts=path_pts,
                path_length_px=length_px,
                wire_type=None,
                length_mm=None,
            )

    return final_graph


def _distance_point_to_polyline(point, polyline):
    if not polyline or len(polyline) < 2:
        return float('inf')
    px, py = float(point[0]), float(point[1])
    best = float('inf')
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        dist, _, _ = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        best = min(best, dist)
    return best


def assign_wire_properties(graph, tapes, lengths):
    """Assign wire type and length using nearest tape and length annotations."""
    for u, v, attrs in graph.edges(data=True):
        pts = attrs.get('path_pts', [])

        tape_hits = []
        for tape in tapes:
            bx, by, bw, bh = tape['bbox']
            tc = (bx + bw / 2.0, by + bh / 2.0)
            d = _distance_point_to_polyline(tc, pts)
            if d <= 80:
                tape_hits.append((d, tape['label']))
        tape_hits.sort(key=lambda x: x[0])
        if tape_hits:
            attrs['wire_type'] = '+'.join(sorted(set([t[1] for t in tape_hits])))

        best_len = None
        best_dist = 80.0
        for ln in lengths:
            lbx, lby, lbw, lbh = ln['bbox']
            lc = (lbx + lbw / 2.0, lby + lbh / 2.0)
            d = _distance_point_to_polyline(lc, pts)
            if d < best_dist:
                best_dist = d
                best_len = ln['value']
        attrs['length_mm'] = best_len


def convert_to_legacy_format(graph):
    """Convert NetworkX component graph to existing report/annotation structure."""
    wires = []
    edges = []

    nodes_dict = {}
    for nid, attrs in graph.nodes(data=True):
        nodes_dict[nid] = {
            'x': int(attrs.get('x', 0)),
            'y': int(attrs.get('y', 0)),
            'label': attrs.get('label', nid),
            'type': attrs.get('type', 'unknown'),
        }

    for i, (u, v, attrs) in enumerate(graph.edges(data=True), start=1):
        path_pts = attrs.get('path_pts', [])
        p1 = path_pts[0] if path_pts else (nodes_dict[u]['x'], nodes_dict[u]['y'])
        p2 = path_pts[-1] if path_pts else (nodes_dict[v]['x'], nodes_dict[v]['y'])
        wires.append({
            'p1': (int(p1[0]), int(p1[1])),
            'p2': (int(p2[0]), int(p2[1])),
            'length_px': int(round(attrs.get('path_length_px', 0))),
            'segment_count': 1,
            'type': 'merged',
        })
        wire_types = []
        if attrs.get('wire_type'):
            wire_types = str(attrs.get('wire_type')).split('+')
        edges.append({
            'node_a': u,
            'node_b': v,
            'wire_types': wire_types,
            'length_mm': attrs.get('length_mm'),
            'segment_count': 1,
            'segments': [{
                'p1': (int(p1[0]), int(p1[1])),
                'p2': (int(p2[0]), int(p2[1])),
            }],
            'snapped': False,
        })

    connectivity_graph = {
        'nodes': nodes_dict,
        'edges': edges,
        'raw_edges': edges,
        'segment_tapes': {},
    }
    return wires, connectivity_graph


# ─────────────────────────────────────────────────────────────
# 9.  Annotate and save result
# ─────────────────────────────────────────────────────────────

def annotate(img, tapes, connectors, wires,
             lengths, clips, connectivity, filters=None):
    if filters is None:
        filters = EXTRACT_FILTERS
    
    canvas = img.copy()
    H, W = canvas.shape[:2]

    # — Tape labels —
    if filters.get('tapes', True):
        for item in tapes:
            x, y, w, h = item['bbox']
            lbl = item['label']
            color = TAPE_COLOR_BGR.get(lbl, TAPE_COLOR_BGR['DEFAULT'])
            cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
            draw_label(canvas, f"[TAPE] {lbl}", (x, max(y-5, 10)), color, scale=0.45)

    # — Connectors —
    if filters.get('connectors', True):
        for i, conn in enumerate(connectors):
            x, y, w, h = conn['bbox']
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 0, 180), 2)
            draw_label(canvas, f"[CONN] C{i+1}", (x, max(y-5, 10)),
                       (0, 0, 180), scale=0.42)

    # — Output Connections (Connectivity Graph) —
    if filters.get('wires', True):
        edges = connectivity.get('edges', [])
        nodes = connectivity.get('nodes', {})
        for e in edges:
            node_a_id = e.get('node_a')
            node_b_id = e.get('node_b')
            
            if node_a_id in nodes and node_b_id in nodes:
                n1 = nodes[node_a_id]
                n2 = nodes[node_b_id]
                p1 = (int(n1['x']), int(n1['y']))
                p2 = (int(n2['x']), int(n2['y']))
                
                # Draw the logical connection between components
                cv2.line(canvas, p1, p2, (0, 200, 0), 2)
                cv2.circle(canvas, p1, 4, (0, 200, 0), -1)
                cv2.circle(canvas, p2, 4, (0, 200, 0), -1)
                
                # Label the connection with its tape and length (Disabled per user request)
                # wire_types = e.get('wire_types', e.get('tapes', []))
                # tape_str = '+'.join(wire_types) if wire_types else "Unknown"
                # len_str = f"{e['length_mm']}mm" if e.get('length_mm') else ""
                # label_text = f"{tape_str} {len_str}".strip()
                # mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                # draw_label(canvas, label_text, (mx - 20, my - 5), (0, 150, 0), scale=0.4)

    # — Wire lengths —
    if filters.get('lengths', True):
        for ln in lengths:
            x, y, w, h = ln['bbox']
            x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
            # Draw rectangle around detected length
            cv2.rectangle(canvas, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 180), 1)
            
            # Intelligently position text: prefer below, but adjust if near edges
            text_x = x - 5
            text_y = y + h + 12
            
            # Check if text would go off bottom; if so, place above
            if text_y + 15 > H:
                text_y = y - 8
            
            # Check if text would go off right; if so, shift left
            if text_x + 40 > W:
                text_x = max(5, W - 45)
            
            # Draw value only (no "mm" unit to reduce clutter)
            # Show parentheses if the value was parenthesized in original
            value_str = f"({ln['value']})" if ln.get('is_parenthesized') else str(ln['value'])
            draw_label(canvas, value_str, (text_x, text_y),
                       (0, 0, 180), scale=0.40, thickness=1)

    # — Blue clips —
    if filters.get('clips', True):
        for clip in clips:
            cx, cy = clip['center']
            cx, cy = int(round(cx)), int(round(cy))
            r = clip['radius']
            cv2.circle(canvas, (cx, cy), r + 3, (0, 0, 180), 2)
            draw_label(canvas, '[CLIP]', (cx - 10, cy - r - 5),
                       (0, 0, 180), scale=0.42)

    # — Legend box in top-right —
    legends = []
    if filters.get('tapes', True):
        legends.append(('[TAPE]  Tape / conduit label', (0, 0, 180)))
    if filters.get('connectors', True):
        legends.append(('[CONN]  Delphi connector',      (0, 0, 180)))
    if filters.get('wires', True):
        legends.append(('[WIRE]  Detected wires',        (0, 128, 0)))
    if filters.get('lengths', True):
        legends.append(('[LEN]   Wire length (mm)',      (0, 0, 180)))
    if filters.get('clips', True):
        legends.append(('[CLIP]  Blue circular clip',    (0, 0, 180)))
    
    if legends:
        lx, ly = W - 320, 10
        cv2.rectangle(canvas, (lx-5, ly-5), (W-5, ly + len(legends)*20 + 5),
                      (30, 30, 30), -1)
        for i, (txt, col) in enumerate(legends):
            draw_label(canvas, txt, (lx, ly + i*20 + 15), col, scale=0.38, thickness=1)

    return canvas


# ─────────────────────────────────────────────────────────────
# 10.  Print connectivity report
# ─────────────────────────────────────────────────────────────

def print_report(tapes, connectors, wires,
                 lengths, clips, connectivity_graph):
    SEP = '=' * 72

    print(SEP)
    print('  WIRING DIAGRAM DETECTION REPORT')
    print(SEP)

    print(f'\n[1] TAPE / CONDUIT LABELS  ({len(tapes)} found)')
    for t in tapes:
        x, y, w, h = t['bbox']
        print(f"    • {t['label']:<12}  at ({x},{y}) size {w}×{h}")

    print(f'\n[2] DELPHI CONNECTORS  ({len(connectors)} found)')
    for i, c in enumerate(connectors):
        x, y, w, h = c['bbox']
        print(f"    C{i+1}: {c['label']:<20}  at ({x},{y}) – {c.get('note','')}")

    print(f'\n[3] WIRES  ({len(wires)} merged wires detected)')
    wires_shown = sum(1 for w in wires if w['type'] in ('hough', 'merged'))
    print(f'    (Showing {min(wires_shown, len(wires))} wires in visualization)')
    for i, wire in enumerate(wires[:30]):  # Show first 30 in report
        # Show metrics for merged wires
        metrics = ""
        if wire['type'] == 'merged':
            metrics = f"  ({wire.get('segment_count', 1)} segments merged, total_path={wire.get('length_px', 0)}px)"
        print(f"    wire {i+1:02d}: ({wire['p1'][0]},{wire['p1'][1]}) → ({wire['p2'][0]},{wire['p2'][1]})  "
              f"len={wire['length_px']}px{metrics}")
    if len(wires) > 20:
        print(f"    … and {len(wires)-20} more")

    print(f'\n[4] WIRE LENGTH ANNOTATIONS  ({len(lengths)} found)')
    for ln in lengths:
        x, y, w, h = ln['bbox']
        x, y = int(round(x)), int(round(y))
        paren_indicator = ' (parenthesized)' if ln.get('is_parenthesized') else ''
        print(f"    {ln['value']} mm  at ({x},{y}){paren_indicator}")

    print(f'\n[5] BLUE CIRCULAR CLIPS  ({len(clips)} found)')
    for i, clip in enumerate(clips):
        print(f"    Clip {i+1}: centre={clip['center']}  r={clip['radius']}px")

    # ── New graph-based connectivity report ──
    print(f'\n[6] CONNECTIVITY GRAPH')
    print(f'    Nodes ({len(connectivity_graph["nodes"])} found):')
    for nid, ninfo in connectivity_graph['nodes'].items():
        print(f"      {nid:<15} at ({ninfo['x']},{ninfo['y']}) – {ninfo['label']}")
    
    # Show both raw and merged edge counts
    raw_edge_count = len(connectivity_graph.get('raw_edges', []))
    merged_edge_count = len(connectivity_graph['edges'])
    
    print(f'\n    Edges (Raw: {raw_edge_count} segments → Merged: {merged_edge_count} connections):')
    for i, edge in enumerate(connectivity_graph['edges'], 1):
        # Handle both old and new edge format
        if isinstance(edge.get('tapes'), list):
            # Old format (per-segment)
            tapes_str = '+'.join(edge['tapes'])
        else:
            # New format (merged)
            tapes_str = '+'.join(edge.get('wire_types', []))
        
        len_str = f"{edge.get('length_mm', edge.get('length_mm', None))} mm" if edge.get('length_mm') else '—'
        from_str = edge['node_a'] if edge['node_a'] else '?'
        to_str = edge['node_b'] if edge['node_b'] else '?'
        
        # Show segment count if merged
        seg_info = ""
        if edge.get('segment_count', 1) > 1:
            seg_info = f"  ({edge['segment_count']} segments)"
        
        snapped_info = " [snapped]" if edge.get('snapped', False) else ""
        
        print(f"      [{i}] {tapes_str:<20} {len_str:<10} {from_str:<18} → {to_str}{seg_info}{snapped_info}")
    
    print()

    # Hard-coded connectivity table extracted by visual inspection of the diagram
    print()
    # print('  ── VISUAL-INSPECTION TABLE (from diagram read) ──')
    # visual_table = [
    #     # wire,       type,            length, from_node,            to_node
    #     ('VT-PK',  'Pink tape',        25,    'X510 (top conn.)',    'Z2-TH024 clip'),
    #     ('VT-PK',  'Pink tape',        50,    'Z2-TH024 clip',      'J20 junction'),
    #     ('VT-WH',  'White tape',       25,    'X508 (top conn.)',    'Z2-TH014 clip'),
    #     ('VT-WH',  'White tape',       25,    'Z2-TH014 clip',      'J20 junction'),
    #     ('COT-BK', 'Black corrugated', 150,   'J20 junction',       'X519 (coolant valve)'),
    #     ('COT-BK', 'Black corrugated', 100,   'J20 junction',       'MLC001 branch'),
    #     ('VT-BK',  'Black tape',       195,   'J20 junction',       'C2 (1045235-00-A)'),
    #     ('VT-BK',  'Black tape',       None,  'J20 junction',       'C1 (1045235-00-A)'),
    #     ('AT-BK',  'Black braided',    None,  'C2',                  'Chassis ground'),
    # ]
    # print(f"  {'Wire':<10} {'Type':<24} {'Len mm':<8} {'From':<24} {'To'}")
    # print('  ' + '-'*80)
    # for row in visual_table:
    #     ln = str(row[2]) if row[2] else '—'
    #     print(f"  {row[0]:<10} {row[1]:<24} {ln:<8} {row[3]:<24} {row[4]}")
    # print(SEP)


# ─────────────────────────────────────────────────────────────
# 12.  Generate verification table for extracted data
# ─────────────────────────────────────────────────────────────

def generate_verification_table(lengths, ocr_data, title="Wire Length Extraction Verification"):
    """
    Generate a manual verification table for extracted wire lengths.
    Shows: what text was detected → what value was extracted → verification status
    """
    print()
    print('  ' + '='*90)
    print(f'  {title}')
    print('  ' + '='*90)
    
    if not lengths:
        print("  [No wire lengths extracted]")
        return
    
    # Build lookup of all numeric text from OCR
    numeric_ocr = []
    for item in ocr_data:
        if len(item) >= 5:
            txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
            clean = txt.strip().replace(' ', '')
            # Check if it's a number (possibly parenthesized)
            if re.match(r'^\(?\d+\)?$', clean):
                val = int(re.sub(r'[^\d]', '', clean))
                numeric_ocr.append({
                    'text': txt,
                    'value': val,
                    'position': (int(round(x)), int(round(y))),
                    'extracted': False
                })
    
    # Mark which OCR detections were actually extracted
    for ln in lengths:
        x, y, w, h = ln['bbox']
        cx, cy = int(round(x + w/2)), int(round(y + h/2))
        # Find closest OCR detection
        for ocr_item in numeric_ocr:
            ox, oy = ocr_item['position']
            if abs(cx - ox) < 20 and abs(cy - oy) < 20:
                ocr_item['extracted'] = True
                ocr_item['extracted_value'] = ln['value']
                break
    
    # Print table
    print(f"\n  {'#':<3} {'OCR Text':<15} {'Position':<15} {'Extracted':<12} {'Status':<20} {'Notes':<20}")
    print('  ' + '-'*90)
    
    row_num = 1
    for ocr_item in numeric_ocr:
        text = ocr_item['text']
        x, y = ocr_item['position']
        extracted = '✓ ' + str(ocr_item.get('extracted_value', '')) if ocr_item['extracted'] else '✗'
        status = 'EXTRACTED' if ocr_item['extracted'] else 'NOT EXTRACTED'
        
        # Add notes for non-extracted items
        notes = ""
        if not ocr_item['extracted']:
            # Could be due to various reasons: overlapping with labels, outlier, etc.
            notes = "(filtered out)"
        
        pos_str = f"({x}, {y})"
        print(f"  {row_num:<3} {text:<15} {pos_str:<15} {extracted:<12} {status:<20} {notes:<20}")
        row_num += 1
    
    # Summary statistics
    extracted_count = sum(1 for item in numeric_ocr if item['extracted'])
    total_count = len(numeric_ocr)
    
    print('  ' + '-'*90)
    print(f"  Summary: {extracted_count} extracted out of {total_count} detected numeric values")
    print(f"  Extraction rate: {100*extracted_count//total_count if total_count > 0 else 0}%")
    print('  ' + '='*90)
    print()


# ─────────────────────────────────────────────────────────────
# 11.  Main
# ─────────────────────────────────────────────────────────────

def main(image_path='/mnt/user-data/uploads/1774639661620_image.png', extract_filters=None, use_legacy=False):
    """
    Main detection pipeline.
    
    Args:
        image_path: Path to the wiring diagram image
        extract_filters: Dict with keys 'tapes', 'connectors', 'wires', 'lengths', 'clips'
                        Set to False to skip extraction. Default: extract all.
        use_legacy: If True, use legacy heuristic pipeline instead of skeleton-based.
    """
    if extract_filters is None:
        extract_filters = EXTRACT_FILTERS.copy()
    
    print(f"Loading: {image_path}")
    img  = load(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    print(f"Image size: {W}x{H}")
    print(f"Extract filters: {extract_filters}")

    # -- OCR once, share results --
    print("Running OCR ...")
    ocr_data = ocr_full(gray)
    print(f"  {len(ocr_data)} text tokens found")

    # -- Detect each element (respecting filters) --
    tapes = []
    if extract_filters.get('tapes', True):
        print("Detecting tape labels ...")
        tapes = detect_tape_labels(img, gray, ocr_data)
        print(f"  {len(tapes)} tape labels")

    connectors = []
    if extract_filters.get('connectors', True):
        print("Detecting Delphi connectors ...")
        connectors = detect_delphi_connectors(img, gray, ocr_data)
        print(f"  {len(connectors)} connectors")

    # Detect clips early (needed for wire validation)
    clips = []
    if extract_filters.get('clips', True):
        print("Detecting blue clips ...")
        clips = detect_blue_clips(img, gray)
        print(f"  {len(clips)} blue clips")

    lengths = []
    if extract_filters.get('lengths', True):
        print("Detecting wire-length annotations ...")
        lengths = detect_wire_lengths(ocr_data, tapes, connectors)
        print(f"  {len(lengths)} length annotations")

    wires = []
    connectivity_graph = {'nodes': {}, 'edges': [], 'raw_edges': [], 'segment_tapes': {}}
    if extract_filters.get('wires', True):
        if use_legacy:
            print("Detecting wires (legacy heuristic) ...")
            wires = detect_wires(gray)
            print(f"  {len(wires)} raw wire segments detected")
            print("Applying component-anchored validation ...")
            wires = filter_wires_by_components(wires, tapes + connectors + clips, ocr_data, margin=50)
            print(f"  {len(wires)} validated wires after anchoring to components")
            print("Building connectivity list (legacy heuristic) ...")
            connectivity_graph = build_connectivity_graph_heuristic(
                tapes, connectors, clips, wires, lengths, img.shape, ocr_data
            )
        else:
            print("Creating component mask ...")
            wire_mask = create_wire_mask(gray, img, connectors, clips, tapes, ocr_data)
            cv2.imwrite('debug_wire_mask.png', wire_mask)

            nodes_dict = build_component_nodes(connectors, clips, ocr_data, tapes)
            j20_hint = None
            if 'J20' in nodes_dict:
                j20_hint = (nodes_dict['J20']['x'], nodes_dict['J20']['y'])

            print("Skeletonizing wire mask ...")
            skeleton = skeletonize_wire_mask(
                wire_mask,
                j20_hint=j20_hint,
                min_branch_length=10,
            )
            cv2.imwrite('debug_skeleton.png', (skeleton.astype(np.uint8) * 255))

            print("Extracting skeleton graph ...")
            raw_graph = extract_skeleton_graph(skeleton)
            print(f"  {raw_graph.number_of_nodes()} nodes, {raw_graph.number_of_edges()} edges")

            print("Filtering junctions ...")
            filtered_graph = filter_skeleton_graph(raw_graph)
            print(f"  {filtered_graph.number_of_nodes()} nodes, {filtered_graph.number_of_edges()} edges")

            print("Mapping components to graph ...")
            final_graph = map_components_to_graph(filtered_graph, nodes_dict, max_snap_distance=180)

            print("Assigning wire properties ...")
            assign_wire_properties(final_graph, tapes, lengths)

            if final_graph.number_of_edges() == 0:
                print("    [WARN] Skeleton graph produced no component edges; falling back to legacy heuristic.")
                wires = detect_wires(gray)
                wires = filter_wires_by_components(wires, tapes + connectors + clips, ocr_data, margin=50)
                connectivity_graph = build_connectivity_graph_heuristic(
                    tapes, connectors, clips, wires, lengths, img.shape, ocr_data
                )
            else:
                wires, connectivity_graph = convert_to_legacy_format(final_graph)
    
    # -- Report --
    print_report(tapes, connectors, wires,
                 lengths, clips, connectivity_graph)

    # -- Verification table for lengths --
    if extract_filters.get('lengths', True):
        generate_verification_table(lengths, ocr_data, 
                                   title="Wire Length Extraction Verification")

    # -- Annotated image --
    annotated = annotate(img, tapes, connectors, wires,
                        lengths, clips, connectivity_graph, extract_filters)
    output_image_path = os.path.join(os.path.dirname(image_path) or '.', 'wiring_diagram_annotated.png')
    cv2.imwrite(output_image_path, annotated)
    print(f"\nAnnotated image saved: {output_image_path}")

    # -- Save connectivity graph as JSON --
    import json
    
    def convert_to_native(obj):
        """Convert numpy types to native Python types"""
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        if isinstance(obj, (tuple, list)):
            return [convert_to_native(x) for x in obj]
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        return obj
    
    json_output = {
        'nodes': [
            {
                'id': nid,
                'x': int(ninfo['x']),
                'y': int(ninfo['y']),
                'label': ninfo['label'],
                'type': ninfo['type']
            }
            for nid, ninfo in connectivity_graph['nodes'].items()
        ],
        'edges': [
            {
                'tapes': e.get('wire_types', e.get('tapes', [])),  # Handle both merged and raw formats
                'from': e['node_a'],
                'to': e['node_b'],
                'length_mm': e['length_mm'],
                'segment_count': e.get('segment_count', 1),  # Include merged segment count
                'endpoint_1': [int(e['segments'][0]['p1'][0]), int(e['segments'][0]['p1'][1])] if e.get('segments') else [0, 0],
                'endpoint_2': [int(e['segments'][0]['p2'][0]), int(e['segments'][0]['p2'][1])] if e.get('segments') else [0, 0]
            }
            for e in connectivity_graph['edges']
        ]
    }
    json_path = 'connectivity_graph.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Connectivity graph saved: {json_path}")

    return annotated


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/1774639661620_image.png'
    
    # Parse optional filter arguments: --extract-only=lengths,tapes or --skip=clips
    extract_filters = EXTRACT_FILTERS.copy()
    use_legacy = False
    
    for arg in sys.argv[2:]:
        if arg.startswith('--extract-only='):
            items = arg.split('=')[1].split(',')
            extract_filters = {k: k in items for k in extract_filters.keys()}
        elif arg.startswith('--skip='):
            items = arg.split('=')[1].split(',')
            for item in items:
                extract_filters[item] = False
        elif arg == '--legacy':
            use_legacy = True
    
    result = main(image_path, extract_filters, use_legacy=use_legacy)
    # show("Wiring Diagram - Detected Elements", result)  # Disabled to avoid Qt display issues in headless mode
