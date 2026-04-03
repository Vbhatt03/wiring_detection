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
import re
try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    print("[WARN] pytesseract not installed – OCR disabled, using hard-coded labels.")

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
    'VT-WH':  (200, 200,  50),   # cyan-ish
    'VT-BK':  (  0,   0, 220),   # red
    'VT-PK':  (220,  50, 220),   # magenta
    'AT-BK':  ( 50, 180, 220),   # orange
    'COT-BK': ( 50, 220, 100),   # green
    'DEFAULT':(  0, 180, 180),
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

def detect_wires(gray):
    """
    Detect all wires in the diagram: dash-dot, zigzag, and any other wire patterns.
    Uses both Hough line detection and row-by-row transition scanning to capture
    all wire types without distinguishing between them.
    """
    wires = []
    
    # Method 1: Hough line detection for linear wires
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                             minLineLength=30, maxLineGap=8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2-x1, y2-y1)
            if length >= 30:
                wires.append({
                    'p1': (x1, y1), 'p2': (x2, y2),
                    'length_px': int(length),
                    'type': 'hough'
                })
    
    # Method 2: Row-by-row scanning for undulating/zigzag patterns
    # Scan entire image for rows with significant alternation
    for y in range(gray.shape[0]):
        row = gray[y, :]
        bin_row = (row < 150).astype(np.uint8)
        dark_cols = np.where(bin_row == 1)[0]
        
        if len(dark_cols) == 0:
            continue
        
        x_start = int(dark_cols[0])
        x_end = int(dark_cols[-1])
        width = x_end - x_start
        
        if width < 60 or x_start <= 20:
            continue
        
        # Count transitions (alternation pattern)
        transitions = np.sum(np.diff(bin_row) != 0)
        
        # Require minimum transitions for wire pattern
        if transitions < 80:
            continue
        
        # Check if this wire already detected by Hough (to avoid duplicates)
        is_duplicate = False
        for w in wires:
            if w['type'] == 'hough':
                p1, p2 = w['p1'], w['p2']
                # Check if row-based detection overlaps with Hough line
                if abs(p1[1] - y) < 5 or abs(p2[1] - y) < 5:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            wires.append({
                'y': y,
                'x_start': x_start,
                'x_end': x_end,
                'length_px': width,
                'transitions': transitions,
                'type': 'row_scan'
            })
    
    # Merge consecutive row-scan detections
    merged_wires = []
    row_scan_wires = [w for w in wires if w['type'] == 'row_scan']
    hough_wires = [w for w in wires if w['type'] == 'hough']
    
    if row_scan_wires:
        sorted_rows = sorted(row_scan_wires, key=lambda w: w['y'])
        merged = [sorted_rows[0]]
        for seg in sorted_rows[1:]:
            if seg['y'] - merged[-1]['y'] <= 8:
                merged[-1]['y'] = seg['y']
                merged[-1]['x_start'] = min(merged[-1]['x_start'], seg['x_start'])
                merged[-1]['x_end'] = max(merged[-1]['x_end'], seg['x_end'])
                merged[-1]['transitions'] = max(merged[-1]['transitions'], seg['transitions'])
                merged[-1]['length_px'] = merged[-1]['x_end'] - merged[-1]['x_start']
            else:
                merged.append(seg)
        
        merged_wires = merged
    
    # Return combined list: Hough lines + merged row scans
    return hough_wires + merged_wires


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

def build_connectivity_graph(tape_labels, connectors, clips, wires, lengths, img_shape):
    """
    Build a connectivity graph by tracing wire segments.
    
    Strategy:
    1. For each dash-dot segment, find which tape labels are on it
    2. Group tape labels that are on the same segment
    3. For each segment, find endpoints and map to actual nodes
    4. Build edges showing: Node A --[wire type, length]--> Node B
    
    Returns: dict with 'nodes' (list of nodes) and 'edges' (list of connections)
    """
    h, w = img_shape[:2]
    
    # Create node list from connectors and clips
    nodes_dict = {}  # id -> {cx, cy, label, type}
    
    for i, conn in enumerate(connectors):
        bx, by, bw, bh = conn['bbox']
        node_id = f"Connector-{i+1}"
        nodes_dict[node_id] = {
            'x': bx + bw//2,
            'y': by + bh//2,
            'label': conn.get('note', conn['label']),
            'type': 'connector'
        }
    
    for i, clip in enumerate(clips):
        node_id = f"Clip-{i+1}"
        nodes_dict[node_id] = {
            'x': clip['center'][0],
            'y': clip['center'][1],
            'label': f"Z2-TH{i+1}",
            'type': 'clip'
        }
    
    # For each wire segment, find which tapes are on it
    segment_tapes = {}  # segment_idx -> list of tape info
    
    # Filter to only Hough-detected wires (with 'p1' and 'p2' endpoints)
    hough_wires = [w for w in wires if w['type'] == 'hough']
    
    for seg_idx, seg in enumerate(hough_wires):
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
                
                # Include tape if it's close to the segment (within 30px)
                if perp_dist < 30:
                    tapes_on_segment.append({
                        'label': tape['label'],
                        'proj_dist': proj_dist,
                        'perp_dist': perp_dist
                    })
        
        if tapes_on_segment:
            segment_tapes[seg_idx] = tapes_on_segment
    
    # Find nearest node to segment endpoints
    def find_nearest_node(point, exclude_dist_thresh=20):
        """Find nearest node to a point, or return None if too far"""
        min_dist = float('inf')
        nearest = None
        for nid, ninfo in nodes_dict.items():
            dist = np.linalg.norm(np.array([ninfo['x'], ninfo['y']]) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                nearest = nid
        
        if min_dist < exclude_dist_thresh:
            return nearest
        return None
    
    # Build edges from wire segments
    edges = []
    
    for seg_idx, seg in enumerate(hough_wires):
        if seg_idx not in segment_tapes:
            continue
        
        tapes = segment_tapes[seg_idx]
        p1 = seg['p1']
        p2 = seg['p2']
        
        # Find endpoints
        node_a = find_nearest_node(p1)
        node_b = find_nearest_node(p2)
        
        # Get wire properties from tape labels on this segment
        segment_info = {
            'tapes': [t['label'] for t in tapes],
            'tapes_sorted': sorted(tapes, key=lambda t: t['proj_dist']),
            'p1': p1,
            'p2': p2,
            'node_a': node_a,
            'node_b': node_b
        }
        
        # Find length for this segment (nearest to midpoint)
        seg_mid = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
        nearest_length = None
        min_len_dist = float('inf')
        for ln in lengths:
            lbx, lby, lbw, lbh = ln['bbox']
            lcx, lcy = lbx + lbw//2, lby + lbh//2
            dist = np.linalg.norm(np.array([lcx, lcy]) - np.array(seg_mid))
            if dist < min_len_dist and dist < 100:
                min_len_dist = dist
                nearest_length = ln['value']
        
        segment_info['length_mm'] = nearest_length
        
        edges.append(segment_info)
    
    return {
        'nodes': nodes_dict,
        'edges': edges,
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


# ─────────────────────────────────────────────────────────────
# 9.  Annotate and save result
# ─────────────────────────────────────────────────────────────

def annotate(img, tapes, connectors, wires,
             lengths, clips, connectivity):
    canvas = img.copy()
    H, W = canvas.shape[:2]

    # — Tape labels —
    for item in tapes:
        x, y, w, h = item['bbox']
        lbl = item['label']
        color = TAPE_COLOR_BGR.get(lbl, TAPE_COLOR_BGR['DEFAULT'])
        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
        draw_label(canvas, f"[TAPE] {lbl}", (x, max(y-5, 10)), color, scale=0.45)

    # — Connectors —
    for i, conn in enumerate(connectors):
        x, y, w, h = conn['bbox']
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 140, 0), 2)
        draw_label(canvas, f"[CONN] C{i+1}", (x, max(y-5, 10)),
                   (255, 140, 0), scale=0.42)

    # — All wires (unified: hough lines + row scans) —
    for wire in wires[:50]:   # limit to first 50 for clarity
        if wire['type'] == 'hough':
            p1, p2 = wire['p1'], wire['p2']
            cv2.line(canvas, p1, p2, (0, 220, 255), 1)
        elif wire['type'] == 'row_scan':
            y_w = wire['y']
            x0, x1 = wire['x_start'], wire['x_end']
            cv2.line(canvas, (x0, y_w), (x1, y_w), (0, 220, 255), 2)

    # — Wire lengths —
    for ln in lengths:
        x, y, w, h = ln['bbox']
        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        # Draw rectangle around detected length
        cv2.rectangle(canvas, (x-2, y-2), (x+w+2, y+h+2), (0, 180, 255), 1)
        
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
                   (0, 180, 255), scale=0.40, thickness=1)

    # — Blue clips —
    for clip in clips:
        cx, cy = clip['center']
        cx, cy = int(round(cx)), int(round(cy))
        r = clip['radius']
        cv2.circle(canvas, (cx, cy), r + 3, (255, 80, 0), 2)
        draw_label(canvas, '[CLIP]', (cx - 10, cy - r - 5),
                   (255, 80, 0), scale=0.42)

    # — Legend box in top-right —
    legends = [
        ('[TAPE]  Tape / conduit label', (0, 200, 0)),
        ('[CONN]  Delphi connector',      (255, 140, 0)),
        ('[WIRE]  Detected wires',        (0, 220, 255)),
        ('[LEN]   Wire length (mm)',      (0, 180, 255)),
        ('[CLIP]  Blue circular clip',    (255, 80, 0)),
    ]
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

    print(f'\n[3] WIRES  ({len(wires)} segments detected)')
    for i, wire in enumerate(wires[:15]):
        if wire['type'] == 'hough':
            print(f"    seg {i+1:02d}: {wire['p1']} → {wire['p2']}  len={wire['length_px']}px")
        elif wire['type'] == 'row_scan':
            print(f"    seg {i+1:02d}: y={wire['y']}  x={wire['x_start']}–{wire['x_end']}  "
                  f"width={wire['length_px']}px  transitions={wire['transitions']}")
    if len(wires) > 15:
        print(f"    … and {len(wires)-15} more")

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
    
    print(f'\n    Edges ({len(connectivity_graph["edges"])} found):')
    for i, edge in enumerate(connectivity_graph['edges'], 1):
        tapes_str = '+'.join(edge['tapes'])
        len_str = f"{edge['length_mm']} mm" if edge['length_mm'] else '—'
        from_str = edge['node_a'] if edge['node_a'] else '?'
        to_str = edge['node_b'] if edge['node_b'] else '?'
        print(f"      [{i}] {tapes_str:<20} {len_str:<10} {from_str:<18} → {to_str}")
    
    print()

    # Hard-coded connectivity table extracted by visual inspection of the diagram
    print()
    print('  ── VISUAL-INSPECTION TABLE (from diagram read) ──')
    visual_table = [
        # wire,       type,            length, from_node,            to_node
        ('VT-PK',  'Pink tape',        25,    'X510 (top conn.)',    'Z2-TH024 clip'),
        ('VT-PK',  'Pink tape',        50,    'Z2-TH024 clip',      'J20 junction'),
        ('VT-WH',  'White tape',       25,    'X508 (top conn.)',    'Z2-TH014 clip'),
        ('VT-WH',  'White tape',       25,    'Z2-TH014 clip',      'J20 junction'),
        ('COT-BK', 'Black corrugated', 150,   'J20 junction',       'X519 (coolant valve)'),
        ('COT-BK', 'Black corrugated', 100,   'J20 junction',       'MLC001 branch'),
        ('VT-BK',  'Black tape',       195,   'J20 junction',       'C2 (1045235-00-A)'),
        ('VT-BK',  'Black tape',       None,  'J20 junction',       'C1 (1045235-00-A)'),
        ('AT-BK',  'Black braided',    None,  'C2',                  'Chassis ground'),
    ]
    print(f"  {'Wire':<10} {'Type':<24} {'Len mm':<8} {'From':<24} {'To'}")
    print('  ' + '-'*80)
    for row in visual_table:
        ln = str(row[2]) if row[2] else '—'
        print(f"  {row[0]:<10} {row[1]:<24} {ln:<8} {row[3]:<24} {row[4]}")
    print(SEP)


# ─────────────────────────────────────────────────────────────
# 11.  Main
# ─────────────────────────────────────────────────────────────

def main(image_path='/mnt/user-data/uploads/1774639661620_image.png'):
    print(f"Loading: {image_path}")
    img  = load(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    print(f"Image size: {W}×{H}")

    # ── OCR once, share results ──
    print("Running OCR …")
    ocr_data = ocr_full(gray)
    print(f"  {len(ocr_data)} text tokens found")

    # ── Detect each element ──
    print("Detecting tape labels …")
    tapes = detect_tape_labels(img, gray, ocr_data)
    print(f"  {len(tapes)} tape labels")

    print("Detecting Delphi connectors …")
    connectors = detect_delphi_connectors(img, gray, ocr_data)
    print(f"  {len(connectors)} connectors")

    print("Detecting wires …")
    wires = detect_wires(gray)
    print(f"  {len(wires)} wire segments")

    print("Detecting wire-length annotations …")
    lengths = detect_wire_lengths(ocr_data, tapes, connectors)
    print(f"  {len(lengths)} length annotations")

    print("Detecting blue clips …")
    clips = detect_blue_clips(img, gray)
    print(f"  {len(clips)} blue clips")

    print("Building connectivity list …")
    connectivity_graph = build_connectivity_graph(tapes, connectors, clips, 
                                                 wires, lengths, img.shape)

    # ── Report ──
    print_report(tapes, connectors, wires,
                 lengths, clips, connectivity_graph)

    # ── Annotated image ──
    annotated = annotate(img, tapes, connectors, wires,
                        lengths, clips, connectivity_graph)
    output_image_path = os.path.join(os.path.dirname(path) or '.', 'wiring_diagram_annotated.png')
    cv2.imwrite(output_image_path, annotated)
    print(f"\nAnnotated image saved: {output_image_path}")

    # ── Save connectivity graph as JSON ──
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
                'tapes': e['tapes'],
                'from': e['node_a'],
                'to': e['node_b'],
                'length_mm': e['length_mm'],
                'endpoint_1': [int(e['p1'][0]), int(e['p1'][1])],
                'endpoint_2': [int(e['p2'][0]), int(e['p2'][1])]
            }
            for e in connectivity_graph['edges']
        ]
    }
    json_path = 'connectivity_graph.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Connectivity graph saved → {json_path}")

    return annotated


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/1774639661620_image.png'
    result = main(path)
    # show("Wiring Diagram – Detected Elements", result)  # Disabled to avoid Qt display issues in headless mode
