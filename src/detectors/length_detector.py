"""Wire length annotation detection module for wiring diagrams."""

import re
import numpy as np

LENGTH_PATTERN = re.compile(r'^[\(\+]*\d{1,4}[\+\)]*$')   # e.g. 0, (0), 25, (25), (50), 150, (+150+)
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

def merge_token_fragments(ocr_data):
    """Merge nearby same-baseline fragments before regex:
       ( + 25 + ) -> (25)
       1 + 00 -> 100
       horizontal gap <= 18 px
       vertical center difference <= max height
    """
    items = []
    for item in ocr_data:
        if len(item) >= 5:
            txt, x, y, w, h = item[0:5]
            angle = item[5] if len(item) > 5 else 0
            conf = item[6] if len(item) > 6 else 50
            items.append({
                'txt': txt, 'x': x, 'y': y, 'w': w, 'h': h, 
                'cy': y + h/2.0, 'angle': angle, 'conf': conf, 'orig': item
            })
            
    items.sort(key=lambda t: t['x'])
    merged = []
    used = set()
    
    for i, t1 in enumerate(items):
        if i in used: continue
        
        current_txt = t1['txt']
        x1 = t1['x']
        y1 = t1['y']
        x2 = x1 + t1['w']
        h_max = t1['h']
        y_min = t1['y']
        conf_max = t1['conf']
        angle_avg = t1['angle']
        
        used.add(i)
        
        while True:
            merged_something = False
            for j, t2 in enumerate(items):
                if j in used: continue
                # horizontal gap
                h_gap = t2['x'] - x2
                if -5 <= h_gap <= 18:
                    # vertical center diff
                    v_diff = abs((y_min + h_max/2.0) - t2['cy'])
                    if v_diff <= max(h_max, t2['h']):
                        current_txt += t2['txt']
                        x2 = t2['x'] + t2['w']
                        h_max = max(h_max, t2['h'])
                        y_min = min(y_min, t2['y'])
                        conf_max = max(conf_max, t2['conf'])
                        used.add(j)
                        merged_something = True
            if not merged_something:
                break
                
        new_w = x2 - x1
        orig = list(t1['orig'])
        orig[0] = current_txt
        orig[1] = x1
        orig[2] = y_min
        orig[3] = new_w
        orig[4] = h_max
        if len(orig) > 5: orig[5] = angle_avg
        if len(orig) > 6: orig[6] = conf_max
        merged.append(tuple(orig))
        
    return merged



def detect_wire_lengths(ocr_data, tapes=None, connectors=None):
    """Extract numeric wire-length annotations from OCR data.
    
    Accepts both horizontal, vertical, and angled text (parenthesized or not).
    Filters out lengths that overlap with tape labels or connector bounding boxes.
    Smartly deduplicates multi-angle detections by proximity and confidence.
    
    Args:
        ocr_data: List of OCR results from ocr_full()
        tapes: List of detected tape labels
        connectors: List of detected connectors
    
    Returns:
        List of detected wire length annotations
    """
    if tapes is None:
        tapes = []
    if connectors is None:
        connectors = []
    
    merged_data = merge_token_fragments(ocr_data)
    
    # Build map of label text positions for filtering
    label_positions = []
    for item in merged_data:
        if len(item) >= 5:
            txt = item[0]
            if LABEL_KEYWORDS.search(txt):
                label_positions.append(item[1:5])
    
    candidates = []
    pre_overlap_count = 0
    tape_reject_count = 0
    
    for item in merged_data:
        if len(item) >= 5:
            txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
            angle = item[5] if len(item) > 5 else 0
            conf = item[6] if len(item) > 6 else 50
        else:
            continue
            
        clean = normalize_length_token(txt)
        if LENGTH_PATTERN.match(clean):
            val = int(re.sub(r'[^\d]', '', clean))
            is_parenthesized = '(' in clean and ')' in clean
            
            # Selective filtering: parenthesized (0-3000) or non-parenthesized multi-digit (10-3000)
            if (is_parenthesized and 0 <= val <= 3000) or (not is_parenthesized and 10 <= val <= 3000):
                pre_overlap_count += 1
                is_overlapping_label = False
                bbox_tolerance = 3 if is_parenthesized else 6
                cx = x + w / 2.0
                cy = y + h / 2.0
                
                # Check against tape labels
                for tape in tapes:
                    tx, ty, tw, th = tape['bbox']
                    if (cx >= tx - bbox_tolerance and cx <= tx + tw + bbox_tolerance and
                        cy >= ty - bbox_tolerance and cy <= ty + th + bbox_tolerance):
                        is_overlapping_label = True
                        tape_reject_count += 1
                        break
                
                # Check against connectors
                if not is_overlapping_label:
                    for conn in connectors:
                        cx_conn, cy_conn, cw_conn, ch_conn = conn['bbox']
                        if (cx >= cx_conn - bbox_tolerance and cx <= cx_conn + cw_conn + bbox_tolerance and
                            cy >= cy_conn - bbox_tolerance and cy <= cy_conn + ch_conn + bbox_tolerance):
                            is_overlapping_label = True
                            break
                
                # Check against label keywords (for non-parenthesized values)
                if not is_overlapping_label and not is_parenthesized:
                    for lx, ly, lw, lh in label_positions:
                        if (cx >= lx - 5 and cx <= lx + lw + 5 and
                            cy >= ly - 5 and cy <= ly + lh + 5):
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
    
    # Smart deduplication: cluster by proximity, pick best value
    common_lengths = {25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300}
    lengths = []
    used = set()
    
    for i, cand in enumerate(candidates):
        if i in used:
            continue
        
        # Find nearby detections within 60px
        cluster_indices = [i]
        cluster_x, cluster_y = cand['bbox'][0], cand['bbox'][1]
        
        for j in range(i + 1, len(candidates)):
            if j not in used:
                x2, y2 = candidates[j]['bbox'][0], candidates[j]['bbox'][1]
                dist = ((cluster_x - x2) ** 2 + (cluster_y - y2) ** 2) ** 0.5
                if dist < 15:
                    cluster_indices.append(j)
                    used.add(j)
        
        # Pick best detection from cluster
        best_idx = i
        best_score = -1
        
        for idx in cluster_indices:
            cand_item = candidates[idx]
            conf = cand_item['conf']
            val = cand_item['value']
            angle = cand_item['angle']
            
            in_common = 10000 if val in common_lengths else 0
            val_score = score_wire_length_value(val)
            norm_angle = min(angle % 180, 180 - (angle % 180))
            angle_score = 100 - min(norm_angle, 90 - abs(norm_angle - 90))
            
            total_score = in_common + conf * 3 + val_score * 2 + angle_score
            
            if total_score > best_score:
                best_score = total_score
                best_idx = idx
        
        best_cand = candidates[best_idx]
        xs = sorted([candidates[idx]['bbox'][0] for idx in cluster_indices])
        ys = sorted([candidates[idx]['bbox'][1] for idx in cluster_indices])
        avg_x = xs[len(xs)//2]
        avg_y = ys[len(ys)//2]
        
        lengths.append({
            'value': best_cand['value'],
            'bbox': (avg_x, avg_y, best_cand['bbox'][2], best_cand['bbox'][3]),
            'is_parenthesized': best_cand['is_parenthesized']
        })
        
    print(f"  [Debug] Length candidates before overlap filter: {pre_overlap_count}")
    print(f"  [Debug] Length candidates rejected by tape overlap: {tape_reject_count}")
    print(f"  [Debug] Length candidates after dedup: {len(lengths)}")
    
    return lengths
def normalize_length_token(txt):
    s = txt.strip().upper().replace(" ", "")
    s = re.sub(r'[\+\-\=\,\.\:\;\'\"]', '', s)
    trans = str.maketrans({
        'O': '0',
        'Q': '0',
        'I': '1',
        'L': '1',
        'S': '5',
        'B': '8',
        '[': '(',
        '{': '(',
        ']': ')',
        '}': ')',
    })
    return s.translate(trans)
