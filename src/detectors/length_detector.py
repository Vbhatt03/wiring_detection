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
       horizontal gap is adaptive based on character height
       vertical center difference <= max height
    """
    ref_h = _median_token_height(ocr_data)
    h_gap_max = max(18, int(ref_h * 1.2))
    
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
                if -5 <= h_gap <= h_gap_max:
                    # vertical center diff
                    v_diff = abs((y_min + h_max/2.0) - t2['cy'])
                    if v_diff <= max(h_max, t2['h']):
                        # Guard: don't merge pure-digit with label keywords, or vice versa
                        cur_is_num = bool(re.match(r'^[\(\)]*\d+[\(\)]*$', current_txt.strip()))
                        t2_is_label = bool(LABEL_KEYWORDS.search(t2['txt'].upper()))
                        t2_is_num = bool(re.match(r'^[\(\)]*\d+[\(\)]*$', t2['txt'].strip()))
                        cur_is_label = bool(LABEL_KEYWORDS.search(current_txt.upper()))
                        t2_is_paren_only = t2['txt'].strip() in ('(', ')', '[', ']', '{', '}')
                        cur_is_paren_only = current_txt.strip() in ('(', ')', '[', ']', '{', '}')
                        # Block: digit ↔ label, or paren-only ↔ label
                        if (cur_is_num and t2_is_label) or (cur_is_label and t2_is_num) or (cur_is_paren_only and t2_is_label) or (cur_is_label and t2_is_paren_only):
                            continue
                        current_txt += t2['txt']
                        x2 = t2['x'] + t2['w']
                        h_max = max(h_max, t2['h'])
                        y_min = min(y_min, t2['y'])
                        conf_max = max(conf_max, t2['conf'])
                        used.add(j)
                        merged_something = True
            if not merged_something:
                break
                
        if re.search(r'[A-Z]', current_txt) and re.search(r'\d', current_txt) and len(current_txt) > 4:
            print(f"  [Debug] Merged suspect token: '{current_txt}' at x={x1}")
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
    Thresholds are adaptive to image dimension and OCR token size.
    
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
    
    # Compute adaptive thresholds based on typical OCR token height
    ref_h = _median_token_height(ocr_data)
    bbox_tol_paren = max(3,  int(ref_h * 0.25))
    bbox_tol_plain = max(6,  int(ref_h * 0.45))
    label_guard    = max(5,  int(ref_h * 0.40))
    dedup_radius   = max(15, int(ref_h * 1.5))
    
    merged_data = merge_token_fragments(ocr_data)
    
    # Skip label_positions check — it's overly protective and causes false negatives
    # Wire lengths appear on wires, not inside label bboxes, so checking against label positions
    # at this stage removes legitimate annotations more than it prevents false positives.
    # Tape and connector checks are sufficient filtering.
    
    # Print debug info after all variables are defined
    print(f"  [Debug] ref_h={ref_h:.1f}, h_gap_max={max(18, int(ref_h*1.2))}, dedup_radius={dedup_radius}")
    print(f"  [Debug] Skipping label_positions check (too many false negatives)")
    
    candidates = []
    pre_overlap_count = 0
    tape_reject_count = 0
    debug_rejected = []
    
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
            is_parenthesized = ('(' in clean) or (')' in clean)
            
            # Debug: track 20 and 105/10
            if val in (20, 55, 105, 10):
                print(f"    [Length Debug] Found txt='{txt}' clean='{clean}' val={val} at ({x},{y}) w={w} h={h}")
            
            # Reject tokens whose bounding box is too short to be real text —
            # wire dashes misread as '1'/'11' have h << ref_h
            if h < ref_h * 0.45:
                if val in (20, 55, 105, 10):
                    print(f"      → Rejected: h={h:.1f} < ref_h*0.45={ref_h*0.45:.1f}")
                continue
            # Reject wide flat bboxes — wire dashes read as '1'/'11' have w/h >> real digits
            # Real digits: w/h < 3.5 even for '111'. Wire dashes: w/h can be 5-15
            if h > 0 and (w / h) > 3.5:
                if val in (20, 55, 105, 10):
                    print(f"      → Rejected: w/h={w/h:.2f} > 3.5")
                continue
            if w*h <40:
                if val in (20, 55, 105, 10):
                    print(f"      → Rejected: area={w*h} < 40")
                continue
            # Selective filtering: parenthesized (0-3000) or non-parenthesized multi-digit (10-3000)
            if (is_parenthesized and 0 <= val <= 3000) or (not is_parenthesized and 10 <= val <= 3000):
                pre_overlap_count += 1
                is_overlapping_label = False
                bbox_tolerance = bbox_tol_paren if is_parenthesized else bbox_tol_plain
                cx = x + w / 2.0
                cy = y + h / 2.0
                
                # Check against tape labels — exempt parenthesized values (they belong inside tape regions)
                if not is_parenthesized:
                    for tape in tapes:
                        tx, ty, tw, th = tape['bbox']
                        if (cx >= tx and cx <= tx + tw and cy >= ty and cy <= ty + th):
                            is_overlapping_label = True
                            tape_reject_count += 1
                            if val in (20, 55, 105, 10):
                                print(f"      → Rejected by tape overlap at tape ({tx},{ty},{tw},{th})")
                            break
                
                # Check against connectors
                if not is_overlapping_label:
                    for conn in connectors:
                        cx_conn, cy_conn, cw_conn, ch_conn = conn['bbox']
                        if (cx >= cx_conn - bbox_tolerance and cx <= cx_conn + cw_conn + bbox_tolerance and
                            cy >= cy_conn - bbox_tolerance and cy <= cy_conn + ch_conn + bbox_tolerance):
                            is_overlapping_label = True
                            if val in (20, 55, 105, 10):
                                print(f"      → Rejected by connector overlap")
                            break
                
                if not is_overlapping_label:
                    if val in (20, 55, 105, 10):
                        print(f"      → ACCEPTED as candidate")
                    candidates.append({
                        'value': val,
                        'bbox': (x, y, w, h),
                        'is_parenthesized': is_parenthesized,
                        'angle': angle,
                        'conf': conf
                    })
            else:
                if val in (20, 55, 105, 10):
                    print(f"      → Rejected: value range check failed (is_paren={is_parenthesized})")
    
    # Smart deduplication: cluster by proximity, pick best value
    common_lengths = {25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300}
    lengths = []
    used = set()
    
    for i, cand in enumerate(candidates):
        if i in used:
            continue
        
        # Find nearby detections (adaptive clustering radius)
        cluster_indices = [i]
        cluster_x, cluster_y = cand['bbox'][0], cand['bbox'][1]
        
        for j in range(i + 1, len(candidates)):
            if j not in used:
                x2, y2 = candidates[j]['bbox'][0], candidates[j]['bbox'][1]
                dist = ((cluster_x - x2) ** 2 + (cluster_y - y2) ** 2) ** 0.5
                if dist < dedup_radius:
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
            is_paren = cand_item['is_parenthesized']
            
            in_common = 10000 if val in common_lengths else 0
            val_score = score_wire_length_value(val)
            norm_angle = min(angle % 180, 180 - (angle % 180))
            angle_score = 100 - min(norm_angle, 90 - abs(norm_angle - 90))
            # Strongly prefer parenthesized candidates — they carry more structural info
            paren_bonus = 5000 if is_paren else 0
            
            total_score = in_common + paren_bonus + conf * 3 + val_score * 2 + angle_score
            
            if total_score > best_score:
                best_score = total_score
                best_idx = idx
        
        best_cand = candidates[best_idx]
        xs = sorted([candidates[idx]['bbox'][0] for idx in cluster_indices])
        ys = sorted([candidates[idx]['bbox'][1] for idx in cluster_indices])
        avg_x = xs[len(xs)//2]
        avg_y = ys[len(ys)//2]
        
        if best_cand['value'] in (20, 55, 105, 10):
            print(f"  [Dedup Debug] Cluster at ({cluster_x},{cluster_y}): candidates={[candidates[idx]['value'] for idx in cluster_indices]} → best={best_cand['value']} (score={best_score})")
        
        lengths.append({
            'value': best_cand['value'],
            'bbox': (avg_x, avg_y, best_cand['bbox'][2], best_cand['bbox'][3]),
            'is_parenthesized': best_cand['is_parenthesized']
        })
        
    print(f"  [Debug] Length candidates before overlap filter: {pre_overlap_count}")
    print(f"  [Debug] Length candidates rejected by tape overlap: {tape_reject_count}")
    print(f"  [Debug] Length candidates after dedup: {len(lengths)}")
    print(f"  [Debug] Final lengths after dedup: {[(l['value'], l['bbox']) for l in lengths if l['value'] in (20, 55, 105, 10)]}")
    
    # Post-dedup fix: add nearby parentheses to finalized lengths
    # Search in original ocr_data (not merged_data) to find parentheses that may have been merged away
    for length in lengths:
        lx, ly, lw, lh = length['bbox']
        lcx, lcy = lx + lw/2, ly + lh/2
        val = length['value']
        
        # Search for ( and ) tokens within proximity (25px default)
        found_paren_left = False
        found_paren_right = False
        
        for item in ocr_data:
            if len(item) >= 5:
                txt = item[0].strip()
                ix, iy, iw, ih = item[1], item[2], item[3], item[4]
                
                # Only match PURE paren tokens: mostly parens/brackets, not mixed text like "4)" or "LABEL(ref)"
                # A pure paren token should be > 50% parens/brackets
                paren_count = sum(1 for c in txt if c in '()[]{}')
                if len(txt) > 0 and paren_count > len(txt) * 0.5:
                    # Left paren: should be to the left of the number
                    if '(' in txt or '[' in txt or '{' in txt:
                        if ix + iw <= lx and abs(iy + ih/2 - lcy) < 20 and (lx - (ix + iw)) < 25:
                            found_paren_left = True
                            if val in (20, 55, 105, 10):
                                print(f"    [Paren Recovery] Found left paren '{txt}' at ({ix},{iy}) for {val}")
                    # Right paren: should be to the right of the number
                    elif ')' in txt or ']' in txt or '}' in txt:
                        if ix >= lx + lw and abs(iy + ih/2 - lcy) < 20 and (ix - (lx + lw)) < 25:
                            found_paren_right = True
                            if val in (20, 55, 105, 10):
                                print(f"    [Paren Recovery] Found right paren '{txt}' at ({ix},{iy}) for {val}")
        
        # Update annotation if parentheses found
        if found_paren_left or found_paren_right:
            if found_paren_left and found_paren_right:
                length['is_parenthesized'] = True
                if val in (20, 55, 105, 10):
                    print(f"    [Paren Recovery] Both parens found for {val} → added ()")
            elif found_paren_left:
                length['is_parenthesized'] = True
                if val in (20, 55, 105, 10):
                    print(f"    [Paren Recovery] Left paren only for {val}")
            elif found_paren_right:
                length['is_parenthesized'] = True
                if val in (20, 55, 105, 10):
                    print(f"    [Paren Recovery] Right paren only for {val}")
    
    # Post-dedup fix: merge adjacent digit-only tokens (e.g., "10" + "5" → "105")
    for i, length in enumerate(lengths):
        lx, ly, lw, lh = length['bbox']
        lcx, lcy = lx + lw/2, ly + lh/2
        val = length['value']
        
        # Look for digit-only tokens adjacent to this length
        for item in merged_data:
            if len(item) >= 5:
                txt = item[0].strip()
                ix, iy, iw, ih = item[1], item[2], item[3], item[4]
                
                # Must be pure digits (no parens)
                if not re.match(r'^\d+$', txt):
                    continue
                
                icx, icy = ix + iw/2, iy + ih/2
                
                # Check proximity and baseline alignment
                if abs(icy - lcy) < 12:  # same baseline
                    h_dist = min(abs((lx + lw) - ix), abs(ix + iw - lx))  # touching distance
                    if h_dist < 15:  # adjacent
                        adj_val = int(txt)
                        # Determine if it's to the left or right
                        if ix + iw <= lx:  # to the left
                            merged_val = int(str(adj_val) + str(val))
                            length['value'] = merged_val
                            print(f"  [Debug] Merged adjacent digits {adj_val}+{val}={merged_val}")
                        elif ix >= lx + lw:  # to the right
                            merged_val = int(str(val) + str(adj_val))
                            length['value'] = merged_val
                            print(f"  [Debug] Merged adjacent digits {val}+{adj_val}={merged_val}")
    
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
        '<': '(',
        '>': ')',
    })
    return s.translate(trans)
def _median_token_height(ocr_data):
    heights = [item[4] for item in ocr_data if len(item) >= 5 and item[4] > 2]
    if not heights:
        return 12.0
    return float(np.median(heights))

