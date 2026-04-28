"""Tape label detection module for segment diagrams."""

import re
import cv2
import numpy as np
from src.detectors.ocr_detector import ocr_region , ocr_upscaled

TAPE_PATTERNS = re.compile(
    r'(VT)\s*-\s*[A-Z]{1,2}', re.IGNORECASE)

TAPE_COLOR_BGR = {
    'VT-WH':  (  0,   0, 180),   # dark red
    'VT-BK':  (  0,   0, 180),   # dark red
    'VT-PK':  (  0,   0, 180),   # dark red
    # 'AT-BK':  (  0,   0, 180),   # dark red
    # 'COT-BK': (  0,   0, 180),   # dark red
    'DEFAULT':(  0,   0, 180),   # dark red
}


def detect_tape_labels(img, gray, ocr_data):
    """
    Detect tape labels using OCR text bounding boxes.
    Uses OCR results directly — no visual fallback (too many false positives).
    """
    found = []

    # Build a spatial index of all tokens for proximity search
    tokens = []
    for item in ocr_data:
        if len(item) >= 5:
            txt, tx, ty, tw, th = item[0], item[1], item[2], int(item[3]), int(item[4])
            tokens.append({'txt': txt.upper(), 'x': int(tx), 'y': int(ty), 'w': tw, 'h': th,
                           'cx': int(tx) + tw // 2, 'cy': int(ty) + th // 2})
    # Pass 0: OCR on upscaled image — catches small text missed by ocr_full()
    upscaled_tokens = []
    for item in ocr_upscaled(gray):
        txt, tx, ty, tw, th = item
        upscaled_tokens.append({'txt': txt.upper(), 'x': tx, 'y': ty, 'w': tw, 'h': th,
                                'cx': tx + tw // 2, 'cy': ty + th // 2})

    # Merge upscaled_tokens into tokens (deduplicate by position ~10px)
    for ut in upscaled_tokens:
        if not any(abs(ut['cx'] - t['cx']) < 10 and abs(ut['cy'] - t['cy']) < 10 for t in tokens):
            tokens.append(ut)
    vt_tokens = [t for t in tokens if re.match(r'^VT', t['txt'], re.IGNORECASE)]
    print(f"    [Tape Debug] VT-prefix tokens after Pass 0: {[(t['txt'], t['x'], t['y']) for t in vt_tokens]}")
    # Pass 1: direct single-token matches
    matched_positions = set()
    for t in tokens:
        m = TAPE_PATTERNS.search(t['txt'])
        if m:
            key = (t['x'], t['y'])
            if key not in matched_positions:
                matched_positions.add(key)
                found.append({
                    'label': m.group(0).upper().replace(" ", ""),
                    'bbox': (t['x'], t['y'], t['w'], t['h']),
                    'source': 'ocr_label'
                })

    # Pass 2: reconstruct split tokens (e.g. "VT" + "BK", "VT-" + "BK")
    # for i, t1 in enumerate(tokens):
    #     if not re.match(r'^VT$|^AT$|^VT-$|^AT-$', t1['txt'], re.IGNORECASE):
    #         continue
    #     # Look for a 2-letter suffix token nearby (within 60px horizontally, 15px vertically)
    #     for t2 in tokens:
    #         if t2 is t1:
    #             continue
    #         hdist = abs(t2['cx'] - (t1['x'] + t1['w']))  # horizontal gap
    #         vdist = abs(t2['cy'] - t1['cy'])
    #         if hdist < 60 and vdist < 15 and re.match(r'^[A-Z]{2}$', t2['txt'], re.IGNORECASE):
    #             combined = t1['txt'].rstrip('-') + '-' + t2['txt']
    #             m = TAPE_PATTERNS.search(combined.upper())
    #             if m:
    #                 # Use bounding box spanning both tokens
    #                 x = min(t1['x'], t2['x'])
    #                 y = min(t1['y'], t2['y'])
    #                 w = max(t1['x'] + t1['w'], t2['x'] + t2['w']) - x
    #                 h = max(t1['h'], t2['h'])
    #                 key = (x, y)
    #                 if key not in matched_positions:
    #                     matched_positions.add(key)
    #                     found.append({
    #                         'label': m.group(0).upper(),
    #                         'bbox': (x, y, w, h),
    #                         'source': 'ocr_reconstructed'
    #                     })
    #                     print(f"    [Tape Detection] Reconstructed: '{t1['txt']}' + '{t2['txt']}' → {m.group(0).upper()}")
    # Pass 3: re-OCR crops around any VT/AT prefix tokens using higher-quality settings
    for t1 in tokens:
        if not re.match(r'^VT$|^AT$|^VT-$|^AT-$', t1['txt'], re.IGNORECASE):
            continue
        # Expand crop rightward to capture the full "VT-BK" label
        pad = 5
        x1 = max(0, t1['x'] - pad)
        y1 = max(0, t1['y'] - pad)
        x2 = min(gray.shape[1], t1['x'] + t1['w'] + 60 + pad)  # 60px rightward
        y2 = min(gray.shape[0], t1['y'] + t1['h'] + pad)
        re_txt = ocr_region(gray, x1, y1, x2, y2).strip().upper()
        m = TAPE_PATTERNS.search(re_txt)
        if m:
            key = (t1['x'], t1['y'])
            if key not in matched_positions:
                matched_positions.add(key)
                found.append({
                    'label': m.group(0).upper().replace(" ", ""),
                    'bbox': (t1['x'], t1['y'], t1['w'], t1['h']),
                    'source': 'ocr_region'
                })
                print(f"    [Tape Detection] Region re-OCR: '{re_txt}' → {m.group(0).upper()}")


    # Proximity-based dedup: same label within 30px center → keep first
    deduped = []
    for tape in found:
        tx, ty, tw, th = tape['bbox']
        tcx, tcy = tx + tw // 2, ty + th // 2
        is_dup = any(
            tape['label'] == kept['label'] and
            abs(tcx - (kept['bbox'][0] + kept['bbox'][2] // 2)) < 30 and
            abs(tcy - (kept['bbox'][1] + kept['bbox'][3] // 2)) < 30
            for kept in deduped
        )
        if not is_dup:
            deduped.append(tape)

    print(f"    [Tape Detection] Found {len(deduped)} tape labels")
    return deduped