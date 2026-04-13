"""Delphi connector detection module for wiring diagrams."""

import cv2
import numpy as np


def detect_delphi_connectors(img, gray, ocr_data, paddleocr_ok=True):
    """
    Detect Delphi connectors in the wiring diagram.
    
    Delphi connectors appear as small rectangular multi-pin symbols
    (drawn with parallel vertical lines inside a rectangle), plus OCR for "DELPHI".
    
    Strategy:
      1. Find "DELPHI" text via OCR and mark vicinity.
      2. Detect small dark rectangles with internal parallel lines (connector body).
    
    Args:
        img: Color image
        gray: Grayscale image
        ocr_data: List of OCR results from ocr_full()
        paddleocr_ok: Whether OCR is available
    
    Returns:
        List of detected connectors
    """
    found = []

    # — OCR: find DELPHI annotations —
    if paddleocr_ok:
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
