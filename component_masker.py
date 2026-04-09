"""Component masking and binary wire-mask generation utilities.

Only masks the specific components from the FINAL output annotation
(connectors, clips, tape-label boxes).  Everything else — text, connector
detail drawings, dimension arrows — is left in the binary and simply forms
small disconnected skeleton fragments that the graph-mapping step ignores
by preferring the largest connected component.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple, Optional

import cv2
import numpy as np


def _as_int_bbox(bbox: Sequence[float]) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def _clip_rect(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    return max(0, x), max(0, y), min(img_w, x + w), min(img_h, y + h)


def _erase_rect(binary: np.ndarray, bbox: Sequence[float], margin: int) -> None:
    """Set pixels within bbox ± margin to 0 on a binary mask."""
    h_img, w_img = binary.shape[:2]
    x, y, bw, bh = _as_int_bbox(bbox)
    x1, y1, x2, y2 = _clip_rect(x - margin, y - margin,
                                    bw + 2 * margin, bh + 2 * margin,
                                    w_img, h_img)
    if x2 > x1 and y2 > y1:
        binary[y1:y2, x1:x2] = 0


def create_wire_mask(
    gray: np.ndarray,
    img_color: np.ndarray,
    connectors: Iterable[Dict],
    clips: Iterable[Dict],
    tapes: Iterable[Dict],
    ocr_data: Iterable[Tuple],
    lengths: Optional[Iterable[Dict]] = None,
) -> np.ndarray:
    """Create a binary mask where wire pixels = 255 and background = 0.

    Only the detected output-annotation components are erased:
      • Connector bodies  (shrunk 2 px so edge-touching wires survive)
      • Clip circles       (tight radius)
      • Tape-label boxes   (exact bbox)
      • Blue / yellow colour highlights

    Phase-2 cleanup:
      • Removes small convex triangular arrowheads from dimension graphics
      • Removes endpoint regions around non-parenthesized length annotations
    """
    if lengths is None:
        lengths = []

    # 1. Binarise – Otsu on lightly blurred grayscale
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # 2. Remove coloured highlights (blue clips, yellow tape backgrounds)
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    binary[cv2.inRange(hsv, (95, 60, 50), (135, 255, 255)) > 0] = 0
    binary[cv2.inRange(hsv, (20, 40, 60), (35, 255, 255)) > 0] = 0

    # 3. Erase ONLY the annotated-output components
    for conn in connectors:
        bbox = conn.get("bbox")
        if bbox is not None:
            _erase_rect(binary, bbox, margin=2)

    for clip in clips:
        center = clip.get("center")
        radius = int(clip.get("radius", 12))
        if center is not None:
            cv2.circle(binary, (int(center[0]), int(center[1])), radius + 3, 0, -1)

    for tape in tapes:
        bbox = tape.get("bbox")
        if bbox is not None:
            _erase_rect(binary, bbox, margin=2)

    # 4. Remove small convex triangular arrowheads (dimension endpoints)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 350:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=3, closed=True)
        vtx = len(approx)
        if vtx < 3 or vtx > 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        cv2.drawContours(binary, [approx], -1, 0, thickness=-1)

    # 5. Remove non-parenthesized length endpoints (dimension-line labels)
    for ln in lengths:
        if ln.get("is_parenthesized", False):
            continue
        bbox = ln.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        cv2.circle(binary, (cx, cy), 20, 0, -1)

    # 6. Gentle morphological close to bridge dash-dot gaps
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=2)

    return binary
