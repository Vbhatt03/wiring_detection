"""OCR text extraction module for wiring diagrams."""

import numpy as np
import cv2

import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

try:
    from paddleocr import PaddleOCR
    import logging
    logging.getLogger("ppocr").setLevel(logging.WARNING)
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    PADDLEOCR_OK = True
except ImportError:
    PADDLEOCR_OK = False

def ocr_upscaled(gray, scale=3):
    """Run OCR on an upscaled version of the image to catch small text.
    
    Returns list of (text, x, y, w, h) in original image coordinates.
    """
    if not PADDLEOCR_OK:
        return []
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if len(upscaled.shape) == 2:
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
        
    result = ocr_model.ocr(upscaled)
    results = []
    if not result or not result[0]:
        return results
        
    for res in result[0]:
        box = res[0]
        text = res[1][0].strip()
        conf = res[1][1] * 100
        
        if not text or conf < 20:
            continue
            
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x_box = min(x_coords)
        y_box = min(y_coords)
        w_box = max(x_coords) - x_box
        h_box = max(y_coords) - y_box
        
        x = int(x_box / scale)
        y = int(y_box / scale)
        w = int(w_box / scale)
        h = int(h_box / scale)
        
        results.append((text, x, y, w, h))
    return results
def ocr_full(gray):
    """Return list of (text, x, y, w, h, angle, confidence) for every detected word."""
    if not PADDLEOCR_OK:
        return []
        
    results = []
    
    if len(gray.shape) == 2:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = gray
        
    result = ocr_model.ocr(bgr)
    if not result or not result[0]:
        return results
        
    for res in result[0]:
        box = res[0]
        text = res[1][0].strip()
        conf = int(res[1][1] * 100)
        
        if not text or conf < 20:
            continue
            
        x_coords = [float(p[0]) for p in box]
        y_coords = [float(p[1]) for p in box]
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)
        
        # Calculate angle
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        if angle_deg < 0:
            angle_deg += 360
            
        results.append((text, x, y, w, h, angle_deg, conf))
        
    return results


def ocr_region(gray, x1, y1, x2, y2):
    """OCR a bounding-box crop."""
    if not PADDLEOCR_OK:
        return ""
        
    # PaddleOCR may fail if region is too small
    if x2 <= x1 or y2 <= y1:
        return ""
        
    crop = gray[y1:y2, x1:x2]
    if len(crop.shape) == 2:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = crop
        
    result = ocr_model.ocr(crop_bgr)
    if not result or not result[0]:
        return ""
        
    texts = [res[1][0] for res in result[0] if res[1][0]]
    return " ".join(texts).strip()
def ocr_full_lengths(gray):
    """OCR pass tuned for numeric wire-length annotations using image tiling.

    Two-pass strategy:
    - Pass 1 (0deg): 480px tiles upscaled 2x = 960px — fits det_limit_side_len exactly.
    - Pass 2 (rotated): 320px tiles upscaled 2x = 640px — at 45deg diagonal ~905px < 960px.
    Both passes avoid PaddleOCR's internal downscaling that makes small numbers invisible.
    """
    if not PADDLEOCR_OK:
        return []

    scale = 2.0
    h_orig, w_orig = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants_gray = [gray, clahe.apply(gray)]
    out = []

    def _ocr_tiles(var_gray, tile_size, overlap, angles):
        y = 0
        while y < h_orig:
            y_end = min(y + tile_size, h_orig)
            x = 0
            while x < w_orig:
                x_end = min(x + tile_size, w_orig)

                tile = var_gray[y:y_end, x:x_end]
                tile_up = cv2.resize(tile, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
                th, tw = tile_up.shape[:2]
                center = (tw / 2.0, th / 2.0)

                for ang in angles:
                    if ang == 0:
                        rotated = cv2.cvtColor(tile_up, cv2.COLOR_GRAY2BGR)
                        M_inv = None
                    else:
                        M = cv2.getRotationMatrix2D(center, ang, 1.0)
                        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
                        nW = int((th * sin) + (tw * cos))
                        nH = int((th * cos) + (tw * sin))
                        M[0, 2] += (nW / 2.0) - center[0]
                        M[1, 2] += (nH / 2.0) - center[1]
                        rot_gray = cv2.warpAffine(tile_up, M, (nW, nH),
                                                  flags=cv2.INTER_CUBIC,
                                                  borderMode=cv2.BORDER_REPLICATE)
                        rotated = cv2.cvtColor(rot_gray, cv2.COLOR_GRAY2BGR)
                        M_inv = cv2.invertAffineTransform(M)

                    result = ocr_model.ocr(rotated, cls=True)
                    if not result or not result[0]:
                        continue

                    for res in result[0]:
                        box  = res[0]
                        text = res[1][0].strip()
                        conf = int(res[1][1] * 100)
                        if not text or conf < 8:
                            continue

                        if M_inv is not None:
                            box = [
                                (M_inv @ np.array([float(p[0]), float(p[1]), 1.0]))[:2].tolist()
                                for p in box
                            ]

                        x_coords = [float(p[0]) for p in box]
                        y_coords = [float(p[1]) for p in box]
                        x_u = min(x_coords)
                        y_u = min(y_coords)
                        w_u = max(x_coords) - x_u
                        h_u = max(y_coords) - y_u

                        orig_x = int(x_u / scale) + x
                        orig_y = int(y_u / scale) + y
                        orig_w = max(1, int(w_u / scale))
                        orig_h = max(1, int(h_u / scale))

                        dx = box[1][0] - box[0][0]
                        dy = box[1][1] - box[0][1]
                        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
                        if angle_deg < 0:
                            angle_deg += 360.0

                        out.append((text, orig_x, orig_y, orig_w, orig_h,
                                    angle_deg, conf))

                x = x_end if x_end == w_orig else x_end - overlap
            y = y_end if y_end == h_orig else y_end - overlap

    for var_gray in variants_gray:
        # Pass 1: horizontal text — 480px tiles, 0deg only
        _ocr_tiles(var_gray, tile_size=480, overlap=40, angles=[0])
        # Pass 2: angled text — 320px tiles (640px upscaled, ~905px at 45deg < 960px limit)
        _ocr_tiles(var_gray, tile_size=320, overlap=30, angles=[30,45,60,75, 90,115,130, 270, 315,345,330])

    return out