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
    """OCR pass tuned for numeric wire-length annotations using combinatorials."""
    if not PADDLEOCR_OK:
        return []

    out = []

    def get_variants(img):
        variants = []
        variants.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        variants.append(cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR))
        return variants

    h_orig, w_orig = gray.shape[:2]
    scale = max(2.0, min(4.0, w_orig / 500.0))
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    for variant_img in get_variants(up):
        h, w = variant_img.shape[:2]
        center = (w / 2.0, h / 2.0)
        
        for ang in [0, 45, 90, 270, 315]:
            if ang == 0:
                rotated_img = variant_img
            else:
                M = cv2.getRotationMatrix2D(center, ang, 1.0)
                cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                M[0, 2] += (nW / 2.0) - center[0]
                M[1, 2] += (nH / 2.0) - center[1]
                
                rotated_img = cv2.warpAffine(variant_img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
            result = ocr_model.ocr(rotated_img, cls=True)
            if not result or not result[0]:
                continue
                
            for res in result[0]:
                box = res[0]
                text = res[1][0].strip()
                conf = int(res[1][1] * 100)
                
                if not text or conf < 8:
                    continue
                    
                if ang != 0:
                    M_inv = cv2.invertAffineTransform(M)
                    new_box = []
                    for p in box:
                        pt = np.array([float(p[0]), float(p[1]), 1.0])
                        new_pt = M_inv.dot(pt)
                        new_box.append([new_pt[0], new_pt[1]])
                    box = new_box
                    
                x_coords = [float(p[0]) for p in box]
                y_coords = [float(p[1]) for p in box]
                x_u = min(x_coords)
                y_u = min(y_coords)
                w_u = max(x_coords) - x_u
                h_u = max(y_coords) - y_u
                
                x_scale = int(x_u / scale)
                y_scale = int(y_u / scale)
                w_norm = max(1, int(w_u / scale))
                h_norm = max(1, int(h_u / scale))
                
                dx = box[1][0] - box[0][0]
                dy = box[1][1] - box[0][1]
                angle_deg = float(np.degrees(np.arctan2(dy, dx)))
                if angle_deg < 0:
                    angle_deg += 360.0
                    
                out.append((text, x_scale, y_scale, w_norm, h_norm, angle_deg, conf))
            
    return out