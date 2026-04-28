"""OCR text extraction module for segment diagrams.

Supports multiple OCR backends: PaddleOCR (default), EasyOCR, Tesseract.
Each backend has per-tuned parameters for optimal performance on diagrams.

Usage:
    from src.detectors.ocr_detector import ocr_full, set_ocr_backend
    set_ocr_backend("easyocr")  # Switch to EasyOCR
    results = ocr_full(gray_image)
"""

import numpy as np
import cv2
import sys as _sys

import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# ==============================================================================
# Backend Management
# ==============================================================================

_OCR_BACKEND = "paddle"  # "paddle" | "easyocr" | "tesseract"
_ocr_models = {}         # Cache for lazy-loaded models

def set_ocr_backend(name: str):
    """Switch active OCR backend. Clears cached model.
    
    Args:
        name: One of "paddle", "easyocr", "tesseract"
    
    Raises:
        AssertionError: If name is not a valid backend
    """
    global _OCR_BACKEND, _ocr_models
    assert name in ("paddle", "easyocr", "tesseract"), \
        f"Invalid backend '{name}'. Must be one of: paddle, easyocr, tesseract"
    _OCR_BACKEND = name
    # Don't clear cache entirely—only clear models we won't use
    print(f"  [OCR] Backend switched to: {name}")
    _sys.stdout.flush()

def get_ocr_backend() -> str:
    """Get current active OCR backend."""
    return _OCR_BACKEND

def _get_paddle_model():
    """Lazy load PaddleOCR model."""
    if "paddle" not in _ocr_models:
        try:
            from paddleocr import PaddleOCR
            import logging
            logging.getLogger("ppocr").setLevel(logging.WARNING)
            _ocr_models["paddle"] = PaddleOCR(use_angle_cls=True, lang='en')
        except ImportError as e:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr") from e
    return _ocr_models["paddle"]

def _get_easyocr_model():
    """Lazy load EasyOCR model."""
    if "easyocr" not in _ocr_models:
        try:
            import easyocr
            _ocr_models["easyocr"] = easyocr.Reader(['en'], gpu=False)
        except ImportError as e:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr") from e
    return _ocr_models["easyocr"]

def _get_tesseract_installed():
    """Check if Tesseract is installed (returns module, not model instance)."""
    try:
        import pytesseract
        return pytesseract
    except ImportError as e:
        raise ImportError("pytesseract not installed. Install with: pip install pytesseract") from e

# Check which backends are available at module load time (WITHOUT loading them)
_available_backends = set()

# Check PaddleOCR availability without loading
try:
    import paddleocr
    _available_backends.add("paddle")
    PADDLEOCR_OK = True
except ImportError:
    PADDLEOCR_OK = False

# Check EasyOCR availability without loading
try:
    import easyocr
    _available_backends.add("easyocr")
except ImportError:
    pass

# Check Tesseract availability without loading
try:
    import pytesseract
    _available_backends.add("tesseract")
except ImportError:
    pass

# Backward compatibility: OCR_OK is True if any backend is available
OCR_OK = len(_available_backends) > 0

if not OCR_OK:
    import warnings
    warnings.warn("No OCR backend available. Install paddleocr, easyocr, or pytesseract.")

# ==============================================================================
# PaddleOCR Backend Implementation
# ==============================================================================

def _ocr_full_paddle(gray):
    """PaddleOCR implementation of ocr_full."""
    model = _get_paddle_model()
    results = []
    
    if len(gray.shape) == 2:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = gray
        
    result = model.ocr(bgr)
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
        
        # Calculate angle from first two points
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        if angle_deg < 0:
            angle_deg += 360
            
        results.append((text, x, y, w, h, angle_deg, conf))
        
    return results

def _ocr_upscaled_paddle(gray, scale=3):
    """PaddleOCR implementation of ocr_upscaled."""
    model = _get_paddle_model()
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if len(upscaled.shape) == 2:
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
        
    result = model.ocr(upscaled)
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

def _ocr_region_paddle(gray, x1, y1, x2, y2):
    """PaddleOCR implementation of ocr_region."""
    model = _get_paddle_model()
    
    if x2 <= x1 or y2 <= y1:
        return ""
        
    crop = gray[y1:y2, x1:x2]
    if len(crop.shape) == 2:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = crop
        
    result = model.ocr(crop_bgr)
    if not result or not result[0]:
        return ""
        
    texts = [res[1][0] for res in result[0] if res[1][0]]
    return " ".join(texts).strip()

def _ocr_full_dimensions_paddle(gray, use_tiling=True):
    """PaddleOCR implementation of ocr_full_dimensions.
    
    Two-pass strategy optimized for PaddleOCR's det_limit_side_len=960:
    - Pass 1 (0deg): 480px tiles × 2× = 960px (exact limit)
    - Pass 2 (rotated): 320px tiles × 2× = 640px (fits 45deg ~905px < 960px)
    """
    model = _get_paddle_model()
    
    if not use_tiling:
        print("  [progress] Single-pass OCR (tiling disabled)...")
        _sys.stdout.flush()
        return _ocr_full_paddle(gray)

    scale = 2.0
    h_orig, w_orig = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants_gray = [gray, clahe.apply(gray)]
    out = []

    def _ocr_tiles(var_gray, tile_size, overlap, angles):
        total_tiles = ((h_orig - 1) // tile_size + 1) * ((w_orig - 1) // tile_size + 1)
        y = 0
        tile_count = 0
        while y < h_orig:
            y_end = min(y + tile_size, h_orig)
            x = 0
            while x < w_orig:
                x_end = min(x + tile_size, w_orig)
                tile_count += 1

                tile = var_gray[y:y_end, x:x_end]
                tile_up = cv2.resize(tile, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
                th, tw = tile_up.shape[:2]
                center = (tw / 2.0, th / 2.0)

                for i, ang in enumerate(angles):
                    if i == 0 and y == 0 and x == 0:
                        angle_list_str = ', '.join(str(a) for a in angles)
                        print(f"  [progress] Processing {len(angles)} angles: [{angle_list_str}] ({total_tiles} tiles each)")
                        _sys.stdout.flush()
                    elif i > 0 and y == 0 and x == 0:
                        print(f"  [progress]   ...angle {ang}° ({tile_count}/{total_tiles} tiles)...")
                        _sys.stdout.flush()
                    elif tile_count % 5 == 0 and i == 0:
                        print(f"  [progress]     tile {tile_count}/{total_tiles}...")
                        _sys.stdout.flush()
                    
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

                    result = model.ocr(rotated, cls=True)
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
        print("  [progress] Pass 1/2 - scanning horizontal text...")
        _sys.stdout.flush()
        _ocr_tiles(var_gray, tile_size=480, overlap=40, angles=[0])
        print(f"  [progress] Pass 1 complete: {len(out)} tokens found")
        _sys.stdout.flush()
        
        print("  [progress] Pass 2/2 - scanning rotated text (11 angles × ~25 tiles)...")
        _sys.stdout.flush()
        _ocr_tiles(var_gray, tile_size=320, overlap=30, angles=[30,45,60,75, 90,115,130, 270, 315,345,330])
        print(f"  [progress] Pass 2 complete: {len(out)} total tokens extracted")
        _sys.stdout.flush()

    return out

# ==============================================================================
# EasyOCR Backend Implementation
# ==============================================================================

def _ocr_full_easyocr(gray):
    """EasyOCR implementation of ocr_full.
    
    EasyOCR has no rotation step limit and handles angles natively.
    Min confidence threshold tuned higher (30) to reduce junk detections.
    """
    reader = _get_easyocr_model()
    results = []
    
    if len(gray.shape) == 2:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = gray
        
    # readtext returns: [([x1,y1], [x2,y1], [x2,y2], [x1,y2]), text, conf]
    ocr_results = reader.readtext(bgr, detail=1)
    if not ocr_results:
        return results
        
    for (bbox, text, conf) in ocr_results:
        text = text.strip()
        conf_int = int(conf * 100)
        
        # EasyOCR min confidence: 30 (higher than Paddle to filter noise)
        if not text or conf_int < 30:
            continue
            
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)
        
        # Calculate angle from first two points (same as Paddle)
        dx = bbox[1][0] - bbox[0][0]
        dy = bbox[1][1] - bbox[0][1]
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        if angle_deg < 0:
            angle_deg += 360
            
        results.append((text, x, y, w, h, angle_deg, conf_int))
        
    return results

def _ocr_upscaled_easyocr(gray, scale=2):
    """EasyOCR implementation of ocr_upscaled.
    
    Tuned for 2× upscaling (vs 3× for Paddle/Tesseract).
    Min confidence: 30.
    """
    reader = _get_easyocr_model()
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if len(upscaled.shape) == 2:
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
        
    ocr_results = reader.readtext(upscaled, detail=1)
    results = []
    if not ocr_results:
        return results
        
    for (bbox, text, conf) in ocr_results:
        text = text.strip()
        conf_int = int(conf * 100)
        
        if not text or conf_int < 30:
            continue
            
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
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

def _ocr_region_easyocr(gray, x1, y1, x2, y2):
    """EasyOCR implementation of ocr_region."""
    reader = _get_easyocr_model()
    
    if x2 <= x1 or y2 <= y1:
        return ""
        
    crop = gray[y1:y2, x1:x2]
    if len(crop.shape) == 2:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    else:
        crop_bgr = crop
        
    ocr_results = reader.readtext(crop_bgr, detail=1)
    if not ocr_results:
        return ""
        
    texts = [text for (_, text, conf) in ocr_results if text.strip() and conf > 0.3]
    return " ".join(texts).strip()

def _ocr_full_dimensions_easyocr(gray, use_tiling=True):
    """EasyOCR implementation of ocr_full_dimensions.
    
    Single-pass tiling (no rotation step, EasyOCR handles angles natively).
    Tile size: 1000px (no det_limit_side_len constraint like Paddle).
    Min confidence: 30.
    """
    reader = _get_easyocr_model()
    
    if not use_tiling:
        print("  [progress] Single-pass OCR (tiling disabled)...")
        _sys.stdout.flush()
        return _ocr_full_easyocr(gray)

    scale = 2.0
    h_orig, w_orig = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants_gray = [gray, clahe.apply(gray)]
    out = []

    print("  [progress] Single-pass tiling (EasyOCR - no rotation passes needed)...")
    _sys.stdout.flush()
    
    for var_gray in variants_gray:
        tile_size = 1000
        overlap = 50
        
        y = 0
        while y < h_orig:
            y_end = min(y + tile_size, h_orig)
            x = 0
            while x < w_orig:
                x_end = min(x + tile_size, w_orig)
                
                tile = var_gray[y:y_end, x:x_end]
                tile_up = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
                if len(tile_up.shape) == 2:
                    tile_up = cv2.cvtColor(tile_up, cv2.COLOR_GRAY2BGR)
                
                ocr_results = reader.readtext(tile_up, detail=1)
                
                for (bbox, text, conf) in ocr_results:
                    text = text.strip()
                    conf_int = int(conf * 100)
                    
                    if not text or conf_int < 30:
                        continue
                    
                    x_coords = [float(p[0]) for p in bbox]
                    y_coords = [float(p[1]) for p in bbox]
                    x_u = min(x_coords)
                    y_u = min(y_coords)
                    w_u = max(x_coords) - x_u
                    h_u = max(y_coords) - y_u
                    
                    orig_x = int(x_u / scale) + x
                    orig_y = int(y_u / scale) + y
                    orig_w = max(1, int(w_u / scale))
                    orig_h = max(1, int(h_u / scale))
                    
                    dx = bbox[1][0] - bbox[0][0]
                    dy = bbox[1][1] - bbox[0][1]
                    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
                    if angle_deg < 0:
                        angle_deg += 360.0
                    
                    out.append((text, orig_x, orig_y, orig_w, orig_h, angle_deg, conf_int))
                
                x = x_end if x_end == w_orig else x_end - overlap
            y = y_end if y_end == h_orig else y_end - overlap
    
    print(f"  [progress] Tiling complete: {len(out)} total tokens extracted")
    _sys.stdout.flush()
    return out

# ==============================================================================
# Tesseract Backend Implementation
# ==============================================================================

def _ocr_full_tesseract(gray):
    """Tesseract implementation of ocr_full.
    
    Tesseract is good for dense text with high confidence threshold (50+).
    Angles are always 0 (axis-aligned boxes only).
    Uses PSM 11 (sparse text) + OEM 3 (best recognition).
    """
    pytesseract = _get_tesseract_installed()
    
    # Upscale to help Tesseract (trained on ~300 DPI)
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    scale = 3.0
    
    config = '--psm 11 --oem 3'
    data = pytesseract.image_to_data(upscaled, config=config, output_type=pytesseract.Output.DATAFRAME)
    
    results = []
    for idx, row in data.iterrows():
        conf = row['conf']
        text = str(row['text']).strip()
        
        # Skip non-text rows and low confidence
        if conf < 0 or conf < 50 or not text:
            continue
        
        # Scale coordinates back to original image
        left = int(row['left'] / scale)
        top = int(row['top'] / scale)
        width = int(row['width'] / scale)
        height = int(row['height'] / scale)
        
        # Tesseract doesn't provide rotated boxes, so angle = 0
        results.append((text, left, top, width, height, 0.0, int(conf)))
    
    return results

def _ocr_upscaled_tesseract(gray, scale=3):
    """Tesseract implementation of ocr_upscaled.
    
    Tesseract needs 3× upscaling (vs 2× for others).
    """
    pytesseract = _get_tesseract_installed()
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    config = '--psm 11 --oem 3'
    data = pytesseract.image_to_data(upscaled, config=config, output_type=pytesseract.Output.DATAFRAME)
    
    results = []
    for idx, row in data.iterrows():
        conf = row['conf']
        text = str(row['text']).strip()
        
        if conf < 0 or conf < 50 or not text:
            continue
        
        left = int(row['left'] / scale)
        top = int(row['top'] / scale)
        width = int(row['width'] / scale)
        height = int(row['height'] / scale)
        
        results.append((text, left, top, width, height))
    
    return results

def _ocr_region_tesseract(gray, x1, y1, x2, y2):
    """Tesseract implementation of ocr_region."""
    pytesseract = _get_tesseract_installed()
    
    if x2 <= x1 or y2 <= y1:
        return ""
    
    crop = gray[y1:y2, x1:x2]
    
    config = '--psm 11 --oem 3'
    text = pytesseract.image_to_string(crop, config=config)
    return text.strip()

def _ocr_full_dimensions_tesseract(gray, use_tiling=True):
    """Tesseract implementation of ocr_full_dimensions.
    
    For Tesseract, dimension extraction is same as ocr_full (no special tiling).
    Tesseract is less rotation-sensitive, so skip rotation passes entirely.
    """
    if not use_tiling:
        print("  [progress] Single-pass OCR (tiling disabled)...")
        _sys.stdout.flush()
        return _ocr_full_tesseract(gray)
    
    # For Tesseract, tiling doesn't provide significant benefit
    # Just use single-pass full image OCR
    print("  [progress] Single-pass OCR (Tesseract - no rotation passes)...")
    _sys.stdout.flush()
    return _ocr_full_tesseract(gray)

# ==============================================================================
# Public Interface (Backend-Agnostic)
# ==============================================================================

def ocr_upscaled(gray, scale=3):
    """Run OCR on an upscaled version of the image to catch small text.
    
    Returns list of (text, x, y, w, h) in original image coordinates.
    Backend-agnostic dispatcher that routes to appropriate implementation.
    
    Args:
        gray: Grayscale input image
        scale: Upscale factor (default 3 for Paddle/Tesseract, 2 for EasyOCR)
    """
    if not OCR_OK:
        return []
    
    # Adjust scale default based on backend
    if _OCR_BACKEND == "easyocr" and scale == 3:
        scale = 2  # EasyOCR prefers 2× upscaling
    
    if _OCR_BACKEND == "paddle":
        return _ocr_upscaled_paddle(gray, scale)
    elif _OCR_BACKEND == "easyocr":
        return _ocr_upscaled_easyocr(gray, scale)
    else:  # tesseract
        return _ocr_upscaled_tesseract(gray, scale)

def ocr_full(gray):
    """Return list of (text, x, y, w, h, angle, confidence) for every detected word.
    
    Backend-agnostic dispatcher that routes to appropriate implementation.
    Returns consistent format across all backends.
    """
    if not OCR_OK:
        return []
    
    if _OCR_BACKEND == "paddle":
        return _ocr_full_paddle(gray)
    elif _OCR_BACKEND == "easyocr":
        return _ocr_full_easyocr(gray)
    else:  # tesseract
        return _ocr_full_tesseract(gray)

def ocr_region(gray, x1, y1, x2, y2):
    """OCR a bounding-box crop.
    
    Backend-agnostic dispatcher that routes to appropriate implementation.
    """
    if not OCR_OK:
        return ""
    
    if _OCR_BACKEND == "paddle":
        return _ocr_region_paddle(gray, x1, y1, x2, y2)
    elif _OCR_BACKEND == "easyocr":
        return _ocr_region_easyocr(gray, x1, y1, x2, y2)
    else:  # tesseract
        return _ocr_region_tesseract(gray, x1, y1, x2, y2)

def ocr_full_dimensions(gray, use_tiling=True):
    """OCR pass tuned for numeric segment-dimension annotations.
    
    Backend-agnostic dispatcher that routes to appropriate implementation.
    Each backend has its own tuning strategy:
    - PaddleOCR: Two-pass rotation (480/320px tiles × 2× scale)
    - EasyOCR: Single-pass tiling (1000px tiles, no rotation passes needed)
    - Tesseract: Full image (no tiling, no rotation passes)
    
    Args:
        gray: Grayscale image array
        use_tiling: If False, scans entire image without tiling. Default True.
    """
    if not OCR_OK:
        return []
    
    if _OCR_BACKEND == "paddle":
        return _ocr_full_dimensions_paddle(gray, use_tiling)
    elif _OCR_BACKEND == "easyocr":
        return _ocr_full_dimensions_easyocr(gray, use_tiling)
    else:  # tesseract
        return _ocr_full_dimensions_tesseract(gray, use_tiling)