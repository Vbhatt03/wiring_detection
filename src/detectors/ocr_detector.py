"""OCR text extraction module for wiring diagrams."""

import numpy as np
import cv2

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False


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
