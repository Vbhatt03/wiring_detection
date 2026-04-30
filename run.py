#!/usr/bin/env python3
"""
Segment Detection Diagram Analyzer - Root Entry Point

Run the segment diagram detector from the root directory.

Usage:
    python run.py automotive_schematic.png
    python run.py /path/to/diagram.png --ocr-backend=paddle
    python run.py /path/to/diagram.png --ocr-backend=easyocr
    python run.py /path/to/diagram.png --ocr-backend=tesseract
    python run.py /path/to/diagram.png --skip=tapes,dimensions --ocr-backend=easyocr
    python run.py /path/to/diagram.png --debug-masks-only
"""

import sys
from src.run_detector import main, EXTRACT_FILTERS

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/1774639661620_image.png'
    
    extract_filters = EXTRACT_FILTERS.copy()
    use_legacy = False
    ocr_use_tiling = True
    ocr_backend = "paddle"  # Default backend
    debug_masks_only = False
    
    # Parse optional arguments
    for arg in sys.argv[2:]:
        if arg.startswith('--extract-only='):
            items = arg.split('=')[1].split(',')
            extract_filters = {k: k in items for k in extract_filters.keys()}
        elif arg.startswith('--skip='):
            items = arg.split('=')[1].split(',')
            for item in items:
                extract_filters[item] = False
        elif arg == '--legacy':
            use_legacy = True
        elif arg == '--ocr-no-tiling':
            ocr_use_tiling = False
        elif arg.startswith('--ocr-backend='):
            ocr_backend = arg.split('=')[1]
        elif arg == '--debug-masks-only':
            debug_masks_only = True
    
    main(image_path, extract_filters, use_legacy=use_legacy,
         ocr_use_tiling=ocr_use_tiling, ocr_backend=ocr_backend,
         debug_masks_only=debug_masks_only)
