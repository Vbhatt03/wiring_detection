#!/usr/bin/env python3
"""
Convenience script to extract ONLY wire lengths from a wiring diagram.
This demonstrates using the new extraction filters feature.

Usage:
    python extract_lengths_only.py <image_path>
    python extract_lengths_only.py  # uses default path
"""

import sys
from wiring_diagram_detector import main, EXTRACT_FILTERS

if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/1774639661620_image.png'
    
    # Extract ONLY wire lengths - skip everything else for faster processing
    lengths_only_filters = {
        'tapes': False,
        'connectors': False,
        'wires': False,
        'lengths': True,      # ← Only extract wire lengths
        'clips': False
    }
    
    print("="*70)
    print("WIRE LENGTH EXTRACTION - SIMPLIFIED MODE")
    print("="*70)
    result = main(image_path, lengths_only_filters)
    
    print("\n" + "="*70)
    print("Done! Check the annotated image and verification table above.")
    print("="*70)
