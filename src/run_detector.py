#!/usr/bin/env python3
"""
Refactored Segment Diagram Detector - Main Orchestrator

This script coordinates the detection of various elements in automotive segment diagrams.
It imports from specialized detector modules for clean separation of concerns.

Usage:
    python -m src.run_detector automotive_schematic.png
    python run_detector.py automotive_schematic.png [--legacy] [--extract-only=tapes,segments]
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import json
import cv2
import numpy as np

# Import detector modules
from .detectors.ocr_detector import (
    ocr_full, OCR_OK, PADDLEOCR_OK, ocr_full_dimensions,
    set_ocr_backend, get_ocr_backend
)
from .detectors.tape_detector import detect_tape_labels, TAPE_COLOR_BGR
from .detectors.connector_detector import detect_delphi_connectors
from .detectors.clip_detector import detect_blue_clips
from .detectors.dimension_detector import detect_segment_dimensions
from .detectors.segment_detector import detect_segments, filter_segments_by_components
from .graph_builders.mask_tracer import trace_mask_connectivity

# Import graph builder modules
from .graph_builders.connectivity_builder import (
    build_connectivity_graph_heuristic,
    build_component_nodes,
    map_components_to_graph,
    assign_segment_properties,
    convert_to_legacy_format,
)

# Import visualization and reporting modules
from .visualization.visualizer import annotate, draw_label
from .visualization.reporter import print_report

# Import skeleton and connectivity modules
from .component_masker import create_segment_mask
# from .skeleton_graph import (
#     skeletonize_segment_mask,
#     extract_skeleton_graph,
#     filter_skeleton_graph,
# )
from .segment_connectivity import (
    Component,
    Segment,
    build_connectivity_graph,
    save_graph_to_json,
    print_connectivity_report,
)


def load(path):
    """Load image from file path."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


EXTRACT_FILTERS = {
    'tapes': True,
    'connectors': True,
    'segments': True,
    'dimensions': True,
    'clips': True,
}


def main(image_path='automotive_schematic.png', extract_filters=None, use_legacy=False,
         ocr_use_tiling=True, ocr_backend="paddle"):
    """
    Main detection pipeline orchestrator.
    
    Args:
        image_path: Path to segment diagram image
        extract_filters: Dict specifying which elements to extract
        use_legacy: If True, use legacy heuristic pipeline instead of skeleton-based
        ocr_use_tiling: If False, OCR scans entire image without tiling. Default True.
        ocr_backend: OCR backend to use: "paddle" (default), "easyocr", or "tesseract"
    """
    if extract_filters is None:
        extract_filters = EXTRACT_FILTERS.copy()
    
    # Switch OCR backend if not default
    if ocr_backend != "paddle":
        set_ocr_backend(ocr_backend)
    
    print(f"Loading: {image_path}")
    img = load(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    print(f"Image size: {W}x{H}")
    print(f"Extract filters: {extract_filters}")

    # Phase 1: OCR
    print("Running OCR ...")
    ocr_data = ocr_full(gray)
    print(f"  {len(ocr_data)} text tokens found")
    print("Running dimension OCR ...")
    import sys as _sys
    _sys.stdout.flush()
    if not ocr_use_tiling:
        print("  (tiling disabled - single pass)")
    ocr_dimensions = ocr_full_dimensions(gray, use_tiling=ocr_use_tiling)
    print(f"  {len(ocr_dimensions)} dimension OCR tokens found")
    # Diagnostic: dump all purely numeric tokens from ocr_dimensions
    import re as _re
    numeric_hits = [(t[0], t[1], t[2]) for t in ocr_dimensions if _re.fullmatch(r'[\(\+]*\d{1,4}[\+\)]*', t[0].strip())]
    print(f"  [Debug] Numeric-pattern tokens in ocr_dimensions: {len(numeric_hits)}")
    for tok in numeric_hits:
        print(f"    '{tok[0]}' at ({tok[1]}, {tok[2]})")
    # Save debug image showing where OCR found numeric tokens
    debug_ocr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    seen_positions = set()
    for tok in ocr_dimensions:
        if _re.fullmatch(r'[\(\+]*\d{1,4}[\+\)]*', tok[0].strip()):
            pos_key = (tok[1] // 5, tok[2] // 5)  # bucket to 5px grid to dedup visually
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                x, y, w, h = tok[1], tok[2], tok[3], tok[4]
                cv2.rectangle(debug_ocr, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(debug_ocr, tok[0], (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite('debug_ocr_numerics.png', debug_ocr)
    print(f"  [Debug] Saved debug_ocr_numerics.png")

    # Phase 2: Element Detection
    tapes = []
    if extract_filters.get('tapes', True):
        print("Detecting tape labels ...")
        tapes = detect_tape_labels(img, gray, ocr_data + list(ocr_dimensions))
        print(f"  {len(tapes)} tape labels")

    connectors = []
    if extract_filters.get('connectors', True):
        print("Detecting Delphi connectors ...")
        connectors = detect_delphi_connectors(img, gray, ocr_data, paddleocr_ok=PADDLEOCR_OK)
        print(f"  {len(connectors)} connectors")

    clips = []
    if extract_filters.get('clips', True):
        print("Detecting blue clips ...")
        clips = detect_blue_clips(img, gray)
        print(f"  {len(clips)} blue clips")

    dimensions = []
    if extract_filters.get('dimensions', True):
        print("Detecting segment-dimension annotations ...")
        dimensions = detect_segment_dimensions(ocr_dimensions, tapes, connectors)
        print(f"  {len(dimensions)} dimension annotations")

    # Phase 3: Segment Detection & Connectivity
    segments = []
    connectivity_graph = {'nodes': {}, 'segments': [], 'raw_traces': [], 'trace_tapes': {}}
    
    if extract_filters.get('segments', True):
        if use_legacy:
            print("Detecting segments (legacy heuristic) ...")
            segments = detect_segments(gray)
            print(f"  {len(segments)} raw segment traces detected")
            print("Applying component-anchored validation ...")
            segments = filter_segments_by_components(segments, tapes + connectors + clips, ocr_data, margin=50)
            print(f"  {len(segments)} validated segments after anchoring to components")
            print("Building connectivity list (legacy heuristic) ...")
            connectivity_graph = build_connectivity_graph_heuristic(
                tapes, connectors, clips, segments, dimensions, img.shape, ocr_data
            )
        else:
            print("Creating segment mask ...")
            segment_mask = create_segment_mask(gray, img, connectors, clips, tapes, ocr_data, dimensions)
            cv2.imwrite('debug_segment_mask.png', segment_mask)

            nodes_dict = build_component_nodes(connectors, clips, ocr_data, tapes)

            print("Tracing segment blobs ...")
            final_graph = trace_mask_connectivity(segment_mask, nodes_dict)

            print("Assigning segment properties ...")
            assign_segment_properties(final_graph, tapes, dimensions)

            if final_graph.number_of_edges() == 0:
                print("    [WARN] Mask tracer produced no segments; falling back to legacy heuristic.")
                segments = detect_segments(gray)
                segments = filter_segments_by_components(segments, tapes + connectors + clips, ocr_data, margin=50)
                connectivity_graph = build_connectivity_graph_heuristic(
                    tapes, connectors, clips, segments, dimensions, img.shape, ocr_data
                )
            else:
                segments, connectivity_graph = convert_to_legacy_format(final_graph)

            if final_graph.number_of_edges() == 0:
                print("        [WARN] Skeleton graph produced no component segments; falling back to legacy heuristic.")
                segments = detect_segments(gray)
                segments = filter_segments_by_components(segments, tapes + connectors + clips, ocr_data, margin=50)
                connectivity_graph = build_connectivity_graph_heuristic(
                    tapes, connectors, clips, segments, dimensions, img.shape, ocr_data
                )
            else:
                segments, connectivity_graph = convert_to_legacy_format(final_graph)
    
    # Phase 4: Reporting
    print_report(tapes, connectors, segments, dimensions, clips, connectivity_graph)

    # if extract_filters.get('dimensions', True):
    #     generate_verification_table(dimensions, ocr_data, 
    #                                title="Segment Dimension Extraction Verification")

    # Phase 5: Output
    annotated = annotate(img, tapes, connectors, segments,
                        dimensions, clips, connectivity_graph, extract_filters)
    output_image_path = os.path.join(os.path.dirname(image_path) or '.', 'segment_diagram_annotated.png')
    cv2.imwrite(output_image_path, annotated)
    print(f"\nAnnotated image saved: {output_image_path}")

    # Save connectivity graph as JSON
    def convert_to_native(obj):
        """Convert numpy types to native Python types"""
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, (tuple, list)):
            return [convert_to_native(x) for x in obj]
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        return obj
    
    json_output = {
        'nodes': [
            {
                'id': nid,
                'x': int(ninfo['x']),
                'y': int(ninfo['y']),
                'label': ninfo['label'],
                'type': ninfo['type']
            }
            for nid, ninfo in connectivity_graph['nodes'].items()
        ],
        'segments': [
            {
                'tapes': e.get('segment_types', e.get('tapes', [])),
                'from': e['node_a'],
                'to': e['node_b'],
                'dimension_mm': e['dimension_mm'],
                'trace_count': e.get('trace_count', 1),
                'endpoint_1': [int(e['traces'][0]['p1'][0]), int(e['traces'][0]['p1'][1])] if e.get('traces') else [0, 0],
                'endpoint_2': [int(e['traces'][0]['p2'][0]), int(e['traces'][0]['p2'][1])] if e.get('traces') else [0, 0]
            }
            for e in connectivity_graph['segments']
        ],
        'routes': []
    }
    json_path = 'connectivity_graph.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Connectivity graph saved: {json_path}")

    return annotated


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'automotive_schematic.png'
    
    extract_filters = EXTRACT_FILTERS.copy()
    use_legacy = False
    
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
    
    main(image_path, extract_filters, use_legacy=use_legacy)
