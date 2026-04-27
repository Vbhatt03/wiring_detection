#!/usr/bin/env python3
"""
Refactored Wiring Diagram Detector - Main Orchestrator

This script coordinates the detection of various elements in automotive wiring diagrams.
It imports from specialized detector modules for clean separation of concerns.

Usage:
    python -m src.run_detector automotive_schematic.png
    python run_detector.py automotive_schematic.png [--legacy] [--extract-only=tapes,wires]
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import json
import cv2
import numpy as np

# Import detector modules
from .detectors.ocr_detector import ocr_full, PADDLEOCR_OK, ocr_full_lengths
from .detectors.tape_detector import detect_tape_labels, TAPE_COLOR_BGR
from .detectors.connector_detector import detect_delphi_connectors
from .detectors.clip_detector import detect_blue_clips
from .detectors.length_detector import detect_wire_lengths
from .detectors.wire_detector import detect_wires, filter_wires_by_components
from .graph_builders.mask_tracer import trace_mask_connectivity

# Import graph builder modules
from .graph_builders.connectivity_builder import (
    build_connectivity_graph_heuristic,
    build_component_nodes,
    map_components_to_graph,
    assign_wire_properties,
    convert_to_legacy_format,
)

# Import visualization and reporting modules
from .visualization.visualizer import annotate, draw_label
from .visualization.reporter import print_report, generate_verification_table

# Import skeleton and connectivity modules
from .component_masker import create_wire_mask
# from .skeleton_graph import (
#     skeletonize_wire_mask,
#     extract_skeleton_graph,
#     filter_skeleton_graph,
# )
from .wiring_connectivity import (
    Component,
    Wire,
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
    'wires': True,
    'lengths': True,
    'clips': True,
}


def main(image_path='automotive_schematic.png', extract_filters=None, use_legacy=False, ocr_use_tiling=True):
    """
    Main detection pipeline orchestrator.
    
    Args:
        image_path: Path to wiring diagram image
        extract_filters: Dict specifying which elements to extract
        use_legacy: If True, use legacy heuristic pipeline instead of skeleton-based
        ocr_use_tiling: If False, OCR scans entire image without tiling. Default True.
    """
    if extract_filters is None:
        extract_filters = EXTRACT_FILTERS.copy()
    
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
    print("Running length OCR ...")
    import sys as _sys
    _sys.stdout.flush()
    if not ocr_use_tiling:
        print("  (tiling disabled - single pass)")
    ocr_lengths = ocr_full_lengths(gray, use_tiling=ocr_use_tiling)
    print(f"  {len(ocr_lengths)} length OCR tokens found")
    # Diagnostic: dump all purely numeric tokens from ocr_lengths
    import re as _re
    numeric_hits = [(t[0], t[1], t[2]) for t in ocr_lengths if _re.fullmatch(r'[\(\+]*\d{1,4}[\+\)]*', t[0].strip())]
    print(f"  [Debug] Numeric-pattern tokens in ocr_lengths: {len(numeric_hits)}")
    for tok in numeric_hits:
        print(f"    '{tok[0]}' at ({tok[1]}, {tok[2]})")
    # Save debug image showing where OCR found numeric tokens
    debug_ocr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    seen_positions = set()
    for tok in ocr_lengths:
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
        tapes = detect_tape_labels(img, gray, ocr_data + list(ocr_lengths))
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

    lengths = []
    if extract_filters.get('lengths', True):
        print("Detecting wire-length annotations ...")
        lengths = detect_wire_lengths(ocr_lengths, tapes, connectors)
        print(f"  {len(lengths)} length annotations")

    # Phase 3: Wire Detection & Connectivity
    wires = []
    connectivity_graph = {'nodes': {}, 'edges': [], 'raw_edges': [], 'segment_tapes': {}}
    
    if extract_filters.get('wires', True):
        if use_legacy:
            print("Detecting wires (legacy heuristic) ...")
            wires = detect_wires(gray)
            print(f"  {len(wires)} raw wire segments detected")
            print("Applying component-anchored validation ...")
            wires = filter_wires_by_components(wires, tapes + connectors + clips, ocr_data, margin=50)
            print(f"  {len(wires)} validated wires after anchoring to components")
            print("Building connectivity list (legacy heuristic) ...")
            connectivity_graph = build_connectivity_graph_heuristic(
                tapes, connectors, clips, wires, lengths, img.shape, ocr_data
            )
        else:
            print("Creating wire mask ...")
            wire_mask = create_wire_mask(gray, img, connectors, clips, tapes, ocr_data, lengths)
            cv2.imwrite('debug_wire_mask.png', wire_mask)

            nodes_dict = build_component_nodes(connectors, clips, ocr_data, tapes)

            print("Tracing wire blobs ...")
            final_graph = trace_mask_connectivity(wire_mask, nodes_dict)

            print("Assigning wire properties ...")
            assign_wire_properties(final_graph, tapes, lengths)

            if final_graph.number_of_edges() == 0:
                print("    [WARN] Mask tracer produced no edges; falling back to legacy heuristic.")
                wires = detect_wires(gray)
                wires = filter_wires_by_components(wires, tapes + connectors + clips, ocr_data, margin=50)
                connectivity_graph = build_connectivity_graph_heuristic(
                    tapes, connectors, clips, wires, lengths, img.shape, ocr_data
                )
            else:
                wires, connectivity_graph = convert_to_legacy_format(final_graph)

            if final_graph.number_of_edges() == 0:
                print("    [WARN] Skeleton graph produced no component edges; falling back to legacy heuristic.")
                wires = detect_wires(gray)
                wires = filter_wires_by_components(wires, tapes + connectors + clips, ocr_data, margin=50)
                connectivity_graph = build_connectivity_graph_heuristic(
                    tapes, connectors, clips, wires, lengths, img.shape, ocr_data
                )
            else:
                wires, connectivity_graph = convert_to_legacy_format(final_graph)
    
    # Phase 4: Reporting
    print_report(tapes, connectors, wires, lengths, clips, connectivity_graph)

    if extract_filters.get('lengths', True):
        generate_verification_table(lengths, ocr_data, 
                                   title="Wire Length Extraction Verification")

    # Phase 5: Output
    annotated = annotate(img, tapes, connectors, wires,
                        lengths, clips, connectivity_graph, extract_filters)
    output_image_path = os.path.join(os.path.dirname(image_path) or '.', 'wiring_diagram_annotated.png')
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
        'edges': [
            {
                'tapes': e.get('wire_types', e.get('tapes', [])),
                'from': e['node_a'],
                'to': e['node_b'],
                'length_mm': e['length_mm'],
                'segment_count': e.get('segment_count', 1),
                'endpoint_1': [int(e['segments'][0]['p1'][0]), int(e['segments'][0]['p1'][1])] if e.get('segments') else [0, 0],
                'endpoint_2': [int(e['segments'][0]['p2'][0]), int(e['segments'][0]['p2'][1])] if e.get('segments') else [0, 0]
            }
            for e in connectivity_graph['edges']
        ]
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
