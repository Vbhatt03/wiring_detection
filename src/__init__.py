"""
Segment Detection - Segment Diagram Analysis Package

This package contains modules for detecting and analyzing automotive segment diagrams.

Subpackages:
- detectors/: Element detection modules (OCR, tapes, connectors, clips, dimensions, segments)
- graph_builders/: Connectivity graph construction and analysis
- visualization/: Image annotation and reporting utilities
"""

__version__ = "1.0.0"
__author__ = "Segment Detection Team"

# Import detector modules
from .detectors import (
    ocr_full,
    ocr_region,
    set_ocr_backend,
    get_ocr_backend,
    detect_tape_labels,
    detect_delphi_connectors,
    detect_blue_clips,
    detect_segment_dimensions,
    detect_segments,
    filter_segments_by_components,
    detect_components,
    TAPE_PATTERNS,
    TAPE_COLOR_BGR,
    DIMENSION_PATTERN,
)

# Import graph builder modules
from .graph_builders import (
    merge_segments_by_component_pair,
    build_connectivity_graph_heuristic,
    classify_segment,
    build_component_nodes,
    map_components_to_graph,
    assign_segment_properties,
    convert_to_legacy_format,
)

# Import visualization and reporting modules
from .visualization import (
    annotate,
    draw_label,
    print_report,
    generate_verification_table,
)

# Import other core modules
from .component_masker import create_segment_mask
from .skeleton_graph import (
    skeletonize_segment_mask,
    extract_skeleton_graph,
    filter_skeleton_graph,
)
from .segment_connectivity import (
    Component,
    Segment,
    Segment,
    build_connectivity_graph,
    save_graph_to_json,
    print_connectivity_report,
)

# Import main orchestrator function and utilities
from .run_detector import main, load, EXTRACT_FILTERS

__all__ = [
    # Detector functions
    'ocr_full',
    'ocr_region',
    'set_ocr_backend',
    'get_ocr_backend',
    'detect_tape_labels',
    'detect_delphi_connectors',
    'detect_blue_clips',
    'detect_segment_dimensions',
    'detect_segments',
    'filter_segments_by_components',
    'detect_components',
    # Graph builder functions
    'merge_segments_by_component_pair',
    'build_connectivity_graph_heuristic',
    'classify_segment',
    'build_component_nodes',
    'map_components_to_graph',
    'assign_segment_properties',
    'convert_to_legacy_format',
    'trace_mask_connectivity',
    # Visualization and reporting
    'annotate',
    'draw_label',
    'print_report',
    'generate_verification_table',
    # Core utilities
    'create_segment_mask',
    'skeletonize_segment_mask',
    'extract_skeleton_graph',
    'filter_skeleton_graph',
    'load',
    'main',
    # Constants
    'EXTRACT_FILTERS',
    'TAPE_PATTERNS',
    'TAPE_COLOR_BGR',
    'DIMENSION_PATTERN',
    # Legacy support
    'Component',
    'Segment',
    'Segment',
    'build_connectivity_graph',
    'save_graph_to_json',
    'print_connectivity_report',
]
