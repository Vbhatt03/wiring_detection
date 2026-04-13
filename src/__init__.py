"""
Wire Detection - Wiring Diagram Analysis Package

This package contains modules for detecting and analyzing automotive wiring diagrams.

Subpackages:
- detectors/: Element detection modules (OCR, tapes, connectors, clips, lengths, wires)
- graph_builders/: Connectivity graph construction and analysis
- visualization/: Image annotation and reporting utilities
"""

__version__ = "1.0.0"
__author__ = "Wire Detection Team"

# Import detector modules
from .detectors import (
    ocr_full,
    ocr_region,
    detect_tape_labels,
    detect_delphi_connectors,
    detect_blue_clips,
    detect_wire_lengths,
    detect_wires,
    filter_wires_by_components,
    detect_components,
    TAPE_PATTERNS,
    TAPE_COLOR_BGR,
    LENGTH_PATTERN,
)

# Import graph builder modules
from .graph_builders import (
    merge_edges_by_component_pair,
    build_connectivity_graph_heuristic,
    classify_wire,
    build_component_nodes,
    map_components_to_graph,
    assign_wire_properties,
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
from .component_masker import create_wire_mask
from .skeleton_graph import (
    skeletonize_wire_mask,
    extract_skeleton_graph,
    filter_skeleton_graph,
)
from .wiring_connectivity import (
    Component,
    Wire,
    Edge,
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
    'detect_tape_labels',
    'detect_delphi_connectors',
    'detect_blue_clips',
    'detect_wire_lengths',
    'detect_wires',
    'filter_wires_by_components',
    'detect_components',
    # Graph builder functions
    'merge_edges_by_component_pair',
    'build_connectivity_graph_heuristic',
    'classify_wire',
    'build_component_nodes',
    'map_components_to_graph',
    'assign_wire_properties',
    'convert_to_legacy_format',
    # Visualization and reporting
    'annotate',
    'draw_label',
    'print_report',
    'generate_verification_table',
    # Core utilities
    'create_wire_mask',
    'skeletonize_wire_mask',
    'extract_skeleton_graph',
    'filter_skeleton_graph',
    'load',
    'main',
    # Constants
    'EXTRACT_FILTERS',
    'TAPE_PATTERNS',
    'TAPE_COLOR_BGR',
    'LENGTH_PATTERN',
    # Legacy support
    'Component',
    'Wire',
    'Edge',
    'build_connectivity_graph',
    'save_graph_to_json',
    'print_connectivity_report',
]
