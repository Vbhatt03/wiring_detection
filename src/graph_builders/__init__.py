"""Graph building modules for segment diagram connectivity analysis.

This package contains modules for building and processing connectivity graphs:
- connectivity_builder: Main connectivity graph construction with heuristic and skeleton-based methods
"""

from .connectivity_builder import (
    merge_segments_by_component_pair,
    build_connectivity_graph_heuristic,
    classify_segment,
    build_component_nodes,
    map_components_to_graph,
    assign_segment_properties,
    convert_to_legacy_format,
)

__all__ = [
    'merge_segments_by_component_pair',
    'build_connectivity_graph_heuristic',
    'classify_segment',
    'build_component_nodes',
    'map_components_to_graph',
    'assign_segment_properties',
    'convert_to_legacy_format',
]
