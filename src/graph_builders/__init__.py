"""Graph building modules for wiring diagram connectivity analysis.

This package contains modules for building and processing connectivity graphs:
- connectivity_builder: Main connectivity graph construction with heuristic and skeleton-based methods
"""

from .connectivity_builder import (
    merge_edges_by_component_pair,
    build_connectivity_graph_heuristic,
    classify_wire,
    build_component_nodes,
    map_components_to_graph,
    assign_wire_properties,
    convert_to_legacy_format,
)

__all__ = [
    'merge_edges_by_component_pair',
    'build_connectivity_graph_heuristic',
    'classify_wire',
    'build_component_nodes',
    'map_components_to_graph',
    'assign_wire_properties',
    'convert_to_legacy_format',
]
