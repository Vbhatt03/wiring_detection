#!/usr/bin/env python3
"""
Connectivity Graph Builder for Wiring Diagrams

Builds a graph connecting wire segments to components (connectors, clips, tapes).
Uses proximity-based matching to link wire endpoints to physical components.
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Component:
    """Represents a physical component in the diagram"""
    id: str
    type: str  # "tape", "connector", "clip"
    position: Tuple[int, int]
    label: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Wire:
    """Represents a wire segment in the diagram"""
    id: str
    endpoints: Tuple[Tuple[int, int], Tuple[int, int]]  # (p1, p2)
    dimension_mm: Optional[float] = None
    confidence: Optional[float] = None
    tape_types: Optional[List[str]] = None


@dataclass
class Edge:
    """Represents a connection between two components via a wire"""
    wire_id: str
    from_component: Optional[Dict]
    to_component: Optional[Dict]
    dimension_mm: Optional[float] = None
    tape_types: Optional[List[str]] = None
    confidence: Optional[float] = None
    status: str = "unverified"  # unverified, correct, incorrect, manual_review


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Manhattan distance between two points"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def find_closest_component(
    point: Tuple[int, int],
    components: List[Component],
    max_distance: float = 50,
    distance_metric: str = "euclidean"
) -> Optional[Component]:
    """Find the closest component to a given point within max_distance"""
    
    distance_fn = euclidean_distance if distance_metric == "euclidean" else manhattan_distance
    
    closest_component = None
    min_distance = max_distance
    
    for component in components:
        dist = distance_fn(point, component.position)
        if dist < min_distance:
            min_distance = dist
            closest_component = component
    
    return closest_component


def build_connectivity_graph(
    components: List[Component],
    wires: List[Wire],
    proximity_threshold: float = 50,
    distance_metric: str = "euclidean"
) -> Dict:
    """
    Build connectivity graph connecting wires to components.
    
    Args:
        components: List of Component objects (connectors, clips, tapes)
        wires: List of Wire objects
        proximity_threshold: Max distance for endpoint-component matching (pixels)
        distance_metric: "euclidean" or "manhattan"
    
    Returns:
        Dict with 'edges', 'nodes', 'orphans', and 'statistics'
    """
    edges = []
    orphans = []
    nodes_dict = {}
    
    # Build nodes dictionary
    for component in components:
        nodes_dict[component.id] = {
            'id': component.id,
            'type': component.type,
            'position': component.position,
            'label': component.label,
            'confidence': component.confidence
        }
    
    # Process each wire
    for wire in wires:
        p1, p2 = wire.endpoints
        
        # Find closest component to each endpoint
        from_component = find_closest_component(
            p1, components, 
            max_distance=proximity_threshold,
            distance_metric=distance_metric
        )
        
        to_component = find_closest_component(
            p2, components,
            max_distance=proximity_threshold,
            distance_metric=distance_metric
        )
        
        # Create edge if both endpoints connected
        if from_component and to_component:
            edge = {
                'wire_id': wire.id,
                'from': {
                    'id': from_component.id,
                    'type': from_component.type,
                    'position': from_component.position,
                    'label': from_component.label
                },
                'to': {
                    'id': to_component.id,
                    'type': to_component.type,
                    'position': to_component.position,
                    'label': to_component.label
                },
                'dimension_mm': wire.dimension_mm,
                'tape_types': wire.tape_types,
                'confidence': wire.confidence,
                'status': 'unverified'
            }
            edges.append(edge)
        
        # Track orphaned wires (not connected to endpoints)
        elif from_component is None or to_component is None:
            orphan_info = {
                'wire_id': wire.id,
                'type': 'wire_endpoint_not_connected',
                'midpoint': [(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2],
                'from_connected': from_component is not None,
                'to_connected': to_component is not None,
                'from_component': from_component.id if from_component else None,
                'to_component': to_component.id if to_component else None,
                'suggests': [
                    'increase proximity_threshold',
                    'improve component detection',
                    'manually verify wire placement'
                ]
            }
            orphans.append(orphan_info)
    
    # Calculate statistics
    stats = {
        'total_wires': len(wires),
        'connected_edges': len(edges),
        'orphaned_wires': len(orphans),
        'connection_rate': (len(edges) / len(wires) * 100) if wires else 0,
        'total_components': len(components),
        'components_used': len(set([c for e in edges for c in [e['from']['id'], e['to']['id']]])),
        'components_unused': len(components) - len(set([c for e in edges for c in [e['from']['id'], e['to']['id']]]))
    }
    
    return {
        'nodes': nodes_dict,
        'edges': edges,
        'orphans': orphans,
        'statistics': stats
    }


def save_graph_to_json(graph: Dict, output_path: str = "connectivity_graph.json") -> None:
    """Save connectivity graph to JSON file"""
    
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for numpy types"""
        def default(self, obj):
            # Handle numpy types
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle standard types
            elif isinstance(obj, tuple):
                return list(obj)
            try:
                return super().default(obj)
            except TypeError:
                return str(obj)
    
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Connectivity graph saved to {output_path}")


def print_connectivity_report(graph: Dict) -> None:
    """Print human-readable connectivity report"""
    stats = graph.get('statistics', {})
    
    print("\n" + "="*80)
    print("CONNECTIVITY GRAPH REPORT")
    print("="*80)
    
    print(f"\nStatistics:")
    print(f"  Total wires detected:        {stats.get('total_wires', 0)}")
    print(f"  Successfully connected:      {stats.get('connected_edges', 0)}")
    print(f"  Orphaned (unconnected):      {stats.get('orphaned_wires', 0)}")
    print(f"  Connection rate:             {stats.get('connection_rate', 0):.1f}%")
    print(f"\nComponents:")
    print(f"  Total detected:              {stats.get('total_components', 0)}")
    print(f"  Used in connections:         {stats.get('components_used', 0)}")
    print(f"  Unused:                      {stats.get('components_unused', 0)}")
    
    # Print edges summary
    edges = graph.get('edges', [])
    if edges:
        print(f"\nEdges ({len(edges)} found):")
        for i, edge in enumerate(edges[:10], 1):  # Show first 10
            from_label = edge['from'].get('label', edge['from']['id'])
            to_label = edge['to'].get('label', edge['to']['id'])
            dimension = edge.get('dimension_mm', '—')
            tapes = '+'.join(edge.get('tape_types', [])) or '—'
            print(f"  [{i}] {from_label:20} → {to_label:20} | {tapes:15} | {dimension} mm")
        
        if len(edges) > 10:
            print(f"  ... and {len(edges) - 10} more edges")
    
    # Print orphaned wires summary
    orphans = graph.get('orphans', [])
    if orphans:
        print(f"\nOrphaned Wires ({len(orphans)} found):")
        for i, orphan in enumerate(orphans[:5], 1):  # Show first 5
            from_ok = "✓" if orphan.get('from_connected') else "✗"
            to_ok = "✓" if orphan.get('to_connected') else "✗"
            print(f"  [{i}] {orphan['wire_id']:20} | From: {from_ok} | To: {to_ok}")
        
        if len(orphans) > 5:
            print(f"  ... and {len(orphans) - 5} more orphaned wires")
    
    print("\n" + "="*80)


def validate_graph(graph: Dict) -> Dict:
    """Validate graph integrity and suggest improvements"""
    issues = []
    warnings = []
    
    stats = graph.get('statistics', {})
    
    # Check connection rate
    if stats.get('connection_rate', 0) < 50:
        warnings.append(
            f"Low connection rate ({stats.get('connection_rate', 0):.1f}%). "
            "Consider increasing proximity_threshold or improving component detection."
        )
    
    # Check for unused components
    if stats.get('components_unused', 0) > 0:
        warnings.append(
            f"{stats.get('components_unused', 0)} components not used in any connections. "
            "May indicate detection or positioning issues."
        )
    
    # Check for orphaned wires
    if stats.get('orphaned_wires', 0) > stats.get('total_wires', 0) * 0.3:
        issues.append(
            f"High orphan rate ({stats.get('orphaned_wires', 0)}/{stats.get('total_wires', 0)}). "
            "Many wires not connecting to components."
        )
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }
