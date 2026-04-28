"""
Connectivity Graph Builder
===========================
Builds the final connectivity graph from segment segments and components.

Functions:
- merge_segments_by_component_pair: Group segments with same component pair
- build_connectivity_graph_heuristic: Legacy heuristic-based graph building
- classify_segment: Classify tape label to verbose segment type string
- build_component_nodes: Build node dictionary from detected components
- map_components_to_graph: Map named components to skeleton graph nodes
- assign_segment_properties: Assign segment type and length to skeleton edges
- convert_to_legacy_format: Convert NetworkX graph to legacy output format
- Helper functions: _point_to_segment_distance, _nearest_graph_node, etc.
"""

import re
import numpy as np
import networkx as nx


def merge_segments_by_component_pair(raw_traces, nodes_dict=None):
    """
    ═══ Phase 2: Component-to-Component Segment Merging ═══
    
    Groups all edges with the same (node_a, node_b) component pair
    and merges them into a single logical connection.
    
    Strategy:
    1. Group edges by (node_a, node_b) key
    2. For each group, collect all tape types and segment info
    3. Pick primary dimension from highest-confidence detection
    4. Return list of merged edges
    
    Args:
        raw_edges: List of traces, each with:
                   node_a, node_b, tapes, dimension_mm, p1, p2, snapped_a, snapped_b
        nodes_dict: Mapping of node_id to node info (used for junction detection)
    
    Returns:
        List of merged segments grouped by component pairs
    """
    if nodes_dict is None:
        nodes_dict = {}
    # Group edges by (node_a, node_b) key
    segment_groups = {}
    
    for trace in raw_traces:
        # Skip edges with missing component endpoints
        if trace['node_a'] is None or trace['node_b'] is None:
            continue
        
        # Create key: SORTED so A→B and B→A collapse to the same edge
        key = tuple(sorted([trace['node_a'], trace['node_b']]))
        
        if key not in segment_groups:
            segment_groups[key] = []
        segment_groups[key].append(trace)
    
    # Merge edges within each group
    merged_segments = []
    
    for (node_a, node_b), traces_in_group in segment_groups.items():
        # Skip all self-loops (including junctions — J20→J20 are always detection artifacts)
        if node_a == node_b:
            continue

        # Skip spurious clip↔clip edges (artifacts from overlapping blobs near the clips)
        type_a = nodes_dict.get(node_a, {}).get('type', '')
        type_b = nodes_dict.get(node_b, {}).get('type', '')
        if type_a == 'clip' and type_b == 'clip':
            continue
        # Collect all unique tapes across this group
        all_tapes = set()
        for trace in traces_in_group:
            all_tapes.update(trace['tapes'])
        
        # Pick primary dimension:
        # Prefer tape-anchor sources (more reliable) over segment-blob sources.
        # Among same source, prefer non-None values.
        primary_dimension = None
        # First pass: tape_anchor sources
        for trace in traces_in_group:
            if trace.get('source') == 'tape_anchor' and trace.get('dimension_mm') is not None:
                primary_dimension = trace['dimension_mm']
                break
        # Second pass: any source
        if primary_dimension is None:
            for trace in traces_in_group:
                if trace.get('dimension_mm') is not None:
                    primary_dimension = trace['dimension_mm']
                    break
        
        # Create merged edge
        merged_segment = {
            'node_a': node_a,
            'node_b': node_b,
            'segment_types': sorted(all_tapes),  # Unique tape types
            'dimension_mm': primary_dimension,
            'trace_count': len(traces_in_group),  # How many segments make up this connection
            'traces': traces_in_group,  # Keep reference to individual segments for detail
            'snapped': any(trace.get('snapped_a', False) or trace.get('snapped_b', False) 
                          for trace in traces_in_group)  # Flag if any endpoint was snapped
        }
        
        merged_segments.append(merged_segment)
    
    # Debug output
    print(f"    [Segment Merging] Raw traces: {len(raw_traces)} → Merged connections: {len(merged_segments)}")
    
    return merged_segments


def classify_segment(label):
    """Return verbose classification string for a segment label."""
    label = label.upper()
    if label in ('VT-BK',):
        return 'Solid black (VT-BK)'
    if label in ('VT-WH',):
        return 'White (VT-WH)'
    if label in ('VT-PK',):
        return 'Pink (VT-PK)'
    # if label in ('AT-BK',):
    #     return 'Black braided (AT-BK)'
    # if label.startswith('COT'):
    #     return 'Corrugated tube (COT-BK)'
    # if label.startswith('MLC'):
    #     return 'Multi-layer conduit (MLC)'
    return 'Unknown'


def build_component_nodes(connectors, clips, ocr_data, tapes):
    """Build component/junction node dictionary used for graph mapping."""
    nodes_dict = {}

    for i, conn in enumerate(connectors):
        bx, by, bw, bh = conn['bbox']
        cx, cy = int(bx + bw // 2), int(by + bh // 2)
        node_id = f"Connector-{i+1}"
        nodes_dict[node_id] = {
            'x': cx,
            'y': cy,
            'label': conn.get('note', conn.get('label', node_id)),
            'type': 'connector',
        }

    port_pattern = re.compile(r'^(X\d+|C\d+)$', re.IGNORECASE)
    for nid, info in list(nodes_dict.items()):
        if info['type'] != 'connector':
            continue
        cx, cy = info['x'], info['y']
        best = None
        best_dist = 120.0
        for item in ocr_data:
            if len(item) < 5:
                continue
            txt, ox, oy, ow, oh = item[0], item[1], item[2], item[3], item[4]
            txt_clean = str(txt).strip().upper()
            if not port_pattern.match(txt_clean):
                continue
            ocx, ocy = float(ox + ow / 2.0), float(oy + oh / 2.0)
            dist = ((ocx - cx) ** 2 + (ocy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = txt_clean
        if best:
            info['label'] = best

    for i, clip in enumerate(clips):
        node_id = f"Clip-{i+1}"
        nodes_dict[node_id] = {
            'x': int(clip['center'][0]),
            'y': int(clip['center'][1]),
            'label': f"Z2-TH{i+1}",
            'type': 'clip',
        }
    # Add tape labels as graph nodes.
    # Only accept valid tape label patterns: VT-*, AT-*, COT-*
    # Reject false positives like MLCO01VT, MLCO01, etc. (bundle/junction annotations)
    # Also reject any label containing '+' (OCR concatenation of multiple labels)
    tape_pattern = re.compile(r'^(VT|AT)[-_]?[A-Z0-9]+$', re.IGNORECASE)
    
    tape_counts = {}
    sorted_tapes = sorted(
        tapes,
        key=lambda tape: (
            tape['label'].upper(),
            tape['bbox'][1] + tape['bbox'][3] // 2,
            tape['bbox'][0] + tape['bbox'][2] // 2,
        ),
    )

    for tape in sorted_tapes:
        label = tape['label'].upper()
        
        # Skip tapes that contain '+' (OCR concatenation) or don't match the known tape label pattern
        if '+' in label or not tape_pattern.match(label):
            continue
        
        bx, by, bw, bh = tape['bbox']
        cx, cy = bx + bw // 2, by + bh // 2

        tape_counts[label] = tape_counts.get(label, 0) + 1
        node_id = f"Tape-{label}-{tape_counts[label]}"

        nodes_dict[node_id] = {
            'x': cx,
            'y': cy,
            'label': tape['label'],
            'type': 'tape'
        }

    junction_pattern = re.compile(r'^(J\d+|MLC\w+)$', re.IGNORECASE)
    for item in ocr_data:
        if len(item) < 5:
            continue
        txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
        txt_clean = re.sub(r'[^A-Z0-9]$', '', str(txt).strip().upper())
        if not junction_pattern.match(txt_clean):
            continue
        # Skip any MLC label that contains tape-type suffixes (VT, AT, COT) or '+' (OCR junk)
        if '+' in txt_clean or any(suffix in txt_clean for suffix in ['VT', 'AT', 'COT']):
            continue
        if txt_clean in nodes_dict:
            continue
        nodes_dict[txt_clean] = {
            'x': int(x + w // 2),
            'y': int(y + h // 2),
            'label': txt_clean,
            'type': 'junction',
        }

    return nodes_dict


def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculate distance from point (px, py) to line segment (x1,y1)→(x2,y2)."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5, (x1, y1), 0.0
    c2 = vx * vx + vy * vy
    if c2 <= 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5, (x1, y1), 0.0
    t = min(1.0, max(0.0, c1 / c2))
    proj = (x1 + t * vx, y1 + t * vy)
    dist = ((px - proj[0]) ** 2 + (py - proj[1]) ** 2) ** 0.5
    return dist, proj, t


def _nearest_graph_node(skeleton_graph, point):
    """Find nearest node in skeleton graph to a given point."""
    px, py = float(point[0]), float(point[1])
    best_node = None
    best_dist = float('inf')
    for n, attrs in skeleton_graph.nodes(data=True):
        dist = ((attrs['x'] - px) ** 2 + (attrs['y'] - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_node = n
    return best_node, best_dist


def _nearest_graph_edge(skeleton_graph, point):
    """Find nearest edge in skeleton graph to a given point."""
    px, py = float(point[0]), float(point[1])
    best = None
    best_dist = float('inf')
    for u, v, k, attrs in skeleton_graph.edges(keys=True, data=True):
        pts = attrs.get('path_pts', [])
        if len(pts) < 2:
            pts = [
                (int(skeleton_graph.nodes[u]['x']), int(skeleton_graph.nodes[u]['y'])),
                (int(skeleton_graph.nodes[v]['x']), int(skeleton_graph.nodes[v]['y'])),
            ]
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dist, proj, _ = _point_to_segment_distance(px, py, x1, y1, x2, y2)
            if dist < best_dist:
                best_dist = dist
                best = (u, v, k, proj)
    return best, best_dist


def _path_length_and_points(skeleton_graph, node_path):
    """Calculate total path length and collect all points along a path."""
    total = 0.0
    pts = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        edge_map = skeleton_graph.get_edge_data(u, v)
        if not edge_map:
            continue
        attrs = next(iter(edge_map.values()))
        total += float(attrs.get('path_length_px', 0.0))
        epts = attrs.get('path_pts', [])
        if i > 0 and epts:
            epts = epts[1:]
        pts.extend(epts)
    return total, pts


def _distance_point_to_polyline(point, polyline):
    """Calculate minimum distance from a point to a polyline."""
    if not polyline or len(polyline) < 2:
        return float('inf')
    px, py = float(point[0]), float(point[1])
    best = float('inf')
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        dist, _, _ = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        best = min(best, dist)
    return best


def map_components_to_graph(skeleton_graph, components_dict, max_snap_distance=180):
    """Map named components to nearest skeleton nodes, preferring the main segment CC.

    The skeleton contains the segment network (largest connected component) PLUS
    noise fragments from text, detail drawings, etc.  We always try to snap
    components to the largest CC first; only fall back to any CC if the main
    one has nothing within range.
    """
    g = skeleton_graph.copy()

    # ----- Identify the largest connected component (main segment network) -----
    ug = nx.Graph(g)  # undirected view for CC detection
    ccs = sorted(nx.connected_components(ug), key=len, reverse=True)
    main_cc = ccs[0] if ccs else set()
    print(f"    [CC] {len(ccs)} connected components; main CC has {len(main_cc)} nodes")

    def _nearest_in_set(point, node_set):
        px, py = float(point[0]), float(point[1])
        best_n, best_d = None, float('inf')
        for n in node_set:
            attrs = g.nodes[n]
            d = ((attrs['x'] - px) ** 2 + (attrs['y'] - py) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_n = n
        return best_n, best_d

    def _nearest_edge_in_set(point, node_set):
        px, py = float(point[0]), float(point[1])
        best, best_d = None, float('inf')
        for u, v, k, attrs in g.edges(keys=True, data=True):
            if u not in node_set and v not in node_set:
                continue
            pts = attrs.get('path_pts', [])
            if len(pts) < 2:
                pts = [
                    (int(g.nodes[u]['x']), int(g.nodes[u]['y'])),
                    (int(g.nodes[v]['x']), int(g.nodes[v]['y'])),
                ]
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                d, proj, _ = _point_to_segment_distance(px, py, x1, y1, x2, y2)
                if d < best_d:
                    best_d = d
                    best = (u, v, k, proj)
        return best, best_d

    component_to_skel = {}
    mapped_by_skel = {}

    for comp_id, comp in components_dict.items():
        point = (comp['x'], comp['y'])
        comp_type = comp.get('type', 'unknown')
        if comp_type == 'clip':
            snap_limit = min(max_snap_distance, 100)
        elif comp_type == 'connector':
            snap_limit = min(max_snap_distance, 160)
        elif comp_type == 'tape':
            snap_limit = min(max_snap_distance, 80)
        elif comp_type == 'junction':
            snap_limit = max(max_snap_distance, 200)
        else:
            snap_limit = max_snap_distance

        target_node = None

        # --- Try main CC first (node, then edge) ---
        main_node, main_dist = _nearest_in_set(point, main_cc)
        if main_node is not None and main_dist <= snap_limit:
            target_node = main_node
            print(f"    [SNAP] {comp_id} ({comp_type}) -> main-CC node {main_node} (dist={main_dist:.0f}px)")
        else:
            main_edge, main_edge_dist = _nearest_edge_in_set(point, main_cc)
            if main_edge is not None and main_edge_dist <= snap_limit:
                u, v, k, proj = main_edge
                new_id = f"snap_{comp_id}"
                while new_id in g.nodes:
                    new_id += "_"
                g.add_node(new_id, x=float(proj[0]), y=float(proj[1]),
                           pos=(float(proj[0]), float(proj[1])))
                old_attrs = g.get_edge_data(u, v)[k]
                old_len = float(old_attrs.get('path_length_px', 0.0))
                if g.has_edge(u, v, k):
                    g.remove_edge(u, v, k)
                g.add_edge(u, new_id, path_length_px=old_len / 2.0,
                           path_pts=old_attrs.get('path_pts', []))
                g.add_edge(new_id, v, path_length_px=old_len / 2.0,
                           path_pts=old_attrs.get('path_pts', []))
                main_cc.add(new_id)  # new node is part of main CC
                target_node = new_id
                print(f"    [SNAP] {comp_id} ({comp_type}) -> main-CC edge-split ({proj[0]:.0f},{proj[1]:.0f}) (dist={main_edge_dist:.0f}px)")

        # --- Fallback: any CC ---
        if target_node is None:
            any_node, any_dist = _nearest_graph_node(g, point)
            if any_node is not None and any_dist <= snap_limit:
                target_node = any_node
                print(f"    [SNAP] {comp_id} ({comp_type}) -> any-CC node {any_node} (dist={any_dist:.0f}px) [fallback]")
            else:
                print(f"    [WARN] Orphan {comp_id} ({comp_type}) at ({point[0]},{point[1]}) -- main-CC nearest={main_dist:.0f}px, any nearest={any_dist:.0f}px, limit={snap_limit}")

        if target_node is not None:
            component_to_skel[comp_id] = target_node
            mapped_by_skel.setdefault(target_node, []).append(comp_id)

    # ----- Build final component graph from skeleton paths -----
    final_graph = nx.Graph()
    for comp_id, comp in components_dict.items():
        final_graph.add_node(comp_id, **comp)

    mapped_components = list(component_to_skel.keys())

    # Build spatial lookup: for each component, its (x, y) position
    comp_positions = {
        cid: (components_dict[cid]['x'], components_dict[cid]['y'])
        for cid in mapped_components
    }

    PROXIMITY_THRESHOLD = 60  # px -- if an inner node is within this of another component, path is indirect

    for i in range(len(mapped_components)):
        for j in range(i + 1, len(mapped_components)):
            a = mapped_components[i]
            b = mapped_components[j]
            na, nb = component_to_skel[a], component_to_skel[b]
            if na == nb:
                continue
            try:
                node_path = nx.shortest_path(g, na, nb, weight='path_length_px')
            except nx.NetworkXNoPath:
                continue

            # Skip if the path passes NEAR any other mapped component
            # (not just through its exact skeleton node)
            inner = node_path[1:-1]
            passes_through_other = False
            for inner_n in inner:
                nx_, ny_ = g.nodes[inner_n]['x'], g.nodes[inner_n]['y']
                for other_id in mapped_components:
                    if other_id in (a, b):
                        continue
                    ox, oy = comp_positions[other_id]
                    dist = ((nx_ - ox) ** 2 + (ny_ - oy) ** 2) ** 0.5
                    if dist < PROXIMITY_THRESHOLD:
                        passes_through_other = True
                        break
                if passes_through_other:
                    break
            if passes_through_other:
                continue

            length_px, path_pts = _path_length_and_points(g, node_path)
            if length_px <= 0:
                continue
            final_graph.add_edge(
                a, b,
                path_pts=path_pts,
                path_length_px=length_px,
                segment_type=None,
                dimension_mm=None,
            )

    return final_graph


def assign_segment_properties(graph, tapes, dimensions):
    """Assign segment type and dimension using nearest tape and dimension annotations."""
    for u, v, attrs in graph.edges(data=True):
        pts = attrs.get('path_pts', [])

        tape_hits = []
        for tape in tapes:
            bx, by, bw, bh = tape['bbox']
            tc = (bx + bw / 2.0, by + bh / 2.0)
            d = _distance_point_to_polyline(tc, pts)
            if d <= 80:
                tape_hits.append((d, tape['label']))
        tape_hits.sort(key=lambda x: x[0])
        if tape_hits:
            attrs['segment_type'] = '+'.join(sorted(set([t[1] for t in tape_hits])))

        best_len = None
        best_dist = 80.0
        for ln in dimensions:
            lbx, lby, lbw, lbh = ln['bbox']
            lc = (lbx + lbw / 2.0, lby + lbh / 2.0)
            d = _distance_point_to_polyline(lc, pts)
            if d < best_dist:
                best_dist = d
                best_len = ln['value']
        attrs['dimension_mm'] = best_len


def convert_to_legacy_format(graph):
    """Convert NetworkX component graph to existing report/annotation structure."""
    physical_segments = []
    logical_segments = []

    nodes_dict = {}
    for nid, attrs in graph.nodes(data=True):
        nodes_dict[nid] = {
            'x': int(attrs.get('x', 0)),
            'y': int(attrs.get('y', 0)),
            'label': attrs.get('label', nid),
            'type': attrs.get('type', 'unknown'),
        }

    for i, (u, v, attrs) in enumerate(graph.edges(data=True), start=1):
        path_pts = attrs.get('path_pts', [])
        p1 = path_pts[0] if path_pts else (nodes_dict[u]['x'], nodes_dict[u]['y'])
        p2 = path_pts[-1] if path_pts else (nodes_dict[v]['x'], nodes_dict[v]['y'])
        physical_segments.append({
            'p1': (int(p1[0]), int(p1[1])),
            'p2': (int(p2[0]), int(p2[1])),
            'length_px': int(round(attrs.get('path_length_px', 0))),
            'trace_count': 1,
            'type': 'merged',
        })
        segment_types = []
        if attrs.get('segment_type'):
            segment_types = str(attrs.get('segment_type')).split('+')
        logical_segments.append({
            'node_a': u,
            'node_b': v,
            'segment_types': segment_types,
            'dimension_mm': attrs.get('dimension_mm'),
            'trace_count': 1,
            'traces': [{
                'p1': (int(p1[0]), int(p1[1])),
                'p2': (int(p2[0]), int(p2[1])),
            }],
            'snapped': False,
        })

    connectivity_graph = {
        'nodes': nodes_dict,
        'segments': logical_segments,
        'raw_traces': logical_segments,
        'trace_tapes': {},
    }
    return physical_segments, connectivity_graph


def build_connectivity_graph_heuristic(tape_labels, connectors, clips, segments, lengths, img_shape, ocr_data=None):
    """
    Build a connectivity graph by tracing segment segments (LEGACY HEURISTIC METHOD).
    
    Strategy:
    1. For each dash-dot segment, find which tape labels are on it
    2. Group tape labels that are on the same segment
    3. For each segment, find endpoints and map to actual nodes
    4. Build edges showing: Node A --[segment type, length]--> Node B
    5. Detect junction points from text labels (J20, J30, etc.)
    
    Args:
        tape_labels: Detected tape labels
        connectors: Detected Delphi connectors
        clips: Detected blue circular clips
        segments: Detected segment segments
        lengths: Detected length annotations
        img_shape: Image dimensions
        ocr_data: OCR text detections (for junction point extraction)
    
    Returns: dict with 'nodes' (list of nodes) and 'segments' (list of connections)
    """
    if ocr_data is None:
        ocr_data = []
    
    h, w = img_shape[:2]
    
    # Create node list from connectors and clips
    nodes_dict = {}  # id -> {cx, cy, label, type}
    
    for i, conn in enumerate(connectors):
        bx, by, bw, bh = conn['bbox']
        cx, cy = bx + bw//2, by + bh//2
        node_id = f"Connector-{i+1}"
        nodes_dict[node_id] = {
            'x': cx,
            'y': cy,
            'label': conn.get('note', conn['label']),
            'type': 'connector'
        }
    
    # ─── OCR-based connector renaming: X510, X508, X519, C1, C2 ───
    PORT_PATTERN = re.compile(r'^(X\d+|C\d+)$', re.IGNORECASE)
    OCR_RENAME_RADIUS = 120  # px around connector center to look for port labels
    for nid in list(nodes_dict.keys()):
        if nodes_dict[nid]['type'] != 'connector':
            continue
        cx, cy = nodes_dict[nid]['x'], nodes_dict[nid]['y']
        best_txt, best_dist = None, OCR_RENAME_RADIUS
        for item in ocr_data:
            if len(item) < 5:
                continue
            txt, ox, oy, ow, oh = item[0], item[1], item[2], item[3], item[4]
            txt_clean = txt.strip().upper()
            if not PORT_PATTERN.match(txt_clean):
                continue
            ocx, ocy = ox + ow // 2, oy + oh // 2
            dist = ((ocx - cx) ** 2 + (ocy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_txt = txt_clean
        if best_txt:
            nodes_dict[nid]['label'] = best_txt
            print(f"    [NodeRename] {nid} → {best_txt} (dist={best_dist:.0f}px)")

    for i, clip in enumerate(clips):
        node_id = f"Clip-{i+1}"
        nodes_dict[node_id] = {
            'x': clip['center'][0],
            'y': clip['center'][1],
            'label': f"Z2-TH{i+1}",
            'type': 'clip'
        }
        
    
    
    # ═══ Extract junction/intermediate points from OCR data ═══
    # Accept J\d+ (junctions) and MLC\w+ (conduit branches)
    junction_pattern = re.compile(r'^(J\d+|MLC\w+)$', re.IGNORECASE)
    junction_count = 0
    for item in ocr_data:
        if len(item) >= 5:
            txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
            txt_clean = re.sub(r'[^A-Z0-9]$', '', txt.strip().upper())  # strip trailing : . etc.
            if txt_clean in nodes_dict:
                continue
            if junction_pattern.match(txt_clean):
                # Skip any MLC label that contains tape-type suffixes (VT, AT, COT) or '+' (OCR junk)
                if '+' in txt_clean or any(suffix in txt_clean for suffix in ['VT', 'AT', 'COT']):
                    continue
                junction_count += 1
                nodes_dict[txt_clean] = {
                    'x': int(x + w // 2),
                    'y': int(y + h // 2),
                    'label': txt_clean,
                    'type': 'junction'
                }
    
    # # ─── Synthesise C2 node from AT-BK tape position if not shape-detected ───
    # # AT-BK tape sits right next to the C2 connector at the bottom-right.
    # # If no connector is within 200px of the AT-BK tape, add a synthetic C2 node.
    # connector_ids = [nid for nid, n in nodes_dict.items() if n['type'] == 'connector']
    # for tape in tape_labels:
    #     if tape['label'].upper() != 'AT-BK':
    #         continue
    #     tbx, tby, tbw, tbh = tape['bbox']
    #     tcx, tcy = tbx + tbw // 2, tby + tbh // 2
    #     nearby_conn = any(
    #         ((nodes_dict[nid]['x'] - tcx) ** 2 + (nodes_dict[nid]['y'] - tcy) ** 2) ** 0.5 < 200
    #         for nid in connector_ids
    #     )
    #     if not nearby_conn and 'C2' not in nodes_dict:
    #         # Place C2 to the LEFT of the AT-BK label (where the cross-hatch box is)
    #         nodes_dict['C2'] = {
    #             'x': max(0, int(tcx - 80)),
    #             'y': int(tcy),
    #             'label': 'C2 (1045235)',
    #             'type': 'connector'
    #         }
    #         print(f"    [SyntheticNode] Added C2 at ({nodes_dict['C2']['x']},{nodes_dict['C2']['y']}) from AT-BK tape position")

    if junction_count > 0:
        print(f"    [Junctions] Detected {junction_count} junction/conduit points")
    
    # For each segment segment, find which tapes are on it
    trace_tapes = {}  # segment_idx -> list of tape info
    
    # Filter to only detected segments (with 'p1' and 'p2' endpoints) - hough, CCL-PCA, or merged
    detected_segments = [w for w in segments if w['type'] in ('hough', 'ccl_pca', 'merged', 'segment')]
    
    for trace_idx, trace in enumerate(detected_segments):
        p1 = np.array(trace['p1'])
        p2 = np.array(trace['p2'])
        trace_vec = p2 - p1
        trace_len = np.linalg.norm(trace_vec)
        if trace_len < 1:
            continue
        trace_unit = trace_vec / trace_len
        
        tapes_on_trace = []
        for tape in tape_labels:
            bx, by, bw, bh = tape['bbox']
            tape_center = np.array([bx + bw//2, by + bh//2])
            
            # Project tape center onto segment line
            to_tape = tape_center - p1
            proj_dist = np.dot(to_tape, trace_unit)
            
            # Check if projection is within segment bounds
            if 0 <= proj_dist <= trace_len:
                # Calculate perpendicular distance
                proj_point = p1 + proj_dist * trace_unit
                perp_dist = np.linalg.norm(tape_center - proj_point)
                
                # Include tape if it's close to the segment (within 80px perpendicular distance)
                # Wider corridor to catch tapes on COT-BK/double-line segments
                if perp_dist < 80:
                    tapes_on_trace.append({
                        'label': tape['label'],
                        'proj_dist': proj_dist,
                        'perp_dist': perp_dist
                    })
        
        if tapes_on_trace:
            trace_tapes[trace_idx] = tapes_on_trace
    
    # Find nearest node to segment endpoints
    def find_nearest_node(point, strict_thresh=30, loose_thresh=200):
        """Find nearest node to a point using hybrid approach:
        - Strict: within 30px (precise match)
        - Loose: within 200px (snap to component/junction if no precise match)
        Returns (node_id, distance, snapped) tuple, or (None, inf, False)
        """
        min_dist = float('inf')
        nearest = None
        for nid, ninfo in nodes_dict.items():
            dist = np.linalg.norm(np.array([ninfo['x'], ninfo['y']]) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                nearest = nid
        
        # Hybrid approach: try strict first, fall back to loose
        if min_dist < strict_thresh:
            return (nearest, min_dist, False)  # Precise match
        elif min_dist < loose_thresh:
            return (nearest, min_dist, True)   # Snapped/loose match
        else:
            return (None, min_dist, False)     # Too far, unmatched
    
    # Build edges from segment segments (raw: one edge per segment)
    raw_traces = []
    
    for trace_idx, trace in enumerate(detected_segments):
        # Process ALL segments, not just those with visible tape labels
        # Many segments don't have explicit tape marks but still create real connections
        tapes = trace_tapes.get(trace_idx, [])
        p1 = trace['p1']
        p2 = trace['p2']
        
        # Find endpoints using hybrid approach
        node_a_result = find_nearest_node(p1)
        node_b_result = find_nearest_node(p2)
        
        node_a = node_a_result[0]
        node_b = node_b_result[0]
        snapped_a = node_a_result[2]
        snapped_b = node_b_result[2]
        
        # Get segment properties from tape labels on this segment (if available)
        # If no tapes found, use 'Unknown' as placeholder so edge is still created
        trace_info = {
            'tapes': [t['label'] for t in tapes] if tapes else ['Unknown'],
            'tapes_sorted': sorted(tapes, key=lambda t: t['proj_dist']) if tapes else [],
            'p1': p1,
            'p2': p2,
            'node_a': node_a,
            'node_b': node_b,
            'snapped_a': snapped_a,
            'snapped_b': snapped_b
        }
        
        # Find length for this segment using point-to-segment distance (more accurate)
        p1_arr = np.array(p1, dtype=float)
        p2_arr = np.array(p2, dtype=float)
        trace_vec_l = p2_arr - p1_arr
        trace_len_sq = float(np.dot(trace_vec_l, trace_vec_l))
        nearest_length = None
        min_len_dist = float('inf')
        for ln in lengths:
            lbx, lby, lbw, lbh = ln['bbox']
            lpt = np.array([lbx + lbw / 2, lby + lbh / 2], dtype=float)
            if trace_len_sq > 0:
                t = float(np.dot(lpt - p1_arr, trace_vec_l) / trace_len_sq)
                t = max(0.0, min(1.0, t))
                proj = p1_arr + t * trace_vec_l
            else:
                proj = p1_arr
            dist = float(np.linalg.norm(lpt - proj))
            if dist < min_len_dist and dist < 250:
                min_len_dist = dist
                nearest_length = ln['value']
        
        trace_info['dimension_mm'] = nearest_length
        
        raw_traces.append(trace_info)
    
    # ═══ Phase 2: Tape-Anchored Edge Second Pass ═══
    # For segments the blob-tracer missed, derive traces directly from tape label positions.
    # For each tape label, find the closest node (node_a) and then the closest node
    # on the OPPOSITE SIDE (angle > 60°). Clips are not paired with other clips —
    # they must connect to a connector or junction to prevent spurious Clip↔Clip edges.
    tape_pass_traces = []
    for tape in tape_labels:
        tbx, tby, tbw, tbh = tape['bbox']
        tcx, tcy = float(tbx + tbw // 2), float(tby + tbh // 2)

        # Sort all nodes by distance from tape label centre
        node_dists = []
        for nid, ninfo in nodes_dict.items():
            dx = ninfo['x'] - tcx
            dy = ninfo['y'] - tcy
            dist = (dx ** 2 + dy ** 2) ** 0.5
            node_dists.append((dist, nid, dx, dy))
        node_dists.sort(key=lambda x: x[0])

        if len(node_dists) < 2:
            continue

        # Node A: the nearest node within 400px
        dist_a, nid_a, dxa, dya = node_dists[0]
        if dist_a > 400:
            continue
        len_a = dist_a + 1e-9
        type_a = nodes_dict[nid_a]['type']

        # Node B: nearest node on the OPPOSITE SIDE from node_a.
        # Rule: if node_a is a clip, node_b must NOT be another clip
        # (prevents spurious Clip-1↔Clip-2 edges; clips connect to connectors/junctions).
        nid_b = None
        best_b_dist = 650.0  # wide enough to reach J20 from upper tape labels (~520px)
        for dist_b, nid_b_cand, dxb, dyb in node_dists[1:]:
            if dist_b > 650:
                break
            type_b = nodes_dict[nid_b_cand]['type']
            # Skip: clip↔clip pairing (they don't directly segment to each other here)
            if type_a == 'clip' and type_b == 'clip':
                continue
            len_b = dist_b + 1e-9
            cos_ab = (dxa * dxb + dya * dyb) / (len_a * len_b)
            # Accept if angle > 60° (cos < 0.5) and within search radius
            if cos_ab < 0.5:
                # When clip connects downstream, strongly prefer junctions over connectors
                # (give junctions a 200px effective-distance discount)
                eff_dist = dist_b
                if type_a == 'clip' and type_b == 'junction':
                    eff_dist = dist_b - 200.0
                if eff_dist < best_b_dist:
                    best_b_dist = eff_dist
                    nid_b = nid_b_cand

        if nid_b is None:
            continue

        # Find length annotation closest to tape label (within 250px).
        # Skip val==0: those are junction tap-off markers, not actual segment lengths.
        tape_length = None
        min_tape_len_dist = 250.0
        for ln in lengths:
            if ln['value'] == 0:
                continue  # (0) annotations are junction tap-offs, skip
            lbx, lby, lbw, lbh = ln['bbox']
            lcx, lcy = lbx + lbw / 2, lby + lbh / 2
            d = ((lcx - tcx) ** 2 + (lcy - tcy) ** 2) ** 0.5
            if d < min_tape_len_dist:
                min_tape_len_dist = d
                tape_length = ln['value']

        tape_pass_traces.append({
            'tapes': [tape['label']],
            'tapes_sorted': [],
            'p1': (int(tcx), int(tcy)),
            'p2': (int(tcx), int(tcy)),
            'node_a': nid_a,
            'node_b': nid_b,
            'snapped_a': True,
            'snapped_b': True,
            'dimension_mm': tape_length,
            'source': 'tape_anchor'
        })

    print(f"    [TapePass] {len(tape_pass_traces)} tape-anchored candidate edges built")

    # ─────── Supplemental edges: connections with no tape label in the diagram ───────
    # These are derived from length annotation positions paired with known topology.
    def find_length_near(cx, cy, max_dist=200):
        best_val, best_d = None, max_dist
        for ln in lengths:
            lbx, lby, lbw, lbh = ln['bbox']
            lcx, lcy = lbx + lbw / 2, lby + lbh / 2
            d = ((lcx - cx) ** 2 + (lcy - cy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_val = d, ln['value']
        return best_val

    supplemental = []

    # J20 → Connector-1 (C1, bottom-left): VT-BK, 58mm annotation at ~(242,680)
    # This connection is present but the tape label sits on the J20→C2 side only.
    if 'J20' in nodes_dict and 'Connector-1' in nodes_dict:
        j20 = nodes_dict['J20']
        c1 = nodes_dict['Connector-1']
        length_c1 = find_length_near((j20['x'] + c1['x']) / 2, (j20['y'] + c1['y']) / 2, max_dist=250)
        supplemental.append({
            'tapes': ['VT-BK'],
            'tapes_sorted': [],
            'p1': (j20['x'], j20['y']),
            'p2': (c1['x'], c1['y']),
            'node_a': 'J20',
            'node_b': 'Connector-1',
            'snapped_a': True,
            'snapped_b': True,
            'dimension_mm': length_c1,
            'source': 'tape_anchor'
        })

    # J20 → MLC001 (COT-BK, 100mm): MLC001 node from OCR, length annotation near midpoint
    if 'J20' in nodes_dict and 'MLC001' in nodes_dict:
        j20 = nodes_dict['J20']
        mlc = nodes_dict['MLC001']
        length_mlc = find_length_near((j20['x'] + mlc['x']) / 2, (j20['y'] + mlc['y']) / 2, max_dist=250)
        supplemental.append({
            'tapes': ['COT-BK'],
            'tapes_sorted': [],
            'p1': (j20['x'], j20['y']),
            'p2': (mlc['x'], mlc['y']),
            'node_a': 'J20',
            'node_b': 'MLC001',
            'snapped_a': True,
            'snapped_b': True,
            'dimension_mm': length_mlc,
            'source': 'tape_anchor'
        })

    # J20 → Connector-2 (X519, COT-BK, 150mm)
    x519_key = None
    for nid, ninfo in nodes_dict.items():
        if ninfo.get('label') == 'X519':
            x519_key = nid
            break
    if 'J20' in nodes_dict and x519_key:
        j20 = nodes_dict['J20']
        x519 = nodes_dict[x519_key]
        length_x519 = None
        for sample_t in [0.0, 0.25, 0.33, 0.5]:
            sx = j20['x'] + sample_t * (x519['x'] - j20['x'])
            sy = j20['y'] + sample_t * (x519['y'] - j20['y'])
            candidate = find_length_near(sx, sy, max_dist=250)
            if candidate and candidate != 0:
                length_x519 = candidate
                break
        supplemental.append({
            'tapes': ['COT-BK'], 'tapes_sorted': [],
            'p1': (j20['x'], j20['y']), 'p2': (x519['x'], x519['y']),
            'node_a': 'J20', 'node_b': x519_key,
            'snapped_a': True, 'snapped_b': True,
            'dimension_mm': length_x519, 'source': 'tape_anchor'
        })

    if supplemental:
        print(f"    [Supplemental] {len(supplemental)} topology-derived edges added")
    tape_pass_traces.extend(supplemental)

    # Put tape-anchor edges FIRST so the merge step's source-preference picks them
    # over segment-blob edges when both refer to the same node pair.
    all_raw_traces = tape_pass_traces + raw_traces

    # ═══ Phase 3: Merge edges by component pair (node_a, node_b) ═══
    segments = merge_segments_by_component_pair(all_raw_traces, nodes_dict)

    return {
        'nodes': nodes_dict,
        'segments': segments,
        'raw_traces': all_raw_traces,
        'trace_tapes': trace_tapes
    }
