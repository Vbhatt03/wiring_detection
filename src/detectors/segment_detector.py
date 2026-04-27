"""
Segment Detection
==============
Detects individual segment traces in the schematic using Connected Components Labeling (CCL)
with PCA-based endpoint extraction and Union-Find based merging.

Strategy:
    PHASE A (Component Detection): Detect filled blobs (components, clips)
    PHASE B (Segment Trace Extraction): Use CCL to find segment traces
    PHASE C (Union-Find Merging): Merge collinear/parallel traces
    PHASE D (Segment Reconstruction): Collect merged groups and output final segments
"""

import cv2
import numpy as np


def detect_components(gray, img_color):
    """
    Detect component bounding boxes so we can use them as segment-break boundaries.
    Returns list of (x, y, w, h) bboxes.
    Strategy: find closed, filled, non-elongated blobs — opposite of segments.
    """
    # Threshold to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S
    )

    component_bboxes = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < 80 or area > 8000:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        fill_ratio = area / (w * h)
        # Components are squarish and dense; segments are elongated/sparse
        if aspect < 4.0 and fill_ratio > 0.25 and min(w, h) > 8:
            component_bboxes.append((x, y, w, h))

    return component_bboxes


def trace_crosses_component(p1, p2, component_bboxes, margin=4):
    """
    Check if the straight line from p1 to p2 passes through any component bbox.
    Uses parametric line sampling.
    
    Args:
        p1, p2: (x, y) endpoints of the line trace
        component_bboxes: List of (x, y, w, h) bounding boxes
        margin: Pixel margin around each bbox to consider as "crossing"
    
    Returns:
        True if the trace crosses any component, False otherwise
    """
    x1, y1 = p1
    x2, y2 = p2
    steps = max(int(np.hypot(x2 - x1, y2 - y1)), 1)
    
    for i in range(steps + 1):
        t = i / steps
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        for (bx, by, bw, bh) in component_bboxes:
            if (bx - margin) <= px <= (bx + bw + margin) and \
               (by - margin) <= py <= (by + bh + margin):
                return True
    return False


def angle_between(p1, p2):
    """Return angle in degrees of the vector from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx)) % 180


def detect_segments(gray):
    """
    Segment detection with Union-Find based trace merging.
    
    Strategy:
    PHASE A (Component Detection):
    1. Detect filled blobs (components like connectors, clips) and create a mask
    
    PHASE B (Segment Trace Extraction):
    2. Create segment mask (dark + edges), remove component regions
    3. Use Connected Components Labeling (CCL) to find segment traces
    4. Extract endpoints using PCA for each trace
    
    PHASE C (Union-Find Merging):
    5. Build Union-Find structure and merge traces based on:
       - Proximity: endpoints < 30px apart
       - Collinearity: angle difference < 15°
       - Component blocking: gaps don't cross components
    
    PHASE D (Segment Reconstruction):
    6. Collect merged groups, find true terminals, output final segments
    
    This method correctly handles dashed segments and respects component boundaries.
    """
    
    img_color = cv2.imread('automotive_schematic.png')
    if img_color is None:
        return []

    # ────────────────────────────────────────────────────────────
    # PHASE A: Component Detection
    # ────────────────────────────────────────────────────────────
    component_bboxes = detect_components(gray, img_color)

    # Build a component mask so we can subtract it from the segment mask
    comp_mask = np.zeros(gray.shape, dtype=np.uint8)
    for (x, y, w, h) in component_bboxes:
        cv2.rectangle(comp_mask, (x, y), (x + w, y + h), 255, -1)

    # ────────────────────────────────────────────────────────────
    # PHASE B: Segment Trace Extraction
    # ────────────────────────────────────────────────────────────
    
    # Segment mask: dark pixels + edges
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 100, 150])  # Lenient dark mask
    color_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    segment_mask = cv2.bitwise_and(color_mask, edges)

    # Remove component regions from segment mask — this is the KEY BREAK
    segment_mask = cv2.bitwise_and(segment_mask, cv2.bitwise_not(comp_mask))

    # Gap filling: dilate and clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bridged_mask = cv2.dilate(segment_mask, kernel, iterations=2)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bridged_mask = cv2.morphologyEx(bridged_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    bridged_mask = cv2.morphologyEx(bridged_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Also remove components from bridged mask to prevent bleed-through
    bridged_mask = cv2.bitwise_and(bridged_mask, cv2.bitwise_not(comp_mask))

    # Connected Components Labeling
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bridged_mask, connectivity=8, ltype=cv2.CV_32S
    )

    # Collect raw traces with PCA endpoints
    raw_traces = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < 20 or area > 6000:
            continue
        
        bbox_aspect = max(w, h) / max(min(w, h), 1)
        if bbox_aspect < 1.3:
            continue
        
        fill_ratio = area / (w * h)
        if fill_ratio < 0.05:
            continue

        blob_points = np.where(labels == label_id)
        if len(blob_points[0]) < 5:
            continue

        points = np.column_stack((blob_points[1], blob_points[0])).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points, mean=None)
        primary_axis = eigenvectors[0]
        projections = points @ primary_axis

        p1 = points[np.argmin(projections)]
        p2 = points[np.argmax(projections)]
        length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if length < 15:
            continue

        raw_traces.append({
            'p1': tuple(p1.astype(int)),
            'p2': tuple(p2.astype(int)),
            'length': length,
            'area': area,
            'axis': primary_axis,
        })

    if not raw_traces:
        return []

    # ────────────────────────────────────────────────────────────
    # PHASE C: Hybrid Graph-Based Merging (improved approach)
    # ────────────────────────────────────────────────────────────
    n = len(raw_traces)
    
    # Build endpoint graph: connect endpoints that should be merged
    GAP_THRESH = 100  # px — allow very long gaps in dashed/traceed segments
    ANGLE_THRESH = 45  # degrees — allow wide angle variations
    
    # Create adjacency graph: endpoint_id -> list of connected endpoints
    endpoint_graph = {}  # (trace_idx, ep_type) -> [(trace_idx, ep_type), ...]
    
    # For each trace, create node IDs for its endpoints
    trace_endpoints = {}  # trace_idx -> {'p1_idx': 0, 'p2_idx': 1}
    for i in range(n):
        trace_endpoints[i] = {'p1_idx': (i, 'p1'), 'p2_idx': (i, 'p2')}
        endpoint_graph[(i, 'p1')] = []
        endpoint_graph[(i, 'p2')] = []
    
    # Test all pairs of traces and connect compatible endpoints
    trace_pairs_tested = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in trace_pairs_tested:
                continue
            trace_pairs_tested.add((i, j))
            
            trace_i = raw_traces[i]
            trace_j = raw_traces[j]
            
            # Precompute angles once per trace
            angle_i = np.degrees(np.arctan2(trace_i['axis'][1], trace_i['axis'][0])) % 180
            angle_j = np.degrees(np.arctan2(trace_j['axis'][1], trace_j['axis'][0])) % 180
            
            # Check all 4 endpoint pairs
            endpoint_pairs = [
                ((i, 'p1'), trace_i['p1'], (j, 'p1'), trace_j['p1']),
                ((i, 'p1'), trace_i['p1'], (j, 'p2'), trace_j['p2']),
                ((i, 'p2'), trace_i['p2'], (j, 'p1'), trace_j['p1']),
                ((i, 'p2'), trace_i['p2'], (j, 'p2'), trace_j['p2']),
            ]
            
            for node_a, pt_a, node_b, pt_b in endpoint_pairs:
                # Criterion 1: Proximity
                dist = np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
                if dist > GAP_THRESH:
                    continue
                
                # Criterion 2: Collinearity (enhanced tolerance)
                angle_diff = abs(angle_i - angle_j)
                angle_diff = min(angle_diff, 180 - angle_diff)
                if angle_diff > ANGLE_THRESH:
                    continue
                
                # Criterion 3: Component blocking
                if trace_crosses_component(pt_a, pt_b, component_bboxes):
                    continue
                
                # All criteria met → connect these endpoints in graph
                if node_b not in endpoint_graph[node_a]:
                    endpoint_graph[node_a].append(node_b)
                if node_a not in endpoint_graph[node_b]:
                    endpoint_graph[node_b].append(node_a)
    
    # ────────────────────────────────────────────────────────────
    # PHASE D: Trace Paths Through Endpoint Graph
    # ────────────────────────────────────────────────────────────
    
    visited = set()
    segments = []
    
    # Find all connected components and trace paths
    for start_node in endpoint_graph:
        if start_node in visited:
            continue
        
        # BFS to find all nodes in this connected component
        component = []
        queue = [start_node]
        visited_local = {start_node}
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            visited.add(node)
            
            for neighbor in endpoint_graph[node]:
                if neighbor not in visited_local:
                    visited_local.add(neighbor)
                    queue.append(neighbor)
        
        # Now trace the actual segment path through this component
        # Start from a terminal node (degree 1) or any node if no terminal exists
        terminal_nodes = [n for n in component if len(endpoint_graph[n]) == 1]
        start = terminal_nodes[0] if terminal_nodes else component[0]
        
        # Trace path: follow graph edges to build segment
        path_nodes = [start]
        prev_node = None
        current = start
        
        while True:
            neighbors = [n for n in endpoint_graph[current] if n != prev_node]
            if not neighbors:
                break
            
            prev_node = current
            current = neighbors[0]  # Follow the edge (should be at most 1 unvisited neighbor)
            path_nodes.append(current)
            
            if len(path_nodes) > len(component) * 2:  # Prevent infinite loops
                break
        
        # Convert path nodes back to trace indices and collect endpoints
        all_points = []
        member_traces = set()
        
        for node in path_nodes:
            trace_idx, ep_type = node
            member_traces.add(trace_idx)
            trace = raw_traces[trace_idx]
            all_points.append(trace['p1'])
            all_points.append(trace['p2'])
        
        if not all_points:
            continue
        
        # Fit PCA to find segment direction and endpoints
        pts = np.array(all_points, dtype=np.float32)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        primary_axis = eigenvectors[0]
        projections = pts @ primary_axis
        
        p1 = pts[np.argmin(projections)]
        p2 = pts[np.argmax(projections)]
        length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        
        # Final length constraints - accept wider range for more flexibility
        if length < 20 or length > 600:
            continue
        
        segments.append({
            'p1': (int(p1[0]), int(p1[1])),
            'p2': (int(p2[0]), int(p2[1])),
            'length_px': int(length),
            'trace_count': len(member_traces),
            'type': 'merged'
        })
    
    # ────────────────────────────────────────────────────────────
    # PHASE E: Deduplication - Remove duplicate segments
    # ────────────────────────────────────────────────────────────
    
    # Remove duplicate segments (same endpoints, possibly reversed)
    deduped_segments = []
    seen_segments = set()
    
    for segment in segments:
        p1 = segment['p1']
        p2 = segment['p2']
        
        # Create normalized key (sorted endpoints for direction invariance)
        key = tuple(sorted([p1, p2]))
        
        if key not in seen_segments:
            seen_segments.add(key)
            deduped_segments.append(segment)
    
    # ═══ DEBUG: Phase instrumentation ═══
    print(f"    [Segment Detection] Raw traces: {n} → Graph merged: {len(segments)} → Deduplicated: {len(deduped_segments)} segments (GAP={GAP_THRESH}px, ANGLE={ANGLE_THRESH}°)")

    return deduped_segments


def filter_segments_by_components(segments, components, ocr_data, margin=50):
    """
    Component-Anchored Validation:
    Filter segments to only those with at least one endpoint near:
    - A detected component (Tape, Connector, Clip)
    - An OCR text region (length annotations like "150mm")
    
    This eliminates dimension lines, borders, and orphan traces.
    Margin (default 50px) controls how close endpoints must be to anchors.
    """
    if not segments:
        return []
    
    validated_segments = []
    
    for segment in segments:
        p1 = np.array(segment['p1'], dtype=np.float32)
        p2 = np.array(segment['p2'], dtype=np.float32)
        
        p1_anchored = False
        p2_anchored = False
        
        # Check if endpoints are near detected components
        for comp in components:
            # Handle clips (have 'center' key)
            if 'center' in comp:
                comp_center = np.array(comp['center'], dtype=np.float32)
            # Handle tapes and connectors (have 'bbox' key)
            elif 'bbox' in comp:
                bbox = comp['bbox']
                comp_center = np.array([
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0
                ], dtype=np.float32)
            else:
                continue
            
            dist_p1 = np.linalg.norm(p1 - comp_center)
            dist_p2 = np.linalg.norm(p2 - comp_center)
            
            if dist_p1 < margin:
                p1_anchored = True
            if dist_p2 < margin:
                p2_anchored = True
        
        # Check if endpoints are near OCR text hits (length annotations, labels)
        for item in ocr_data:
            # Handle tuple with (text, x, y, w, h, [angle, conf])
            if len(item) >= 4:
                x, y, w, h = item[1], item[2], item[3], item[4]
                text_center = np.array([x + w/2, y + h/2], dtype=np.float32)
                
                dist_p1 = np.linalg.norm(p1 - text_center)
                dist_p2 = np.linalg.norm(p2 - text_center)
                
                if dist_p1 < margin:
                    p1_anchored = True
                if dist_p2 < margin:
                    p2_anchored = True
        
        # Accept segment if at least ONE endpoint is anchored to a component or label
        if p1_anchored or p2_anchored:
            segment['p1_anchored'] = p1_anchored
            segment['p2_anchored'] = p2_anchored
            validated_segments.append(segment)
    
    return validated_segments
