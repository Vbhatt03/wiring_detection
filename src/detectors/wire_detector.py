"""
Wire Detection
==============
Detects individual wire segments in the schematic using Connected Components Labeling (CCL)
with PCA-based endpoint extraction and Union-Find based merging.

Strategy:
    PHASE A (Component Detection): Detect filled blobs (components, clips)
    PHASE B (Wire Segment Extraction): Use CCL to find wire segments
    PHASE C (Union-Find Merging): Merge collinear/parallel segments
    PHASE D (Wire Reconstruction): Collect merged groups and output final wires
"""

import cv2
import numpy as np


def detect_components(gray, img_color):
    """
    Detect component bounding boxes so we can use them as wire-break boundaries.
    Returns list of (x, y, w, h) bboxes.
    Strategy: find closed, filled, non-elongated blobs — opposite of wires.
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
        # Components are squarish and dense; wires are elongated/sparse
        if aspect < 4.0 and fill_ratio > 0.25 and min(w, h) > 8:
            component_bboxes.append((x, y, w, h))

    return component_bboxes


def segment_crosses_component(p1, p2, component_bboxes, margin=4):
    """
    Check if the straight line from p1 to p2 passes through any component bbox.
    Uses parametric line sampling.
    
    Args:
        p1, p2: (x, y) endpoints of the line segment
        component_bboxes: List of (x, y, w, h) bounding boxes
        margin: Pixel margin around each bbox to consider as "crossing"
    
    Returns:
        True if the segment crosses any component, False otherwise
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


def detect_wires(gray):
    """
    Wire detection with Union-Find based segment merging.
    
    Strategy:
    PHASE A (Component Detection):
    1. Detect filled blobs (components like connectors, clips) and create a mask
    
    PHASE B (Wire Segment Extraction):
    2. Create wire mask (dark + edges), remove component regions
    3. Use Connected Components Labeling (CCL) to find wire segments
    4. Extract endpoints using PCA for each segment
    
    PHASE C (Union-Find Merging):
    5. Build Union-Find structure and merge segments based on:
       - Proximity: endpoints < 30px apart
       - Collinearity: angle difference < 15°
       - Component blocking: gaps don't cross components
    
    PHASE D (Wire Reconstruction):
    6. Collect merged groups, find true terminals, output final wires
    
    This method correctly handles dashed wires and respects component boundaries.
    """
    
    img_color = cv2.imread('automotive_schematic.png')
    if img_color is None:
        return []

    # ────────────────────────────────────────────────────────────
    # PHASE A: Component Detection
    # ────────────────────────────────────────────────────────────
    component_bboxes = detect_components(gray, img_color)

    # Build a component mask so we can subtract it from the wire mask
    comp_mask = np.zeros(gray.shape, dtype=np.uint8)
    for (x, y, w, h) in component_bboxes:
        cv2.rectangle(comp_mask, (x, y), (x + w, y + h), 255, -1)

    # ────────────────────────────────────────────────────────────
    # PHASE B: Wire Segment Extraction
    # ────────────────────────────────────────────────────────────
    
    # Wire mask: dark pixels + edges
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 100, 150])  # Lenient dark mask
    color_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    wire_mask = cv2.bitwise_and(color_mask, edges)

    # Remove component regions from wire mask — this is the KEY BREAK
    wire_mask = cv2.bitwise_and(wire_mask, cv2.bitwise_not(comp_mask))

    # Gap filling: dilate and clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bridged_mask = cv2.dilate(wire_mask, kernel, iterations=2)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bridged_mask = cv2.morphologyEx(bridged_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    bridged_mask = cv2.morphologyEx(bridged_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Also remove components from bridged mask to prevent bleed-through
    bridged_mask = cv2.bitwise_and(bridged_mask, cv2.bitwise_not(comp_mask))

    # Connected Components Labeling
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bridged_mask, connectivity=8, ltype=cv2.CV_32S
    )

    # Collect raw segments with PCA endpoints
    raw_segments = []
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

        raw_segments.append({
            'p1': tuple(p1.astype(int)),
            'p2': tuple(p2.astype(int)),
            'length': length,
            'area': area,
            'axis': primary_axis,
        })

    if not raw_segments:
        return []

    # ────────────────────────────────────────────────────────────
    # PHASE C: Hybrid Graph-Based Merging (improved approach)
    # ────────────────────────────────────────────────────────────
    n = len(raw_segments)
    
    # Build endpoint graph: connect endpoints that should be merged
    GAP_THRESH = 100  # px — allow very long gaps in dashed/segmented wires
    ANGLE_THRESH = 45  # degrees — allow wide angle variations
    
    # Create adjacency graph: endpoint_id -> list of connected endpoints
    endpoint_graph = {}  # (seg_idx, ep_type) -> [(seg_idx, ep_type), ...]
    
    # For each segment, create node IDs for its endpoints
    segment_endpoints = {}  # seg_idx -> {'p1_idx': 0, 'p2_idx': 1}
    for i in range(n):
        segment_endpoints[i] = {'p1_idx': (i, 'p1'), 'p2_idx': (i, 'p2')}
        endpoint_graph[(i, 'p1')] = []
        endpoint_graph[(i, 'p2')] = []
    
    # Test all pairs of segments and connect compatible endpoints
    segment_pairs_tested = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in segment_pairs_tested:
                continue
            segment_pairs_tested.add((i, j))
            
            seg_i = raw_segments[i]
            seg_j = raw_segments[j]
            
            # Precompute angles once per segment
            angle_i = np.degrees(np.arctan2(seg_i['axis'][1], seg_i['axis'][0])) % 180
            angle_j = np.degrees(np.arctan2(seg_j['axis'][1], seg_j['axis'][0])) % 180
            
            # Check all 4 endpoint pairs
            endpoint_pairs = [
                ((i, 'p1'), seg_i['p1'], (j, 'p1'), seg_j['p1']),
                ((i, 'p1'), seg_i['p1'], (j, 'p2'), seg_j['p2']),
                ((i, 'p2'), seg_i['p2'], (j, 'p1'), seg_j['p1']),
                ((i, 'p2'), seg_i['p2'], (j, 'p2'), seg_j['p2']),
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
                if segment_crosses_component(pt_a, pt_b, component_bboxes):
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
    wires = []
    
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
        
        # Now trace the actual wire path through this component
        # Start from a terminal node (degree 1) or any node if no terminal exists
        terminal_nodes = [n for n in component if len(endpoint_graph[n]) == 1]
        start = terminal_nodes[0] if terminal_nodes else component[0]
        
        # Trace path: follow graph edges to build wire
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
        
        # Convert path nodes back to segment indices and collect endpoints
        all_points = []
        member_segs = set()
        
        for node in path_nodes:
            seg_idx, ep_type = node
            member_segs.add(seg_idx)
            seg = raw_segments[seg_idx]
            all_points.append(seg['p1'])
            all_points.append(seg['p2'])
        
        if not all_points:
            continue
        
        # Fit PCA to find wire direction and endpoints
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
        
        wires.append({
            'p1': (int(p1[0]), int(p1[1])),
            'p2': (int(p2[0]), int(p2[1])),
            'length_px': int(length),
            'segment_count': len(member_segs),
            'type': 'merged'
        })
    
    # ────────────────────────────────────────────────────────────
    # PHASE E: Deduplication - Remove duplicate wires
    # ────────────────────────────────────────────────────────────
    
    # Remove duplicate wires (same endpoints, possibly reversed)
    deduped_wires = []
    seen_wires = set()
    
    for wire in wires:
        p1 = wire['p1']
        p2 = wire['p2']
        
        # Create normalized key (sorted endpoints for direction invariance)
        key = tuple(sorted([p1, p2]))
        
        if key not in seen_wires:
            seen_wires.add(key)
            deduped_wires.append(wire)
    
    # ═══ DEBUG: Phase instrumentation ═══
    print(f"    [Wire Detection] Raw segments: {n} → Graph merged: {len(wires)} → Deduplicated: {len(deduped_wires)} wires (GAP={GAP_THRESH}px, ANGLE={ANGLE_THRESH}°)")

    return deduped_wires


def filter_wires_by_components(wires, components, ocr_data, margin=50):
    """
    Component-Anchored Validation:
    Filter wires to only those with at least one endpoint near:
    - A detected component (Tape, Connector, Clip)
    - An OCR text region (length annotations like "150mm")
    
    This eliminates dimension lines, borders, and orphan segments.
    Margin (default 50px) controls how close endpoints must be to anchors.
    """
    if not wires:
        return []
    
    validated_wires = []
    
    for wire in wires:
        p1 = np.array(wire['p1'], dtype=np.float32)
        p2 = np.array(wire['p2'], dtype=np.float32)
        
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
        
        # Accept wire if at least ONE endpoint is anchored to a component or label
        if p1_anchored or p2_anchored:
            wire['p1_anchored'] = p1_anchored
            wire['p2_anchored'] = p2_anchored
            validated_wires.append(wire)
    
    return validated_wires
