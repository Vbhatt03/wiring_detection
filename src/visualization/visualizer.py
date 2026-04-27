"""
Visualization & Annotation
===========================
Functions for annotating detected elements on the image canvas.
"""

import cv2


# Tape label colors
TAPE_COLOR_BGR = {
    'VT-WH':  (  0,   0, 180),   # dark red
    'VT-BK':  (  0,   0, 180),   # dark red
    'VT-PK':  (  0,   0, 180),   # dark red
    'AT-BK':  (  0,   0, 180),   # dark red
    'COT-BK': (  0,   0, 180),   # dark red
    'DEFAULT':(  0,   0, 180),   # dark red
}


def draw_label(canvas, text, pt, color=(0, 200, 0), scale=0.45, thickness=1):
    """Draw a text label on the canvas."""
    x, y = pt
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def annotate(img, tapes, connectors, wires,
             dimensions, clips, connectivity, filters=None):
    """Annotate detected elements on the image.
    
    Args:
        img: Input image
        tapes: List of detected tape labels
        connectors: List of detected connectors
        wires: List of detected wires
        dimensions: List of detected dimension annotations
        clips: List of detected clips
        connectivity: Connectivity graph dict with 'nodes' and 'edges'
        filters: Dict of extraction filters (which elements to show)
    
    Returns:
        Annotated image canvas
    """
    if filters is None:
        filters = {
            'tapes': True,
            'connectors': True,
            'wires': True,
            'dimensions': True,
            'clips': True,
        }
    
    canvas = img.copy()
    H, W = canvas.shape[:2]

    # — Tape labels —
    if filters.get('tapes', True):
        for item in tapes:
            x, y, w, h = item['bbox']
            lbl = item['label']
            color = TAPE_COLOR_BGR.get(lbl, TAPE_COLOR_BGR['DEFAULT'])
            cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
            draw_label(canvas, f"[TAPE] {lbl}", (x, max(y-5, 10)), color, scale=0.45)

    # — Connectors —
    if filters.get('connectors', True):
        for i, conn in enumerate(connectors):
            x, y, w, h = conn['bbox']
            cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 0, 180), 2)
            draw_label(canvas, f"[CONN] C{i+1}", (x, max(y-5, 10)),
                       (0, 0, 180), scale=0.42)

    # — Output Connections (Connectivity Graph) —
    if filters.get('wires', True):
        edges = connectivity.get('edges', [])
        nodes = connectivity.get('nodes', {})
        for e in edges:
            node_a_id = e.get('node_a')
            node_b_id = e.get('node_b')
            
            if node_a_id in nodes and node_b_id in nodes:
                n1 = nodes[node_a_id]
                n2 = nodes[node_b_id]
                p1 = (int(n1['x']), int(n1['y']))
                p2 = (int(n2['x']), int(n2['y']))
                
                # Draw the logical connection between components
                cv2.line(canvas, p1, p2, (0, 200, 0), 2)
                cv2.circle(canvas, p1, 4, (0, 200, 0), -1)
                cv2.circle(canvas, p2, 4, (0, 200, 0), -1)
                
                # Label the connection with its tape and dimension (Disabled per user request)
                # wire_types = e.get('wire_types', e.get('tapes', []))
                # tape_str = '+'.join(wire_types) if wire_types else "Unknown"
                # dim_str = f"{e['dimension_mm']}mm" if e.get('dimension_mm') else ""
                # label_text = f"{tape_str} {dim_str}".strip()
                # mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                # draw_label(canvas, label_text, (mx - 20, my - 5), (0, 150, 0), scale=0.4)

    # — Wire dimensions —
    if filters.get('dimensions', True):
        for ln in dimensions:
            x, y, w, h = ln['bbox']
            x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
            # Draw rectangle around detected length
            cv2.rectangle(canvas, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 180), 1)
            
            # Intelligently position text: prefer below, but adjust if near edges
            text_x = x - 5
            text_y = y + h + 12
            
            # Check if text would go off bottom; if so, place above
            if text_y + 15 > H:
                text_y = y - 8
            
            # Check if text would go off right; if so, shift left
            if text_x + 40 > W:
                text_x = max(5, W - 45)
            
            # Draw value only (no "mm" unit to reduce clutter)
            # Show parentheses if the value was parenthesized in original
            value_str = f"({ln['value']})" if ln.get('is_parenthesized') else str(ln['value'])
            draw_label(canvas, value_str, (text_x, text_y),
                       (0, 0, 180), scale=0.40, thickness=1)

    # — Blue clips —
    if filters.get('clips', True):
        for clip in clips:
            cx, cy = clip['center']
            cx, cy = int(round(cx)), int(round(cy))
            r = clip['radius']
            cv2.circle(canvas, (cx, cy), r + 3, (0, 0, 180), 2)
            draw_label(canvas, '[CLIP]', (cx - 10, cy - r - 5),
                       (0, 0, 180), scale=0.42)

    # — Legend box in top-right —
    legends = []
    if filters.get('tapes', True):
        legends.append(('[TAPE]  Tape / conduit label', (0, 0, 180)))
    if filters.get('connectors', True):
        legends.append(('[CONN]  Delphi connector',      (0, 0, 180)))
    if filters.get('wires', True):
        legends.append(('[WIRE]  Detected wires',        (0, 128, 0)))
    if filters.get('dimensions', True):
        legends.append(('[DIM]   Wire dimension (mm)',      (0, 0, 180)))
    if filters.get('clips', True):
        legends.append(('[CLIP]  Blue circular clip',    (0, 0, 180)))
    
    if legends:
        lx, ly = W - 320, 10
        cv2.rectangle(canvas, (lx-5, ly-5), (W-5, ly + len(legends)*20 + 5),
                      (30, 30, 30), -1)
        for i, (txt, col) in enumerate(legends):
            draw_label(canvas, txt, (lx, ly + i*20 + 15), col, scale=0.38, thickness=1)

    return canvas
