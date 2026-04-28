"""
Reporting & Verification
===========================
Functions for generating connectivity reports and verification tables.
"""

import re


def print_report(tapes, connectors, segments,
                 lengths, clips, connectivity_graph):
    """Print a connectivity report for all detected elements.
    
    Args:
        tapes: List of detected tape labels
        connectors: List of detected connectors
        segments: List of detected segments
        lengths: List of detected length annotations
        clips: List of detected clips
        connectivity_graph: Connectivity graph dict with nodes and edges
    """
    SEP = '=' * 72

    print(SEP)
    print('  SEGMENT DIAGRAM DETECTION REPORT')
    print(SEP)

    print(f'\n[1] TAPE / CONDUIT LABELS  ({len(tapes)} found)')
    for t in tapes:
        x, y, w, h = t['bbox']
        print(f"    • {t['label']:<12}  at ({x},{y}) size {w}×{h}")

    print(f'\n[2] DELPHI CONNECTORS  ({len(connectors)} found)')
    for i, c in enumerate(connectors):
        x, y, w, h = c['bbox']
        print(f"    C{i+1}: {c['label']:<20}  at ({x},{y}) – {c.get('note','')}")

    print(f'\n[3] SEGMENTS  ({len(segments)} merged segments detected)')
    segments_shown = sum(1 for w in segments if w['type'] in ('hough', 'merged'))
    print(f'    (Showing {min(segments_shown, len(segments))} segments in visualization)')
    for i, segment in enumerate(segments[:30]):  # Show first 30 in report
        # Show metrics for merged segments
        metrics = ""
        if segment['type'] == 'merged':
            metrics = f"  ({segment.get('trace_count', 1)} traces merged, total_path={segment.get('length_px', 0)}px)"
        print(f"    segment {i+1:02d}: ({segment['p1'][0]},{segment['p1'][1]}) → ({segment['p2'][0]},{segment['p2'][1]})  "
              f"len={segment['length_px']}px{metrics}")
    if len(segments) > 20:
        print(f"    … and {len(segments)-20} more")

    print(f'\n[4] SEGMENT LENGTH ANNOTATIONS  ({len(lengths)} found)')
    for ln in lengths:
        x, y, w, h = ln['bbox']
        x, y = int(round(x)), int(round(y))
        paren_indicator = ' (parenthesized)' if ln.get('is_parenthesized') else ''
        print(f"    {ln['value']} mm  at ({x},{y}){paren_indicator}")

    print(f'\n[5] BLUE CIRCULAR CLIPS  ({len(clips)} found)')
    for i, clip in enumerate(clips):
        print(f"    Clip {i+1}: centre={clip['center']}  r={clip['radius']}px")

    # ── New graph-based connectivity report ──
    print(f'\n[6] CONNECTIVITY GRAPH')
    print(f'    Nodes ({len(connectivity_graph["nodes"])} found):')
    for nid, ninfo in connectivity_graph['nodes'].items():
        print(f"      {nid:<15} at ({ninfo['x']},{ninfo['y']}) – {ninfo['label']}")
    
    # Show both raw and merged edge counts
    raw_trace_count = len(connectivity_graph.get('raw_traces', []))
    merged_segment_count = len(connectivity_graph['segments'])
    
    print(f'\n    Segments (Raw: {raw_trace_count} traces → Merged: {merged_segment_count} connections):')
    for i, segment in enumerate(connectivity_graph['segments'], 1):
        # Handle both old and new segment format
        if isinstance(segment.get('tapes'), list):
            # Old format (per-segment)
            tapes_str = '+'.join(segment['tapes'])
        else:
            # New format (merged)
            tapes_str = '+'.join(segment.get('segment_types', []))
        
        dim_str = f"{segment.get('dimension_mm', segment.get('dimension_mm', None))} mm" if segment.get('dimension_mm') else '—'
        from_str = segment['node_a'] if segment['node_a'] else '?'
        to_str = segment['node_b'] if segment['node_b'] else '?'
        
        # Show trace count if merged
        trace_info = ""
        if segment.get('trace_count', 1) > 1:
            trace_info = f"  ({segment['trace_count']} traces)"
        
        snapped_info = " [snapped]" if segment.get('snapped', False) else ""
        
        print(f"      [{i}] {tapes_str:<20} {dim_str:<10} {from_str:<18} → {to_str}{trace_info}{snapped_info}")
    
    print()


# def generate_verification_table(lengths, ocr_data, title="Segment Length Extraction Verification"):
#     """
#     Generate a manual verification table for extracted segment lengths.
#     Shows: what text was detected → what value was extracted → verification status
    
#     Args:
#         lengths: List of extracted length annotations
#         ocr_data: List of all OCR detections
#         title: Title for the verification table
#     """
#     print()
#     print('  ' + '='*90)
#     print(f'  {title}')
#     print('  ' + '='*90)
    
#     if not lengths:
#         print("  [No segment lengths extracted]")
#         return
    
#     # Build lookup of all numeric text from OCR
#     numeric_ocr = []
#     for item in ocr_data:
#         if len(item) >= 5:
#             txt, x, y, w, h = item[0], item[1], item[2], item[3], item[4]
#             clean = txt.strip().replace(' ', '')
#             # Check if it's a number (possibly parenthesized)
#             if re.match(r'^\(?\d+\)?$', clean):
#                 val = int(re.sub(r'[^\d]', '', clean))
#                 numeric_ocr.append({
#                     'text': txt,
#                     'value': val,
#                     'position': (int(round(x)), int(round(y))),
#                     'extracted': False
#                 })
    
#     # Mark which OCR detections were actually extracted
#     for ln in lengths:
#         x, y, w, h = ln['bbox']
#         cx, cy = int(round(x + w/2)), int(round(y + h/2))
#         # Find closest OCR detection
#         for ocr_item in numeric_ocr:
#             ox, oy = ocr_item['position']
#             if abs(cx - ox) < 20 and abs(cy - oy) < 20:
#                 ocr_item['extracted'] = True
#                 ocr_item['extracted_value'] = ln['value']
#                 break
    
#     # Print table
#     print(f"\n  {'#':<3} {'OCR Text':<15} {'Position':<15} {'Extracted':<12} {'Status':<20} {'Notes':<20}")
#     print('  ' + '-'*90)
    
#     row_num = 1
#     for ocr_item in numeric_ocr:
#         text = ocr_item['text']
#         x, y = ocr_item['position']
#         extracted = '✓ ' + str(ocr_item.get('extracted_value', '')) if ocr_item['extracted'] else '✗'
#         status = 'EXTRACTED' if ocr_item['extracted'] else 'NOT EXTRACTED'
        
#         # Add notes for non-extracted items
#         notes = ""
#         if not ocr_item['extracted']:
#             # Could be due to various reasons: overlapping with labels, outlier, etc.
#             notes = "(filtered out)"
        
#         pos_str = f"({x}, {y})"
#         print(f"  {row_num:<3} {text:<15} {pos_str:<15} {extracted:<12} {status:<20} {notes:<20}")
#         row_num += 1
    
#     # Summary statistics
#     extracted_count = sum(1 for item in numeric_ocr if item['extracted'])
#     total_count = len(numeric_ocr)
    
#     print('  ' + '-'*90)
#     print(f"  Summary: {extracted_count} extracted out of {total_count} detected numeric values")
#     print(f"  Extraction rate: {100*extracted_count//total_count if total_count > 0 else 0}%")
#     print('  ' + '='*90)
#     print()
