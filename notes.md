# Wire Detection System - Component Detection Overview

The wire detection system detects electrical components in automotive wiring diagrams through specialized detector modules.

## Component Detection Methods

Each component type is detected via a dedicated detector module in `src/detectors/`:

| Component | Script | Detection Method | Key Libraries |
|-----------|--------|------------------|----------------|
| **Text** | `ocr_detector.py` | PaddleOCR with upscaling | PaddleOCR |
| **Tape Labels** | `tape_detector.py` | OCR text matching + regex patterns (VT-*, AT-*, MLC*) | OpenCV, regex |
| **Connectors** | `connector_detector.py` | Shape detection (rectangles w/ internal lines) + OCR "DELPHI" | OpenCV contours |
| **Blue Clips** | `clip_detector.py` | HSV blue mask + HoughCircles (circular shapes) | OpenCV HSV/circles |
| **Wire Lengths** | `length_detector.py` | Numeric OCR pattern matching + outlier filtering | PaddleOCR, regex |
| **Wires** | `wire_detector.py` | Connected Components Labeling on dark/edge pixels | OpenCV CCL |

All detectors are orchestrated by `run_detector.py:main()` which sequentially calls each detector and compiles results into a connectivity graph.

---

## Detection Methods - Alternative & Legacy Approaches

### OCR Text Detection

**Current Method**: `ocr_full()` - Single-pass PaddleOCR with basic angle calculation

**Alternative Method**: `ocr_full_lengths()` - Two-pass specialized OCR for numeric annotations
- **Pass 1**: 480px tiles at 0° (horizontal text) — catches most numeric labels
- **Pass 2**: 320px tiles at 11 rotation angles [30°, 45°, 60°, 75°, 90°, 115°, 130°, 270°, 315°, 345°, 330°]
- **Technique**: Image tiling + 2x upscaling to avoid PaddleOCR internal downscaling
- **Use case**: Numeric wire length annotations that appear at various angles

**Additional Helper**: `ocr_region()` - OCR a specific bounding box crop (used for re-checking detected regions)

---

### Tape Label Detection

**Current Method**: Multi-pass OCR-driven approach
- **Pass 1**: Direct token matching — single OCR tokens matching TAPE_PATTERNS regex (VT-*, AT-*, MLC*)
- **Pass 3**: Region re-OCR — re-OCR crops around detected VT/AT prefix tokens with 60px rightward expansion

**Legacy Method** (commented out): **Pass 2 - Token Reconstruction**
- Merges nearby OCR tokens: "VT" + "BK" → "VT-BK"
- Proximity check: 60px horizontal gap, 15px vertical alignment
- **Why disabled**: Generated false positives from unrelated nearby text
- **Code location**: Lines 60-92 in `tape_detector.py` (commented block)

**Deduplication**: Proximity-based within 30px center distance using label name

---

### Wire Length Detection

**Current Method**: `detect_wire_lengths()` with adaptive thresholds
- Numeric pattern matching: `^[\(\+]*\d{1,4}[\+\)]*$` (e.g., 0, (25), +150+)
- Outlier filtering based on wire length scoring function
- Deduplication by spatial proximity (merge within 20px)

**Pre-processing**: `merge_token_fragments()`
- Merges nearby same-baseline OCR fragments: "(" + "25" + ")" → "(25)"
- Horizontal gap threshold: adaptive based on median token height
- **Guard logic**: Prevents merging numeric tokens with label keywords (VT-, DELPHI, etc.)
- **Output**: Cleaned OCR data ready for pattern matching

**Scoring Function**: `score_wire_length_value()`
- Prefers round multiples of 25mm (automotive standard)
- Filters unrealistic values (< 10mm or > 600mm)
- Weights common lengths (25, 50, 75, 100, 150, 200, 250)

---

### Wire Detection

**Current Method**: Connected Components Labeling + PCA Endpoints + Endpoint Graph Merging
1. Wire mask creation: HSV dark + Canny edges AND operation
2. Component subtraction: Remove filled blobs as topology barriers
3. Gap bridging: 9x9 ellipse dilation (2 iterations)
4. Connected Components Labeling: Extract individual segments
5. PCA endpoint extraction: Project pixels onto primary axis to find true wire endpoints
6. Endpoint graph + BFS: Three-criterion merging
   - Proximity: < 100px apart
   - Collinearity: angle difference < 45°
   - Component blocking: gap doesn't cross detected components

**Component Detection** (helper): `detect_components()`
- Identifies filled blobs (connectors, clips) as electrical topology barriers
- Criteria: aspect ratio < 4.0, fill ratio > 0.25, area 80-8000px
- **Purpose**: Create mask to prevent wires from incorrectly snapping to component edges

**Merging Philosophy**: Endpoint graph (adjacency list) with BFS instead of Union-Find
- **Advantage**: Explicit edges show merge candidates clearly
- **Debugging**: Graph structure directly visualizable
- **Flexibility**: Easy to add/modify merge criteria
- **Complexity**: O(n²) pairwise testing acceptable for typical 5-30 segments

### Validation Challenges

- Ground truth difficult to obtain (requires manual labeling)
- Different interpretations of "correct" edge for same diagram
- Cascading errors from earlier phases

---

## Debugging Tips

### Wire Not Detected

**Diagnostics**:
```python
# Check wire mask
cv2.imshow('Wire Mask', wire_mask)

# Check after dilation
cv2.imshow('Dilated', dilated_mask)

# Check CCL labels
labels = cv2.connectedComponentsWithStats(dilated_mask)[1]
cv2.imshow('CCL Labels', labels.astype(np.uint8) * 10)

# Check filtering
print(f"Segments before filtering: {num_labels}")
print(f"Segments after filtering: {len(valid_segments)}")
```

**Common Causes**:
1. Wire color not in mask range (adjust inversion threshold)
2. Component masking too aggressive (increase erase margin)
3. Dilation not bridging gaps (increase kernel size/iterations)
4. Merged with adjacent wire (reduce angle/proximity thresholds)

### Component Snapping Wrong

**Diagnostics**:
```python
# Visualize component snap points
cv2.circle(img, snap_point, 5, (0, 255, 0), -1)

# Print snap distances
print(f"Snap to node: {dist_to_node}px at {nearest_node}")
```

**Common Causes**:
1. Snap distance threshold too large
2. Component detection failed (check HSV/shape parameters)
3. Graph has no nodes nearby (check phase 3 output)

---
