# Wire Detection System - Technical Notes

## Extraction Process Overview

The wire detection system extracts electrical topology from automotive wiring diagrams through a multi-phase pipeline. 

**IMPORTANT**: This system implements **TWO extraction approaches**:
1. **Default (Modern)**: Skeleton-based wire extraction (recommended, more robust)
2. **Legacy**: Connected Components Labeling + endpoint graph (fallback, `use_legacy=True`)

This document describes both approaches. The default skeleton-based method is used unless explicitly overridden.

---

## Phase 1: OCR Text Detection

### Purpose
Extract text elements from the diagram including:
- Tape labels (VT-WH, VT-BK, AT-BK, COT-BK, VT-PK)
- Connector identifiers (C1, C2, X508, X510, X519)
- Junction labels (J20, MLC001, etc.)
- Length annotations (numeric measurements)

### Implementation

**Library Used**: PaddleOCR

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
results = ocr.ocr(image)
```

**Method**: Two-pass tiling OCR with selective angle rotation
- Pass 1: 480px tiles at 0° (horizontal text) - handles most text
- Pass 2: 320px tiles at 11 angles [30, 45, 60, 75, 90, 115, 130, 270, 315, 345, 330]

**Rationale**: 
- Wiring diagrams often contain text at arbitrary angles
- Two-pass strategy: first pass catches most horizontal text efficiently
- Second pass with selective angles covers rotated labels (8 cardinal + diagonals)
- Image tiling + upscaling avoids PaddleOCR internal downscaling
- Deduplicates results to avoid redundant detections
- More efficient than blind 36-angle rotation

**Libraries Tested**:
1. **PaddleOCR** (CHOSEN) - Multi-language support, fast, angle classification
2. **Tesseract OCR** - Limited rotation support, slower
3. **EasyOCR** - Similar capabilities to PaddleOCR but slower

**Why PaddleOCR**: Industry-standard for Chinese/multilingual documents, built-in angle classification, good balance of speed and accuracy.

---

## Phase 2: Component Detection

### 2.1 Tape Label Detection

**Components Detected**: Small colored rectangles with associated text labels

**Process**:
1. Morphological shape detection (8-30px wide, aspect ratio 3:1-15:1)
2. OCR text matching using regex patterns (TAPE_PATTERNS)
3. Hungarian algorithm to pair detected shapes with text labels

**Libraries Used**:
- OpenCV (cv2.matchTemplate, cv2.morphologyEx)
- scipy.optimize (linear_sum_assignment for Hungarian matching)

**Why This Works**:
- Tape labels have distinctive small rectangular shape
- Regex patterns filter tape-specific text (e.g., "VT-WH")
- Hungarian matching handles 1-to-1 correspondence between shapes and text

### 2.2 Connector Detection

**Components Detected**: Delphi electrical connector bodies

**Process**:
1. Detect small rectangles via contour analysis
2. Identify internal structural lines (characteristic of Delphi connectors)
3. Match with OCR text for connector IDs

**Libraries Used**: OpenCV (cv2.findContours, cv2.contourArea, cv2.moments)

**Implementation Details**:
- Connectors are **NOT** added as graph nodes automatically
- Instead, serve as anchor points for edge derivation
- Snap distance threshold: 160px

**Why Separate from Other Components**:
- Connectors have unique structural signature (internal lines)
- Simplifies electrical topology interpretation
- Allows connector-specific snapping logic in graph construction

### 2.3 Clip Detection

**Components Detected**: Blue Z-marker clips (wire bundle markers)

**Process**:
1. HSV color filtering (blue range specific to clip markers)
2. Connected component labeling
3. Circular shape detection via HoughCircles
4. Density filtering (fill ratio > 25%)

**Libraries Used**: 
- OpenCV (cv2.inRange, cv2.HoughCircles, cv2.connectedComponentsWithStats)

**Implementation Details**:
- Clips are **circular** or nearly circular
- Distinguished from junctions by shape and color
- Snap distance threshold: 100px
- Clips typically don't connect to other clips (filtered as artifacts)

**Why This Works**:
- Clips have distinctive blue color (HSV-specific range)
- Circular shape is unique among diagram elements
- HoughCircles effectively finds circular features

### 2.4 Length Annotation Detection

**Components Detected**: Wire length measurements in millimeters

**Process**:
1. Extract numeric text via OCR with numeric pattern filtering
2. Locate measurements along wire paths
3. Deduplicate nearby detections (merge within 20px)
4. Outlier filtering (remove unrealistic values)

**Libraries Used**: 
- PaddleOCR (with custom post-processing)
- OpenCV (spatial proximity filtering)

**Why Separate Pipeline**:
- Numeric OCR is noisy without dedicated filtering
- Deduplication necessary due to multi-angle OCR
- Outlier filtering removes false positives from similar-looking patterns

---

## Phase 3: Wire Extraction - Core Algorithm

This is the **most complex and critical** phase. The system supports TWO different implementations:

### CURRENT IMPLEMENTATION: Skeleton-Based Approach (Default)

The default mode (in run_detector.py, `use_legacy=False`) uses:
1. Wire mask creation from HSV + Canny edges
2. **Skeletonization** (scikit-image) to 1-pixel centerlines
3. **Graph extraction** using sknw library
4. **Component mapping** to skeleton nodes
5. **Property assignment** (tape labels, lengths)

See **Phase 4: Skeletonization & Graph Extraction** below for details.

### LEGACY IMPLEMENTATION: Connected Components + Endpoint Graph (CCL-Based)

Alternative mode (available with `use_legacy=True`) uses Connected Components Labeling:
1. Wire mask creation and dilation
2. **Connected Components Labeling (CCL)** to find segments
3. **PCA-based endpoint extraction** for each segment
4. **Endpoint graph + BFS merging** to reconstruct wires

See **Phase 3A: Connected Components Legacy Approach** below for details.

Both approaches are mathematically sound but the skeleton-based method is more robust for real diagrams.

---

## Phase 3A: Wire Extraction - Connected Components Legacy Approach

This describes the CCL-based implementation, kept for reference and fallback use.

### 3.1 Wire Mask Creation

**Purpose**: Extract wire pixels from the diagram image

**Process**:
```
1. Component Detection Phase (separate pass)
   - Detect filled blobs (connectors, clips) as topology barriers
   - Create component mask for later use

2. Wire Pixel Extraction (detect_wires function)
   a. HSV dark color filtering: cv2.inRange(HSV, dark_min, dark_max)
      - Captures dark wire colors in HSV space
   b. Canny edge detection: cv2.Canny(gray, 30, 100)
      - Captures wire boundary edges
   c. Combine with AND: wire_mask = HSV_mask & Canny_mask
      - Only pixels dark AND having edges are wire pixels
   d. Remove component regions: wire_mask = wire_mask & NOT(component_mask)
      - Respects electrical topology boundaries

3. Gap Bridging
   a. Dilate with 9x9 ellipse kernel (2 iterations) to connect dashes
   b. Morphological close (3x3, 1 iteration) for cleanup
   c. Morphological open (3x3, 1 iteration) to remove small artifacts
   d. Remove components again from bridged mask (prevent bleed-through)
```

**Libraries Used**: OpenCV (cv2.inRange, cv2.Canny, cv2.dilate, cv2.morphologyEx, cv2.bitwise_and)

**Key Design Decisions**:
- **Dual filtering (HSV + Canny)**: Ensures wire pixels match both color AND edge criteria
  - Filters out large dark areas that aren't wire edges
  - Prevents thick connector bodies from being mistaken for wires
- **Component masking**: Remove electrical components before and after dilation
  - Prevents wires from snapping incorrectly to component boundaries
  - Respects electrical topology (wires don't pass through components)
  - Second masking prevents dilation from creating bridges across components
- **Gentle morphology**: 3x3 kernel chosen to bridge dashes without merging separate wires

### 3.2 Historical Approaches: What Didn't Work

#### Attempt 1: HoughLinesP (Line Segment Detection)
**Approach**: Detect line segments in wire mask using probabilistic Hough transform

**How It Worked**:
```python
lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=50, minLineLength=20, maxLineGap=20)
```

**Why It Failed**:
- **Dashed wires broke into fragments**: maxLineGap parameter failed to bridge dashes
  - Typical dash gap: 10-50px depending on zoom level
  - Setting maxLineGap=50 caused perpendicular wires to incorrectly merge
  - Setting maxLineGap=20 resulted in one physical wire split into 20+ segments
- **26 false positives** after component filtering
- HoughLinesP designed for **continuous lines**, not dashed patterns
- No concept of "endpoints" in HoughLinesP output

**Lesson Learned**: 
- Hough transforms assume continuous patterns
- Dashed/dotted patterns require different approach
- Need explicit endpoint extraction, not just line parameters

#### Attempt 2: Simple Dilate + CCL
**Approach**: Dilate wire mask heavily to close gaps, then use Connected Components Labeling

**How It Worked**:
```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
dilated = cv2.dilate(mask, kernel, iterations=2)
num_labels, labels = cv2.connectedComponentsWithStats(dilated)
```

**Why It Partially Worked**:
- Merged dashed wire segments into connected blobs
- Better than HoughLinesP for fragmented patterns
- **Only produced 5 raw segments** instead of needed 14 final wires
- Over-dilation caused **false merges** between nearby parallel wires
- No endpoint information (dilated blobs lost wire direction)

**Why Endpoint Extraction Failed**:
- Dilated blobs lost sharp endpoints
- Using blob centroid caused **incorrect merges** in branching areas
- Different angled dashes failed to bridge at all angles

**Lesson Learned**:
- Dilation alone insufficient for complex topologies
- Need additional structural information beyond just connectivity
- Endpoint precision critical for correct merging

### 3.3 Current Approach: Connected Components + PCA (Recommended)

**Purpose**: Extract individual wire segments with precise endpoints

**Process**:
```
1. Create wire mask from HSV dark color AND Canny edges (both must match)
2. Remove detected component regions from wire mask (KEY BREAK)
3. Gap filling: Dilate with 9x9 ellipse kernel (2 iterations) to bridge dashes
4. Morphological cleanup: MORPH_CLOSE and MORPH_OPEN (1 iteration each)
5. Remove components again from bridged mask (prevent bleed-through)
6. Connected Components Labeling: cv2.connectedComponentsWithStats()
7. For each component:
   a. Filter by area (20-6000 px) and bbox aspect ratio (> 1.3)
   b. Filter by fill ratio (> 5% density)
   c. PCA analysis to find primary axis (cv2.PCACompute)
   d. Project all pixels onto primary axis
   e. Find min/max projections as wire endpoints
```

**Libraries Used**:
- OpenCV (cv2.connectedComponentsWithStats, cv2.PCACompute)
- NumPy (eigenvalue decomposition, principal component analysis)

**Why This Works Well**:
- **Precise endpoints**: PCA finds actual wire termination points
- **Handles dashes**: 9x9 dilation bridges gaps effectively
- **Robust**: Multiple filtering criteria (area, elongation, fill ratio)
- **Fast**: O(n log n) complexity, scales well
- **Geometric**: Aspect ratio naturally distinguishes wires from noise

**Mathematical Foundation**:
```
For each connected component:
- Center = mean of all pixel positions
- Covariance matrix C = (pixel - center)^T @ (pixel - center)
- PCA eigenvectors point along wire axis (highest eigenvalue)
- Project all pixels: p_i · v_1 (dot product with primary axis)
- Endpoints = argmin/argmax(projections)
```

### 3.4 Wire Segment Merging: Endpoint Graph with BFS Traversal

**Purpose**: Merge segmented wire pieces back into complete wires

**Problem Being Solved**:
- Dashed wires detected as 2-5 separate segments depending on gap pattern
- Simple CCL + PCA produces fragmented result
- Need intelligent merging respecting electrical topology

**Algorithm Overview**:
Uses endpoint graph (adjacency list) instead of Union-Find:
```
PHASE A: Component Detection
- Separate pass to detect filled blobs (connectors, clips, junctions)
- Criteria: aspect ratio < 4.0, fill ratio > 0.25, area 80-8000px
- Create component mask as topology barrier

PHASE B: Wire Segment Extraction (as described above)

PHASE C: Endpoint Graph Building with THREE Criteria
For each pair of segments and endpoint pair:
1. Proximity Check: Endpoints < 100px apart
   - Allows large gaps in dashed wires
2. Collinearity Check: Primary axes angle difference < 45°
   - Permits wider angle variation for flexibility
3. Component-Free Path: Gap doesn't cross any detected component
   - Respects electrical isolation
   - Prevents merging segments separated by connector

If all 3 criteria pass, add edge to endpoint graph

PHASE D: Path Tracing via BFS
- Find connected components in endpoint graph (not Union-Find)
- For each component: trace path via BFS from terminal node
- Collect all segment endpoints in path
- Re-apply PCA to find true final wire endpoints

PHASE E: Deduplication
- Remove duplicate wires (same endpoints, possibly reversed)
```

**Data Structure**: Endpoint Graph (Adjacency List)
```python
endpoint_graph = {}  # (seg_idx, ep_type) -> list of connected endpoints
for seg_i in range(n):
    for seg_j in range(i+1, n):
        for endpoint_pair in all_4_combinations:
            if meets_3_criteria(seg_i, seg_j, endpoint_pair):
                endpoint_graph[endpoint_a].append(endpoint_b)
                endpoint_graph[endpoint_b].append(endpoint_a)

# Then BFS to find connected components and trace wire paths
for start in endpoints:
    component = bfs_connected_component(start)
    wire_path = trace_path_from_terminal(component)
```

**Why This Approach** (instead of Union-Find):
- Explicit edge representation shows merge candidates clearly
- BFS naturally finds connected components
- Graph structure directly maps to wire path traversal
- Easier to debug and add merge criteria
- O(n^2) pairwise testing, acceptable for typical diagrams (n=10-30 segments)

**Key Parameters** (actual values from code):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| GAP_THRESH | 100px | Allows large gaps in heavily dashed wires |
| ANGLE_THRESH | 45° | Permits wider angle variation for curved wire paths |
| MIN_CCL_AREA | 20px² | Filters noise fragments |
| MAX_CCL_AREA | 6000px² | Prevents over-merging large blobs |
| COMPONENT_ASPECT | < 4.0 | Distinguishes compact components from elongated wires |
| COMPONENT_FILL | > 0.25 | Density threshold for meaningful blob |

**Results**:
- **Detected: 14 wires** (100% match to expected)
- **Component-aware**: Respects connectors/clips as boundaries
- Graph-based: Clear representation of segment relationships
- Topology-aware: Multi-criteria avoids spurious merges

**Algorithm Design Notes**:
Implementation uses endpoint graph (adjacency list) with BFS instead of Union-Find:
- Simpler to understand: explicit edges represent merge candidates
- Graph structure naturally maps to wire path tracing
- BFS ensures connected component discovery
- Flexible for adding more complex merge criteria in future

---

## Phase 4: Skeletonization & Graph Extraction (Modern Default Implementation)

**Note**: This is the current default approach used when `use_legacy=False` (recommended).
Convert wire mask to skeleton (1-pixel-wide centerline), extract graph structure

### Process
```
1. Skeletonize: scikit-image thin skeletonization
   - Iteratively remove outer pixels while preserving connectivity
   - Result: 1-pixel-wide wire paths
   
2. Prune: Remove isolated pixels and short spurs
   - min_branch_length = 10px (prune spurs shorter than this)
   - Cleans up noise from morphological operations
   
3. Extract Graph: Use sknw library
   - Convert skeleton to MultiGraph
   - Nodes at junction points (degree ≠ 2)
   - Edges contain actual path pixels
   
4. Filter Graph: Multiple cleanup passes
   - Consolidate nearby junctions (radius 15px)
   - Contract degree-2 pass-through nodes
   - Bridge gap endpoints (radius 50px)
   - Re-contract after bridging
```

**Libraries Used**:
- scikit-image (cv2.ximgproc.thinning or skimage.morphology.skeletonize)
- sknw (skeleton to graph conversion)
- NetworkX (graph data structure)

**Why Skeletonization**:
- Reduces 2-3 pixel wide wires to single path representation
- Enables junction detection (nodes with degree > 2)
- Makes endpoint extraction cleaner than CCL+PCA
- Standard approach in circuit diagram analysis

### Alternative Approaches Tested

#### Approach: Direct skeleton without CCL pre-filtering
**Problem**: Skeletonization of dilated blobs created spurious junction nodes
**Solution**: Combine CCL endpoint detection with skeletonization

---

## Phase 5: Component-to-Graph Mapping & Edge Assignment

NOTE: This phase is currently under development and not fully accurate.

### Conceptual Overview

Components (connectors, clips, junctions) must be mapped to the skeleton graph:

```
1. Identify main wire network (largest connected component)
2. For each component:
   - Snap to nearest skeleton node (prefer main CC)
   - If no nearby node: snap to nearest edge, create synthetic node
3. Build component graph showing logical connections
```

**Snap Distance Limits**:
- Clips: 100px
- Connectors: 160px  
- Junctions: 200px

### Known Limitations

- **Connection accuracy**: Heuristic-based endpoint matching may fail
- **Component-to-edge snapping**: Creating synthetic nodes changes connectivity
- **Dimension assignment**: Length/wire_type matching not fully validated
- **Missing circular logic**: No feedback loop to validate component positions

**Why Not Fully Implemented**:
- Requires electrical knowledge to validate connections
- Different diagram conventions need different thresholds
- Dependency on other components' accuracy (cascading errors)

---

## Phase 6: Output Generation

### Output Formats

**1. JSON Connectivity Graph** (`connectivity_graph.json`)
```json
{
  "nodes": [
    {"id": "connector_1", "type": "connector", "x": 100, "y": 150},
    {"id": "J20", "type": "junction", "x": 250, "y": 150}
  ],
  "edges": [
    {"source": "connector_1", "target": "J20", "wire_types": ["VT-WH"], "length_mm": 500}
  ]
}
```

**2. Annotated Image**
- Rectangles around detected components
- Labels with component IDs
- Wire paths highlighted

**3. Console Report**
- Detected element counts
- Extracted nodes and edges
- Verification table

---

## Critical Design Decisions

### Why Component Masking Before Wire Detection?

**Decision**: Remove components from wire mask before skeleton extraction

**Rationale**:
1. **Electrical correctness**: Wires don't pass "through" components in real circuits
2. **Endpoint accuracy**: Connector bodies shouldn't be wire endpoints
3. **Graph clarity**: Components appear as nodes, not part of wire path
4. **Topology preservation**: Enables correct component-to-junction connectivity

**Impact**: Prevents artifacts where wire endpoints snap to connector edges

### Why PCA Over Moment-Based Endpoints?

**Decision**: Use Principal Component Analysis to find wire endpoints

**Candidates**:
- Blob centroid: Doesn't give endpoint info
- Convex hull extrema: Noisy with dilated blobs
- PCA projection: Finds actual wire axis, robust to shape

**Why PCA Wins**:
- Mathematically principled (eigenvalue decomposition)
- Handles elongated shapes naturally
- Robust to noise and dilation artifacts
- Clear interpretation: primary eigenvector = wire axis

### Why Endpoint Graph + BFS Over Other Approaches?

**Decision**: Use endpoint graph (adjacency list) with BFS instead of Union-Find or scipy.sparse.csgraph

**Rationale**:
1. **Clarity**: Explicit edges between endpoints show merge relationships
2. **Debuggability**: Graph structure directly visualizable and traceable
3. **Flexibility**: Easy to add additional merge criteria or constraints
4. **Directness**: BFS naturally finds connected components, path tracing is straightforward
5. **Simplicity**: No need for path compression or rank heuristics

**Alternatives Considered**:
- Union-Find: Simpler O(n log n) complexity, but harder to debug merge decisions
- Scipy sparse.csgraph: Too heavyweight for pairwise segment merging
- Simple BFS/DFS: Not enough structure for merge criterion validation

**Actual Complexity**:
- Pairwise testing: O(n^2) where n = number of segments (typically 5-30)
- BFS for each component: O(m) where m = number of edges
- Total: O(n^2 + m) - acceptable for typical diagrams

---

## Parameter Tuning Guide

| Parameter | Actual Value | Range | Sensitivity |
|-----------|---------------|-------|-------------|
| GAP_THRESH | 100px | 50-150px | High (controls merging aggressiveness) |
| ANGLE_THRESH | 45° | 20-60° | Medium (prevents false merges) |
| DILATION_KERNEL | 9x9 ellipse | 7x7 to 11x11 | High (affects gap bridging) |
| DILATION_ITER | 2 | 1-3 | Medium (more iterations = more merging) |
| COMPONENT_ASPECT | 4.0 | 3.0-6.0 | Medium (affects component vs wire classification) |
| COMPONENT_FILL | 0.25 (25%) | 0.15-0.35 | Low (affects noise filtering) |
| MIN_BBOX_ASPECT | 1.3 | 1.0-2.0 | Medium (filters thin blobs) |
| MIN_FILL_RATIO | 0.05 (5%) | 0.02-0.15 | Medium (filters sparse blobs) |

**Tuning Strategy**:
1. Test on diverse diagrams
2. Adjust GAP_THRESH first (most sensitive)
3. Then ANGLE_THRESH (prevents false merges)
4. Fine-tune morphological parameters last

---

## Known Issues & Future Improvements

### Current Issues

1. **Dashed Wire Inconsistency**
   - Dash pattern varies by diagram zoom level
   - Single kernel size may not work for all inputs
   - Solution: Adaptive gap detection based on dash frequency analysis

2. **Component Snapping Ambiguity**
   - When multiple components near wire endpoint
   - Current: snap to nearest (may be wrong)
   - Solution: Use component type preferences + distance weighting

3. **Junction Detection Brittleness**
   - OCR-based junction detection misses unlabeled junctions
   - Skeletonization finds junctions but no semantic info
   - Solution: Hybrid approach combining both methods

### Future Improvements

- Machine learning for component classification (instead of heuristics)
- Validation against known electrical rules
- Interactive correction UI for user feedback
- Adaptive parameters based on diagram characteristics
- Support for different diagram standards (SAE, IEC, etc.)
- 3D harness reconstruction from 2D diagram

---

## Testing & Validation

### Metrics Used

1. **Completeness**: # Detected wires / # Expected wires
2. **Accuracy**: # Correctly merged / # Total detections
3. **Precision**: # Correct edges / # Detected edges
4. **False positives**: # Incorrect detections

### Current Performance

On test diagram with 14 wires:
- Detected: 14 wires (100% completeness)
- Correctly merged: 14/14 (100% accuracy)
- Edge accuracy: Dependent on Phase 5 (not fully validated)

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
