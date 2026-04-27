# Wire Detection System - Component Detection Overview

The wire detection system detects electrical components in automotive wiring diagrams through specialized detector modules.

## Component Detection Methods

Each component type is detected via a dedicated detector module in `src/detectors/`:

| Component | Script | Detection Method | Key Libraries |
|-----------|--------|------------------|----------------|
| **Text** | `ocr_detector.py` | PaddleOCR with upscaling | PaddleOCR |
| **Tape Labels** | `tape_detector.py` | OCR text matching + regex patterns (VT-*, AT-*) | OpenCV, regex |
| **Connectors** | `connector_detector.py` | Shape detection (rectangles w/ internal lines) + OCR "DELPHI" | OpenCV contours |
| **Blue Clips** | `clip_detector.py` | HSV blue mask + HoughCircles (circular shapes) | OpenCV HSV/circles |
| **Wire Dimensions** | `dimension_detector.py` | Numeric OCR pattern matching + outlier filtering | PaddleOCR, regex |
| **Wires** | `wire_detector.py` | Connected Components Labeling on dark/edge pixels | OpenCV CCL |

All detectors are orchestrated by `run_detector.py:main()` which sequentially calls each detector and compiles results into a connectivity graph.

---

## Detection Methods - Alternative & Legacy Approaches

### OCR Text Detection

**Primary Method**: `ocr_full()` - Single-pass PaddleOCR for general text tokens (connector labels, junction IDs, tape prefixes)

**Complementary Specialized Pass**: `ocr_full_dimensions()` - Runs alongside `ocr_full()` for numeric-only annotations
- **Pass 1**: 480px tiles at 0° (horizontal text) — catches most numeric labels
- **Pass 2**: 320px tiles at 11 rotation angles [30°, 45°, 60°, 75°, 90°, 115°, 130°, 270°, 315°, 345°, 330°]
- **Technique**: Image tiling + 2x upscaling to avoid PaddleOCR internal downscaling
- **Use case**: Wire dimension annotations that appear at various angles
- **Integration**: Results merged into `detect_tape_labels()` and fed exclusively to `detect_wire_dimensions()`

**Additional Helper**: `ocr_region()` - OCR a specific bounding box crop (used for re-checking detected regions)

---

### Tape Label Detection

**Current Method**: Multi-source OCR with targeted re-OCR
1. **Token aggregation**: Merges tokens from `ocr_full()`, `ocr_full_dimensions()`, and `ocr_upscaled()` (3x upscaling for small text)
   - Deduplicates by 10px center proximity
2. **Direct regex matching**: `TAPE_PATTERNS = (VT|AT)\s*-\s*[A-Z]{1,2}`
   - Catches full labels in single tokens: `VT-BK`, `VT-WH`, `VT-PK`, `AT-BK`, etc.
   - Matches optional whitespace around hyphens
3. **Region re-OCR**: For any token matching `^VT$|^AT$|^VT-$|^AT-$`
   - Crops grayscale ±60px rightward expansion
   - Re-OCRs crop with `ocr_region()` to catch split/missing fragments
   - **Example**: Bare `AT` token triggers re-OCR → recovers full `AT-BK` label
   - **Note**: `COT-BK` requires full string in OCR (no prefix trigger for region re-OCR)
4. **Deduplication**: Proximity-based within 30px center distance; same label → keep first occurrence

---

### Wire Dimension Detection

**Current Method**: `detect_wire_dimensions()` with adaptive thresholds
- Numeric pattern matching: `^[\(\+]*\d{1,4}[\+\)]*$` (e.g., 0, (25), +150+)
- Outlier filtering based on wire dimension scoring function
- Deduplication by spatial proximity (merge within 20px)

**Pre-processing**: `merge_token_fragments()`
- Merges nearby same-baseline OCR fragments: "(" + "25" + ")" → "(25)"
- Horizontal gap threshold: adaptive based on median token height
- **Guard logic**: Prevents merging numeric tokens with label keywords (VT-, DELPHI, etc.)
- **Output**: Cleaned OCR data ready for pattern matching

**Scoring Function**: `score_wire_dimension_value()`
- Prefers round multiples of 25mm (automotive standard)
- Filters unrealistic values (< 10mm or > 600mm)
- Weights common dimensions (25, 50, 75, 100, 150, 200, 250)

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

---
## Running the Pipeline (`run.py`)

### Phases (all modes)

| Phase | Step | Description |
|-------|------|-------------|
| 1 | OCR | `ocr_full()` (general text) + `ocr_full_dimensions()` (numeric annotations, 11 rotation angles) |
| 2 | Detection | Tape labels, connectors, clips, wire dimensions (each gated by extract filters) |
| 3 | Connectivity | Wire mask or wire blob detection → graph building (see below) |
| 4 | Output | `wiring_diagram_annotated.png` + `connectivity_graph.json` |

---

### Default (no flags) — Mask-Tracer Pipeline

1. `create_wire_mask()` — binary mask of wire pixels with component regions erased
2. `build_component_nodes()` — flat dict of connectors, clips, tapes, junctions
3. `trace_mask_connectivity()` — morphological closing → CCL noise removal → seed each component → multi-source BFS flood → wherever two labels meet = connection
4. `assign_wire_properties()` — attaches `wire_type` and `dimension_mm` to each edge (80px proximity to path polyline)
5. `convert_to_legacy_format()` — converts NetworkX graph to reporter format
6. **Fallback**: if BFS produces 0 edges, automatically runs heuristic pipeline

---

### `--legacy` — Heuristic Pipeline

1. `detect_wires()` — 5-phase blob detector (HSV+Canny → CCL → PCA endpoints → endpoint graph → BFS merge)
2. `filter_wires_by_components()` — drops wires with endpoints > 50px from any component
3. `build_connectivity_graph_heuristic()`:
   - Pass 1: project tape labels onto wire segments (80px perpendicular corridor)
   - Pass 2 (tape-anchor): pair nodes via angle heuristic (>60° separation, <650px)
   - Supplemental hardcoded edges: J20→Connector-1, J20→MLC001, J20→X519

---

### `--extract-only=<items>` / `--skip=<items>`

Valid items: `tapes`, `connectors`, `wires`, `dimensions`, `clips`

- `--extract-only=tapes,connectors` — disables all detectors except those listed
- `--skip=clips,dimensions` — disables only those detectors; all others run
- Both flags are composable with `--legacy`