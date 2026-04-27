# Segment Detection System - Component Detection Overview

The segment detection system detects electrical components in automotive segment diagrams through specialized detector modules.

## Component Detection Methods

Each component type is detected via a dedicated detector module in `src/detectors/`:

| Component | Script | Detection Method | Key Libraries |
|-----------|--------|------------------|----------------|
| **Text** | `ocr_detector.py` | PaddleOCR with upscaling | PaddleOCR |
| **Tape Labels** | `tape_detector.py` | OCR text matching + regex patterns (VT-*, AT-*) | OpenCV, regex |
| **Connectors** | `connector_detector.py` | Shape detection (rectangles w/ internal lines) + OCR "DELPHI" | OpenCV contours |
| **Blue Clips** | `clip_detector.py` | HSV blue mask + HoughCircles (circular shapes) | OpenCV HSV/circles |
| **Segment Dimensions** | `dimension_detector.py` | Numeric OCR pattern matching + outlier filtering | PaddleOCR, regex |
| **Segments** | `segment_detector.py` / `mask_tracer.py` | Mask tracing (default) or Canny+CCL (legacy `--legacy` flag) | OpenCV Canny/HSV/CCL

All detectors are orchestrated by `run_detector.py:main()` which sequentially calls each detector and compiles results into a connectivity graph.

---

## Detection Methods - Alternative & Legacy Approaches

### OCR Text Detection

**Primary Method**: `ocr_full()` - Single-pass PaddleOCR for general text tokens (connector labels, junction IDs, tape prefixes)

**Complementary Specialized Pass**: `ocr_full_dimensions()` - Runs alongside `ocr_full()` for numeric-only annotations
- **Pass 1**: 480px tiles at 0° (horizontal text) — catches most numeric labels
- **Pass 2**: 320px tiles at 11 rotation angles [30°, 45°, 60°, 75°, 90°, 115°, 130°, 270°, 315°, 345°, 330°]
- **Technique**: Image tiling + 2x upscaling to avoid PaddleOCR internal downscaling
- **Use case**: Segment dimension annotations that appear at various angles
- **Integration**: Results merged into `detect_tape_labels()` and fed exclusively to `detect_segment_dimensions()`

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

### Segment Dimension Detection

**Current Method**: `detect_segment_dimensions()` with adaptive thresholds
- Numeric pattern matching: `^[\(\+]*\d{1,4}[\+\)]*$` (e.g., 0, (25), +150+)
- Outlier filtering based on segment dimension scoring function
- Deduplication by spatial proximity (merge within 20px)

**Pre-processing**: `merge_token_fragments()`
- Merges nearby same-baseline OCR fragments: "(" + "25" + ")" → "(25)"
- Horizontal gap threshold: adaptive based on median token height
- **Guard logic**: Prevents merging numeric tokens with label keywords (VT-, DELPHI, etc.)
- **Output**: Cleaned OCR data ready for pattern matching

**Scoring Function**: `score_segment_dimension_value()`
- Prefers round multiples of 25mm (automotive standard)
- Filters unrealistic values (< 10mm or > 600mm)
- Weights common dimensions (25, 50, 75, 100, 150, 200, 250)

---

### Segment Detection

**Default Method**: Binary Mask Tracing via `create_segment_mask()` + `trace_mask_connectivity()`

**Step 1 — `create_segment_mask()` (component_masker.py)** — Create clean binary mask:
1. **Otsu thresholding**: Gaussian blur (3×3) + inverted Otsu threshold → dark pixels = white (255), background = black (0)
2. **Remove colour highlights**: Erase blue clip blobs (HSV 95-135 H) and yellow tape backgrounds (HSV 20-35 H) via range masking
3. **Erase detected components**: Zero-out connector bboxes (±2px margin), clip circles (radius+3px), tape label boxes (±2px margin)
4. **Remove arrowheads**: Find small (20–350px area) convex 3–4 vertex contours → erase (dimension line end arrows)
5. **Remove dimension label endpoints**: Black out 20px circle around each non-parenthesized dimension annotation center
6. **Morphological close**: 3×3 ellipse close (2 iterations) to bridge remaining dash-dot gaps

**Step 2 — `trace_mask_connectivity()` (mask_tracer.py)** — BFS flood-fill from components:
1. **Mask cleaning**: Additional morphological close with directional kernels (horizontal 1×45, vertical 45×1, isotropic 19×19 ellipse ×2)
2. **Noise removal**: CCL to identify and filter tiny blobs (text chars, arrowheads) below `MIN_SEGMENT_AREA=80px`
3. **Component seeding**: Seed each component (connector, clip, junction) with circular region of influence (radius `COMPONENT_EXPAND=25px`, +15px for junctions)
4. **Multi-source BFS**: From each component's seed pixels, flood-fill outward along connected white pixels (8-connectivity)
5. **Connectivity mapping**: Label each seed pixel with its component ID; wherever two different component IDs meet on a pixel boundary → segment connection
6. **Graph building**: Convert labeled connectivity into NetworkX graph (one edge per component pair)

**Legacy Method** (`--legacy` flag): Canny Edge Detection + HSV Masking + Connected Components Labeling
1. **Edge detection**: Canny edge detection (thresholds 30-100)
2. **Dark mask**: HSV color range for dark pixels (V < 150, S < 100)
3. **Segment mask**: AND operation between Canny edges and dark HSV mask
4. **Component subtraction**: Remove filled blobs (connectors, clips) as topology barriers
5. **Gap bridging**: 9x9 ellipse dilation (2 iterations), then morphological close/open
6. **Connected Components Labeling (CCL)**: Extract individual segment traces from binary mask
7. **Heuristic pairing**: Tape-label anchoring to pair components into connections

**Component Detection** (helper): `detect_components()`
- Identifies filled blobs (connectors, clips) as electrical topology barriers
- Criteria: aspect ratio < 4.0, fill ratio > 0.25, area 80-8000px
- **Purpose**: Create mask to prevent segments from incorrectly snapping to component edges

---
## Running the Pipeline (`run.py`)

### Phases (all modes)

| Phase | Step | Description |
|-------|------|-------------|
| 1 | OCR | `ocr_full()` (general text) + `ocr_full_dimensions()` (numeric annotations, 11 rotation angles) |
| 2 | Detection | Tape labels, connectors, clips, segment dimensions (each gated by extract filters) |
| 3 | Connectivity | `create_segment_mask()` + `trace_mask_connectivity()` BFS flood-fill → graph building |
| 4 | Output | `segment_diagram_annotated.png` + `connectivity_graph.json` |

---

### Default (no flags) — Mask-Tracer Pipeline

1. `create_segment_mask()` — Otsu threshold + component erasure → clean binary mask of segment pixels only
2. `build_component_nodes()` — Build dict of connectors, clips, tapes, and junctions from OCR and detection results
3. `trace_mask_connectivity()` — Morphological cleaning → CCL noise removal → seed each component → multi-source BFS → wherever two labels meet = segment connection
4. `assign_segment_properties()` — Attaches `segment_type` (tape label) and `dimension_mm` to each segment (80px proximity to path polyline)
5. `convert_to_legacy_format()` — Converts NetworkX graph to reporter-friendly dictionary format
6. **Fallback**: If mask tracer produces 0 segments, automatically falls back to legacy CCL-based heuristic pipeline

### `--legacy` — Canny Edge Detection + CCL Pipeline

1. `detect_segments(gray)` — Canny edge detection + HSV dark mask → CCL noise removal → extract individual segment traces with PCA endpoints
2. `filter_segments_by_components()` — drops segments with endpoints > 50px from any component
3. `build_connectivity_graph_heuristic()`:
   - Pass 1: project tape labels onto segment traces (80px perpendicular corridor)
   - Pass 2 (tape-anchor): pair nodes via angle heuristic (>60° separation, <650px)
   - Supplemental hardcoded segments: J20→Connector-1, J20→MLC001, J20→X519

---

### `--extract-only=<items>` / `--skip=<items>`

Valid items: `tapes`, `connectors`, `segments`, `dimensions`, `clips`

- `--extract-only=tapes,connectors` — disables all detectors except those listed
- `--skip=clips,dimensions` — disables only those detectors; all others run
- Both flags are composable with `--legacy`