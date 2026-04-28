# Segment Detection System

Automated extraction and analysis of automotive segment diagram schematics. This system identifies electrical components (connectors, clips, junctions) and segments, then constructs a connectivity graph representing the electrical topology.

## Overview

The Segment Detection System processes high-resolution automotive segment diagrams to extract:
- **Segments**: Individual segment traces and merged segments
- **Connectors**: Delphi-style electrical connectors 
- **Junctions**: Labeled connection points in the diagram
- **Clips**: Blue Z-marker clips indicating segment harness bundles
- **Tape Labels**: Segment type identifiers (VT-WH, VT-BK, etc.)
- **Dimension Annotations**: Segment dimension measurements in millimeters

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Basic detection (PaddleOCR by default)
python run.py /path/to/diagram.png

# Switch OCR backend
python run.py diagram.png --ocr-backend=easyocr
python run.py diagram.png --ocr-backend=tesseract

# With filtering options
python run.py diagram.png --extract-only=tapes,segments,clips

# Skip certain elements
python run.py diagram.png --skip=dimensions

# Combine OCR and filter flags
python run.py diagram.png --ocr-backend=easyocr --skip=clips
```

### Output

The detector generates:
- `connectivity_graph.json`: Complete graph structure with nodes, segments, and routes
- Annotated image with detected elements labeled
- Console report showing detected components and connections

## Project Structure

```
src/
├── run_detector.py              # Main orchestrator
├── component_masker.py          # Segment mask extraction
├── skeleton_graph.py            # Skeleton-based segment graph
├── segment_connectivity.py       # Legacy connectivity logic
│
├── detectors/                   # Element detection modules
│   ├── segment_detector.py         # Segment extraction (CCL + PCA)
│   ├── ocr_detector.py          # Text recognition
│   ├── tape_detector.py         # Tape label detection
│   ├── connector_detector.py    # Connector detection
│   ├── clip_detector.py         # Clip detection
│   └── dimension_detector.py       # Dimension annotation extraction
│
├── graph_builders/              # Graph construction modules
│   └── connectivity_builder.py  # Component-to-graph mapping
│
└── visualization/               # Output modules
    ├── visualizer.py            # Image annotation
    └── reporter.py              # Text reporting
```

## Key Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| OpenCV (cv2) | Image processing, morphology, connected components | ≥4.5.0 |
| PaddleOCR | Text recognition (OCR, default) | 2.8.1 |
| EasyOCR | Alternative OCR backend (`--ocr-backend=easyocr`) | ≥1.6 |
| pytesseract | Alternative OCR backend (`--ocr-backend=tesseract`) | ≥0.3 |
| NetworkX | Graph representation | ≥2.8 |
| NumPy | Numerical operations | ≥1.20.0 |

## Detection Pipeline

### Phase 1: OCR Text Detection
- Extract tape labels, connectors IDs, junction identifiers, and dimension annotations
- Two separate passes: `ocr_full()` for general text, `ocr_full_dimensions()` for small numeric annotations
- Backend-selectable via `--ocr-backend`: `paddle` (default), `easyocr`, `tesseract`
- **PaddleOCR**: Tiled OCR with 11 rotation angles (2-pass), 2× upscale — tuned for `det_limit_side_len=960`
- **EasyOCR**: Single-pass tiling (1000px tiles), no rotation passes — angle handled natively
- **Tesseract**: Full-image single pass, 3× upscale, `--psm 11 --oem 3` — no rotation support

### Phase 2: Component Detection
- **Tape labels**: Shape detection (small rectangles) + OCR matching
- **Connectors**: Delphi-style rectangle patterns with internal structure
- **Clips**: Blue HSV color range + circular shape detection
- **Dimensions**: Numeric pattern recognition with deduplication

### Phase 3: Segment Extraction
- **Default (Mask-Tracer)**: Binary mask from Otsu thresholding + component erasure → morphological closing to bridge gaps → multi-source BFS flood-fill from component seeds → connections where two component labels meet
- **Legacy (`--legacy`)**: Canny edge detection + HSV dark mask → Connected Components Labeling (CCL) → PCA endpoint extraction → heuristic tape-label pairing
- **Fallback**: If mask-tracer produces 0 segments, automatically falls back to legacy CCL pipeline

### Phase 4: Output Generation
- Convert detected elements to connectivity graph
- Annotate image with rectangles and labels
- Export to JSON format

## Limitations & Future Work

- **Connection accuracy**: Heuristic-based segment derivation may miss or misidentify connections
- **Dimension mapping**: Segment type and dimension assignment not fully accurate
- **Component classification**: Distinguishes connectors/clips from junctions but limited semantic understanding
- **Dashed segment handling**: Currently works well but could be optimized further

## Configuration

See [notes.md](notes.md) for detailed information on extraction methods, libraries tested, and technical decisions.
