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
# Basic detection
python run.py /path/to/diagram.png

# With filtering options
python run.py diagram.png --extract-only=tapes,segments,clips

# Skip certain elements
python run.py diagram.png --skip=dimensions
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
| PaddleOCR | Text recognition (OCR) | 2.8.1 |
| scikit-image | Skeletonization | ≥0.19.0 |
| sknw | Skeleton graph extraction | ≥0.15 |
| NetworkX | Graph representation | ≥2.8 |
| NumPy | Numerical operations | ≥1.20.0 |

## Detection Pipeline

### Phase 1: OCR Text Detection
- Extract tape labels, connectors IDs, junction identifiers, and dimension annotations
- Multi-angle OCR (36 angles) for robust rotation-invariant detection

### Phase 2: Component Detection
- **Tape labels**: Shape detection (small rectangles) + OCR matching
- **Connectors**: Delphi-style rectangle patterns with internal structure
- **Clips**: Blue HSV color range + circular shape detection
- **Dimensions**: Numeric pattern recognition with deduplication

### Phase 3: Segment Extraction
- **Mask creation**: Invert grayscale + remove colored overlays
- **Skeletonization**: Thin segment paths to 1-pixel centerlines
- **Graph extraction**: Use Connected Components + PCA for endpoint detection
- **Graph filtering**: Consolidate nearby nodes, bridge gaps, prune spurs

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
