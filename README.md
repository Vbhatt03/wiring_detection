# Wire Detection System

Automated extraction and analysis of automotive wiring diagram schematics. This system identifies electrical components (connectors, clips, junctions) and wires, then constructs a connectivity graph representing the electrical topology.

## Overview

The Wire Detection System processes high-resolution automotive wiring diagrams to extract:
- **Wires**: Individual wire segments and merged paths
- **Connectors**: Delphi-style electrical connectors 
- **Junctions**: Labeled connection points in the diagram
- **Clips**: Blue Z-marker clips indicating wire harness bundles
- **Tape Labels**: Wire type identifiers (VT-WH, VT-BK, etc.)
- **Length Annotations**: Wire length measurements in millimeters

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
python run.py diagram.png --extract-only=tapes,wires,clips

# Skip certain elements
python run.py diagram.png --skip=lengths
```

### Output

The detector generates:
- `connectivity_graph.json`: Complete graph structure with nodes and edges
- Annotated image with detected elements labeled
- Console report showing detected components and connections

## Project Structure

```
src/
├── run_detector.py              # Main orchestrator
├── component_masker.py          # Wire mask extraction
├── skeleton_graph.py            # Skeleton-based wire graph
├── wiring_connectivity.py       # Legacy connectivity logic
│
├── detectors/                   # Element detection modules
│   ├── wire_detector.py         # Wire extraction (CCL + PCA)
│   ├── ocr_detector.py          # Text recognition
│   ├── tape_detector.py         # Tape label detection
│   ├── connector_detector.py    # Connector detection
│   ├── clip_detector.py         # Clip detection
│   └── length_detector.py       # Length annotation extraction
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
- Extract tape labels, connectors IDs, junction identifiers, and length annotations
- Multi-angle OCR (36 angles) for robust rotation-invariant detection

### Phase 2: Component Detection
- **Tape labels**: Shape detection (small rectangles) + OCR matching
- **Connectors**: Delphi-style rectangle patterns with internal structure
- **Clips**: Blue HSV color range + circular shape detection
- **Lengths**: Numeric pattern recognition with deduplication

### Phase 3: Wire Extraction
- **Mask creation**: Invert grayscale + remove colored overlays
- **Skeletonization**: Thin wire paths to 1-pixel centerlines
- **Graph extraction**: Use Connected Components + PCA for endpoint detection
- **Graph filtering**: Consolidate nearby nodes, bridge gaps, prune spurs

### Phase 4: Output Generation
- Convert detected elements to connectivity graph
- Annotate image with rectangles and labels
- Export to JSON format

## Limitations & Future Work

- **Connection accuracy**: Heuristic-based edge derivation may miss or misidentify connections
- **Dimension mapping**: Wire type and length assignment not fully accurate
- **Component classification**: Distinguishes connectors/clips from junctions but limited semantic understanding
- **Dashed wire handling**: Currently works well but could be optimized further

## Configuration

See [notes.md](notes.md) for detailed information on extraction methods, libraries tested, and technical decisions.
