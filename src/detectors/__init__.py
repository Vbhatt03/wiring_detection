"""Detection modules for wiring diagram elements.

This package contains specialized detection modules for different diagram elements:
- ocr_detector: Optical character recognition for text extraction
- tape_detector: Tape/conduit label detection
- connector_detector: Delphi connector detection
- clip_detector: Blue circular clip detection
- length_detector: Wire length annotation detection
- wire_detector: Wire segment detection and filtering
"""

from .ocr_detector import ocr_full, ocr_region, PADDLEOCR_OK
from .tape_detector import detect_tape_labels, TAPE_PATTERNS, TAPE_COLOR_BGR
from .connector_detector import detect_delphi_connectors
from .clip_detector import detect_blue_clips
from .length_detector import detect_wire_lengths, score_wire_length_value, LENGTH_PATTERN, LABEL_KEYWORDS
from .wire_detector import detect_wires, filter_wires_by_components, detect_components

__all__ = [
    'ocr_full',
    'ocr_region',
    'PADDLEOCR_OK',
    'detect_tape_labels',
    'detect_delphi_connectors',
    'detect_blue_clips',
    'detect_wire_lengths',
    'score_wire_length_value',
    'detect_wires',
    'filter_wires_by_components',
    'detect_components',
    'TAPE_PATTERNS',
    'TAPE_COLOR_BGR',
    'LENGTH_PATTERN',
    'LABEL_KEYWORDS',
]
