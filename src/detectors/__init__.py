"""Detection modules for segment diagram elements.

This package contains specialized detection modules for different diagram elements:
- ocr_detector: Optical character recognition for text extraction
- tape_detector: Tape/conduit label detection
- connector_detector: Delphi connector detection
- clip_detector: Blue circular clip detection
- dimension_detector: Segment dimension annotation detection
- segment_detector: Segment segment detection and filtering
"""

from .ocr_detector import (
    ocr_full, ocr_region, PADDLEOCR_OK, OCR_OK,
    set_ocr_backend, get_ocr_backend
)
from .tape_detector import detect_tape_labels, TAPE_PATTERNS, TAPE_COLOR_BGR
from .connector_detector import detect_delphi_connectors
from .clip_detector import detect_blue_clips
from .dimension_detector import detect_segment_dimensions, score_segment_dimension_value, DIMENSION_PATTERN, LABEL_KEYWORDS
from .segment_detector import detect_segments, filter_segments_by_components, detect_components

__all__ = [
    'ocr_full',
    'ocr_region',
    'PADDLEOCR_OK',
    'OCR_OK',
    'set_ocr_backend',
    'get_ocr_backend',
    'detect_tape_labels',
    'detect_delphi_connectors',
    'detect_blue_clips',
    'detect_segment_dimensions',
    'score_segment_dimension_value',
    'detect_segments',
    'filter_segments_by_components',
    'detect_components',
    'TAPE_PATTERNS',
    'TAPE_COLOR_BGR',
    'LENGTH_PATTERN',
    'LABEL_KEYWORDS',
]
