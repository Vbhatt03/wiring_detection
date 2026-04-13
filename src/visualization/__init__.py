"""Visualization and reporting modules for wiring diagrams.

This package contains modules for visualizing and reporting results:
- visualizer: Image annotation and drawing utilities
- reporter: Report generation and verification tables
"""

from .visualizer import annotate, draw_label, TAPE_COLOR_BGR
from .reporter import print_report, generate_verification_table

__all__ = [
    'annotate',
    'draw_label',
    'TAPE_COLOR_BGR',
    'print_report',
    'generate_verification_table',
]
