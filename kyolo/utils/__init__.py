"""Utility helpers for :mod:`kyolo`: COCO metadata and result visualization.

This subpackage is a dependency-light leaf: it does not import anything else
from :mod:`kyolo`, so it is safe to import from any other module without risking
circular imports. The visualization helpers keep their ``matplotlib`` import
lazy, so importing this package works even if ``matplotlib`` is unavailable.
"""

from __future__ import annotations

from .coco import COCO_CLASS_NAMES, COCO_CLASSES, get_class_names
from .visualization import draw_detections_batch, visualize_detections

__all__ = [
    "COCO_CLASSES",
    "COCO_CLASS_NAMES",
    "get_class_names",
    "visualize_detections",
    "draw_detections_batch",
]
