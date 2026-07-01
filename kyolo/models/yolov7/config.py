"""Scale configuration for YOLOv7."""

from __future__ import annotations

YOLOV7_CONFIG = {
    "tiny": (0.50, 1),
    "": (1.00, 2),
    "base": (1.00, 2),
    "x": (1.25, 3),
}


__all__ = ["YOLOV7_CONFIG"]
