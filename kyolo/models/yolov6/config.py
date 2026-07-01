"""Scale configuration for YOLOv6."""

from __future__ import annotations

YOLOV6_CONFIG = {
    "n": (0.33, 0.25, False),
    "s": (0.33, 0.50, False),
    "m": (0.60, 0.75, True),
    "l": (1.00, 1.00, True),
}


__all__ = ["YOLOV6_CONFIG"]
