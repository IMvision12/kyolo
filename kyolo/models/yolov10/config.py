"""Scale configuration for YOLOv10."""

from __future__ import annotations

YOLOV10_CONFIG = {
    "n": (0.33, 0.25, 1024, False),
    "s": (0.33, 0.50, 1024, False),
    "m": (0.67, 0.75, 768, True),
    "b": (0.67, 1.00, 512, True),
    "l": (1.00, 1.00, 512, True),
    "x": (1.00, 1.25, 512, True),
}


__all__ = ["YOLOV10_CONFIG"]
