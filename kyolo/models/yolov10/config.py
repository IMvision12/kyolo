"""Scale configuration for YOLOv10."""

from __future__ import annotations

YOLOV10_CONFIG = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "b": (0.67, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


YOLOV10_CIB_SLOTS = {
    "n": {22},
    "s": {8, 22},
    "m": {8, 19, 22},
    "b": {8, 13, 19, 22},
    "l": {8, 13, 19, 22},
    "x": {6, 8, 13, 19, 22},
}


__all__ = ["YOLOV10_CONFIG", "YOLOV10_CIB_SLOTS"]
