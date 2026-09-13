"""kyolo detector models.

Import a ready-to-build variant directly and call it::

    from kyolo.models import yolov8n, yolo11s, yolov9c
    model = yolov8n(nc=80)                       # -> keras.Model
    feats = model(images)                        # raw [P3, P4, P5] feature list

Pass weights=<path> to load an already-converted Keras checkpoint. kyolo does
not download or convert the official (AGPL-3.0) weights for you; convert a ``.pt``
you supply yourself with the per-model converter first (see the README)::

    model = yolov8n(weights="yolov8n.weights.h5")         # converted Keras file only
    model = yolov8n(nc=3, weights="yolov8n.weights.h5")   # fine-tune: class branch re-initialised

The loader behind ``weights=`` is ``load_pretrained_weights``; use it directly
to load into a model you built yourself.

Every factory returns a plain keras.Model. Feed the outputs to
kyolo.postprocessing.YOLOPostprocessor for detections, or wrap the model in
kyolo.training.YOLODetector to train / fine-tune. The family classes (YOLOv8,
YOLO11, ...) are also exported.
"""

from __future__ import annotations

from .base import load_pretrained_weights
from .yolo11 import (
    YOLO11,
    build_yolo11,
    yolo11l,
    yolo11m,
    yolo11n,
    yolo11s,
    yolo11x,
)
from .yolo12 import (
    YOLO12,
    build_yolo12,
    yolo12l,
    yolo12m,
    yolo12n,
    yolo12s,
    yolo12x,
)
from .yolo26 import (
    YOLO26,
    build_yolo26,
    yolo26l,
    yolo26m,
    yolo26n,
    yolo26s,
    yolo26x,
)
from .yolov5 import (
    YOLOv5,
    build_yolov5,
    yolov5l,
    yolov5m,
    yolov5n,
    yolov5s,
    yolov5x,
)
from .yolov8 import (
    YOLOv8,
    build_yolov8,
    yolov8l,
    yolov8m,
    yolov8n,
    yolov8s,
    yolov8x,
)
from .yolov9 import (
    YOLOv9,
    build_yolov9,
    yolov9c,
    yolov9e,
    yolov9m,
    yolov9s,
    yolov9t,
)
from .yolov10 import (
    YOLOv10,
    build_yolov10,
    yolov10b,
    yolov10l,
    yolov10m,
    yolov10n,
    yolov10s,
    yolov10x,
)

MODEL_NAMES = [
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    "yolov9e",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10b",
    "yolov10l",
    "yolov10x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
    "yolo12n",
    "yolo12s",
    "yolo12m",
    "yolo12l",
    "yolo12x",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
]


def list_models():
    """Return the list of available variant factory names."""
    return list(MODEL_NAMES)


__all__ = [
    "YOLOv5",
    "build_yolov5",
    "YOLOv8",
    "build_yolov8",
    "YOLOv9",
    "build_yolov9",
    "YOLOv10",
    "build_yolov10",
    "YOLO11",
    "build_yolo11",
    "YOLO12",
    "build_yolo12",
    "YOLO26",
    "build_yolo26",
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    "yolov9e",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10b",
    "yolov10l",
    "yolov10x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
    "yolo12n",
    "yolo12s",
    "yolo12m",
    "yolo12l",
    "yolo12x",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
    "list_models",
    "MODEL_NAMES",
    "load_pretrained_weights",
]
