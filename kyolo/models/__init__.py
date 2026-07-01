"""kyolo detector models.

Import a ready-to-build variant directly and call it::

    from kyolo.models import yolov8n, yolo11s, yolov9c
    model = yolov8n(nc=80)                       # -> keras.Model
    feats = model(images)                        # raw [P3, P4, P5] feature list

Pass convert_weights=True to auto-download + convert + load the official COCO
checkpoint, or weights=<path/url> to load your own::

    model = yolov8n(convert_weights=True)                 # official COCO weights
    model = yolov8n(weights="yolov8n.weights.h5")         # converted Keras file
    model = yolov8n(nc=80, weights="yolov8n.pt")          # convert a PyTorch file

Every factory returns a plain keras.Model. Feed the outputs to
kyolo.postprocessing.YOLOPostprocessor for detections, or wrap the model in
kyolo.training.YOLODetector to train / fine-tune. The family classes (YOLOv8,
YOLO11, ...) are also exported.
"""

from __future__ import annotations

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
from .yolov6 import (
    YOLOv6,
    build_yolov6,
    yolov6l,
    yolov6m,
    yolov6n,
    yolov6s,
)
from .yolov7 import (
    YOLOv7,
    build_yolov7,
    yolov7,
    yolov7_tiny,
    yolov7_x,
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
    "yolov6n",
    "yolov6s",
    "yolov6m",
    "yolov6l",
    "yolov7",
    "yolov7_tiny",
    "yolov7_x",
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
    "YOLOv6",
    "build_yolov6",
    "YOLOv7",
    "build_yolov7",
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
    "yolov6n",
    "yolov6s",
    "yolov6m",
    "yolov6l",
    "yolov7",
    "yolov7_tiny",
    "yolov7_x",
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
]
