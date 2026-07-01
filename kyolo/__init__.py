"""kyolo — the YOLO object-detection family in pure Keras 3.

Backend-agnostic (TensorFlow / JAX / PyTorch) implementations of YOLOv5, v6,
v7, v8, v9, v10, YOLO11, YOLO12 and YOLO26, with preprocessing, postprocessing,
training/fine-tuning support and PyTorch->Keras weight-conversion utilities.

Quickstart
----------
    import keras
    from kyolo import yolov8n, YOLOPreprocessor, YOLOPostprocessor

    model = yolov8n(nc=80)
    pre = YOLOPreprocessor(image_size=640)
    post = YOLOPostprocessor(nc=80)

    batch = pre(image)                 # {"images", "ratio", "pad"}
    feats = model(batch["images"])     # list of 3 raw feature maps
    detections = post(feats)           # (B, max_det, 6): x1,y1,x2,y2,score,cls

Every model variant is a factory function (``yolov5n``, ``yolo11s``, ``yolov9c``,
...). See ``kyolo.models.MODEL_NAMES`` for the full list.
"""

from __future__ import annotations

from . import models
from .losses import YOLODetectionLoss
from .models import *  # noqa: F401,F403  (per-variant factories + family classes)
from .models import MODEL_NAMES, list_models
from .postprocessing import (
    NonMaxSuppression,
    YOLOPostprocessor,
    detections_to_list,
)
from .preprocessing import YOLOPreprocessor
from .training import YOLODetector
from .version import __version__, version

__all__ = [
    "__version__",
    "version",
    "models",
    "list_models",
    "MODEL_NAMES",
    # per-variant factories + family classes (re-exported from kyolo.models)
    *models.__all__,
    # pre / post
    "YOLOPreprocessor",
    "YOLOPostprocessor",
    "NonMaxSuppression",
    "detections_to_list",
    # training / loss
    "YOLODetector",
    "YOLODetectionLoss",
]
