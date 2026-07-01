"""kyolo — the YOLO object-detection family in pure Keras 3.

Backend-agnostic (TensorFlow / JAX / PyTorch) implementations of YOLOv5, v6,
v7, v8, v9, v10, YOLO11, YOLO12 and YOLO26, with preprocessing, postprocessing,
training/fine-tuning support and PyTorch->Keras weight-conversion utilities.

Quickstart
----------
    import keras
    from kyolo import get_model, YOLOPreprocessor, YOLOPostprocessor

    model = get_model("yolov8n", nc=80)
    pre = YOLOPreprocessor(image_size=640)
    post = YOLOPostprocessor(nc=80)

    batch = pre(image)                 # {"images", "ratio", "pad"}
    feats = model(batch["images"])     # list of 3 raw feature maps
    detections = post(feats)           # (B, max_det, 6): x1,y1,x2,y2,score,cls
"""

from __future__ import annotations

__version__ = "0.1.0"

from .losses import YOLODetectionLoss
from .models import (
    YOLO11,
    YOLO12,
    YOLO26,
    YOLOv5,
    YOLOv6,
    YOLOv7,
    YOLOv8,
    YOLOv9,
    YOLOv10,
    get_model,
    list_models,
)
from .postprocessing import (
    NonMaxSuppression,
    YOLOPostprocessor,
    detections_to_list,
)
from .preprocessing import YOLOPreprocessor
from .training import YOLODetector

__all__ = [
    "__version__",
    # models
    "get_model",
    "list_models",
    "YOLOv5",
    "YOLOv6",
    "YOLOv7",
    "YOLOv8",
    "YOLOv9",
    "YOLOv10",
    "YOLO11",
    "YOLO12",
    "YOLO26",
    # pre / post
    "YOLOPreprocessor",
    "YOLOPostprocessor",
    "NonMaxSuppression",
    "detections_to_list",
    # training / loss
    "YOLODetector",
    "YOLODetectionLoss",
]
