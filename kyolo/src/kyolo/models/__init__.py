"""kyolo detector models.

Every builder returns a plain ``keras.Model`` whose output is the list of three
raw pyramid feature maps ``[P3, P4, P5]`` (each ``B x H x W x (4*reg_max + nc)``,
channels_last by default). Use :class:`kyolo.postprocessing.YOLOPostprocessor`
to turn them into detections and :class:`kyolo.losses.YOLODetectionLoss` /
:class:`kyolo.training.YOLODetector` to train.

Use :func:`get_model` for string-based construction, or the family factories
(:func:`YOLOv8`, :func:`YOLO11`, ...) directly.
"""

from __future__ import annotations

from .yolo11 import YOLO11, build_yolo11
from .yolo12 import YOLO12, build_yolo12
from .yolo26 import YOLO26, build_yolo26
from .yolov5 import YOLOv5, build_yolov5
from .yolov6 import YOLOv6, build_yolov6
from .yolov7 import YOLOv7, build_yolov7
from .yolov8 import YOLOv8, build_yolov8
from .yolov9 import YOLOv9, build_yolov9
from .yolov10 import YOLOv10, build_yolov10

__all__ = [
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
]

# (builder, variant) for each registered model name.
_REGISTRY = {}


def _register(prefix, builder, variants):
    for v in variants:
        _REGISTRY[f"{prefix}{v}"] = (builder, v)


_register("yolov5", build_yolov5, ["n", "s", "m", "l", "x"])
_register("yolov6", build_yolov6, ["n", "s", "m", "l"])
_register("yolov8", build_yolov8, ["n", "s", "m", "l", "x"])
_register("yolov9", build_yolov9, ["t", "s", "m", "c", "e"])
_register("yolov10", build_yolov10, ["n", "s", "m", "b", "l", "x"])
_register("yolo11", build_yolo11, ["n", "s", "m", "l", "x"])
_register("yolo12", build_yolo12, ["n", "s", "m", "l", "x"])
_register("yolo26", build_yolo26, ["n", "s", "m", "l", "x"])

# YOLOv7 uses non-single-letter variants.
_REGISTRY["yolov7"] = (build_yolov7, "")
_REGISTRY["yolov7-tiny"] = (build_yolov7, "tiny")
_REGISTRY["yolov7-x"] = (build_yolov7, "x")

# NMS-free families (decoded with end_to_end post-processing).
END_TO_END_MODELS = {n for n in _REGISTRY if n.startswith(("yolov10", "yolo26"))}


def list_models():
    """Return the sorted list of registered model names."""
    return sorted(_REGISTRY)


def get_model(
    name,
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    **kwargs,
):
    """Build a detector by name.

    Args:
        name: e.g. ``"yolov8n"``, ``"yolo11m"``, ``"yolov9c"``, ``"yolov7-tiny"``.
        nc: number of classes.
        input_shape: input image shape (H, W, C) for channels_last.
        data_format: ``"channels_last"`` or ``"channels_first"``.
        deploy: build reparameterizable blocks in fused (inference) form.

    Returns:
        A ``keras.Model`` outputting the raw ``[P3, P4, P5]`` feature list.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}. Available: {list_models()}"
        )
    builder, variant = _REGISTRY[key]
    return builder(
        variant=variant, nc=nc, input_shape=input_shape,
        data_format=data_format, deploy=deploy, **kwargs,
    )


def is_end_to_end(name):
    """Whether ``name`` is an NMS-free (end-to-end) model."""
    return name.lower() in END_TO_END_MODELS
