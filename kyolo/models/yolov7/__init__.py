"""YOLOv7 — model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV7_CONFIG
from .yolov7_model import build_yolov7, YOLOv7


def yolov7(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv7-base detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov7(
        "",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov7_tiny(
    nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs
):
    """Build a YOLOv7-tiny detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov7(
        "tiny",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov7_x(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv7-x detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov7(
        "x",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


__all__ = [
    "YOLOv7",
    "build_yolov7",
    "YOLOV7_CONFIG",
    "yolov7",
    "yolov7_tiny",
    "yolov7_x",
]
