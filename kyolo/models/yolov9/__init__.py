"""YOLOv9 — model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV9_CONFIG
from .yolov9_model import build_yolov9, YOLOv9


def yolov9t(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv9-t detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov9(
        "t",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov9s(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv9-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov9(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov9m(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv9-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov9(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov9c(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv9-c detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov9(
        "c",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov9e(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv9-e detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov9(
        "e",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


__all__ = [
    "YOLOv9",
    "build_yolov9",
    "YOLOV9_CONFIG",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    "yolov9e",
]
