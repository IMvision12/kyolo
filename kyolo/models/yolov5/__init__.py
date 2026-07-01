"""YOLOv5 — model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV5_CONFIG
from .yolov5_model import build_yolov5, YOLOv5


def yolov5n(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv5-n detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov5(
        "n",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov5s(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv5-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov5(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov5m(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv5-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov5(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov5l(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv5-l detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov5(
        "l",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov5x(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv5-x detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov5(
        "x",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


__all__ = [
    "YOLOv5",
    "build_yolov5",
    "YOLOV5_CONFIG",
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
]
