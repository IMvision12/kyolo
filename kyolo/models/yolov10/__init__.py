"""YOLOv10 — model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV10_CONFIG
from .yolov10_model import build_yolov10, YOLOv10


def yolov10n(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv10-n detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov10(
        "n",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov10s(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv10-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov10(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov10m(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv10-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov10(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov10b(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv10-b detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov10(
        "b",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov10l(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv10-l detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov10(
        "l",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolov10x(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLOv10-x detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolov10(
        "x",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


__all__ = [
    "YOLOv10",
    "build_yolov10",
    "YOLOV10_CONFIG",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10b",
    "yolov10l",
    "yolov10x",
]
