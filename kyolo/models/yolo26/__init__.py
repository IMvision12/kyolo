"""YOLO26 — model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLO26_CONFIG
from .yolo26_model import build_yolo26, YOLO26


def yolo26n(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO26-n detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo26(
        "n",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo26s(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO26-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo26(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo26m(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO26-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo26(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo26l(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO26-l detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo26(
        "l",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo26x(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO26-x detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo26(
        "x",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


__all__ = [
    "YOLO26",
    "build_yolo26",
    "YOLO26_CONFIG",
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
]
