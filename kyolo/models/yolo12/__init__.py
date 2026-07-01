"""YOLO12 — model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLO12_CONFIG
from .yolo12_model import build_yolo12, YOLO12


def yolo12n(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO12-n detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo12(
        "n",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo12s(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO12-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo12(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo12m(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO12-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo12(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo12l(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO12-l detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo12(
        "l",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


def yolo12x(nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Build a YOLO12-x detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.
    """
    return build_yolo12(
        "x",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )


__all__ = [
    "YOLO12",
    "build_yolo12",
    "YOLO12_CONFIG",
    "yolo12n",
    "yolo12s",
    "yolo12m",
    "yolo12l",
    "yolo12x",
]
