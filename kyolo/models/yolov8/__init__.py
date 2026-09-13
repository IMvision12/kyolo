"""YOLOv8 model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from ..base import load_pretrained_weights
from .config import YOLOV8_CONFIG
from .yolov8_model import YOLOv8, build_yolov8


def yolov8n(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    **kwargs,
):
    """Build a YOLOv8-n detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here. A checkpoint with a
            different class count loads too: everything but the head's
            classification branch is transferred (see
            ``kyolo.models.load_pretrained_weights``).
    """
    model = build_yolov8(
        "n",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        load_pretrained_weights(model, weights)
    return model


def yolov8s(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    **kwargs,
):
    """Build a YOLOv8-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here. A checkpoint with a
            different class count loads too: everything but the head's
            classification branch is transferred (see
            ``kyolo.models.load_pretrained_weights``).
    """
    model = build_yolov8(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        load_pretrained_weights(model, weights)
    return model


def yolov8m(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    **kwargs,
):
    """Build a YOLOv8-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here. A checkpoint with a
            different class count loads too: everything but the head's
            classification branch is transferred (see
            ``kyolo.models.load_pretrained_weights``).
    """
    model = build_yolov8(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        load_pretrained_weights(model, weights)
    return model


def yolov8l(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    **kwargs,
):
    """Build a YOLOv8-l detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here. A checkpoint with a
            different class count loads too: everything but the head's
            classification branch is transferred (see
            ``kyolo.models.load_pretrained_weights``).
    """
    model = build_yolov8(
        "l",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        load_pretrained_weights(model, weights)
    return model


def yolov8x(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    **kwargs,
):
    """Build a YOLOv8-x detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here. A checkpoint with a
            different class count loads too: everything but the head's
            classification branch is transferred (see
            ``kyolo.models.load_pretrained_weights``).
    """
    model = build_yolov8(
        "x",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        load_pretrained_weights(model, weights)
    return model


__all__ = [
    "YOLOv8",
    "build_yolov8",
    "YOLOV8_CONFIG",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
]
