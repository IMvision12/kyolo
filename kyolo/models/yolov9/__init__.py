"""YOLOv9 model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV9_SPECS
from .yolov9_model import YOLOv9, build_yolov9


def yolov9t(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=False,
    weights=None,
    **kwargs,
):
    """Build a YOLOv9-t detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused form. Defaults to
            False (unfused), matching how the official checkpoints ship; the
            fused graph is mathematically equivalent.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here.
    """
    model = build_yolov9(
        "t",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        model.load_weights(weights)
    return model


def yolov9s(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=False,
    weights=None,
    **kwargs,
):
    """Build a YOLOv9-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused form. Defaults to
            False (unfused), matching how the official checkpoints ship; the
            fused graph is mathematically equivalent.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here.
    """
    model = build_yolov9(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        model.load_weights(weights)
    return model


def yolov9m(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=False,
    weights=None,
    **kwargs,
):
    """Build a YOLOv9-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused form. Defaults to
            False (unfused), matching how the official checkpoints ship; the
            fused graph is mathematically equivalent.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here.
    """
    model = build_yolov9(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        model.load_weights(weights)
    return model


def yolov9c(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=False,
    weights=None,
    **kwargs,
):
    """Build a YOLOv9-c detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused form. Defaults to
            False (unfused), matching how the official checkpoints ship; the
            fused graph is mathematically equivalent.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here.
    """
    model = build_yolov9(
        "c",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        model.load_weights(weights)
    return model


def yolov9e(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=False,
    weights=None,
    **kwargs,
):
    """Build a YOLOv9-e detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape. Use (H, W, C) for channels_last or
            (C, H, W) for channels_first.
        data_format: "channels_last", "channels_first", or None to use the
            global keras.config.image_data_format() (the default).
        deploy: build reparameterizable blocks in fused form. Defaults to
            False (unfused), matching how the official checkpoints ship; the
            fused graph is mathematically equivalent.
        weights: optional path to a converted Keras checkpoint (".weights.h5"
            or ".keras") to load after building. kyolo does not download or
            convert the official (AGPL-3.0) weights for you: convert a ".pt"
            you supply yourself with the per-model converter
            (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
            and pass the resulting ".weights.h5" here.
    """
    model = build_yolov9(
        "e",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if weights is not None:
        model.load_weights(weights)
    return model


__all__ = [
    "YOLOv9",
    "build_yolov9",
    "YOLOV9_SPECS",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    "yolov9e",
]
