"""YOLOv8 model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV8_CONFIG
from .yolov8_model import YOLOv8, build_yolov8


def yolov8n(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
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
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov8n" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov8(
        "n",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if convert_weights or weights is not None:
        from ...conversion.pretrained import load_pretrained

        load_pretrained(
            model,
            "yolov8n",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov8s(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
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
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov8s" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov8(
        "s",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if convert_weights or weights is not None:
        from ...conversion.pretrained import load_pretrained

        load_pretrained(
            model,
            "yolov8s",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov8m(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
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
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov8m" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov8(
        "m",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if convert_weights or weights is not None:
        from ...conversion.pretrained import load_pretrained

        load_pretrained(
            model,
            "yolov8m",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov8l(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
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
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov8l" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov8(
        "l",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if convert_weights or weights is not None:
        from ...conversion.pretrained import load_pretrained

        load_pretrained(
            model,
            "yolov8l",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov8x(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
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
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov8x" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov8(
        "x",
        nc=nc,
        input_shape=input_shape,
        data_format=data_format,
        deploy=deploy,
        **kwargs,
    )
    if convert_weights or weights is not None:
        from ...conversion.pretrained import load_pretrained

        load_pretrained(
            model,
            "yolov8x",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
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
