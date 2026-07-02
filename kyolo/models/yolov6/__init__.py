"""YOLOv6 model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV6_CONFIG
from .yolov6_model import YOLOv6, build_yolov6


def yolov6n(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv6-n detector.

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
        convert_weights: if True, download the official "yolov6n" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov6(
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
            "yolov6n",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov6s(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv6-s detector.

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
        convert_weights: if True, download the official "yolov6s" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov6(
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
            "yolov6s",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov6m(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv6-m detector.

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
        convert_weights: if True, download the official "yolov6m" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov6(
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
            "yolov6m",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov6l(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv6-l detector.

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
        convert_weights: if True, download the official "yolov6l" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov6(
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
            "yolov6l",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


__all__ = [
    "YOLOv6",
    "build_yolov6",
    "YOLOV6_CONFIG",
    "yolov6n",
    "yolov6s",
    "yolov6m",
    "yolov6l",
]
