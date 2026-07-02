"""YOLOv10 model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV10_CONFIG
from .yolov10_model import YOLOv10, build_yolov10


def yolov10n(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv10-n detector.

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
        convert_weights: if True, download the official "yolov10n" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov10(
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
            "yolov10n",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov10s(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv10-s detector.

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
        convert_weights: if True, download the official "yolov10s" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov10(
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
            "yolov10s",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov10m(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv10-m detector.

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
        convert_weights: if True, download the official "yolov10m" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov10(
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
            "yolov10m",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov10b(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv10-b detector.

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
        convert_weights: if True, download the official "yolov10b" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov10(
        "b",
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
            "yolov10b",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov10l(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv10-l detector.

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
        convert_weights: if True, download the official "yolov10l" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov10(
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
            "yolov10l",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov10x(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv10-x detector.

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
        convert_weights: if True, download the official "yolov10x" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov10(
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
            "yolov10x",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


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
