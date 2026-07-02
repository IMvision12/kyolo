"""YOLO11 model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLO11_CONFIG
from .yolo11_model import YOLO11, build_yolo11


def yolo11n(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLO11-n detector.

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
        convert_weights: if True, download the official "yolo11n" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolo11(
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
            "yolo11n",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolo11s(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLO11-s detector.

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
        convert_weights: if True, download the official "yolo11s" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolo11(
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
            "yolo11s",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolo11m(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLO11-m detector.

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
        convert_weights: if True, download the official "yolo11m" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolo11(
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
            "yolo11m",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolo11l(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLO11-l detector.

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
        convert_weights: if True, download the official "yolo11l" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolo11(
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
            "yolo11l",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolo11x(
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLO11-x detector.

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
        convert_weights: if True, download the official "yolo11x" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolo11(
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
            "yolo11x",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


__all__ = [
    "YOLO11",
    "build_yolo11",
    "YOLO11_CONFIG",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
]
