"""YOLOv5 model, config, per-variant factories and weight conversion."""

from __future__ import annotations

from .config import YOLOV5_CONFIG
from .yolov5_model import YOLOv5, build_yolov5


def yolov5n(
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv5-n detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape.
        data_format: "channels_last" or "channels_first".
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov5n" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov5(
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
            "yolov5n",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov5s(
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv5-s detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape.
        data_format: "channels_last" or "channels_first".
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov5s" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov5(
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
            "yolov5s",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov5m(
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv5-m detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape.
        data_format: "channels_last" or "channels_first".
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov5m" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov5(
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
            "yolov5m",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov5l(
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv5-l detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape.
        data_format: "channels_last" or "channels_first".
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov5l" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov5(
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
            "yolov5l",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


def yolov5x(
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    **kwargs,
):
    """Build a YOLOv5-x detector.

    Returns a ``keras.Model`` whose output is the raw ``[P3, P4, P5]`` feature list.

    Args:
        nc: number of classes.
        input_shape: input image shape.
        data_format: "channels_last" or "channels_first".
        deploy: build reparameterizable blocks in fused (inference) form.
        weights: optional checkpoint to load after building. A ".weights.h5" or
            ".keras" file is loaded directly; a ".pt" / ".pth" file (or an http
            URL) is converted from PyTorch first.
        convert_weights: if True, download the official "yolov5x" checkpoint,
            convert it to Keras and load it. Needs the "conversion" extra
            (torch + ultralytics) and nc=80 (the COCO checkpoint). Auto-download
            covers the ultralytics-family models; for others pass weights=<path>.
        cache_dir: where converted weights are cached (default ~/.cache/kyolo).
    """
    model = build_yolov5(
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
            "yolov5x",
            weights=weights,
            convert_weights=convert_weights,
            cache_dir=cache_dir,
            nc=nc,
        )
    return model


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
