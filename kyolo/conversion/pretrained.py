"""Download, convert and load official pretrained weights.

This is what a model factory routes to when you pass ``convert_weights=True`` or
``weights=...``. It follows the KerasFormers download-convert-cache pattern:

1. Resolve the source checkpoint (auto-download the official ``.pt`` via
   ``ultralytics``, or use a path / URL you provide).
2. Convert it to Keras with the order-based transfer (:mod:`kyolo.conversion`).
3. Cache the converted ``.weights.h5`` so later calls load instantly.
4. Load the weights into the model.

The official YOLO checkpoints are AGPL-3.0. kyolo does not redistribute them;
they are fetched from their source on demand.
"""

from __future__ import annotations

import os

from .convert import convert_weights as _run_conversion
from .file_downloader import DEFAULT_CACHE, download_file

__all__ = ["load_pretrained", "default_cache_dir"]

# kyolo factory name -> official ".pt" filename stem for the ultralytics loader.
_PT_STEM = {
    "yolov7": "yolov7",
    "yolov7_tiny": "yolov7-tiny",
    "yolov7_x": "yolov7x",
}


def default_cache_dir(cache_dir=None) -> str:
    """Resolve the cache directory (arg > ``KYOLO_CACHE`` env > ``~/.cache/kyolo``)."""
    return cache_dir or os.environ.get("KYOLO_CACHE") or DEFAULT_CACHE


def _is_url(s) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


def _resolve_local(path_or_url, cache_dir, force):
    if _is_url(path_or_url):
        return download_file(path_or_url, cache_dir=cache_dir, force_download=force)
    return path_or_url


def _download_official_pt(name, cache_dir, force):
    """Fetch the official ``.pt`` for ``name`` via the ultralytics package."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "convert_weights=True needs `ultralytics` (and torch) to fetch the "
            "official checkpoint. Install with: pip install kyolo[conversion]. "
            "Otherwise pass weights='/path/to/checkpoint.pt' to convert a file "
            "you supply yourself."
        ) from e

    stem = _PT_STEM.get(name, name)
    try:
        yolo = YOLO(f"{stem}.pt")  # ultralytics downloads + caches the checkpoint
    except Exception as e:
        raise RuntimeError(
            f"Could not auto-download official weights for {name!r} via ultralytics "
            f"({e}). This is expected for non-ultralytics families (e.g. YOLOv6, "
            f"YOLOv7): download the .pt yourself and pass weights='/path/to.pt'."
        ) from e

    path = getattr(yolo, "ckpt_path", None)
    if not path or not os.path.exists(str(path)):
        path = f"{stem}.pt"
    return str(path)


def load_pretrained(
    model,
    name,
    weights=None,
    convert_weights=False,
    cache_dir=None,
    method="order",
    nc=80,
    force_download=False,
):
    """Load weights into ``model`` (see module docstring).

    Args:
        model: a built kyolo Keras model.
        name: the factory name, e.g. ``"yolov8n"`` (used for caching + lookup).
        weights: optional checkpoint path or http URL. A ``.pt``/``.pth`` file is
            converted; a ``.weights.h5``/``.keras`` file is loaded directly.
        convert_weights: if True, fetch + convert + cache + load the official
            COCO checkpoint (requires ``nc == 80``).
        cache_dir: cache directory for downloads and converted weights.
        method: transfer strategy, ``"order"`` (default) or ``"name"``.
        nc: number of classes (must be 80 when ``convert_weights=True``).
        force_download: bypass caches and re-fetch / re-convert.

    Returns:
        The same ``model``, with weights loaded.
    """
    cache = default_cache_dir(cache_dir)
    os.makedirs(cache, exist_ok=True)

    if convert_weights:
        if nc != 80:
            raise ValueError(
                f"convert_weights=True loads the official COCO checkpoint (nc=80), "
                f"but nc={nc}. For custom classes, load the model without "
                f"convert_weights and fine-tune, or pass weights=<your checkpoint>."
            )
        cached = os.path.join(cache, f"{name}.weights.h5")
        if os.path.exists(cached) and not force_download:
            print(f"Loading cached converted weights: {cached}")
            model.load_weights(cached)
            return model

        pt = (
            _resolve_local(weights, cache, force_download)
            if weights is not None
            else _download_official_pt(name, cache, force_download)
        )
        _run_conversion(model, pt, output_path=cached, method=method, verbose=True)
        return model

    if weights is not None:
        wpath = _resolve_local(weights, cache, force_download)
        if str(wpath).endswith((".pt", ".pth")):
            # convert a PyTorch checkpoint straight into the model (no file saved)
            _run_conversion(model, wpath, output_path=False, method=method, verbose=True)
        else:
            model.load_weights(wpath)
    return model
