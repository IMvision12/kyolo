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
# YOLOv5 maps to the anchor-free "u" checkpoints (yolov5nu.pt, ...), which use
# the same DFL Detect head kyolo builds; the original anchor-based yolov5n.pt is
# a different head and will not convert.
_PT_STEM = {
    "yolov5n": "yolov5nu",
    "yolov5s": "yolov5su",
    "yolov5m": "yolov5mu",
    "yolov5l": "yolov5lu",
    "yolov5x": "yolov5xu",
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


def _warn_on_misses(report, name):
    """Warn loudly if the transfer left some variables unmatched.

    A faithful conversion transfers every variable. Non-zero misses mean the
    kyolo architecture for ``name`` diverges from the official checkpoint (e.g.
    an approximated family), so those variables keep their random init.
    """
    skipped = report.get("skipped", 0) if isinstance(report, dict) else 0
    if skipped:
        total = report.get("total", "?")
        print(
            f"WARNING: {name}: {skipped}/{total} variables were NOT matched and "
            f"keep their random initialization. The converted model will not "
            f"reproduce the official outputs. This kyolo variant's architecture "
            f"differs from the official checkpoint."
        )


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
    method="name",
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
        method: transfer strategy, ``"name"`` (default) or ``"order"``.
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
        report = _run_conversion(model, pt, output_path=cached, method=method, verbose=True)
        _warn_on_misses(report, name)
        return model

    if weights is not None:
        wpath = _resolve_local(weights, cache, force_download)
        if str(wpath).endswith((".pt", ".pth")):
            # convert a PyTorch checkpoint straight into the model (no file saved)
            report = _run_conversion(model, wpath, output_path=False, method=method, verbose=True)
            _warn_on_misses(report, name)
        else:
            model.load_weights(wpath)
    return model
