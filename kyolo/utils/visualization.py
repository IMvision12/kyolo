"""Matplotlib-based visualization helpers for object-detection results.

These helpers draw predicted boxes and labels on top of an image. ``matplotlib``
ships with the base install, but it is imported lazily inside the functions so
that ``import kyolo.utils`` (and the rest of the package) keeps working even if
``matplotlib`` has been removed from the environment.

Only :mod:`numpy` and :mod:`matplotlib` are required here, plus ``keras`` for
converting backend tensors to numpy via ``keras.ops.convert_to_numpy``.
"""

from __future__ import annotations

import keras
import numpy as np

from .coco import COCO_CLASS_NAMES

__all__ = [
    "visualize_detections",
    "draw_detections_batch",
]


# A small, high-contrast palette cycled through for successive boxes.
_COLORS = [
    "red",
    "blue",
    "lime",
    "orange",
    "magenta",
    "cyan",
    "yellow",
    "deeppink",
    "chartreuse",
    "dodgerblue",
]


def _import_matplotlib():
    """Import and return the ``pyplot`` and ``patches`` modules, lazily.

    Raises:
        ImportError: With an install hint when matplotlib is not available.
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "matplotlib is required for visualization. It ships with kyolo, so "
            "reinstall it with `pip install matplotlib` (or `pip install kyolo`)."
        ) from exc
    return plt, patches


def _to_numpy(x):
    """Convert a numpy array or a Keras/backend tensor to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if x is None:
        return None
    # ``convert_to_numpy`` handles tensors from any Keras backend.
    try:
        return keras.ops.convert_to_numpy(x)
    except (TypeError, ValueError):
        return np.asarray(x)


def _prepare_image(image):
    """Normalize an image to an ``(H, W, 3)`` float array in ``[0, 1]``.

    Accepts ``(H, W, 3)``, ``(3, H, W)`` or ``(H, W)`` layouts, with values in
    either ``[0, 1]`` or ``[0, 255]`` (auto-detected).
    """
    img = _to_numpy(image)
    img = np.asarray(img)

    # Channels-first (3, H, W) -> channels-last (H, W, 3).
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    # Grayscale (H, W) -> (H, W, 3).
    elif img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    img = img.astype("float32")

    # Auto-detect [0, 255] range and normalize to [0, 1] for display.
    if img.size > 0 and float(img.max()) > 1.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


def visualize_detections(
    image,
    detections,
    class_names=None,
    score_threshold=0.0,
    save_path=None,
    show=False,
    title="Detections",
):
    """Draw detection boxes and labels on a single image.

    Args:
        image: A single image as a numpy array or Keras tensor. Supported
            shapes are ``(H, W, 3)``, ``(3, H, W)`` or ``(H, W)``; values may be
            in ``[0, 1]`` or ``[0, 255]`` (auto-detected and normalized).
        detections: Array/tensor of shape ``(N, 6)`` where each row is
            ``[x1, y1, x2, y2, score, class_id]``. An empty ``(0, 6)`` array is
            accepted and renders the bare image.
        class_names: Optional sequence mapping ``class_id -> name``. Defaults to
            the 80 COCO class names.
        score_threshold: Detections with ``score`` below this value are skipped.
        save_path: If given, the figure is written to this path with
            ``Figure.savefig``.
        show: If ``True``, call ``pyplot.show()`` to display the figure.
        title: Base title for the plot; the count of drawn objects is appended.

    Returns:
        The created ``matplotlib.figure.Figure``.
    """
    plt, patches = _import_matplotlib()

    if class_names is None:
        class_names = COCO_CLASS_NAMES

    img = _prepare_image(image)

    dets = _to_numpy(detections)
    dets = np.asarray(dets, dtype="float32")
    if dets.ndim == 1:
        dets = dets.reshape(1, -1) if dets.size else dets.reshape(0, 6)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")

    drawn = 0
    for det in dets:
        if det.shape[0] < 6:
            continue
        x1, y1, x2, y2, score, cls = det[:6]
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        cls = int(cls)
        color = _COLORS[drawn % len(_COLORS)]

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        if 0 <= cls < len(class_names):
            name = class_names[cls]
        else:
            name = f"class_{cls}"
        ax.text(
            x1,
            y1 - 5,
            f"{name}: {float(score):.2f}",
            color=color,
            fontsize=10,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 2},
        )
        drawn += 1

    ax.set_title(f"{title}: {drawn} object{'s' if drawn != 1 else ''}", fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def draw_detections_batch(images, detections_list, class_names=None, **kw):
    """Visualize a batch by looping over images and their detections.

    Args:
        images: An iterable/sequence of images, each accepted by
            :func:`visualize_detections`.
        detections_list: A sequence of per-image detection arrays of shape
            ``(N_i, 6)``, aligned with ``images``.
        class_names: Optional class-name sequence forwarded to
            :func:`visualize_detections`.
        **kw: Additional keyword arguments forwarded to
            :func:`visualize_detections` (e.g. ``score_threshold``, ``show``,
            ``title``). Note ``save_path`` is applied to every figure, so pass
            it only when a single output path is intended.

    Returns:
        A list of ``matplotlib.figure.Figure`` objects, one per image.
    """
    return [
        visualize_detections(image, detections, class_names=class_names, **kw)
        for image, detections in zip(images, detections_list)
    ]
