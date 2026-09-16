"""Reading and writing YOLO ``.txt`` label files.

One file per image, one object per line, coordinates normalized to ``[0, 1]``::

    <class> <x_center> <y_center> <width> <height>

Polygon lines (``<class> x1 y1 x2 y2 ...``), which segmentation datasets use,
are accepted and reduced to their enclosing box.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = [
    "label_path_for",
    "load_yolo_label",
    "write_yolo_label",
    "normalized_to_xyxy",
    "xyxy_to_normalized",
]

_IMAGES_DIR = f"{os.sep}images{os.sep}"
_LABELS_DIR = f"{os.sep}labels{os.sep}"


def label_path_for(image_path):
    """Map an image path to its label path.

    Follows the Ultralytics layout, where the two live in sibling trees and only
    the last ``images`` directory in the path becomes ``labels``::

        datasets/coco/images/train/0.jpg -> datasets/coco/labels/train/0.txt

    If the path contains no ``images`` directory the label is taken to sit
    beside the image, with a ``.txt`` extension.
    """
    text = str(Path(image_path))
    if _IMAGES_DIR in text:
        head, tail = text.rsplit(_IMAGES_DIR, 1)
        text = head + _LABELS_DIR + tail
    return Path(text).with_suffix(".txt")


def parse_line(values):
    """One label line -> ``(class_index, (cx, cy, w, h))`` normalized."""
    class_index = int(float(values[0]))
    coords = np.asarray(values[1:], dtype="float32")

    if coords.size == 4:
        return class_index, coords
    if coords.size >= 6 and coords.size % 2 == 0:
        points = coords.reshape(-1, 2)
        low = points.min(axis=0)
        high = points.max(axis=0)
        center = (low + high) / 2.0
        extent = high - low
        return class_index, np.concatenate([center, extent]).astype("float32")

    raise ValueError(
        f"expected 4 box values or an even number (>= 6) of polygon values, got {coords.size}."
    )


def load_yolo_label(path, nc=None):
    """Read a label file.

    Missing files are treated as "no objects", which is how YOLO datasets encode
    background-only images.

    Args:
        path: the ``.txt`` file.
        nc: when given, class indices outside ``[0, nc)`` are rejected.

    Returns:
        ``(classes, boxes)`` with ``classes`` ``(N,)`` int32 and ``boxes``
        ``(N, 4)`` float32 normalized ``(cx, cy, w, h)``. Degenerate boxes are
        dropped.
    """
    path = Path(path)
    if not path.is_file():
        return np.zeros((0,), dtype="int32"), np.zeros((0, 4), dtype="float32")

    classes = []
    boxes = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            values = line.split()
            if not values:
                continue
            try:
                class_index, box = parse_line(values)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: malformed label line: {exc}") from exc
            if nc is not None and not 0 <= class_index < nc:
                raise ValueError(
                    f"{path}:{number}: class index {class_index} is outside [0, {nc})."
                )
            classes.append(class_index)
            boxes.append(box)

    if not boxes:
        return np.zeros((0,), dtype="int32"), np.zeros((0, 4), dtype="float32")

    classes = np.asarray(classes, dtype="int32")
    boxes = np.stack(boxes).astype("float32")
    boxes[:, :2] = np.clip(boxes[:, :2], 0.0, 1.0)
    boxes[:, 2:] = np.clip(boxes[:, 2:], 0.0, 1.0)

    keep = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    return classes[keep], boxes[keep]


def write_yolo_label(path, classes, boxes):
    """Write normalized ``(cx, cy, w, h)`` boxes to a label file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_index, box in zip(np.asarray(classes).reshape(-1), np.asarray(boxes).reshape(-1, 4)):
        coords = " ".join(f"{float(v):.6g}" for v in box)
        lines.append(f"{int(class_index)} {coords}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def normalized_to_xyxy(boxes, width, height):
    """Normalized ``(cx, cy, w, h)`` -> pixel ``(x1, y1, x2, y2)``."""
    boxes = np.asarray(boxes, dtype="float32").reshape(-1, 4)
    center_x = boxes[:, 0] * width
    center_y = boxes[:, 1] * height
    half_w = boxes[:, 2] * width / 2.0
    half_h = boxes[:, 3] * height / 2.0
    return np.stack(
        [center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h], axis=-1
    ).astype("float32")


def xyxy_to_normalized(boxes, width, height):
    """Pixel ``(x1, y1, x2, y2)`` -> normalized ``(cx, cy, w, h)``."""
    boxes = np.asarray(boxes, dtype="float32").reshape(-1, 4)
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0 / width
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0 / height
    box_w = (boxes[:, 2] - boxes[:, 0]) / width
    box_h = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([center_x, center_y, box_w, box_h], axis=-1).astype("float32")
