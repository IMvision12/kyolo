"""COCO class metadata and small helpers for naming detection classes.

This module is intentionally dependency-free (standard library only) so it can
be imported from anywhere in :mod:`kyolo` without pulling in a backend or any
heavy visualization dependency.
"""

from __future__ import annotations

__all__ = [
    "COCO_CLASSES",
    "COCO_CLASS_NAMES",
    "get_class_names",
]


# Mapping ``{class_id -> class_name}`` for the 80 COCO detection classes, in the
# canonical order used by the YOLO family of models.
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


# List of the 80 COCO class names ordered by class id (``COCO_CLASS_NAMES[i]``
# is the name of class ``i``).
COCO_CLASS_NAMES = [COCO_CLASSES[i] for i in range(len(COCO_CLASSES))]


def get_class_names(nc=80, names=None):
    """Return a list of class names of length ``nc``.

    Args:
        nc: Number of classes. Ignored when ``names`` is provided.
        names: Optional explicit sequence of class names. When given it is
            returned as a list unchanged (and defines the effective length).

    Returns:
        A list of ``nc`` class-name strings:

        * If ``names`` is provided, ``list(names)``.
        * Else if ``nc == 80``, the canonical :data:`COCO_CLASS_NAMES`.
        * Otherwise generic placeholder names ``["class_0", "class_1", ...]``.
    """
    if names is not None:
        return list(names)
    if nc == 80:
        return list(COCO_CLASS_NAMES)
    return [f"class_{i}" for i in range(nc)]
