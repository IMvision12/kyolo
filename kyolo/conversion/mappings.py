"""Substring name mappings for the NAME-BASED conversion path.

These are *best-effort* helpers for :func:`kyolo.conversion.transfer_torch_to_keras`.
They are applied *after* the generic leaf-suffix rules (``kernel->weight``,
``gamma->weight``, ``beta->bias``, ``moving_mean->running_mean``,
``moving_variance->running_var``) have already turned a Keras variable path such
as ``model.0.conv/kernel`` into a candidate Torch key ``model.0.conv.weight``.

Each mapping is an *ordered* ``dict`` of ``old -> new`` substring replacements
applied in insertion order to that candidate key. Because the :mod:`kyolo`
builders deliberately name their sub-layers after the reference PyTorch modules
(dotted paths like ``model.0.conv``/``model.0.bn``), the identity mapping
(an empty dict) already resolves the vast majority of parameters, and that is
the pragmatic default here.

Where a mapping *is* needed it is almost always the detection head, whose module
layout differs between YOLO generations and between the Ultralytics and
WongKinYiu code bases. Treat every entry below as a starting point to be
verified against a real checkpoint -- not as a guarantee. When a mapping proves
insufficient, extend it: the order-based transfer (``method="order"``) is not a
usable fallback here, because kyolo's CSP blocks are built in a different order
than the reference registers them (see :mod:`kyolo.conversion.convert`).
"""

from __future__ import annotations

from typing import Dict

__all__ = ["DEFAULT_MAPPING", "NAME_MAPPINGS", "get_mapping"]


DEFAULT_MAPPING: Dict[str, str] = {}


NAME_MAPPINGS: Dict[str, Dict[str, str]] = {
    "yolov5": {},
    "yolov8": {},
    "yolov10": {},
    "yolo11": {},
    "yolo12": {},
    "yolo26": {},
    "yolov9": {},
}


def family_of(model_name: str) -> str:
    """Best-effort map of a concrete model name (e.g. ``yolov8n``) to a family.

    Falls back to an empty string (-> :data:`DEFAULT_MAPPING`) when the name is
    not recognised.
    """
    name = model_name.lower().strip()

    for family in (
        "yolov5",
        "yolov8",
        "yolov9",
        "yolov10",
        "yolo11",
        "yolo12",
        "yolo26",
    ):
        if name.startswith(family):
            return family
    return ""


def get_mapping(model_name: str) -> Dict[str, str]:
    """Return the substring mapping for ``model_name`` (family-resolved).

    Unknown names yield :data:`DEFAULT_MAPPING`.
    """
    return NAME_MAPPINGS.get(family_of(model_name), DEFAULT_MAPPING)
