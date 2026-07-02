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
insufficient, prefer the order-based transfer (``method="order"``), which does
not depend on names at all.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["DEFAULT_MAPPING", "NAME_MAPPINGS", "get_mapping"]


# Applied when no family-specific mapping is requested. Identity by default:
# the dotted kyolo naming already mirrors the reference module hierarchy.
DEFAULT_MAPPING: Dict[str, str] = {}


# Per-family overrides. Keyed by the architecture family (not the size suffix),
# so "yolov8n", "yolov8s", ... all share the "yolov8" entry. Kept intentionally
# minimal / documented as tunable -- fill these in per checkpoint as needed.
NAME_MAPPINGS: Dict[str, Dict[str, str]] = {
    # Ultralytics-style checkpoints (v5/v8/v10/11/12/26). Their state_dict keys
    # are already ``model.<idx>.<module>.<param>`` which matches the kyolo
    # dotted naming, so the identity mapping is the correct starting point.
    "yolov5": {},
    "yolov8": {},
    "yolov10": {},
    "yolo11": {},
    "yolo12": {},
    "yolo26": {},
    # WongKinYiu YOLOv9. Structural names align; the head may need attention.
    "yolov9": {},
}


def _family_of(model_name: str) -> str:
    """Best-effort map of a concrete model name (e.g. ``yolov8n``) to a family.

    Falls back to an empty string (-> :data:`DEFAULT_MAPPING`) when the name is
    not recognised.
    """
    name = model_name.lower().strip()
    # Order matters: check longer / more specific prefixes first so that e.g.
    # "yolov10" is not swallowed by "yolov1".
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
    return NAME_MAPPINGS.get(_family_of(model_name), DEFAULT_MAPPING)
