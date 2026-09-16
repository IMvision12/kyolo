"""Freezing helpers for fine-tuning."""

from __future__ import annotations

import re

__all__ = ["freeze_backbone"]


_STAGE_RE = re.compile(r"^model-(\d+)(?:-|$)")


def freeze_backbone(model, through=None):
    """Set ``trainable=False`` on the backbone stages of a kyolo model.

    Freezes every layer of ``model.0`` .. ``model.<through>``. ``through``
    defaults to ``model.backbone_end``, the last stage before the neck (SPPF /
    C2PSA / PSA / SPPELAN / the last A2C2f, depending on the family), so the
    neck and detection head stay trainable. Frozen BatchNormalization layers
    run in inference mode, which is the usual recipe for small datasets.

    Call this before ``compile()`` (or re-compile afterwards).

    Args:
        model: a detector built by a kyolo factory (or any model whose layers
            follow the ``model-<stage>-...`` naming).
        through: last stage index to freeze (inclusive). Defaults to
            ``model.backbone_end``.

    Returns:
        The list of layers that were frozen.
    """
    if through is None:
        through = getattr(model, "backbone_end", None)
        if through is None:
            raise ValueError(
                "model has no `backbone_end` attribute; pass `through=<last backbone stage>`."
            )
    frozen = []
    for layer in model.layers:
        match = _STAGE_RE.match(layer.name)
        if match and int(match.group(1)) <= through:
            layer.trainable = False
            frozen.append(layer)
    return frozen
