"""YOLOv9 - GELAN (RepNCSPELAN4 / ELAN1 + ADown / AConv + SPPELAN), DFL head.

Each variant (``t``, ``s``, ``m``, ``c``, ``e``) is a distinct topology, so the
model is assembled by interpreting the per-variant spec in :mod:`.config`, which
mirrors the Ultralytics YOLOv9 ``.yaml`` files layer for layer. The ``e`` variant
includes the dual-branch programmable-gradient path (``CBLinear`` / ``CBFuse``).

Reparameterizable blocks are built unfused (``deploy=False``) by default because
that is the form the official checkpoints ship in; a fused inference graph is
mathematically equivalent and can be requested with ``deploy=True``.
"""

from __future__ import annotations

from ...layers import (
    aconv,
    adown,
    cbfuse,
    cblinear,
    conv_bn,
    elan1,
    rep_ncspelan4,
    sppelan,
)
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, make_divisible
from .config import YOLOV9_SPECS

__all__ = ["YOLOV9_SPECS", "build_yolov9", "YOLOv9"]


def _round(c):
    """Round a module output channel to a multiple of 8 (Ultralytics parse_model).

    Only the top-level output channel (``args[0]``) is rounded; internal widths
    such as a RepNCSPELAN4 ``c3``/``c4`` are passed through unchanged, matching
    how ``parse_model`` scales the modules.
    """
    return make_divisible(c, 8)


def _gather(frm, prev, outputs):
    """Resolve a spec ``from`` field into the layer input(s)."""
    if frm == -1:
        return prev
    if isinstance(frm, int):
        return outputs[frm]
    return [prev if j == -1 else outputs[j] for j in frm]


def _build_layer(op, x, a, deploy, data_format, ax, name):
    """Build a single spec layer and return its output tensor (or tensor list)."""
    if op == "Conv":
        c2, k, s = a
        return conv_bn(x, _round(c2), k, s, data_format=data_format, name=name)
    if op == "RepNCSPELAN4":
        c2, c3, c4, n = a
        return rep_ncspelan4(
            x, _round(c2), c3, c4, n=n, deploy=deploy, data_format=data_format, name=name
        )
    if op == "ELAN1":
        c2, c3, c4 = a
        return elan1(x, _round(c2), c3, c4, data_format=data_format, name=name)
    if op == "ADown":
        return adown(x, _round(a[0]), data_format=data_format, name=name)
    if op == "AConv":
        return aconv(x, _round(a[0]), data_format=data_format, name=name)
    if op == "SPPELAN":
        c2, c3 = a
        return sppelan(x, _round(c2), c3, 5, data_format=data_format, name=name)
    if op == "Upsample":
        return layers.UpSampling2D(2, data_format=data_format, interpolation="nearest", name=name)(
            x
        )
    if op == "Concat":
        return layers.Concatenate(axis=ax, name=name)(x)
    if op == "CBLinear":
        return cblinear(x, [_round(c) for c in a[0]], data_format=data_format, name=name)
    if op == "CBFuse":
        return cbfuse(x, a[0], data_format=data_format, name=name)
    if op == "Identity":
        return x
    raise ValueError(f"unknown YOLOv9 spec op {op!r}")


def build_yolov9(
    variant="c",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=False,
    reg_max=16,
    **kwargs,
):
    """Assemble a YOLOv9 detector from its per-variant spec (see :mod:`.config`)."""
    if variant not in YOLOV9_SPECS:
        raise ValueError(
            f"unknown YOLOv9 variant {variant!r}; expected one of {list(YOLOV9_SPECS)}"
        )
    spec = YOLOV9_SPECS[variant]
    data_format = resolve_data_format(data_format)
    ax = concat_axis(data_format)

    inp = image_input(input_shape, data_format)
    outputs = []
    prev = inp
    detect_from = None
    detect_idx = None
    for i, (frm, op, a) in enumerate(spec):
        if op == "Detect":
            detect_from = frm
            detect_idx = i
            break
        x = _gather(frm, prev, outputs)
        y = _build_layer(op, x, a, deploy, data_format, ax, f"model.{i}")
        outputs.append(y)
        prev = y

    backbone_end = next(i for i, (_, op, _) in enumerate(spec) if op == "Upsample") - 1
    feats = [outputs[j] for j in detect_from]
    return finalize_detector(
        inp,
        feats,
        nc=nc,
        reg_max=reg_max,
        data_format=data_format,
        name=f"yolov9{variant}",
        head_name=f"model.{detect_idx}",
        backbone_end=backbone_end,
    )


def YOLOv9(
    variant="c",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=False,
    **kwargs,
):
    """Factory for a YOLOv9 detector (variant in ``{t, s, m, c, e}``)."""
    return build_yolov9(variant, nc, input_shape, data_format, deploy, **kwargs)
