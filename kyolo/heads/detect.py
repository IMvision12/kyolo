"""Anchor-free decoupled detection head (shared by every kyolo model).

Each pyramid level gets an independent regression branch (two 3x3 convs + a
1x1 conv producing ``4 * reg_max`` distribution logits) and classification
branch (two 3x3 convs + a 1x1 conv producing ``nc`` class logits). The two are
concatenated so a level's output has ``4 * reg_max + nc`` channels - the format
consumed by :class:`kyolo.postprocessing.YOLOPostprocessor` and
:class:`kyolo.losses.YOLODetectionLoss`.

Sub-layer names mirror the Ultralytics ``Detect`` module (``cv2.{i}.{0,1,2}``
for the box branch and ``cv3.{i}.{0,1,2}`` for the class branch), so the
converted weights line up.
"""

from __future__ import annotations

import math

import keras

from ..layers.common import channels_of, concat_axis, conv_bn, dw_conv, resolve_data_format
from ..layers.knames import layers

__all__ = ["detect_head"]


_BOX_BIAS = 1.0


_BIAS_IMGSZ = 640


def cls_bias(nc, stride, imgsz=_BIAS_IMGSZ):
    """Ultralytics ``Detect.bias_init`` class-branch bias for one pyramid level.

    ``log(5 / nc / (imgsz / stride) ** 2)``: an initial per-class probability
    calibrated so that roughly 5 objects are expected across the level's
    ``(imgsz / stride) ** 2`` cells. Because it divides by the cell count it is
    **stride-dependent** (about -11.5 / -10.2 / -8.8 at strides 8 / 16 / 32 for
    ``nc=80``), i.e. far more confident-of-background than a flat ``p=0.01``
    prior, which keeps the initial BCE term -- summed over ``B * A * nc`` mostly
    negative entries -- the same order of magnitude as the box and DFL terms.
    """
    return math.log(5.0 / nc / (imgsz / stride) ** 2)


def detect_head(
    feats,
    nc=80,
    reg_max=16,
    cls_dw=False,
    act=True,
    data_format=None,
    strides=(8, 16, 32),
    name="detect",
):
    """Build the decoupled head over a list of pyramid feature maps.

    Args:
        feats: list of feature tensors (P3, P4, P5, ...).
        nc: number of classes.
        reg_max: DFL bin count (box branch has ``4 * reg_max`` outputs).
        cls_dw: if ``True`` the first two class-branch convs are depth-wise
            separable (the lightweight head used by YOLO11/12/26); otherwise
            plain convs (YOLOv8 style).
        act: conv activation (``True`` -> SiLU, the YOLO default, or a string).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        strides: per-level strides, used only to pick the stride-dependent
            class-branch bias (see :func:`cls_bias`). Must be one per level.
        name: dotted name prefix.

    Returns:
        list of ``(B, H, W, 4*reg_max + nc)`` (or channels_first) tensors.
    """
    data_format = resolve_data_format(data_format)
    ax = concat_axis(data_format)
    ch = [channels_of(f, data_format) for f in feats]
    c2 = max(16, ch[0] // 4, reg_max * 4)
    c3 = max(ch[0], min(nc, 100))
    if len(strides) != len(feats):
        raise ValueError(
            f"detect_head got {len(feats)} feature maps but {len(strides)} strides "
            f"({strides!r}); there must be exactly one stride per level."
        )

    outputs = []
    for i, f in enumerate(feats):
        reg = conv_bn(f, c2, 3, 1, act=act, data_format=data_format, name=f"{name}.cv2.{i}.0")
        reg = conv_bn(reg, c2, 3, 1, act=act, data_format=data_format, name=f"{name}.cv2.{i}.1")
        reg = layers.Conv2D(
            4 * reg_max,
            1,
            use_bias=True,
            data_format=data_format,
            bias_initializer=keras.initializers.Constant(_BOX_BIAS),
            name=f"{name}.cv2.{i}.2",
        )(reg)

        if cls_dw:
            cls = dw_conv(
                f, ch[i], 3, 1, act=act, data_format=data_format, name=f"{name}.cv3.{i}.0.0"
            )
            cls = conv_bn(
                cls, c3, 1, 1, act=act, data_format=data_format, name=f"{name}.cv3.{i}.0.1"
            )
            cls = dw_conv(
                cls, c3, 3, 1, act=act, data_format=data_format, name=f"{name}.cv3.{i}.1.0"
            )
            cls = conv_bn(
                cls, c3, 1, 1, act=act, data_format=data_format, name=f"{name}.cv3.{i}.1.1"
            )
        else:
            cls = conv_bn(f, c3, 3, 1, act=act, data_format=data_format, name=f"{name}.cv3.{i}.0")
            cls = conv_bn(cls, c3, 3, 1, act=act, data_format=data_format, name=f"{name}.cv3.{i}.1")
        cls = layers.Conv2D(
            nc,
            1,
            use_bias=True,
            data_format=data_format,
            bias_initializer=keras.initializers.Constant(cls_bias(nc, strides[i])),
            name=f"{name}.cv3.{i}.2",
        )(cls)

        out = layers.Concatenate(axis=ax, name=f"{name}.concat.{i}")([reg, cls])
        outputs.append(out)
    return outputs
