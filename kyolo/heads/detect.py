"""Anchor-free decoupled detection head (shared by every kyolo model).

Each pyramid level gets an independent regression branch (two 3x3 convs + a
1x1 conv producing ``4 * reg_max`` distribution logits) and classification
branch (two 3x3 convs + a 1x1 conv producing ``nc`` class logits). The two are
concatenated so a level's output has ``4 * reg_max + nc`` channels — the format
consumed by :class:`kyolo.postprocessing.YOLOPostprocessor` and
:class:`kyolo.losses.YOLODetectionLoss`.

Sub-layer names mirror the Ultralytics ``Detect`` module (``cv2.{i}.{0,1,2}``
for the box branch and ``cv3.{i}.{0,1,2}`` for the class branch), so the
converted weights line up.
"""

from __future__ import annotations

import math

import keras
from keras import layers

from ..layers.common import channels_of, concat_axis, conv_bn, dw_conv

__all__ = ["detect_head"]

# Bias so the initial class probability is ~0.01 (stabilises early training).
_CLS_BIAS = -math.log((1 - 0.01) / 0.01)


def detect_head(
    feats,
    nc=80,
    reg_max=16,
    cls_dw=False,
    data_format="channels_last",
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
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: dotted name prefix.

    Returns:
        list of ``(B, H, W, 4*reg_max + nc)`` (or channels_first) tensors.
    """
    ax = concat_axis(data_format)
    ch = [channels_of(f, data_format) for f in feats]
    c2 = max(16, ch[0] // 4, reg_max * 4)
    c3 = max(ch[0], min(nc, 100))

    outputs = []
    for i, f in enumerate(feats):
        # --- box (regression) branch ---
        reg = conv_bn(f, c2, 3, 1, data_format=data_format, name=f"{name}.cv2.{i}.0")
        reg = conv_bn(reg, c2, 3, 1, data_format=data_format, name=f"{name}.cv2.{i}.1")
        reg = layers.Conv2D(
            4 * reg_max,
            1,
            use_bias=True,
            data_format=data_format,
            name=f"{name}.cv2.{i}.2",
        )(reg)

        # --- class branch ---
        if cls_dw:
            cls = dw_conv(f, ch[i], 3, 1, data_format=data_format, name=f"{name}.cv3.{i}.0.0")
            cls = conv_bn(cls, c3, 1, 1, data_format=data_format, name=f"{name}.cv3.{i}.0.1")
            cls = dw_conv(cls, c3, 3, 1, data_format=data_format, name=f"{name}.cv3.{i}.1.0")
            cls = conv_bn(cls, c3, 1, 1, data_format=data_format, name=f"{name}.cv3.{i}.1.1")
        else:
            cls = conv_bn(f, c3, 3, 1, data_format=data_format, name=f"{name}.cv3.{i}.0")
            cls = conv_bn(cls, c3, 3, 1, data_format=data_format, name=f"{name}.cv3.{i}.1")
        cls = layers.Conv2D(
            nc,
            1,
            use_bias=True,
            data_format=data_format,
            bias_initializer=keras.initializers.Constant(_CLS_BIAS),
            name=f"{name}.cv3.{i}.2",
        )(cls)

        out = layers.Concatenate(axis=ax, name=f"{name}.concat.{i}")([reg, cls])
        outputs.append(out)
    return outputs
