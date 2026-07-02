"""Shared model-assembly utilities for every kyolo detector.

Model files here follow a common recipe:

    inputs = image_input(input_shape, data_format)
    feats  = build_backbone_neck(inputs, ...)     # list [P3, P4, P5]
    model  = finalize_detector(inputs, feats, nc, reg_max, ...)

``finalize_detector`` attaches the shared decoupled head and returns a plain
``keras.Model`` whose output is the list of three raw feature maps
``[P3, P4, P5]`` (each ``B x H x W x (4*reg_max + nc)``). Keeping the model a
standard functional graph means it is trainable, fine-tunable and serialisable
with no special machinery - training targets/loss live in
:mod:`kyolo.losses` and post-processing in :mod:`kyolo.postprocessing`.
"""

from __future__ import annotations

import math

import keras

from ..heads.detect import detect_head
from ..layers.common import resolve_data_format
from ..layers.knames import kname, layers

__all__ = [
    "make_divisible",
    "scale_channels",
    "scale_depth",
    "image_input",
    "finalize_detector",
]


def make_divisible(x, divisor=8):
    """Round ``x`` up to the nearest multiple of ``divisor``."""
    return int(math.ceil(x / divisor) * divisor)


def scale_channels(channels, width, divisor=8, max_channels=None):
    """Scale a base channel count by ``width`` (kept divisible by ``divisor``)."""
    if max_channels is not None:
        channels = min(channels, max_channels)
    return make_divisible(channels * width, divisor)


def scale_depth(n, depth):
    """Scale a repeat count by ``depth`` (minimum 1)."""
    return max(round(n * depth), 1)


def image_input(input_shape=(640, 640, 3), data_format=None, name="images"):
    """Create the image ``Input`` tensor for the given shape / data format."""
    data_format = resolve_data_format(data_format)
    if input_shape is None:
        input_shape = (640, 640, 3)
    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3-tuple, got {input_shape}")

    if data_format == "channels_first":
        # accept either (C,H,W) or (H,W,C) and normalise to (C,H,W)
        if input_shape[0] in (1, 3):
            c, h, w = input_shape
        else:
            h, w, c = input_shape
        shape = (c, h, w)
    else:
        if input_shape[-1] in (1, 3):
            h, w, c = input_shape
        else:
            c, h, w = input_shape
        shape = (h, w, c)
    return layers.Input(shape=shape, name=name)


def finalize_detector(
    inputs,
    feats,
    nc=80,
    reg_max=16,
    cls_dw=False,
    head_act=True,
    data_format=None,
    name="kyolo",
    head_name="head",
    strides=(8, 16, 32),
    end_to_end=False,
):
    """Attach the shared head and return the assembled ``keras.Model``.

    The returned model carries a few attributes (``nc``, ``reg_max``,
    ``strides``, ``num_levels``, ``end_to_end``) that the loss / post-processor
    and :class:`kyolo.training.YOLODetector` read. ``head_name`` is the
    layer-name prefix for the head (kept stable across variants so a single
    weight mapping works for a whole family). ``end_to_end`` marks NMS-free
    models (YOLOv10 / YOLO26).
    """
    data_format = resolve_data_format(data_format)
    outputs = detect_head(
        feats,
        nc=nc,
        reg_max=reg_max,
        cls_dw=cls_dw,
        act=head_act,
        data_format=data_format,
        name=head_name,
    )
    model = keras.Model(inputs=inputs, outputs=outputs, name=kname(name))
    # metadata (plain python attrs; safe on a functional model)
    model.nc = nc
    model.reg_max = reg_max
    model.strides = tuple(strides)
    model.num_levels = len(outputs)
    model.data_format = data_format
    model.end_to_end = end_to_end
    return model
