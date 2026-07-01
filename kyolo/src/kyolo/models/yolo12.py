"""YOLO12 — area-attention (A2C2f) backbone + PAN-FPN, lightweight DFL head.

The head/neck attention stages use :func:`kyolo.layers.a2c2f` (area attention).
The neck aggregation stages, which in the reference alternate attention with
C3k blocks, are approximated here with ``area=1`` A2C2f / C3k2 blocks.
"""

from __future__ import annotations

from keras import layers

from ..layers import a2c2f, c3k2, conv_bn
from ..layers.common import concat_axis
from .base import finalize_detector, image_input, scale_channels, scale_depth

__all__ = ["YOLO12_CONFIG", "build_yolo12", "YOLO12"]

# variant: (depth, width, max_channels)
YOLO12_CONFIG = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


def build_yolo12(
    variant="n", nc=80, input_shape=(640, 640, 3), data_format="channels_last",
    deploy=True, reg_max=16, **kwargs,
):
    d, w, mc = YOLO12_CONFIG[variant]
    ax = concat_axis(data_format)

    def ch(c):
        return scale_channels(c, w, max_channels=mc)

    def nd(x):
        return scale_depth(x, d)

    def up(x, name):
        return layers.UpSampling2D(2, data_format=data_format, interpolation="nearest", name=name)(x)

    def cat(xs, name):
        return layers.Concatenate(axis=ax, name=name)(xs)

    inp = image_input(input_shape, data_format)

    # --- backbone ---
    x = conv_bn(inp, ch(64), 3, 2, data_format=data_format, name="model.0")
    x = conv_bn(x, ch(128), 3, 2, data_format=data_format, name="model.1")
    x = c3k2(x, ch(256), nd(2), use_c3k=False, e=0.25, data_format=data_format, name="model.2")
    x = conv_bn(x, ch(256), 3, 2, data_format=data_format, name="model.3")
    x = c3k2(x, ch(512), nd(2), use_c3k=False, e=0.25, data_format=data_format, name="model.4")
    p3 = x
    x = conv_bn(x, ch(512), 3, 2, data_format=data_format, name="model.5")
    x = a2c2f(x, ch(512), nd(4), area=4, data_format=data_format, name="model.6")
    p4 = x
    x = conv_bn(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    x = a2c2f(x, ch(1024), nd(4), area=1, data_format=data_format, name="model.8")
    p5 = x

    # --- neck ---
    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = a2c2f(t, ch(512), nd(2), area=1, data_format=data_format, name="model.11")
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = a2c2f(t, ch(256), nd(2), area=1, data_format=data_format, name="model.14")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.15"), p4n], "cat2")
    p4o = a2c2f(t, ch(512), nd(2), area=1, data_format=data_format, name="model.17")
    t = cat([conv_bn(p4o, ch(512), 3, 2, data_format=data_format, name="model.18"), p5], "cat3")
    p5o = a2c2f(t, ch(1024), nd(2), area=1, data_format=data_format, name="model.20")

    return finalize_detector(
        inp, [p3o, p4o, p5o], nc=nc, reg_max=reg_max, cls_dw=True,
        data_format=data_format, name=f"yolo12{variant}",
    )


def YOLO12(variant="n", nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Factory for a YOLO12 detector (variant in ``{n, s, m, l, x}``)."""
    return build_yolo12(variant, nc, input_shape, data_format, deploy, **kwargs)
