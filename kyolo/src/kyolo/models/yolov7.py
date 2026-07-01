"""YOLOv7 — ELAN backbone + SPPCSPC + ELAN-PAN neck, DFL head.

A faithful-in-spirit ELAN detector. YOLOv7's downsampling "MP" modules are
approximated with stride-2 convolutions, and the original anchor-based lead/aux
heads are replaced by the shared anchor-free DFL head used across kyolo.
"""

from __future__ import annotations

from keras import layers

from ..layers import conv_bn, elan, sppcspc
from ..layers.common import concat_axis
from .base import finalize_detector, image_input, scale_channels

__all__ = ["YOLOV7_CONFIG", "build_yolov7", "YOLOv7"]

# variant: (width, elan_depth)
YOLOV7_CONFIG = {
    "tiny": (0.50, 1),
    "": (1.00, 2),
    "base": (1.00, 2),
    "x": (1.25, 3),
}


def build_yolov7(
    variant="", nc=80, input_shape=(640, 640, 3), data_format="channels_last",
    deploy=True, reg_max=16, **kwargs,
):
    if variant not in YOLOV7_CONFIG:
        raise KeyError(f"unknown YOLOv7 variant {variant!r}; choose from {list(YOLOV7_CONFIG)}")
    w, depth = YOLOV7_CONFIG[variant]
    ax = concat_axis(data_format)

    def ch(c):
        return scale_channels(c, w)

    def block(x, c2, mid, name):
        return elan(x, ch(c2), ch(mid), depth=depth, data_format=data_format, name=name)

    def up(x, name):
        return layers.UpSampling2D(2, data_format=data_format, interpolation="nearest", name=name)(x)

    def cat(xs, name):
        return layers.Concatenate(axis=ax, name=name)(xs)

    inp = image_input(input_shape, data_format)

    # --- backbone ---
    x = conv_bn(inp, ch(32), 3, 1, data_format=data_format, name="backbone.stem0")
    x = conv_bn(x, ch(64), 3, 2, data_format=data_format, name="backbone.stem1")
    x = conv_bn(x, ch(64), 3, 1, data_format=data_format, name="backbone.stem2")
    x = conv_bn(x, ch(128), 3, 2, data_format=data_format, name="backbone.stem3")
    x = block(x, 256, 128, "backbone.elan1")
    x = conv_bn(x, ch(256), 3, 2, data_format=data_format, name="backbone.down1")
    x = block(x, 512, 256, "backbone.elan2")
    p3 = x
    x = conv_bn(x, ch(512), 3, 2, data_format=data_format, name="backbone.down2")
    x = block(x, 1024, 512, "backbone.elan3")
    p4 = x
    x = conv_bn(x, ch(1024), 3, 2, data_format=data_format, name="backbone.down3")
    x = block(x, 1024, 512, "backbone.elan4")
    x = sppcspc(x, ch(512), data_format=data_format, name="backbone.sppcspc")
    p5 = x

    # --- ELAN-PAN neck ---
    p4r = conv_bn(p4, ch(256), 1, 1, data_format=data_format, name="neck.reduce_p4")
    t = cat([up(conv_bn(p5, ch(256), 1, 1, data_format=data_format, name="neck.reduce_p5"), "up0"), p4r], "cat0")
    p4n = block(t, 256, 128, "neck.elan_p4")
    p3r = conv_bn(p3, ch(128), 1, 1, data_format=data_format, name="neck.reduce_p3")
    t = cat([up(conv_bn(p4n, ch(128), 1, 1, data_format=data_format, name="neck.reduce_n4"), "up1"), p3r], "cat1")
    p3o = block(t, 128, 64, "neck.elan_p3")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="neck.down0"), p4n], "cat2")
    p4o = block(t, 256, 128, "neck.elan_n4")
    t = cat([conv_bn(p4o, ch(512), 3, 2, data_format=data_format, name="neck.down1"), p5], "cat3")
    p5o = block(t, 512, 256, "neck.elan_n5")

    return finalize_detector(
        inp, [p3o, p4o, p5o], nc=nc, reg_max=reg_max, data_format=data_format,
        name=f"yolov7{variant}" if variant else "yolov7",
    )


def YOLOv7(variant="", nc=80, input_shape=(640, 640, 3), data_format="channels_last", deploy=True, **kwargs):
    """Factory for a YOLOv7 detector (variant in ``{'', 'tiny', 'x'}``)."""
    return build_yolov7(variant, nc, input_shape, data_format, deploy, **kwargs)
