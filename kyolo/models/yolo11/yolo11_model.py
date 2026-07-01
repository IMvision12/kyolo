"""YOLO11 — C3k2 backbone with C2PSA attention + PAN-FPN, lightweight DFL head."""

from __future__ import annotations

from keras import layers

from ...layers import c2psa, c3k2, conv_bn, sppf
from ...layers.common import concat_axis
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLO11_CONFIG

__all__ = ["YOLO11_CONFIG", "build_yolo11", "YOLO11"]


# variant: (depth, width, max_channels)
def build_yolo11(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    reg_max=16,
    **kwargs,
):
    d, w, mc = YOLO11_CONFIG[variant]
    ax = concat_axis(data_format)

    def ch(c):
        return scale_channels(c, w, max_channels=mc)

    def nd(x):
        return scale_depth(x, d)

    def up(x, name):
        return layers.UpSampling2D(2, data_format=data_format, interpolation="nearest", name=name)(
            x
        )

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
    x = c3k2(x, ch(512), nd(2), use_c3k=True, data_format=data_format, name="model.6")
    p4 = x
    x = conv_bn(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    x = c3k2(x, ch(1024), nd(2), use_c3k=True, data_format=data_format, name="model.8")
    x = sppf(x, ch(1024), 5, data_format=data_format, name="model.9")
    x = c2psa(x, ch(1024), nd(2), data_format=data_format, name="model.10")
    p5 = x

    # --- neck ---
    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = c3k2(t, ch(512), nd(2), use_c3k=False, data_format=data_format, name="model.13")
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = c3k2(t, ch(256), nd(2), use_c3k=False, data_format=data_format, name="model.16")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.17"), p4n], "cat2")
    p4o = c3k2(t, ch(512), nd(2), use_c3k=False, data_format=data_format, name="model.19")
    t = cat([conv_bn(p4o, ch(512), 3, 2, data_format=data_format, name="model.20"), p5], "cat3")
    p5o = c3k2(t, ch(1024), nd(2), use_c3k=True, data_format=data_format, name="model.22")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        cls_dw=True,
        data_format=data_format,
        name=f"yolo11{variant}",
    )


def YOLO11(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    **kwargs,
):
    """Factory for a YOLO11 detector (variant in ``{n, s, m, l, x}``)."""
    return build_yolo11(variant, nc, input_shape, data_format, deploy, **kwargs)
