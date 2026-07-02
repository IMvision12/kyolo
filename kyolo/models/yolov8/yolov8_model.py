"""YOLOv8 - CSP (C2f) backbone + PAN-FPN neck, anchor-free DFL head."""

from __future__ import annotations

from ...layers import c2f, conv_bn, sppf
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLOV8_CONFIG

__all__ = ["YOLOV8_CONFIG", "build_yolov8", "YOLOv8"]


# variant: (depth, width, max_channels)
def build_yolov8(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    reg_max=16,
    **kwargs,
):
    d, w, mc = YOLOV8_CONFIG[variant]
    data_format = resolve_data_format(data_format)
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
    x = c2f(x, ch(128), nd(3), shortcut=True, data_format=data_format, name="model.2")
    x = conv_bn(x, ch(256), 3, 2, data_format=data_format, name="model.3")
    x = c2f(x, ch(256), nd(6), shortcut=True, data_format=data_format, name="model.4")
    p3 = x
    x = conv_bn(x, ch(512), 3, 2, data_format=data_format, name="model.5")
    x = c2f(x, ch(512), nd(6), shortcut=True, data_format=data_format, name="model.6")
    p4 = x
    x = conv_bn(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    x = c2f(x, ch(1024), nd(3), shortcut=True, data_format=data_format, name="model.8")
    x = sppf(x, ch(1024), 5, data_format=data_format, name="model.9")
    p5 = x

    # --- neck (PAN-FPN) ---
    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = c2f(t, ch(512), nd(3), shortcut=False, data_format=data_format, name="model.12")
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = c2f(t, ch(256), nd(3), shortcut=False, data_format=data_format, name="model.15")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.16"), p4n], "cat2")
    p4o = c2f(t, ch(512), nd(3), shortcut=False, data_format=data_format, name="model.18")
    t = cat([conv_bn(p4o, ch(512), 3, 2, data_format=data_format, name="model.19"), p5], "cat3")
    p5o = c2f(t, ch(1024), nd(3), shortcut=False, data_format=data_format, name="model.21")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        data_format=data_format,
        name=f"yolov8{variant}",
    )


def YOLOv8(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    **kwargs,
):
    """Factory for a YOLOv8 detector (variant in ``{n, s, m, l, x}``)."""
    return build_yolov8(variant, nc, input_shape, data_format, deploy, **kwargs)
