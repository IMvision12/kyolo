"""YOLOv5 - CSP (C3) backbone + PAN-FPN neck.

The head here is the modern anchor-free DFL head (kyolo uses one head for the
whole family), so these weights are architecture-compatible with the v5 *P5*
backbone but not with the original anchor-based v5 detection layer.
"""

from __future__ import annotations

from ...layers import c3, conv_bn, sppf
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLOV5_CONFIG

__all__ = ["YOLOV5_CONFIG", "build_yolov5", "YOLOv5"]


# variant: (depth, width)
def build_yolov5(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    reg_max=16,
    **kwargs,
):
    d, w = YOLOV5_CONFIG[variant]
    data_format = resolve_data_format(data_format)
    ax = concat_axis(data_format)

    def ch(c):
        return scale_channels(c, w)

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
    x = conv_bn(inp, ch(64), 6, 2, padding=2, data_format=data_format, name="model.0")
    x = conv_bn(x, ch(128), 3, 2, data_format=data_format, name="model.1")
    x = c3(x, ch(128), nd(3), shortcut=True, data_format=data_format, name="model.2")
    x = conv_bn(x, ch(256), 3, 2, data_format=data_format, name="model.3")
    x = c3(x, ch(256), nd(6), shortcut=True, data_format=data_format, name="model.4")
    p3 = x
    x = conv_bn(x, ch(512), 3, 2, data_format=data_format, name="model.5")
    x = c3(x, ch(512), nd(9), shortcut=True, data_format=data_format, name="model.6")
    p4 = x
    x = conv_bn(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    x = c3(x, ch(1024), nd(3), shortcut=True, data_format=data_format, name="model.8")
    x = sppf(x, ch(1024), 5, data_format=data_format, name="model.9")
    p5 = x

    # --- neck (FPN top-down + PAN bottom-up) ---
    p5r = conv_bn(p5, ch(512), 1, 1, data_format=data_format, name="model.10")
    t = cat([up(p5r, "up0"), p4], "cat0")
    p4t = c3(t, ch(512), nd(3), shortcut=False, data_format=data_format, name="model.13")
    p4r = conv_bn(p4t, ch(256), 1, 1, data_format=data_format, name="model.14")
    t = cat([up(p4r, "up1"), p3], "cat1")
    p3o = c3(t, ch(256), nd(3), shortcut=False, data_format=data_format, name="model.17")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.18"), p4r], "cat2")
    p4o = c3(t, ch(512), nd(3), shortcut=False, data_format=data_format, name="model.20")
    t = cat([conv_bn(p4o, ch(512), 3, 2, data_format=data_format, name="model.21"), p5r], "cat3")
    p5o = c3(t, ch(1024), nd(3), shortcut=False, data_format=data_format, name="model.23")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        data_format=data_format,
        name=f"yolov5{variant}",
        head_name="model.24",  # matches the Ultralytics yolov5u Detect module index
        backbone_end=9,  # model.0 - model.9 (SPPF)
    )


def YOLOv5(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    **kwargs,
):
    """Factory for a YOLOv5 detector (variant in ``{n, s, m, l, x}``)."""
    return build_yolov5(variant, nc, input_shape, data_format, deploy, **kwargs)
