"""YOLOv9 - GELAN (RepNCSPELAN4 + ADown + SPPELAN) backbone/neck, DFL head.

Implements the deploy-time GELAN topology (the reparameterized inference model,
equivalent to ``gelan-c``). The auxiliary PGI branches used only during the
original v9 training are not built. Channel widths are scaled per variant.
"""

from __future__ import annotations

from ...layers import adown, conv_bn, rep_ncspelan4, sppelan
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, scale_channels
from .config import YOLOV9_CONFIG

__all__ = ["YOLOV9_CONFIG", "build_yolov9", "YOLOv9"]


# variant: width multiplier (GELAN-c channel structure is scaled by this)
def build_yolov9(
    variant="c",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    reg_max=16,
    **kwargs,
):
    w = YOLOV9_CONFIG[variant]
    data_format = resolve_data_format(data_format)
    ax = concat_axis(data_format)

    def ch(c):
        return scale_channels(c, w)

    def gelan(x, c2, c3, c4, name):
        return rep_ncspelan4(
            x, ch(c2), ch(c3), ch(c4), n=1, deploy=deploy, data_format=data_format, name=name
        )

    def up(x, name):
        return layers.UpSampling2D(2, data_format=data_format, interpolation="nearest", name=name)(
            x
        )

    def cat(xs, name):
        return layers.Concatenate(axis=ax, name=name)(xs)

    inp = image_input(input_shape, data_format)

    # --- backbone (GELAN) ---
    x = conv_bn(inp, ch(64), 3, 2, data_format=data_format, name="model.0")
    x = conv_bn(x, ch(128), 3, 2, data_format=data_format, name="model.1")
    x = gelan(x, 256, 128, 64, "model.2")
    x = adown(x, ch(256), data_format=data_format, name="model.3")
    x = gelan(x, 512, 256, 128, "model.4")
    p3 = x
    x = adown(x, ch(512), data_format=data_format, name="model.5")
    x = gelan(x, 512, 512, 256, "model.6")
    p4 = x
    x = adown(x, ch(512), data_format=data_format, name="model.7")
    x = gelan(x, 512, 512, 256, "model.8")
    x = sppelan(x, ch(512), ch(256), 5, data_format=data_format, name="model.9")
    p5 = x

    # --- neck ---
    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = gelan(t, 512, 512, 256, "model.12")
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = gelan(t, 256, 256, 128, "model.15")
    t = cat([adown(p3o, ch(256), data_format=data_format, name="model.16"), p4n], "cat2")
    p4o = gelan(t, 512, 512, 256, "model.18")
    t = cat([adown(p4o, ch(512), data_format=data_format, name="model.19"), p5], "cat3")
    p5o = gelan(t, 512, 512, 256, "model.21")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        data_format=data_format,
        name=f"yolov9{variant}",
    )


def YOLOv9(
    variant="c",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    **kwargs,
):
    """Factory for a YOLOv9 detector (variant in ``{t, s, m, c, e}``)."""
    return build_yolov9(variant, nc, input_shape, data_format, deploy, **kwargs)
