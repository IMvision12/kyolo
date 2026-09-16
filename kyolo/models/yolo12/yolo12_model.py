"""YOLO12 - area-attention (A2C2f) backbone + PAN-FPN, lightweight DFL head.

Mirrors the Ultralytics ``yolo12.yaml`` layer for layer. The backbone stages 6
and 8 use area-attention :func:`kyolo.layers.a2c2f` (``a2=True``, each inner
entry two ABlocks); the neck A2C2f stages use ``a2=False`` (C3k blocks, no
attention), and the final P5 stage is a :func:`kyolo.layers.c3k2` (``C3k2`` with
``c3k=True``).
"""

from __future__ import annotations

from ...layers import a2c2f, c3k2, conv_bn
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLO12_CONFIG

__all__ = ["YOLO12_CONFIG", "build_yolo12", "YOLO12"]


def build_yolo12(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    reg_max=16,
    **kwargs,
):
    d, w, mc = YOLO12_CONFIG[variant]
    data_format = resolve_data_format(data_format)
    ax = concat_axis(data_format)

    c3k_early = variant in ("m", "l", "x")
    a2_kwargs = {
        "a2": True,
        "residual": variant in ("l", "x"),
        "mlp_ratio": 1.2 if variant in ("l", "x") else 2.0,
    }

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

    x = conv_bn(inp, ch(64), 3, 2, data_format=data_format, name="model.0")
    x = conv_bn(x, ch(128), 3, 2, data_format=data_format, name="model.1")
    x = c3k2(x, ch(256), nd(2), use_c3k=c3k_early, e=0.25, data_format=data_format, name="model.2")
    x = conv_bn(x, ch(256), 3, 2, data_format=data_format, name="model.3")
    x = c3k2(x, ch(512), nd(2), use_c3k=c3k_early, e=0.25, data_format=data_format, name="model.4")
    p3 = x
    x = conv_bn(x, ch(512), 3, 2, data_format=data_format, name="model.5")
    x = a2c2f(x, ch(512), nd(4), area=4, data_format=data_format, name="model.6", **a2_kwargs)
    p4 = x
    x = conv_bn(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    x = a2c2f(x, ch(1024), nd(4), area=1, data_format=data_format, name="model.8", **a2_kwargs)
    p5 = x

    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = a2c2f(t, ch(512), nd(2), a2=False, data_format=data_format, name="model.11")
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = a2c2f(t, ch(256), nd(2), a2=False, data_format=data_format, name="model.14")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.15"), p4n], "cat2")
    p4o = a2c2f(t, ch(512), nd(2), a2=False, data_format=data_format, name="model.17")
    t = cat([conv_bn(p4o, ch(512), 3, 2, data_format=data_format, name="model.18"), p5], "cat3")
    p5o = c3k2(t, ch(1024), nd(2), use_c3k=True, data_format=data_format, name="model.20")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        cls_dw=True,
        data_format=data_format,
        name=f"yolo12{variant}",
        head_name="model.21",
        backbone_end=8,
    )


def YOLO12(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    **kwargs,
):
    """Factory for a YOLO12 detector (variant in ``{n, s, m, l, x}``)."""
    return build_yolo12(variant, nc, input_shape, data_format, deploy, **kwargs)
