"""YOLOv10 - C2f/C2fCIB backbone with SCDown + PSA, NMS-free (end-to-end) head.

Uses the shared DFL head; run inference through
``YOLOPostprocessor(..., end_to_end=True)`` to get the NMS-free top-k selection
that YOLOv10 is designed for.
"""

from __future__ import annotations

from ...layers import c2f, c2f_cib, conv_bn, psa, scdown, sppf
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLOV10_CONFIG

__all__ = ["YOLOV10_CONFIG", "build_yolov10", "YOLOv10"]


# variant: (depth, width, max_channels, deep_cib)
#   deep_cib -> use C2fCIB at backbone stage 8 (larger variants)
def build_yolov10(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    reg_max=16,
    **kwargs,
):
    d, w, mc, deep_cib = YOLOV10_CONFIG[variant]
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
    x = scdown(x, ch(512), 3, 2, data_format=data_format, name="model.5")
    x = c2f(x, ch(512), nd(6), shortcut=True, data_format=data_format, name="model.6")
    p4 = x
    x = scdown(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    if deep_cib:
        x = c2f_cib(x, ch(1024), nd(3), shortcut=True, data_format=data_format, name="model.8")
    else:
        x = c2f(x, ch(1024), nd(3), shortcut=True, data_format=data_format, name="model.8")
    x = sppf(x, ch(1024), 5, data_format=data_format, name="model.9")
    x = psa(x, ch(1024), data_format=data_format, name="model.10")
    p5 = x

    # --- neck ---
    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = c2f(t, ch(512), nd(3), shortcut=False, data_format=data_format, name="model.13")
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = c2f(t, ch(256), nd(3), shortcut=False, data_format=data_format, name="model.16")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.17"), p4n], "cat2")
    p4o = c2f(t, ch(512), nd(3), shortcut=False, data_format=data_format, name="model.19")
    t = cat([scdown(p4o, ch(512), 3, 2, data_format=data_format, name="model.20"), p5], "cat3")
    p5o = c2f_cib(t, ch(1024), nd(3), shortcut=True, data_format=data_format, name="model.22")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        cls_dw=True,
        data_format=data_format,
        name=f"yolov10{variant}",
        end_to_end=True,
    )


def YOLOv10(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    **kwargs,
):
    """Factory for a YOLOv10 detector (variant in ``{n, s, m, b, l, x}``)."""
    return build_yolov10(variant, nc, input_shape, data_format, deploy, **kwargs)
