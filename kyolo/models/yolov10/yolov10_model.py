"""YOLOv10 - C2f/C2fCIB backbone with SCDown + PSA, DFL head.

kyolo builds the standard one-to-many (``cv2``/``cv3``) head, i.e. the branch
the official checkpoints train with NMS-style assignment; the parallel
``one2one_*`` head that gives YOLOv10 its NMS-free inference is not reproduced.
Decode these models with the default (NMS) ``YOLOPostprocessor``; the NMS-free
top-k decode (``end_to_end=True``) would return duplicate boxes here.
"""

from __future__ import annotations

from ...layers import c2f, c2f_cib, conv_bn, psa, scdown, sppf
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLOV10_CIB_SLOTS, YOLOV10_CONFIG

__all__ = ["YOLOV10_CONFIG", "build_yolov10", "YOLOv10"]


def build_yolov10(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    reg_max=16,
    **kwargs,
):
    d, w, mc = YOLOV10_CONFIG[variant]
    data_format = resolve_data_format(data_format)
    ax = concat_axis(data_format)

    cib_slots = YOLOV10_CIB_SLOTS[variant]
    cib_lk = variant in ("n", "s")

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

    def stage(x, c2, n, idx, c2f_shortcut):
        """C2fCIB when this stage is a CIB slot for the variant, else plain C2f."""
        name = f"model.{idx}"
        if idx in cib_slots:
            return c2f_cib(x, c2, n, shortcut=True, lk=cib_lk, data_format=data_format, name=name)
        return c2f(x, c2, n, shortcut=c2f_shortcut, data_format=data_format, name=name)

    inp = image_input(input_shape, data_format)

    x = conv_bn(inp, ch(64), 3, 2, data_format=data_format, name="model.0")
    x = conv_bn(x, ch(128), 3, 2, data_format=data_format, name="model.1")
    x = c2f(x, ch(128), nd(3), shortcut=True, data_format=data_format, name="model.2")
    x = conv_bn(x, ch(256), 3, 2, data_format=data_format, name="model.3")
    x = c2f(x, ch(256), nd(6), shortcut=True, data_format=data_format, name="model.4")
    p3 = x
    x = scdown(x, ch(512), 3, 2, data_format=data_format, name="model.5")
    x = stage(x, ch(512), nd(6), 6, True)
    p4 = x
    x = scdown(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    x = stage(x, ch(1024), nd(3), 8, True)
    x = sppf(x, ch(1024), 5, data_format=data_format, name="model.9")
    x = psa(x, ch(1024), data_format=data_format, name="model.10")
    p5 = x

    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = stage(t, ch(512), nd(3), 13, False)
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = stage(t, ch(256), nd(3), 16, False)
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.17"), p4n], "cat2")
    p4o = stage(t, ch(512), nd(3), 19, False)
    t = cat([scdown(p4o, ch(512), 3, 2, data_format=data_format, name="model.20"), p5], "cat3")
    p5o = stage(t, ch(1024), nd(3), 22, True)

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        cls_dw=True,
        data_format=data_format,
        name=f"yolov10{variant}",
        head_name="model.23",
        backbone_end=10,
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
