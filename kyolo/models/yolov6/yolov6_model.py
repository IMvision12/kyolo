"""YOLOv6 - EfficientRep backbone + Rep-PAN neck, anchor-free DFL head.

Reparameterizable RepVGG blocks are built with the ``deploy`` flag: ``True``
gives the fused single-conv inference graph, ``False`` the multi-branch training
graph. The medium/large variants use CSPStackRep (``bepc3``) blocks instead of
plain RepBlocks.
"""

from __future__ import annotations

from keras import layers

from ...layers import bepc3, conv_bn, rep_block, rep_conv, sppf
from ...layers.common import concat_axis
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLOV6_CONFIG

__all__ = ["YOLOV6_CONFIG", "build_yolov6", "YOLOv6"]


# variant: (depth, width, use_csp)
def build_yolov6(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    reg_max=16,
    **kwargs,
):
    d, w, use_csp = YOLOV6_CONFIG[variant]
    ax = concat_axis(data_format)

    def ch(c):
        return scale_channels(c, w)

    def nd(x):
        return scale_depth(x, d)

    def stage(x, c, n, name):
        if use_csp:
            return bepc3(x, c, n, deploy=deploy, data_format=data_format, name=name)
        return rep_block(x, c, n, deploy=deploy, data_format=data_format, name=name)

    def up(x, name):
        return layers.UpSampling2D(2, data_format=data_format, interpolation="nearest", name=name)(
            x
        )

    def cat(xs, name):
        return layers.Concatenate(axis=ax, name=name)(xs)

    inp = image_input(input_shape, data_format)

    # --- EfficientRep backbone ---
    x = rep_conv(inp, ch(64), 3, 2, deploy=deploy, data_format=data_format, name="backbone.stem")
    x = rep_conv(x, ch(128), 3, 2, deploy=deploy, data_format=data_format, name="backbone.down2")
    x = stage(x, ch(128), nd(6), "backbone.erb2")
    x = rep_conv(x, ch(256), 3, 2, deploy=deploy, data_format=data_format, name="backbone.down3")
    x = stage(x, ch(256), nd(12), "backbone.erb3")
    p3 = x
    x = rep_conv(x, ch(512), 3, 2, deploy=deploy, data_format=data_format, name="backbone.down4")
    x = stage(x, ch(512), nd(18), "backbone.erb4")
    p4 = x
    x = rep_conv(x, ch(1024), 3, 2, deploy=deploy, data_format=data_format, name="backbone.down5")
    x = stage(x, ch(1024), nd(6), "backbone.erb5")
    x = sppf(x, ch(1024), 5, data_format=data_format, name="backbone.sppf")
    p5 = x

    # --- Rep-PAN neck ---
    fpn0 = conv_bn(p5, ch(256), 1, 1, data_format=data_format, name="neck.reduce0")
    t = cat([up(fpn0, "up0"), p4], "cat0")
    fout0 = stage(t, ch(256), nd(12), "neck.rep_p4")
    fpn1 = conv_bn(fout0, ch(128), 1, 1, data_format=data_format, name="neck.reduce1")
    t = cat([up(fpn1, "up1"), p3], "cat1")
    p3o = stage(t, ch(128), nd(12), "neck.rep_p3")
    t = cat([conv_bn(p3o, ch(128), 3, 2, data_format=data_format, name="neck.down2"), fpn1], "cat2")
    p4o = stage(t, ch(256), nd(12), "neck.rep_n3")
    t = cat([conv_bn(p4o, ch(256), 3, 2, data_format=data_format, name="neck.down1"), fpn0], "cat3")
    p5o = stage(t, ch(512), nd(12), "neck.rep_n4")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        data_format=data_format,
        name=f"yolov6{variant}",
    )


def YOLOv6(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format="channels_last",
    deploy=True,
    **kwargs,
):
    """Factory for a YOLOv6 detector (variant in ``{n, s, m, l}``)."""
    return build_yolov6(variant, nc, input_shape, data_format, deploy, **kwargs)
