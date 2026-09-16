"""YOLO26 - end-to-end, NMS-free detector (2025).

Mirrors the Ultralytics ``yolo26.yaml``: a C3k2 + SPPF + C2PSA backbone (the
YOLO11 lineage) feeding the shared head. YOLO26 is natively **DFL-free**
(``reg_max = 1``: the head regresses the four box distances directly).

Being DFL-free changes how it is trained, not just decoded: with no bins to
classify, :class:`kyolo.losses.BboxLoss` supervises the box distances with an
image-size-normalized L1 instead of the Distribution Focal Loss, reported as
``l1`` rather than ``dfl``.

Notes / approximations:
  * Like kyolo's YOLOv10, only the ``cv2``/``cv3`` (one-to-many) head is built;
    the checkpoint's ``one2one_cv2``/``one2one_cv3`` deployment head that makes
    the official model NMS-free is not reproduced (its weights are simply
    skipped on conversion). Decode with the default (NMS) ``YOLOPostprocessor``;
    the NMS-free top-k decode (``end_to_end=True``) would return duplicate boxes.
  * STAL (small-target-aware label assignment) *is* reproduced, in
    :class:`kyolo.ops.TaskAlignedAssigner` -- it is stride-driven rather than
    architectural, so every family gets it, as in Ultralytics.
  * ProgLoss is implemented as :class:`kyolo.losses.E2EDetectionLoss`, but it
    supervises a one-to-many *and* a one-to-one branch, so it only becomes
    usable once the one2one head above exists.
"""

from __future__ import annotations

from ...layers import c2psa, c3k2, conv_bn, sppf
from ...layers.common import concat_axis, resolve_data_format
from ...layers.knames import layers
from ..base import finalize_detector, image_input, scale_channels, scale_depth
from .config import YOLO26_CONFIG

__all__ = ["YOLO26_CONFIG", "build_yolo26", "YOLO26"]


def build_yolo26(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    reg_max=1,
    **kwargs,
):
    d, w, mc = YOLO26_CONFIG[variant]
    data_format = resolve_data_format(data_format)
    ax = concat_axis(data_format)

    c3k_early = variant in ("m", "l", "x")

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
    x = c3k2(x, ch(512), nd(2), use_c3k=True, data_format=data_format, name="model.6")
    p4 = x
    x = conv_bn(x, ch(1024), 3, 2, data_format=data_format, name="model.7")
    x = c3k2(x, ch(1024), nd(2), use_c3k=True, data_format=data_format, name="model.8")
    x = sppf(x, ch(1024), 5, cv1_act=False, shortcut=True, data_format=data_format, name="model.9")
    x = c2psa(x, ch(1024), nd(2), data_format=data_format, name="model.10")
    p5 = x

    t = cat([up(p5, "up0"), p4], "cat0")
    p4n = c3k2(t, ch(512), nd(2), use_c3k=True, data_format=data_format, name="model.13")
    t = cat([up(p4n, "up1"), p3], "cat1")
    p3o = c3k2(t, ch(256), nd(2), use_c3k=True, data_format=data_format, name="model.16")
    t = cat([conv_bn(p3o, ch(256), 3, 2, data_format=data_format, name="model.17"), p4n], "cat2")
    p4o = c3k2(t, ch(512), nd(2), use_c3k=True, data_format=data_format, name="model.19")
    t = cat([conv_bn(p4o, ch(512), 3, 2, data_format=data_format, name="model.20"), p5], "cat3")

    p5o = c3k2(t, ch(1024), nd(1), attn=True, data_format=data_format, name="model.22")

    return finalize_detector(
        inp,
        [p3o, p4o, p5o],
        nc=nc,
        reg_max=reg_max,
        cls_dw=True,
        data_format=data_format,
        name=f"yolo26{variant}",
        head_name="model.23",
        backbone_end=10,
    )


def YOLO26(
    variant="n",
    nc=80,
    input_shape=(640, 640, 3),
    data_format=None,
    deploy=True,
    **kwargs,
):
    """Factory for a YOLO26 detector (variant in ``{n, s, m, l, x}``); NMS-free."""
    return build_yolo26(variant, nc, input_shape, data_format, deploy, **kwargs)
