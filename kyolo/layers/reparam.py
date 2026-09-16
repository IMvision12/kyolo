"""Reparameterizable blocks (RepVGG-style) and the GELAN family.

Used by YOLOv9: the GELAN blocks (``rep_ncspelan4``, ``elan1``, ``adown``,
``aconv``, ``sppelan``) plus the ``e`` variant's dual-branch fusion
(``cblinear`` / ``cbfuse``), all built on the RepVGG-style ``rep_conv``.

Every reparameterizable block supports a ``deploy`` flag:

* ``deploy=False`` -> the multi-branch graph (3x3 conv-bn + 1x1 conv-bn, summed
  then activated). This is the form the official checkpoints ship in, so it is
  the default for weight conversion, and it is what you fine-tune before fusing.
* ``deploy=True``  -> a single fused ``Conv2D(bias) + act`` branch. Mathematically
  equivalent to the unfused graph and faster for inference.
"""

from __future__ import annotations

from keras import ops

from .common import act_layer, autopad, channels_of, concat_axis, conv_bn
from .knames import layers

__all__ = [
    "rep_conv",
    "rep_bottleneck",
    "rep_ncsp",
    "rep_ncspelan4",
    "elan1",
    "adown",
    "aconv",
    "sppelan",
    "cblinear",
    "cbfuse",
]


def rep_conv(
    x,
    c2,
    kernel_size=3,
    strides=1,
    groups=1,
    act=True,
    deploy=False,
    data_format="channels_last",
    name="rep",
):
    """RepVGG-style reparameterizable convolution.

    Training graph = 3x3 branch + 1x1 branch, summed then activated. Deploy
    graph = one fused 3x3 conv with bias. Mirrors the Ultralytics ``RepConv``,
    which is instantiated with ``bn=False`` throughout YOLOv9/GELAN, so there is
    no identity-BN branch (only the two conv-bn branches).
    """
    if deploy:
        pad = autopad(kernel_size, None)
        if strides > 1:
            x = layers.ZeroPadding2D(pad, data_format=data_format, name=f"{name}.pad")(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        y = layers.Conv2D(
            c2,
            kernel_size,
            strides,
            padding=conv_pad,
            groups=groups,
            use_bias=True,
            data_format=data_format,
            name=f"{name}.conv",
        )(x)
        a = act_layer(act, name=f"{name}.act")
        return a(y) if a is not None else y

    branch_3 = conv_bn(
        x,
        c2,
        kernel_size,
        strides,
        groups=groups,
        act=False,
        data_format=data_format,
        name=f"{name}.conv1",
    )
    branch_1 = conv_bn(
        x,
        c2,
        1,
        strides,
        groups=groups,
        act=False,
        data_format=data_format,
        name=f"{name}.conv2",
    )
    y = layers.Add(name=f"{name}.add")([branch_3, branch_1])
    a = act_layer(act, name=f"{name}.act")
    return a(y) if a is not None else y


def rep_bottleneck(
    x,
    c2,
    shortcut=True,
    groups=1,
    e=0.5,
    deploy=False,
    data_format="channels_last",
    name="repbottleneck",
):
    """RepNBottleneck: RepConv(3x3) -> Conv(3x3) with optional residual."""
    c1 = channels_of(x, data_format)
    c_ = int(c2 * e)
    y = rep_conv(x, c_, 3, 1, deploy=deploy, data_format=data_format, name=f"{name}.cv1")
    y = conv_bn(y, c2, 3, 1, groups=groups, data_format=data_format, name=f"{name}.cv2")
    if shortcut and c1 == c2:
        y = layers.Add(name=f"{name}.add")([x, y])
    return y


def rep_ncsp(
    x,
    c2,
    n=1,
    shortcut=True,
    groups=1,
    e=0.5,
    deploy=False,
    data_format="channels_last",
    name="repncsp",
):
    """CSP block whose bottlenecks are RepNBottlenecks (YOLOv9)."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    a = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    for i in range(n):
        a = rep_bottleneck(
            a,
            c_,
            shortcut=shortcut,
            groups=groups,
            e=1.0,
            deploy=deploy,
            data_format=data_format,
            name=f"{name}.m.{i}",
        )
    b = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv2")
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")([a, b])
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv3")


def rep_ncspelan4(
    x,
    c2,
    c3,
    c4,
    n=1,
    deploy=False,
    data_format="channels_last",
    name="gelan",
):
    """RepNCSPELAN4 - the core GELAN aggregation block of YOLOv9.

    ``c3`` is the split width, ``c4`` the branch width. ``cv2`` and ``cv3`` are
    each ``RepNCSP -> Conv(3x3)`` sequences (named ``.0`` and ``.1``).
    """
    ax = concat_axis(data_format)
    y = conv_bn(x, c3, 1, 1, data_format=data_format, name=f"{name}.cv1")
    y0, y1 = ops.split(y, 2, axis=ax)
    parts = [y0, y1]

    b = rep_ncsp(y1, c4, n=n, deploy=deploy, data_format=data_format, name=f"{name}.cv2.0")
    b = conv_bn(b, c4, 3, 1, data_format=data_format, name=f"{name}.cv2.1")
    parts.append(b)

    c = rep_ncsp(b, c4, n=n, deploy=deploy, data_format=data_format, name=f"{name}.cv3.0")
    c = conv_bn(c, c4, 3, 1, data_format=data_format, name=f"{name}.cv3.1")
    parts.append(c)

    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(parts)
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv4")


def elan1(x, c2, c3, c4, data_format="channels_last", name="elan1"):
    """YOLOv9 ELAN1 block (the RepNCSPELAN4 topology with plain 3x3 convs).

    Structurally a :func:`rep_ncspelan4` where ``cv2``/``cv3`` are single 3x3
    convolutions instead of ``RepCSP -> Conv`` sequences (used at the first
    backbone stage of the smaller GELAN variants).
    """
    ax = concat_axis(data_format)
    y = conv_bn(x, c3, 1, 1, data_format=data_format, name=f"{name}.cv1")
    y0, y1 = ops.split(y, 2, axis=ax)
    b = conv_bn(y1, c4, 3, 1, data_format=data_format, name=f"{name}.cv2")
    c = conv_bn(b, c4, 3, 1, data_format=data_format, name=f"{name}.cv3")
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")([y0, y1, b, c])
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv4")


def adown(x, c2, data_format="channels_last", name="adown"):
    """YOLOv9 ADown downsampling block (avg-pool + max-pool dual branch).

    The leading ``2x2`` average pool is stride-1 ``valid`` (no padding), matching
    Ultralytics' ``F.avg_pool2d(x, 2, 1, 0)``.
    """
    ax = concat_axis(data_format)
    c_ = c2 // 2
    x = layers.AveragePooling2D(
        pool_size=2, strides=1, padding="valid", data_format=data_format, name=f"{name}.avg"
    )(x)
    x1, x2 = ops.split(x, 2, axis=ax)
    x1 = conv_bn(x1, c_, 3, 2, data_format=data_format, name=f"{name}.cv1")
    x2 = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="same", data_format=data_format, name=f"{name}.max"
    )(x2)
    x2 = conv_bn(x2, c_, 1, 1, data_format=data_format, name=f"{name}.cv2")
    return layers.Concatenate(axis=ax, name=f"{name}.concat")([x1, x2])


def aconv(x, c2, data_format="channels_last", name="aconv"):
    """YOLOv9 AConv downsampling block (stride-1 ``valid`` avg-pool + 3x3 s2 conv).

    Mirrors Ultralytics' ``AConv``: ``F.avg_pool2d(x, 2, 1, 0)`` followed by a
    single ``Conv(c2, 3, 2)``.
    """
    x = layers.AveragePooling2D(
        pool_size=2, strides=1, padding="valid", data_format=data_format, name=f"{name}.avg"
    )(x)
    return conv_bn(x, c2, 3, 2, data_format=data_format, name=f"{name}.cv1")


def sppelan(x, c2, c3, k=5, data_format="channels_last", name="sppelan"):
    """YOLOv9 SPP-ELAN block."""
    ax = concat_axis(data_format)
    y = conv_bn(x, c3, 1, 1, data_format=data_format, name=f"{name}.cv1")
    pools = [y]
    for i in range(3):
        pools.append(
            layers.MaxPooling2D(
                pool_size=k,
                strides=1,
                padding="same",
                data_format=data_format,
                name=f"{name}.cv{i + 2}",
            )(pools[-1])
        )
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(pools)
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv5")


def spatial_hw(x, data_format):
    """Static (H, W) of a feature tensor for the given data format."""
    shape = x.shape
    return (shape[1], shape[2]) if data_format == "channels_last" else (shape[2], shape[3])


def cblinear(x, c2s, data_format="channels_last", name="cblinear"):
    """YOLOv9-e CBLinear: a single 1x1 conv whose output is split into groups.

    ``c2s`` is the list of per-group output channel counts. Returns a list of
    tensors (one per group) consumed by :func:`cbfuse`. The conv keeps a bias
    and has no BatchNorm, matching Ultralytics' ``CBLinear``.
    """
    ax = concat_axis(data_format)
    y = layers.Conv2D(sum(c2s), 1, use_bias=True, data_format=data_format, name=f"{name}.conv")(x)
    if len(c2s) == 1:
        return [y]
    split_points = []
    running = 0
    for c in c2s[:-1]:
        running += c
        split_points.append(running)
    return list(ops.split(y, split_points, axis=ax))


def cbfuse(xs, idx, data_format="channels_last", name="cbfuse"):
    """YOLOv9-e CBFuse: select, resize and sum multi-scale CBLinear features.

    ``xs`` is ``[cblinear_out_0, ..., cblinear_out_{n-1}, target]`` where each
    ``cblinear_out_i`` is a list of tensors and ``target`` is the running
    feature map. For each source we take element ``idx[i]``, nearest-resize it to
    the target's spatial size and sum everything (matching Ultralytics'
    ``torch.sum(torch.stack(res + xs[-1:]), dim=0)``).
    """
    target = xs[-1]
    th, tw = spatial_hw(target, data_format)
    parts = []
    for i, src in enumerate(xs[:-1]):
        sel = src[idx[i]]
        sh, sw = spatial_hw(sel, data_format)
        if (sh, sw) != (th, tw):
            sel = ops.image.resize(
                sel, size=(th, tw), interpolation="nearest", data_format=data_format
            )
        parts.append(sel)
    parts.append(target)
    return layers.Add(name=f"{name}.add")(parts)
