"""CSP / SPP building blocks.

Functional builders (they take a tensor and a dotted ``name`` prefix and return
a tensor). Sub-layer names mirror the official PyTorch modules so the weight
mappings in :mod:`kyolo.conversion` stay simple, e.g. a ``C2f`` produces
``{name}.cv1``, ``{name}.cv2`` and ``{name}.m.{i}`` bottlenecks.
"""

from __future__ import annotations

from keras import ops

from .common import channels_of, concat_axis, conv_bn
from .knames import layers

__all__ = [
    "bottleneck",
    "c3",
    "c2f",
    "c3k",
    "c3k2",
    "sppf",
]


def bottleneck(
    x,
    c2,
    shortcut=True,
    groups=1,
    kernels=(3, 3),
    e=0.5,
    data_format="channels_last",
    name="bottleneck",
):
    """Standard residual bottleneck: Conv(k0) -> Conv(k1) with optional add."""
    c1 = channels_of(x, data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, c_, kernels[0], 1, data_format=data_format, name=f"{name}.cv1")
    y = conv_bn(y, c2, kernels[1], 1, groups=groups, data_format=data_format, name=f"{name}.cv2")
    if shortcut and c1 == c2:
        y = layers.Add(name=f"{name}.add")([x, y])
    return y


def c3(
    x,
    c2,
    n=1,
    shortcut=True,
    groups=1,
    e=0.5,
    bottleneck_kernels=(1, 3),
    data_format="channels_last",
    name="c3",
):
    """CSP bottleneck with 3 convolutions (YOLOv5 backbone/neck block)."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    a = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    for i in range(n):
        a = bottleneck(
            a,
            c_,
            shortcut=shortcut,
            groups=groups,
            kernels=bottleneck_kernels,
            e=1.0,
            data_format=data_format,
            name=f"{name}.m.{i}",
        )
    b = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv2")
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")([a, b])
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv3")


def c2f(
    x,
    c2,
    n=1,
    shortcut=False,
    groups=1,
    e=0.5,
    data_format="channels_last",
    name="c2f",
):
    """Faster CSP with 2 convolutions and dense gradient concatenation (YOLOv8)."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, 2 * c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    y0, y1 = ops.split(y, 2, axis=ax)
    outputs = [y0, y1]
    cur = y1
    for i in range(n):
        cur = bottleneck(
            cur,
            c_,
            shortcut=shortcut,
            groups=groups,
            kernels=(3, 3),
            e=1.0,
            data_format=data_format,
            name=f"{name}.m.{i}",
        )
        outputs.append(cur)
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(outputs)
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv2")


def c3k(
    x,
    c2,
    n=1,
    shortcut=True,
    groups=1,
    e=0.5,
    k=3,
    data_format="channels_last",
    name="c3k",
):
    """C3 variant with a configurable bottleneck kernel size (YOLO11 inner block)."""
    return c3(
        x,
        c2,
        n=n,
        shortcut=shortcut,
        groups=groups,
        e=e,
        bottleneck_kernels=(k, k),
        data_format=data_format,
        name=name,
    )


def c3k2(
    x,
    c2,
    n=1,
    use_c3k=False,
    shortcut=True,
    groups=1,
    e=0.5,
    data_format="channels_last",
    name="c3k2",
):
    """YOLO11 C3k2 block: a C2f whose inner blocks are C3k or Bottleneck."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, 2 * c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    y0, y1 = ops.split(y, 2, axis=ax)
    outputs = [y0, y1]
    cur = y1
    for i in range(n):
        if use_c3k:
            cur = c3k(
                cur,
                c_,
                n=2,
                shortcut=shortcut,
                groups=groups,
                data_format=data_format,
                name=f"{name}.m.{i}",
            )
        else:
            # Ultralytics C3k2 uses the default Bottleneck (e=0.5, hidden = c_/2),
            # unlike C2f which pins e=1.0. Matching this is required for the
            # official YOLO11 weights to load.
            cur = bottleneck(
                cur,
                c_,
                shortcut=shortcut,
                groups=groups,
                kernels=(3, 3),
                e=0.5,
                data_format=data_format,
                name=f"{name}.m.{i}",
            )
        outputs.append(cur)
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(outputs)
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv2")


def sppf(x, c2, k=5, act=True, data_format="channels_last", name="sppf"):
    """Spatial Pyramid Pooling - Fast (three chained max-pools).

    ``act`` selects the conv activation (``True`` -> SiLU, the YOLO default, or a
    string activation name).
    """
    ax = concat_axis(data_format)
    c1 = channels_of(x, data_format)
    c_ = c1 // 2
    y = conv_bn(x, c_, 1, 1, act=act, data_format=data_format, name=f"{name}.cv1")
    pools = [y]
    for i in range(3):
        pools.append(
            layers.MaxPooling2D(
                pool_size=k,
                strides=1,
                padding="same",
                data_format=data_format,
                name=f"{name}.m.{i}",
            )(pools[-1])
        )
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(pools)
    return conv_bn(y, c2, 1, 1, act=act, data_format=data_format, name=f"{name}.cv2")
