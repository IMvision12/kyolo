"""Reparameterizable blocks (RepVGG-style) and the GELAN family.

Used by YOLOv6 (EfficientRep / Rep-PAN, CSPStackRep) and YOLOv9 (GELAN:
RepNCSPELAN4, ADown, SPPELAN).

Every reparameterizable block supports a ``deploy`` flag:

* ``deploy=True``  -> a single fused ``Conv2D(bias) + act`` branch. This is the
  form the *converted* / inference checkpoints ship in, and the one used for
  fast inference.
* ``deploy=False`` -> the multi-branch training graph (3x3 conv-bn + 1x1
  conv-bn + optional identity bn), which is what you fine-tune with before
  fusing.
"""

from __future__ import annotations

from keras import layers, ops

from .common import act_layer, autopad, channels_of, concat_axis, conv_bn

__all__ = [
    "rep_conv",
    "rep_block",
    "rep_bottleneck",
    "rep_ncsp",
    "rep_ncspelan4",
    "adown",
    "sppelan",
    "bottlerep",
    "bepc3",
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

    Training graph = 3x3 branch + 1x1 branch (+ identity BN when shapes match),
    summed then activated. Deploy graph = one fused 3x3 conv with bias.
    """
    c1 = channels_of(x, data_format)
    axis = -1 if data_format == "channels_last" else 1

    if deploy:
        pad = autopad(kernel_size, None)
        if strides > 1:
            x = layers.ZeroPadding2D(pad, data_format=data_format, name=f"{name}.pad")(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        y = layers.Conv2D(
            c2, kernel_size, strides, padding=conv_pad, groups=groups,
            use_bias=True, data_format=data_format, name=f"{name}.conv",
        )(x)
        a = act_layer(act, name=f"{name}.act")
        return a(y) if a is not None else y

    # --- training (multi-branch) ---
    branch_3 = conv_bn(
        x, c2, kernel_size, strides, groups=groups, act=False,
        data_format=data_format, name=f"{name}.conv1",
    )
    branch_1 = conv_bn(
        x, c2, 1, strides, groups=groups, act=False,
        data_format=data_format, name=f"{name}.conv2",
    )
    outs = [branch_3, branch_1]
    if c1 == c2 and strides == 1:
        idbn = layers.BatchNormalization(
            axis=axis, momentum=0.97, epsilon=1e-3, name=f"{name}.bn"
        )(x)
        outs.append(idbn)
    y = layers.Add(name=f"{name}.add")(outs)
    a = act_layer(act, name=f"{name}.act")
    return a(y) if a is not None else y


def rep_block(
    x, c2, n=1, deploy=False, data_format="channels_last", name="repblock"
):
    """A stack of ``n`` RepConv 3x3 layers (YOLOv6 EfficientRep stage / Rep-PAN)."""
    y = rep_conv(x, c2, 3, 1, deploy=deploy, data_format=data_format, name=f"{name}.conv1")
    for i in range(n - 1):
        y = rep_conv(
            y, c2, 3, 1, deploy=deploy, data_format=data_format, name=f"{name}.block.{i}"
        )
    return y


def bottlerep(
    x, c2, shortcut=True, deploy=False, data_format="channels_last", name="bottlerep"
):
    """YOLOv6 BottleRep: two RepConvs with an optional residual add."""
    c1 = channels_of(x, data_format)
    y = rep_conv(x, c2, 3, 1, deploy=deploy, data_format=data_format, name=f"{name}.conv1")
    y = rep_conv(y, c2, 3, 1, deploy=deploy, data_format=data_format, name=f"{name}.conv2")
    if shortcut and c1 == c2:
        y = layers.Add(name=f"{name}.add")([x, y])
    return y


def bepc3(
    x, c2, n=1, e=0.5, deploy=False, data_format="channels_last", name="bepc3"
):
    """YOLOv6 CSPStackRep / BepC3 block (used by the medium & large variants)."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    a = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    for i in range(n):
        a = bottlerep(a, c_, shortcut=True, deploy=deploy, data_format=data_format, name=f"{name}.m.{i}")
    b = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv2")
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")([a, b])
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv3")


# --------------------------------------------------------------------------- #
# GELAN family (YOLOv9)
# --------------------------------------------------------------------------- #
def rep_bottleneck(
    x, c2, shortcut=True, groups=1, e=0.5, deploy=False,
    data_format="channels_last", name="repbottleneck",
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
    x, c2, n=1, shortcut=True, groups=1, e=0.5, deploy=False,
    data_format="channels_last", name="repncsp",
):
    """CSP block whose bottlenecks are RepNBottlenecks (YOLOv9)."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    a = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    for i in range(n):
        a = rep_bottleneck(
            a, c_, shortcut=shortcut, groups=groups, e=1.0, deploy=deploy,
            data_format=data_format, name=f"{name}.m.{i}",
        )
    b = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv2")
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")([a, b])
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv3")


def rep_ncspelan4(
    x, c2, c3, c4, n=1, deploy=False, data_format="channels_last", name="gelan",
):
    """RepNCSPELAN4 — the core GELAN aggregation block of YOLOv9.

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


def adown(x, c2, data_format="channels_last", name="adown"):
    """YOLOv9 ADown downsampling block (avg-pool + max-pool dual branch)."""
    ax = concat_axis(data_format)
    c_ = c2 // 2
    x = layers.AveragePooling2D(
        pool_size=2, strides=1, padding="same", data_format=data_format, name=f"{name}.avg"
    )(x)
    x1, x2 = ops.split(x, 2, axis=ax)
    x1 = conv_bn(x1, c_, 3, 2, data_format=data_format, name=f"{name}.cv1")
    x2 = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="same", data_format=data_format, name=f"{name}.max"
    )(x2)
    x2 = conv_bn(x2, c_, 1, 1, data_format=data_format, name=f"{name}.cv2")
    return layers.Concatenate(axis=ax, name=f"{name}.concat")([x1, x2])


def sppelan(x, c2, c3, k=5, data_format="channels_last", name="sppelan"):
    """YOLOv9 SPP-ELAN block."""
    ax = concat_axis(data_format)
    y = conv_bn(x, c3, 1, 1, data_format=data_format, name=f"{name}.cv1")
    pools = [y]
    for i in range(3):
        pools.append(
            layers.MaxPooling2D(
                pool_size=k, strides=1, padding="same", data_format=data_format,
                name=f"{name}.cv{i + 2}",
            )(pools[-1])
        )
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(pools)
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv5")
