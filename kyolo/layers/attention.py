"""Attention blocks for YOLOv10 / YOLO11 / YOLO12.

Contains:
  * ``mhsa``       - multi-head self-attention with a depth-wise positional
                     encoding (the YOLO11 ``Attention`` module).
  * ``psa_block``  - attention + FFN residual block (YOLO11 ``PSABlock``).
  * ``c2psa``      - CSP wrapper around ``psa_block`` (YOLO11 ``C2PSA``).
  * ``psa``        - single-path PSA (YOLOv10 ``PSA``).
  * ``scdown``     - spatial-channel decoupled downsample (YOLOv10 ``SCDown``).
  * ``cib`` / ``c2f_cib`` - compact inverted block (YOLOv10 large stages).
  * ``area_attention`` / ``a2c2f`` - area attention (YOLO12 ``AAttn`` / ``A2C2f``).

All spatial reshapes use ``keras.ops`` so the code is backend-agnostic. Channel
counts are static (derived from the fixed model width), while ``H``/``W`` may be
dynamic.
"""

from __future__ import annotations

import keras
from keras import ops

from .blocks import c3k
from .common import channels_of, concat_axis, conv_bn
from .knames import kname, layers

__all__ = [
    "mhsa",
    "psa_block",
    "c2psa",
    "psa",
    "scdown",
    "cib",
    "c2f_cib",
    "rep_vgg_dw",
    "area_attention",
    "ablock",
    "a2c2f",
]


def to_nhwc(x, data_format):
    return x if data_format == "channels_last" else ops.transpose(x, (0, 2, 3, 1))


def from_nhwc(x, data_format):
    return x if data_format == "channels_last" else ops.transpose(x, (0, 3, 1, 2))


def mhsa(x, num_heads=4, attn_ratio=0.5, data_format="channels_last", name="attn"):
    """Multi-head self-attention over the spatial grid (YOLO11 ``Attention``).

    ``qkv``/``proj``/``pe`` are all ``Conv(act=False)`` (conv+bn), matching the
    reference. Positional information is injected via a depth-wise conv on the
    value features.
    """
    dim = channels_of(x, data_format)
    head_dim = dim // num_heads
    key_dim = int(head_dim * attn_ratio)
    scale = key_dim**-0.5
    nh_kd = key_dim * num_heads
    h = dim + nh_kd * 2

    xh = to_nhwc(x, data_format)

    H = xh.shape[1]
    W = xh.shape[2]
    N = H * W

    qkv = conv_bn(x, h, 1, 1, act=False, data_format=data_format, name=f"{name}.qkv")
    qkv = to_nhwc(qkv, data_format)

    qkv = ops.reshape(qkv, (-1, N, num_heads, key_dim * 2 + head_dim))
    q = qkv[..., :key_dim]
    k = qkv[..., key_dim : 2 * key_dim]
    v = qkv[..., 2 * key_dim :]

    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))

    attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale
    attn = ops.softmax(attn, axis=-1)
    out = ops.matmul(attn, v)
    out = ops.transpose(out, (0, 2, 1, 3))
    out = ops.reshape(out, (-1, H, W, dim))
    out = from_nhwc(out, data_format)

    v_img = ops.reshape(ops.transpose(v, (0, 2, 1, 3)), (-1, H, W, dim))
    v_img = from_nhwc(v_img, data_format)
    pe = conv_bn(
        v_img, dim, 3, 1, groups=dim, act=False, data_format=data_format, name=f"{name}.pe"
    )
    out = layers.Add(name=f"{name}.add_pe")([out, pe])
    return conv_bn(out, dim, 1, 1, act=False, data_format=data_format, name=f"{name}.proj")


def ffn(x, c, data_format, name):
    y = conv_bn(x, c * 2, 1, 1, data_format=data_format, name=f"{name}.0")
    return conv_bn(y, c, 1, 1, act=False, data_format=data_format, name=f"{name}.1")


def psa_block(
    x, num_heads=4, attn_ratio=0.5, shortcut=True, data_format="channels_last", name="psablock"
):
    """YOLO11 PSABlock: residual attention followed by residual FFN."""
    c = channels_of(x, data_format)
    a = mhsa(
        x, num_heads=num_heads, attn_ratio=attn_ratio, data_format=data_format, name=f"{name}.attn"
    )
    x = layers.Add(name=f"{name}.add_attn")([x, a]) if shortcut else a
    f = ffn(x, c, data_format, name=f"{name}.ffn")
    x = layers.Add(name=f"{name}.add_ffn")([x, f]) if shortcut else f
    return x


def c2psa(x, c2, n=1, e=0.5, data_format="channels_last", name="c2psa"):
    """YOLO11 C2PSA: split, run PSABlocks on one half, fuse."""
    ax = concat_axis(data_format)
    c = int(c2 * e)
    num_heads = max(1, c // 64)
    y = conv_bn(x, 2 * c, 1, 1, data_format=data_format, name=f"{name}.cv1")
    a, b = ops.split(y, 2, axis=ax)
    for i in range(n):
        b = psa_block(
            b, num_heads=num_heads, attn_ratio=0.5, data_format=data_format, name=f"{name}.m.{i}"
        )
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")([a, b])
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv2")


def psa(x, c2, e=0.5, data_format="channels_last", name="psa"):
    """YOLOv10 PSA: split, single attention + FFN on one half, fuse."""
    ax = concat_axis(data_format)
    c = int(c2 * e)
    num_heads = max(1, c // 64)
    y = conv_bn(x, 2 * c, 1, 1, data_format=data_format, name=f"{name}.cv1")
    a, b = ops.split(y, 2, axis=ax)
    att = mhsa(b, num_heads=num_heads, attn_ratio=0.5, data_format=data_format, name=f"{name}.attn")
    b = layers.Add(name=f"{name}.add_attn")([b, att])
    f = ffn(b, c, data_format, name=f"{name}.ffn")
    b = layers.Add(name=f"{name}.add_ffn")([b, f])
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")([a, b])
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv2")


def scdown(x, c2, kernel_size=3, strides=2, data_format="channels_last", name="scdown"):
    """YOLOv10 spatial-channel decoupled downsample: 1x1 pointwise then dw stride."""
    y = conv_bn(x, c2, 1, 1, data_format=data_format, name=f"{name}.cv1")
    return conv_bn(
        y,
        c2,
        kernel_size,
        strides,
        groups=c2,
        act=False,
        data_format=data_format,
        name=f"{name}.cv2",
    )


def rep_vgg_dw(x, dim, data_format="channels_last", name="repdw"):
    """RepVGGDW: parallel 7x7 + 3x3 depth-wise convs, summed then activated.

    Built in the unfused (training) form the official YOLOv10 checkpoints ship
    in: two depth-wise ``Conv(act=False)`` branches named ``conv`` (7x7) and
    ``conv1`` (3x3), added, then SiLU. This matches the reference exactly, so the
    weights convert one-to-one (a deploy-time fuse would merge the 3x3 into the
    7x7 kernel, but the released weights are not fused).
    """
    a = conv_bn(x, dim, 7, 1, groups=dim, act=False, data_format=data_format, name=f"{name}.conv")
    b = conv_bn(x, dim, 3, 1, groups=dim, act=False, data_format=data_format, name=f"{name}.conv1")
    y = layers.Add(name=f"{name}.add")([a, b])
    return layers.Activation("swish", name=f"{name}.act")(y)


def cib(x, c2, shortcut=True, e=0.5, lk=False, data_format="channels_last", name="cib"):
    """YOLOv10 Compact Inverted Block.

    ``lk`` (large-kernel) swaps the central 3x3 depth-wise conv for a
    :func:`rep_vgg_dw` (7x7 + 3x3), matching the Ultralytics ``C2fCIB(..., lk=True)``
    stages (nano stage 22; small stages 8 and 22).
    """
    c1 = channels_of(x, data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, c1, 3, 1, groups=c1, data_format=data_format, name=f"{name}.cv1.0")
    y = conv_bn(y, 2 * c_, 1, 1, data_format=data_format, name=f"{name}.cv1.1")
    if lk:
        y = rep_vgg_dw(y, 2 * c_, data_format=data_format, name=f"{name}.cv1.2")
    else:
        y = conv_bn(y, 2 * c_, 3, 1, groups=2 * c_, data_format=data_format, name=f"{name}.cv1.2")
    y = conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv1.3")
    y = conv_bn(y, c2, 3, 1, groups=c2, data_format=data_format, name=f"{name}.cv1.4")
    if shortcut and c1 == c2:
        y = layers.Add(name=f"{name}.add")([x, y])
    return y


def c2f_cib(
    x, c2, n=1, shortcut=False, e=0.5, lk=False, data_format="channels_last", name="c2fcib"
):
    """YOLOv10 C2fCIB: a C2f whose inner blocks are CIB (``lk`` -> large-kernel)."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, 2 * c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    y0, y1 = ops.split(y, 2, axis=ax)
    outs = [y0, y1]
    cur = y1
    for i in range(n):
        cur = cib(
            cur, c_, shortcut=shortcut, e=1.0, lk=lk, data_format=data_format, name=f"{name}.m.{i}"
        )
        outs.append(cur)
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(outs)
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv2")


def area_attention(x, num_heads=4, area=1, data_format="channels_last", name="aattn"):
    """YOLO12 area attention (``AAttn``).

    The feature map is split into ``area`` horizontal strips, self-attention is
    computed independently within each strip, then the strips are merged back.
    ``area == 1`` recovers ordinary global attention. A depth-wise conv adds
    positional encoding.
    """
    dim = channels_of(x, data_format)
    head_dim = dim // num_heads
    all_head = head_dim * num_heads
    scale = head_dim**-0.5

    xh = to_nhwc(x, data_format)
    H = xh.shape[1]
    W = xh.shape[2]
    N = H * W
    if area > 1 and N % area != 0:
        area = 1
    seq = N // area

    qkv = conv_bn(x, all_head * 3, 1, 1, act=False, data_format=data_format, name=f"{name}.qkv")
    qkv = to_nhwc(qkv, data_format)

    qkv = ops.reshape(qkv, (-1, seq, num_heads, head_dim * 3))
    q = qkv[..., :head_dim]
    k = qkv[..., head_dim : 2 * head_dim]
    v = qkv[..., 2 * head_dim :]
    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))

    attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale
    attn = ops.softmax(attn, axis=-1)
    out = ops.matmul(attn, v)
    out = ops.transpose(out, (0, 2, 1, 3))
    out = ops.reshape(out, (-1, H, W, all_head))
    out = from_nhwc(out, data_format)

    v_img = ops.transpose(v, (0, 2, 1, 3))
    v_img = ops.reshape(v_img, (-1, H, W, all_head))
    v_img = from_nhwc(v_img, data_format)

    pe = conv_bn(
        v_img,
        dim,
        7,
        1,
        groups=dim,
        act=False,
        use_bias=True,
        data_format=data_format,
        name=f"{name}.pe",
    )
    out = layers.Add(name=f"{name}.add_pe")([out, pe])
    return conv_bn(out, dim, 1, 1, act=False, data_format=data_format, name=f"{name}.proj")


class _ResidualScale(keras.layers.Layer):
    """``x + gamma * y`` with a learnable per-channel ``gamma`` (YOLO12 A2C2f).

    Mirrors the Ultralytics ``A2C2f`` residual: the ``gamma`` is a bare
    parameter initialised to 0.01. The weight leaf is named ``scale`` so the
    converter maps it to the Torch ``gamma`` parameter (see
    :mod:`kyolo.conversion.convert`).
    """

    def __init__(self, channels, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.data_format = data_format

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(self.channels,),
            initializer=keras.initializers.Constant(0.01),
            trainable=True,
        )

    def call(self, inputs):
        x, y = inputs
        shape = (1, 1, 1, -1) if self.data_format == "channels_last" else (1, -1, 1, 1)
        return x + ops.reshape(self.scale, shape) * y

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def ablock(x, num_heads, area=1, mlp_ratio=2.0, data_format="channels_last", name="ablock"):
    """YOLO12 ABlock: residual area-attention followed by a residual MLP.

    Mirrors the Ultralytics ``ABlock``: ``x = x + attn(x)`` then
    ``x = x + mlp(x)``, where ``mlp`` is ``Conv(1x1) -> Conv(1x1, act=False)``
    with hidden width ``int(dim * mlp_ratio)``.
    """
    c = channels_of(x, data_format)
    att = area_attention(
        x, num_heads=num_heads, area=area, data_format=data_format, name=f"{name}.attn"
    )
    x = layers.Add(name=f"{name}.add_attn")([x, att])
    hidden = int(c * mlp_ratio)
    mlp = conv_bn(x, hidden, 1, 1, data_format=data_format, name=f"{name}.mlp.0")
    mlp = conv_bn(mlp, c, 1, 1, act=False, data_format=data_format, name=f"{name}.mlp.1")
    return layers.Add(name=f"{name}.add_mlp")([x, mlp])


def a2c2f(
    x,
    c2,
    n=1,
    a2=True,
    area=1,
    mlp_ratio=2.0,
    e=0.5,
    shortcut=True,
    residual=False,
    data_format="channels_last",
    name="a2c2f",
):
    """YOLO12 A2C2f: C2f-style aggregation over area-attention or C3k blocks.

    Each of the ``n`` inner entries (``m.{i}``) is, following Ultralytics'
    ``A2C2f``:

    * ``a2=True``  -> a sequence of two :func:`ablock` area-attention blocks
      (``m.{i}.0`` and ``m.{i}.1``); ``num_heads`` is ``hidden // 32``.
    * ``a2=False`` -> a single :func:`kyolo.layers.c3k` block (``n=2``), i.e. no
      attention (the configuration used by the YOLO12 neck).

    ``residual`` (used by the L/X scales, together with ``mlp_ratio=1.2``) adds a
    learnable per-channel residual ``x + gamma * y``; it only takes effect when
    ``a2=True`` (matching Ultralytics' ``gamma if a2 and residual``).
    """
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    outs = [y]
    cur = y
    num_heads = max(1, c_ // 32)
    for i in range(n):
        if a2:
            for j in range(2):
                cur = ablock(
                    cur,
                    num_heads=num_heads,
                    area=area,
                    mlp_ratio=mlp_ratio,
                    data_format=data_format,
                    name=f"{name}.m.{i}.{j}",
                )
        else:
            cur = c3k(
                cur,
                c_,
                n=2,
                shortcut=shortcut,
                k=3,
                data_format=data_format,
                name=f"{name}.m.{i}",
            )
        outs.append(cur)
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(outs)
    y = conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv2")
    if a2 and residual:
        y = _ResidualScale(c2, data_format=data_format, name=kname(name))([x, y])
    return y
