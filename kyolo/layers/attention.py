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

from keras import ops

from .common import channels_of, concat_axis, conv_bn
from .knames import layers

__all__ = [
    "mhsa",
    "psa_block",
    "c2psa",
    "psa",
    "scdown",
    "cib",
    "c2f_cib",
    "area_attention",
    "a2c2f",
]


def _to_nhwc(x, data_format):
    return x if data_format == "channels_last" else ops.transpose(x, (0, 2, 3, 1))


def _from_nhwc(x, data_format):
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

    xh = _to_nhwc(x, data_format)
    # spatial dims are static for a fixed input size; batch stays dynamic (-1)
    H = xh.shape[1]
    W = xh.shape[2]
    N = H * W

    qkv = conv_bn(x, h, 1, 1, act=False, data_format=data_format, name=f"{name}.qkv")
    qkv = _to_nhwc(qkv, data_format)
    # (B, N, heads, key_dim*2 + head_dim)
    qkv = ops.reshape(qkv, (-1, N, num_heads, key_dim * 2 + head_dim))
    q = qkv[..., :key_dim]
    k = qkv[..., key_dim : 2 * key_dim]
    v = qkv[..., 2 * key_dim :]
    # -> (B, heads, N, d)
    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))

    attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale  # (B,heads,N,N)
    attn = ops.softmax(attn, axis=-1)
    out = ops.matmul(attn, v)  # (B, heads, N, head_dim)
    out = ops.transpose(out, (0, 2, 1, 3))  # (B, N, heads, head_dim)
    out = ops.reshape(out, (-1, H, W, dim))
    out = _from_nhwc(out, data_format)

    # depth-wise positional encoding on the value features, laid out as an image
    v_img = ops.reshape(ops.transpose(v, (0, 2, 1, 3)), (-1, H, W, dim))
    v_img = _from_nhwc(v_img, data_format)
    pe = conv_bn(
        v_img, dim, 3, 1, groups=dim, act=False, data_format=data_format, name=f"{name}.pe"
    )
    out = layers.Add(name=f"{name}.add_pe")([out, pe])
    return conv_bn(out, dim, 1, 1, act=False, data_format=data_format, name=f"{name}.proj")


def _ffn(x, c, data_format, name):
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
    f = _ffn(x, c, data_format, name=f"{name}.ffn")
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
    f = _ffn(b, c, data_format, name=f"{name}.ffn")
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


def cib(x, c2, shortcut=True, e=0.5, data_format="channels_last", name="cib"):
    """YOLOv10 Compact Inverted Block."""
    c1 = channels_of(x, data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, c1, 3, 1, groups=c1, data_format=data_format, name=f"{name}.cv1.0")
    y = conv_bn(y, 2 * c_, 1, 1, data_format=data_format, name=f"{name}.cv1.1")
    y = conv_bn(y, 2 * c_, 3, 1, groups=2 * c_, data_format=data_format, name=f"{name}.cv1.2")
    y = conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv1.3")
    y = conv_bn(y, c2, 3, 1, groups=c2, data_format=data_format, name=f"{name}.cv1.4")
    if shortcut and c1 == c2:
        y = layers.Add(name=f"{name}.add")([x, y])
    return y


def c2f_cib(x, c2, n=1, shortcut=False, e=0.5, data_format="channels_last", name="c2fcib"):
    """YOLOv10 C2fCIB: a C2f whose inner blocks are CIB."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, 2 * c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    y0, y1 = ops.split(y, 2, axis=ax)
    outs = [y0, y1]
    cur = y1
    for i in range(n):
        cur = cib(cur, c_, shortcut=shortcut, e=1.0, data_format=data_format, name=f"{name}.m.{i}")
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

    xh = _to_nhwc(x, data_format)
    H = xh.shape[1]
    W = xh.shape[2]
    N = H * W
    if area > 1 and N % area != 0:
        area = 1  # fall back to global attention if the grid is not divisible
    seq = N // area  # static sequence length per area group

    qkv = conv_bn(x, all_head * 3, 1, 1, act=False, data_format=data_format, name=f"{name}.qkv")
    qkv = _to_nhwc(qkv, data_format)  # (B,H,W,3*all_head)
    # partition sequence dim into `area` groups -> (B*area, seq, 3*all_head) via -1 batch
    qkv = ops.reshape(qkv, (-1, seq, num_heads, head_dim * 3))
    q = qkv[..., :head_dim]
    k = qkv[..., head_dim : 2 * head_dim]
    v = qkv[..., 2 * head_dim :]
    q = ops.transpose(q, (0, 2, 1, 3))  # (Ba, heads, seq, head_dim)
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))

    attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale
    attn = ops.softmax(attn, axis=-1)
    out = ops.matmul(attn, v)  # (Ba, heads, seq, head_dim)
    out = ops.transpose(out, (0, 2, 1, 3))  # (Ba, seq, heads, head_dim)
    out = ops.reshape(out, (-1, H, W, all_head))
    out = _from_nhwc(out, data_format)

    v_img = ops.transpose(v, (0, 2, 1, 3))
    v_img = ops.reshape(v_img, (-1, H, W, all_head))
    v_img = _from_nhwc(v_img, data_format)
    pe = conv_bn(
        v_img, dim, 7, 1, groups=dim, act=False, data_format=data_format, name=f"{name}.pe"
    )
    out = layers.Add(name=f"{name}.add_pe")([out, pe])
    return conv_bn(out, dim, 1, 1, act=False, data_format=data_format, name=f"{name}.proj")


def a2c2f(
    x,
    c2,
    n=1,
    area=1,
    num_heads=4,
    mlp_ratio=1.2,
    e=0.5,
    data_format="channels_last",
    name="a2c2f",
):
    """YOLO12 A2C2f: C2f-style aggregation with area-attention + MLP blocks."""
    ax = concat_axis(data_format)
    c_ = int(c2 * e)
    y = conv_bn(x, c_, 1, 1, data_format=data_format, name=f"{name}.cv1")
    outs = [y]
    cur = y
    for i in range(n):
        att = area_attention(
            cur, num_heads=num_heads, area=area, data_format=data_format, name=f"{name}.m.{i}.attn"
        )
        cur = layers.Add(name=f"{name}.m.{i}.add_attn")([cur, att])
        hidden = int(c_ * mlp_ratio)
        mlp = conv_bn(cur, hidden, 1, 1, data_format=data_format, name=f"{name}.m.{i}.mlp.0")
        mlp = conv_bn(mlp, c_, 1, 1, act=False, data_format=data_format, name=f"{name}.m.{i}.mlp.1")
        cur = layers.Add(name=f"{name}.m.{i}.add_mlp")([cur, mlp])
        outs.append(cur)
    y = layers.Concatenate(axis=ax, name=f"{name}.concat")(outs)
    return conv_bn(y, c2, 1, 1, data_format=data_format, name=f"{name}.cv2")
