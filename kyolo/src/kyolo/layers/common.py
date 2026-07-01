"""Fundamental convolutional primitives shared by every YOLO architecture.

Everything here is written in pure ``keras.ops`` / ``keras.layers`` so the same
code runs on the TensorFlow, JAX and PyTorch backends without modification.

Naming convention
-----------------
Each helper builds its sub-layers with dotted names such as ``{name}.conv``,
``{name}.bn`` and ``{name}.act``.  This mirrors the module hierarchy of the
official (Ultralytics / WongKinYiu) PyTorch implementations, which makes the
weight-conversion mappings in :mod:`kyolo.conversion` far simpler.
"""

from __future__ import annotations

from keras import layers

__all__ = [
    "autopad",
    "act_layer",
    "conv_bn",
    "dw_conv",
    "concat_axis",
    "channels_of",
]


def channels_of(tensor, data_format="channels_last"):
    """Return the channel dimension of ``tensor`` for the given data format."""
    return tensor.shape[-1] if data_format == "channels_last" else tensor.shape[1]


def concat_axis(data_format="channels_last"):
    """Return the channel concatenation axis for the given data format."""
    return -1 if data_format == "channels_last" else 1


def autopad(kernel_size, padding=None, dilation=1):
    """Compute 'same'-style symmetric padding for a convolution.

    Mirrors Ultralytics' ``autopad`` helper. Returns the per-side pad amount.
    """
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


def act_layer(act, name=None):
    """Resolve an activation specification into a Keras layer.

    Args:
        act: ``True`` -> SiLU/swish (YOLO default), a string activation name,
            a callable/layer, or ``False``/``None`` for the identity.
        name: Optional layer name.

    Returns:
        A ``keras.layers.Layer`` (``Activation`` or the passed callable), or
        ``None`` when no activation should be applied.
    """
    if act is True:
        return layers.Activation("swish", name=name)  # SiLU == swish(beta=1)
    if act is False or act is None:
        return None
    if isinstance(act, str):
        return layers.Activation(act, name=name)
    # Already a layer / callable.
    return act


def conv_bn(
    x,
    filters,
    kernel_size=1,
    strides=1,
    padding=None,
    groups=1,
    dilation=1,
    act=True,
    use_bias=False,
    data_format="channels_last",
    bn_momentum=0.97,
    bn_epsilon=1e-3,
    name="conv",
):
    """Conv2D -> BatchNormalization -> activation (the ``Conv`` module).

    This is the atomic building block of the entire YOLO family. Stride-2
    convolutions use explicit symmetric zero-padding followed by ``valid``
    convolution so the output spatial size exactly matches the PyTorch
    reference (which pads with ``k // 2``).

    Args:
        x: Input tensor.
        filters: Number of output channels.
        kernel_size: Convolution kernel size.
        strides: Convolution stride.
        padding: Explicit per-side padding; ``None`` -> :func:`autopad`.
        groups: Grouped-convolution group count.
        dilation: Dilation rate.
        act: Activation spec (see :func:`act_layer`). Default SiLU.
        use_bias: Whether the convolution keeps a bias (``False`` when followed
            by BatchNorm, which is the standard case).
        data_format: ``"channels_last"`` or ``"channels_first"``.
        name: Dotted name prefix for the sub-layers.

    Returns:
        The activated, batch-normalized feature tensor.
    """
    pad = autopad(kernel_size, padding, dilation)
    if strides > 1:
        x = layers.ZeroPadding2D(
            padding=pad, data_format=data_format, name=f"{name}.pad"
        )(x)
        conv_padding = "valid"
    else:
        # stride 1 with symmetric autopad is equivalent to "same".
        conv_padding = "same"

    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=conv_padding,
        groups=groups,
        dilation_rate=dilation,
        use_bias=use_bias,
        data_format=data_format,
        name=f"{name}.conv",
    )(x)

    axis = -1 if data_format == "channels_last" else 1
    x = layers.BatchNormalization(
        axis=axis,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        name=f"{name}.bn",
    )(x)

    activation = act_layer(act, name=f"{name}.act")
    if activation is not None:
        x = activation(x)
    return x


def dw_conv(
    x,
    filters,
    kernel_size=3,
    strides=1,
    dilation=1,
    act=True,
    data_format="channels_last",
    name="dwconv",
):
    """Depth-wise separable style convolution (``groups == in_channels``).

    Used by the lightweight variants (e.g. YOLOv10 ``SCDown`` and several nano
    configs). Requires ``in_channels == filters`` for a true depth-wise conv;
    Keras handles the general grouped case via ``groups``.
    """
    c1 = channels_of(x, data_format)
    groups = int(c1)  # depth-wise: one group per input channel
    return conv_bn(
        x,
        filters,
        kernel_size=kernel_size,
        strides=strides,
        groups=groups,
        dilation=dilation,
        act=act,
        data_format=data_format,
        name=name,
    )
