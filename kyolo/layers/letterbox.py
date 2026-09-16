from __future__ import annotations

import keras
from keras import ops

__all__ = ["Letterbox"]


@keras.saving.register_keras_serializable(package="kyolo")
class Letterbox(keras.layers.Layer):
    """Resize while preserving aspect ratio, padding the remainder with ``color``.

    Args:
        new_shape: Target ``(height, width)`` or a single int for a square.
        color: RGB pad colour in ``[0, 255]`` (YOLO uses ``(114, 114, 114)``).
        color_divisor: ``color`` is divided by this before filling. The default
            ``255`` matches images already scaled to ``[0, 1]``; pass ``1.0``
            when the images being padded are themselves in ``0-255`` units.
        auto: If ``True`` pad to the nearest multiple of ``stride`` (minimal
            rectangle) instead of the full square.
        scale_fill: Stretch to ``new_shape`` ignoring aspect ratio.
        scaleup: Allow upscaling (``False`` only ever shrinks).
        stride: Stride used when ``auto`` is ``True``.

    Call returns ``(image, ratio, padding)`` where ``ratio`` is ``(r, r)`` and
    ``padding`` is the ``(left, top)`` padding actually applied, in pixels.
    Together they invert the transform: ``x_orig = (x_letterboxed - left) / r``.
    See :func:`kyolo.ops.scale_boxes`.
    """

    def __init__(
        self,
        new_shape=(640, 640),
        color=(114, 114, 114),
        color_divisor=255.0,
        auto=False,
        scale_fill=False,
        scaleup=True,
        stride=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        self.new_shape = tuple(new_shape)
        self.color = tuple(color)
        self.color_divisor = float(color_divisor)
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.color_norm = ops.convert_to_tensor(
            [c / self.color_divisor for c in self.color], dtype="float32"
        )

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype="float32")
        single = len(inputs.shape) == 3
        if single:
            inputs = ops.expand_dims(inputs, axis=0)

        batch = ops.shape(inputs)[0]

        cur_h, cur_w = inputs.shape[1], inputs.shape[2]
        if cur_h is None or cur_w is None:
            raise ValueError(
                "Letterbox needs statically-known spatial dimensions because it "
                f"resizes to a computed size, but got shape {tuple(inputs.shape)}. "
                "Resize the images to a known size first, or build with a concrete "
                "input shape. The batch dimension may stay dynamic."
            )

        target_h, target_w = self.new_shape

        r = min(target_h / cur_h, target_w / cur_w)
        if not self.scaleup:
            r = min(r, 1.0)

        new_w = int(round(cur_w * r))
        new_h = int(round(cur_h * r))

        dw = target_w - new_w
        dh = target_h - new_h
        if self.auto:
            dw = dw % self.stride
            dh = dh % self.stride
        elif self.scale_fill:
            dw, dh = 0, 0
            new_w, new_h = target_w, target_h

        dw_half = dw / 2.0
        dh_half = dh / 2.0

        resized = ops.image.resize(
            inputs,
            size=[new_h, new_w],
            interpolation="bilinear",
            antialias=False,
            data_format="channels_last",
        )

        top = max(int(round(dh_half - 0.1)), 0)
        bottom = max(int(round(dh_half + 0.1)), 0)
        left = max(int(round(dw_half - 0.1)), 0)
        right = max(int(round(dw_half + 0.1)), 0)

        padded = ops.pad(
            resized,
            [[0, 0], [top, bottom], [left, right], [0, 0]],
            mode="constant",
            constant_values=0.0,
        )
        final = self.fill_border(padded, top, bottom, left, right)

        ratio = ops.broadcast_to(ops.convert_to_tensor([[r, r]], dtype="float32"), (batch, 2))
        pad = ops.broadcast_to(
            ops.convert_to_tensor([[float(left), float(top)]], dtype="float32"), (batch, 2)
        )

        if single:
            final = ops.squeeze(final, axis=0)
            ratio = ops.squeeze(ratio, axis=0)
            pad = ops.squeeze(pad, axis=0)
        return final, ratio, pad

    def fill_border(self, padded, top, bottom, left, right):

        h, w = padded.shape[1], padded.shape[2]
        ys = ops.arange(h, dtype="int32")
        xs = ops.arange(w, dtype="int32")
        top_m = ops.expand_dims(ys < top, 1)
        bot_m = ops.expand_dims(ys >= (h - bottom), 1)
        left_m = ops.expand_dims(xs < left, 0)
        right_m = ops.expand_dims(xs >= (w - right), 0)
        border = ops.logical_or(ops.logical_or(top_m, bot_m), ops.logical_or(left_m, right_m))
        border = ops.reshape(border, (1, h, w, 1))
        color = ops.reshape(self.color_norm, (1, 1, 1, 3))
        return ops.where(border, color, padded)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            img = (input_shape[0], self.new_shape[0], self.new_shape[1], input_shape[-1])
            return [img, (input_shape[0], 2), (input_shape[0], 2)]
        img = (self.new_shape[0], self.new_shape[1], input_shape[-1])
        return [img, (2,), (2,)]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "new_shape": self.new_shape,
                "color": self.color,
                "color_divisor": self.color_divisor,
                "auto": self.auto,
                "scale_fill": self.scale_fill,
                "scaleup": self.scaleup,
                "stride": self.stride,
            }
        )
        return config
