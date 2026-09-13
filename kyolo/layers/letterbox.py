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
    ``padding`` is ``(dw_half, dh_half)``.
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
        cur_h = ops.shape(inputs)[1]
        cur_w = ops.shape(inputs)[2]

        target_h, target_w = self.new_shape
        r_h = ops.cast(target_h, "float32") / ops.cast(cur_h, "float32")
        r_w = ops.cast(target_w, "float32") / ops.cast(cur_w, "float32")
        r = ops.minimum(r_h, r_w)
        if not self.scaleup:
            r = ops.minimum(r, 1.0)

        new_w = ops.cast(ops.round(ops.cast(cur_w, "float32") * r), "int32")
        new_h = ops.cast(ops.round(ops.cast(cur_h, "float32") * r), "int32")

        dw = target_w - new_w
        dh = target_h - new_h
        if self.auto:
            dw = dw % self.stride
            dh = dh % self.stride
        elif self.scale_fill:
            dw, dh = 0, 0
            new_w, new_h = target_w, target_h

        dw_half = ops.cast(dw, "float32") / 2.0
        dh_half = ops.cast(dh, "float32") / 2.0

        # Letterbox always works on channels_last (H, W, C) tensors; pin the
        # resize data_format so it ignores a channels_first global Keras config.
        resized = ops.image.resize(
            inputs,
            size=[new_h, new_w],
            interpolation="bilinear",
            antialias=False,
            data_format="channels_last",
        )

        top = ops.maximum(ops.cast(ops.round(dh_half - 0.1), "int32"), 0)
        bottom = ops.maximum(ops.cast(ops.round(dh_half + 0.1), "int32"), 0)
        left = ops.maximum(ops.cast(ops.round(dw_half - 0.1), "int32"), 0)
        right = ops.maximum(ops.cast(ops.round(dw_half + 0.1), "int32"), 0)

        paddings = ops.convert_to_tensor(
            [[0, 0], [top, bottom], [left, right], [0, 0]], dtype="int32"
        )
        padded = ops.pad(resized, paddings, mode="constant", constant_values=0.0)
        final = self._fill_border(padded, top, bottom, left, right)

        ratio = ops.broadcast_to(ops.expand_dims(ops.stack([r, r], 0), 0), [batch, 2])
        pad = ops.broadcast_to(ops.expand_dims(ops.stack([dw_half, dh_half], 0), 0), [batch, 2])

        if single:
            final = ops.squeeze(final, axis=0)
            ratio = ops.squeeze(ratio, axis=0)
            pad = ops.squeeze(pad, axis=0)
        return final, ratio, pad

    def _fill_border(self, padded, top, bottom, left, right):
        b, h, w, c = (
            ops.shape(padded)[0],
            ops.shape(padded)[1],
            ops.shape(padded)[2],
            ops.shape(padded)[3],
        )
        ys = ops.arange(h, dtype="int32")
        xs = ops.arange(w, dtype="int32")
        top_m = ops.expand_dims(ys < top, 1)
        bot_m = ops.expand_dims(ys >= (h - bottom), 1)
        left_m = ops.expand_dims(xs < left, 0)
        right_m = ops.expand_dims(xs >= (w - right), 0)
        border = ops.logical_or(
            ops.logical_or(top_m, bot_m), ops.logical_or(left_m, right_m)
        )  # (h, w)
        border = ops.reshape(border, (1, h, w, 1))
        border = ops.broadcast_to(border, [b, h, w, c])
        color = ops.broadcast_to(ops.reshape(self.color_norm, [1, 1, 1, 3]), [b, h, w, 3])
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
