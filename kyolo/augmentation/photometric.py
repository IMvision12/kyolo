"""Colour-space augmentation. Geometry (and therefore boxes) is untouched."""

from __future__ import annotations

import keras
from keras import ops

from .base import DetectionAugmentation

__all__ = ["RandomHSV"]


@keras.saving.register_keras_serializable(package="kyolo")
class RandomHSV(DetectionAugmentation):
    """Randomly scale hue, saturation and value, per image.

    Mirrors Ultralytics' ``RandomHSV``: a gain is drawn per channel as
    ``1 + U(-1, 1) * gain``, hue *wraps* while saturation and value *clip*.
    Multiplying (rather than shifting) the hue is what Ultralytics does, and it
    is deliberately gentle near hue ``0``.

    Boxes, labels and mask pass through unchanged.

    Args:
        hgain / sgain / vgain: maximum fractional change of hue / saturation /
            value. Ultralytics' defaults are ``0.015 / 0.7 / 0.4``.
        value_range: the ``(low, high)`` range of the incoming pixels. The
            conversion to HSV needs unit-range inputs, so anything else is
            rescaled and restored around it. Use ``(0, 255)`` for raw images.
        prob: per-sample probability of applying the jitter.
    """

    def __init__(
        self,
        hgain=0.015,
        sgain=0.7,
        vgain=0.4,
        value_range=(0.0, 1.0),
        prob=1.0,
        **kwargs,
    ):
        super().__init__(prob=prob, **kwargs)
        for name, gain in (("hgain", hgain), ("sgain", sgain), ("vgain", vgain)):
            if gain < 0:
                raise ValueError(f"{name} must be non-negative; got {gain}.")
        low, high = (float(v) for v in value_range)
        if high <= low:
            raise ValueError(f"value_range must be increasing; got {value_range!r}.")
        self.hgain = float(hgain)
        self.sgain = float(sgain)
        self.vgain = float(vgain)
        self.value_range = (low, high)

    def augment(self, sample):
        if self.hgain == 0.0 and self.sgain == 0.0 and self.vgain == 0.0:
            return sample

        images = sample["images"]
        batch = ops.shape(images)[0]
        low, high = self.value_range
        scale = high - low

        unit = ops.clip((images - low) / scale, 0.0, 1.0)

        gains = ops.convert_to_tensor([[self.hgain, self.sgain, self.vgain]], dtype="float32")
        gains = 1.0 + self.uniform((batch, 3), -1.0, 1.0) * gains
        gains = ops.reshape(gains, (-1, 1, 1, 3))
        gate = self.should_apply(batch, rank=4)
        gains = 1.0 + (gains - 1.0) * gate

        hsv = ops.image.rgb_to_hsv(unit) * gains
        hue = ops.mod(hsv[..., :1], 1.0)
        sat_val = ops.clip(hsv[..., 1:], 0.0, 1.0)
        jittered = ops.image.hsv_to_rgb(ops.concatenate([hue, sat_val], axis=-1))

        out = dict(sample)
        out["images"] = ops.clip(jittered, 0.0, 1.0) * scale + low
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hgain": self.hgain,
                "sgain": self.sgain,
                "vgain": self.vgain,
                "value_range": self.value_range,
            }
        )
        return config
