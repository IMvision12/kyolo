"""Image preprocessing for kyolo detectors (letterbox + normalize)."""

from __future__ import annotations

from typing import List, Union

import keras
from keras import ops

from ..layers.common import resolve_data_format
from ..layers.letterbox import Letterbox

__all__ = ["YOLOPreprocessor"]


@keras.saving.register_keras_serializable(package="kyolo")
class YOLOPreprocessor(keras.layers.Layer):
    """Resize (letterbox) + scale + optional mean/std normalize.

    Calling the layer on an image (``(H, W, C)``) or a batch (``(B, H, W, C)``)
    returns a dict::

        {"images": (B, S, S, 3), "ratio": (B, 2), "pad": (B, 2)}

    ``images`` is ``(B, S, S, 3)`` for channels_last or ``(B, 3, S, S)`` for
    channels_first. ``ratio`` and ``pad`` let you map detections back to
    original coordinates: ``x_orig = (x_letterboxed - pad_x) / ratio``.

    Args:
        image_size: square target side ``S``.
        normalize: divide by 255 (auto-detected) and apply ``mean``/``std``.
        letterbox: aspect-preserving resize + pad; if ``False`` a plain resize.
        auto: minimal-rectangle padding to a multiple of ``stride`` (rarely used
            for batched inference - keep ``False`` for a fixed square).
        stride: stride for ``auto`` padding.
        pad_color: RGB pad colour (0-255).
        mean / std: optional per-channel normalization (in [0, 1] scale).
        data_format: layout of the returned ``images`` ("channels_last",
            "channels_first", or None for the global Keras config). Inputs are
            always accepted as channels_last (H, W, C); only the output layout
            changes.
    """

    def __init__(
        self,
        image_size: int = 640,
        normalize: bool = True,
        letterbox: bool = True,
        auto: bool = False,
        stride: int = 32,
        pad_color=(114, 114, 114),
        mean=None,
        std=None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.normalize = normalize
        self.do_letterbox = letterbox
        self.auto = auto
        self.stride = stride
        self.pad_color = tuple(pad_color)
        self.mean = mean
        self.std = std
        self.data_format = resolve_data_format(data_format)
        self._mean = ops.convert_to_tensor(mean if mean is not None else [0.0, 0.0, 0.0], "float32")
        self._std = ops.convert_to_tensor(std if std is not None else [1.0, 1.0, 1.0], "float32")
        if self.do_letterbox:
            self.lb = Letterbox(
                new_shape=image_size, color=pad_color, auto=auto, stride=stride, scaleup=True
            )

    def _fix_channels(self, x):
        c = x.shape[-1]
        if c == 1:
            x = ops.repeat(x, 3, axis=-1)
        elif c == 4:
            x = x[..., :3]
        elif c not in (3, None):
            raise ValueError(f"unsupported channel count: {c}")
        return x

    def call(self, inputs):
        x = ops.convert_to_tensor(inputs)
        x = ops.cast(x, "float32")
        single = len(x.shape) == 3
        if single:
            x = ops.expand_dims(x, 0)
        x = self._fix_channels(x)

        # scale to [0, 1] if it looks like [0, 255]
        x = ops.where(ops.max(x) > 1.0, x / 255.0, x)

        if self.do_letterbox:
            x, ratio, pad = self.lb(x)
        else:
            b = ops.shape(x)[0]
            # Process in channels_last regardless of the global config; the
            # output is transposed to self.data_format at the end of call().
            x = ops.image.resize(
                x,
                size=[self.image_size, self.image_size],
                interpolation="bilinear",
                data_format="channels_last",
            )
            ratio = ops.ones((b, 2), dtype="float32")
            pad = ops.zeros((b, 2), dtype="float32")

        if self.normalize and (self.mean is not None or self.std is not None):
            x = (x - self._mean) / self._std

        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)

        return {"images": x, "ratio": ratio, "pad": pad}

    def from_files(self, paths: Union[str, List[str]]):
        """Load image file(s) with keras and preprocess them."""
        if isinstance(paths, str):
            paths = [paths]
        imgs = []
        for p in paths:
            img = keras.utils.load_img(p)
            imgs.append(keras.utils.img_to_array(img))
        # letterbox handles differing sizes -> stack after resize by looping
        batch = [self.call(img) for img in imgs]
        images = ops.concatenate([b["images"] for b in batch], axis=0)
        ratio = ops.concatenate([b["ratio"] for b in batch], axis=0)
        pad = ops.concatenate([b["pad"] for b in batch], axis=0)
        return {"images": images, "ratio": ratio, "pad": pad}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "normalize": self.normalize,
                "letterbox": self.do_letterbox,
                "auto": self.auto,
                "stride": self.stride,
                "pad_color": self.pad_color,
                "mean": self.mean,
                "std": self.std,
                "data_format": self.data_format,
            }
        )
        return config
