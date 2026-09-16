"""Image preprocessing for kyolo detectors (letterbox + normalize)."""

from __future__ import annotations

from typing import List, Union

import keras
from keras import ops

from ..layers.common import resolve_data_format
from ..layers.letterbox import Letterbox

__all__ = ["YOLOPreprocessor"]

_INPUT_RANGES = ((0, 1), (0, 255))


@keras.saving.register_keras_serializable(package="kyolo")
class YOLOPreprocessor(keras.layers.Layer):
    """Resize (letterbox) + scale + optional mean/std normalize.

    Calling the layer on an image (``(H, W, C)``) or a batch (``(B, H, W, C)``)
    returns a dict::

        {"images": (B, S, S, 3), "ratio": (B, 2), "pad": (B, 2)}

    ``images`` is ``(B, S, S, 3)`` for channels_last or ``(B, 3, S, S)`` for
    channels_first. ``ratio`` and the applied ``(left, top)`` ``pad`` map
    detections back to original coordinates -- ``x_orig = (x - pad_x) / ratio``
    -- which is what :func:`kyolo.ops.scale_boxes` does::

        batch = preprocessor(image)
        detections = postprocessor(model(batch["images"]))
        detections = scale_boxes(
            detections, batch["ratio"], batch["pad"], image.shape[:2]
        )

    Note that ``ratio``/``pad`` only describe an invertible transform when
    ``letterbox=True`` and ``scale_fill=False``; a plain resize
    (``letterbox=False``) still reports an identity ratio.

    Args:
        image_size: square target side ``S``.
        normalize: if ``True`` (default) pixel values are scaled to ``[0, 1]``
            (see ``input_range``) and then, when given, ``mean``/``std`` are
            applied. If ``False`` pixel values are passed through untouched:
            no scaling, no mean/std, and ``pad_color`` is used as-is.
        input_range: value range of the *input* pixels, used only when
            ``normalize=True``. ``(0, 255)`` always divides by 255, ``(0, 1)``
            never does. The default ``None`` auto-detects per image: integer
            inputs (e.g. ``uint8``) are 0-255, and a float image is treated as
            0-255 when its own maximum exceeds 1. Images in a batch never
            influence each other; pass an explicit range to avoid the
            heuristic altogether (e.g. for very dark 0-255 float images).
        letterbox: aspect-preserving resize + pad; if ``False`` a plain resize.
        auto: minimal-rectangle padding to a multiple of ``stride`` (rarely used
            for batched inference - keep ``False`` for a fixed square).
        stride: stride for ``auto`` padding.
        pad_color: RGB pad colour in 0-255 units, e.g. YOLO's ``(114, 114, 114)``.
            It is scaled to ``[0, 1]`` together with the image when
            ``normalize=True`` and used verbatim otherwise.
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
        input_range=None,
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
        if input_range is not None:
            input_range = tuple(int(v) for v in input_range)
            if input_range not in _INPUT_RANGES:
                raise ValueError(
                    f"input_range must be None, (0, 1) or (0, 255); got {input_range!r}."
                )
        self.image_size = image_size
        self.normalize = normalize
        self.input_range = input_range
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
            # The pad colour must be in the same units as the image it fills:
            # /255 when the image has been scaled to [0, 1], raw otherwise.
            self.lb = Letterbox(
                new_shape=image_size,
                color=pad_color,
                color_divisor=255.0 if normalize else 1.0,
                auto=auto,
                stride=stride,
                scaleup=True,
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

    def _to_unit_range(self, x, integer_input):
        """Scale a float ``(B, H, W, C)`` batch to ``[0, 1]`` per ``input_range``."""
        if self.input_range == (0, 1):
            return x
        if self.input_range == (0, 255) or integer_input:
            return x / 255.0
        # Auto-detect for float inputs, judging every image on its own maximum
        # so a 0-255 image and a [0, 1] image in the same batch are both right.
        per_image_max = ops.max(x, axis=(1, 2, 3), keepdims=True)
        return ops.where(per_image_max > 1.0, x / 255.0, x)

    def call(self, inputs):
        x = ops.convert_to_tensor(inputs)
        dtype = keras.backend.standardize_dtype(x.dtype)
        integer_input = dtype.startswith(("int", "uint")) or dtype == "bool"
        x = ops.cast(x, "float32")
        single = len(x.shape) == 3
        if single:
            x = ops.expand_dims(x, 0)
        x = self._fix_channels(x)

        if self.normalize:
            x = self._to_unit_range(x, integer_input)

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

    def compute_output_shape(self, input_shape):
        """Declared so symbolic / functional use never has to trace ``call``."""
        # Inputs are accepted as (H, W, C) or (B, H, W, C); the output is always
        # batched, so a single image becomes B=1.
        batch = input_shape[0] if len(input_shape) == 4 else 1
        if self.auto:
            # Minimal-rectangle padding: the spatial size depends on the input's
            # aspect ratio, so it is not a fixed square.
            height = width = None
        else:
            height = width = self.image_size
        if self.data_format == "channels_first":
            images = (batch, 3, height, width)
        else:
            images = (batch, height, width, 3)
        return {"images": images, "ratio": (batch, 2), "pad": (batch, 2)}

    def from_files(self, paths: Union[str, List[str]]):
        """Load image file(s) with keras and preprocess them."""
        if isinstance(paths, str):
            paths = [paths]
        imgs = []
        for p in paths:
            img = keras.utils.load_img(p)
            # uint8 so the 0-255 range is known, not guessed
            imgs.append(keras.utils.img_to_array(img, dtype="uint8"))
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
                "input_range": self.input_range,
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
