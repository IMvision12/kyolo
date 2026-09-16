"""Shared plumbing for kyolo's detection-augmentation layers.

Every augmentation in :mod:`kyolo.augmentation` is a ``keras.layers.Layer`` that
takes and returns the same *sample dict* the training loop already speaks::

    {
        "images": (B, H, W, 3) float   # channels_first is transposed internally
        "boxes":  (B, M, 4)    float   # xyxy, in pixels, matching "images"
        "labels": (B, M)       int
        "mask":   (B, M)       float   # 1 for a real box, 0 for padding
    }

That is the format :class:`kyolo.training.YOLODetector` consumes, so an
augmentation pipeline drops straight into a data pipeline with nothing to
translate.

Two conventions make the layers composable and keep them traceable on all three
backends:

* **Fixed-size, masked boxes.** Boxes always live in a padded ``(B, M, 4)``
  tensor and validity is carried by ``mask`` rather than by a ragged length.
  Augmentations that drop boxes zero the mask; augmentations that combine images
  (mosaic, mixup) grow ``M`` and can be trimmed back with ``max_boxes``.
* **Vectorized randomness.** Per-sample decisions are drawn as tensors and
  applied with ``ops.where`` over the whole batch instead of branching in
  Python, so a layer behaves identically eagerly, inside ``jit``, and under
  ``tf.data``. Randomness comes from a ``keras.random.SeedGenerator``, which
  makes a seeded pipeline reproducible.

Images are assumed to be in the same units as ``pad_value`` (the default
``114/255`` matches YOLO's grey pad on ``[0, 1]`` images; pass ``pad_value=114``
for ``0-255`` images).
"""

from __future__ import annotations

import keras
from keras import ops

from ..layers.common import resolve_data_format

__all__ = ["DetectionAugmentation", "DEFAULT_PAD_VALUE"]

DEFAULT_PAD_VALUE = 114.0 / 255.0

SAMPLE_KEYS = ("images", "boxes", "labels", "mask")


class DetectionAugmentation(keras.layers.Layer):
    """Base class for detection augmentations.

    Subclasses implement :meth:`augment`, which receives (and returns) a sample
    dict whose ``images`` are channels-last. The base class handles data-format
    conversion, dtype normalization, the ``training`` passthrough and the
    per-sample probability draw.

    Args:
        prob: per-sample probability of applying the augmentation. Layers that
            are always applied when augmenting (``RandomPerspective``) ignore it.
        pad_value: scalar fill for pixels with no source content, in the images'
            own units. Defaults to YOLO's grey ``114/255``.
        max_boxes: trim the output box list to this many boxes, keeping the
            valid ones. Only meaningful for layers that grow the box count
            (mosaic, mixup, copy-paste); ``None`` leaves the count alone.
        data_format: layout of ``images``; ``None`` uses the Keras default.
        seed: seed for this layer's :class:`keras.random.SeedGenerator`.
    """

    def __init__(
        self,
        prob=1.0,
        pad_value=DEFAULT_PAD_VALUE,
        max_boxes=None,
        data_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1]; got {prob}.")
        if max_boxes is not None and max_boxes < 1:
            raise ValueError(f"max_boxes must be a positive int or None; got {max_boxes}.")
        self.prob = float(prob)
        self.pad_value = float(pad_value)
        self.max_boxes = max_boxes
        self.data_format = resolve_data_format(data_format)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def build(self, input_shape=None):
        """Nothing shape-dependent to create; the only variable is the RNG seed."""
        self.built = True

    def augment(self, sample):
        """Augment a channels-last sample dict. Implemented by subclasses."""
        raise NotImplementedError

    def call(self, inputs, training=True):
        if not training:
            return dict(inputs)

        sample, input_dtype = self.unpack(inputs)
        sample = self.augment(sample)
        if self.max_boxes is not None:
            sample = self.trim(sample, self.max_boxes)
        return self.pack(sample, input_dtype)

    def unpack(self, inputs):
        """Validate the sample dict and move it to channels-last float32.

        Returns the sample and the incoming image dtype. The dtype is returned
        rather than stored on the layer so that a single pipeline can be shared
        by several loader threads (``PyDataset(workers=...)``) without them
        overwriting each other's state.
        """
        if not isinstance(inputs, dict):
            raise TypeError(
                "detection augmentations take a dict with keys "
                f"{SAMPLE_KEYS}; got {type(inputs).__name__}. Wrap a bare image "
                'tensor as {"images": images, "boxes": ..., "labels": ..., "mask": ...}.'
            )
        missing = [key for key in SAMPLE_KEYS if key not in inputs]
        if missing:
            raise ValueError(
                f"sample dict is missing {missing}; augmentations need all of {SAMPLE_KEYS}."
            )

        images = ops.convert_to_tensor(inputs["images"])
        input_dtype = keras.backend.standardize_dtype(images.dtype)
        images = ops.cast(images, "float32")
        if len(images.shape) != 4:
            raise ValueError(
                "augmentations are batched: `images` must be rank 4 "
                f"(B, H, W, C), got shape {tuple(images.shape)}."
            )
        if self.data_format == "channels_first":
            images = ops.transpose(images, (0, 2, 3, 1))

        sample = {
            "images": images,
            "boxes": ops.cast(ops.convert_to_tensor(inputs["boxes"]), "float32"),
            "labels": ops.cast(ops.convert_to_tensor(inputs["labels"]), "int32"),
            "mask": ops.cast(ops.convert_to_tensor(inputs["mask"]), "float32"),
        }
        extras = {k: v for k, v in inputs.items() if k not in SAMPLE_KEYS}
        sample.update(extras)
        return sample, input_dtype

    def pack(self, sample, input_dtype):
        """Restore the caller's data format and image dtype."""
        images = sample["images"]
        if self.data_format == "channels_first":
            images = ops.transpose(images, (0, 3, 1, 2))
        out = dict(sample)
        out["images"] = ops.cast(images, input_dtype)
        return out

    @staticmethod
    def static_hw(images):
        """Return the statically-known ``(height, width)`` of a batch.

        Augmentations resize, crop and pad by computed amounts, so -- like
        :class:`kyolo.layers.Letterbox` -- they need the spatial dims to be known
        at trace time. The batch dimension may stay dynamic.
        """
        height, width = images.shape[1], images.shape[2]
        if height is None or width is None:
            raise ValueError(
                "augmentations need statically-known spatial dimensions because "
                "they crop and pad by computed amounts, but got images of shape "
                f"{tuple(images.shape)}. Letterbox to a fixed size first. The "
                "batch dimension may stay dynamic."
            )
        return int(height), int(width)

    def uniform(self, shape, minval=0.0, maxval=1.0):
        """Draw ``U(minval, maxval)`` from this layer's seed generator."""
        return keras.random.uniform(
            shape, minval=minval, maxval=maxval, dtype="float32", seed=self.seed_generator
        )

    def should_apply(self, batch, rank=1):
        """Per-sample ``{0., 1.}`` gate at rate ``prob``, shaped for broadcasting.

        ``rank`` is the rank to broadcast against: ``1`` gives ``(B,)``, ``3``
        gives ``(B, 1, 1)`` for box tensors, ``4`` gives ``(B, 1, 1, 1)`` for
        images.
        """
        draw = self.uniform((batch,)) < self.prob
        gate = ops.cast(draw, "float32")
        return ops.reshape(gate, (-1,) + (1,) * (rank - 1))

    @staticmethod
    def partner_indices(batch, offset=1):
        """Indices pairing each sample with another one in the same batch.

        Mixing augmentations (mosaic, mixup, copy-paste) need a second image.
        Ultralytics draws it from the whole dataset; drawing it from the current
        batch instead keeps the layer a pure tensor op -- no dataset handle, no
        extra file reads -- which is what lets these run batched on device. A
        fixed roll (rather than a random permutation) guarantees a sample is
        never mixed with itself, which a permutation cannot promise.
        """
        return ops.mod(ops.arange(batch, dtype="int32") + offset, batch)

    @staticmethod
    def gather_samples(sample, indices):
        """Reindex a sample dict along the batch axis."""
        out = dict(sample)
        for key in SAMPLE_KEYS:
            out[key] = ops.take(sample[key], indices, axis=0)
        return out

    @staticmethod
    def concat_boxes(samples):
        """Concatenate the box/label/mask lists of several samples per image."""
        return {
            "boxes": ops.concatenate([s["boxes"] for s in samples], axis=1),
            "labels": ops.concatenate([s["labels"] for s in samples], axis=1),
            "mask": ops.concatenate([s["mask"] for s in samples], axis=1),
        }

    @staticmethod
    def trim(sample, max_boxes):
        """Keep at most ``max_boxes`` boxes per image, valid ones first.

        Combining augmentations multiply the box count (four-fold for mosaic),
        and most of the result is padding. Sorting by validity and slicing keeps
        the tensor small without a ragged gather: the sort key is
        ``-mask * (M + 1) + position``, which orders every valid box (by
        original position) ahead of every padded one, deterministically.
        """
        boxes = ops.convert_to_tensor(sample["boxes"])
        count = boxes.shape[1]
        if count is None:
            raise ValueError("trimming needs a statically-known box count.")
        if count <= max_boxes:
            return sample

        labels = ops.convert_to_tensor(sample["labels"])
        mask = ops.cast(ops.convert_to_tensor(sample["mask"]), "float32")

        positions = ops.cast(ops.arange(count), "float32")
        key = -mask * float(count + 1) + positions
        order = ops.argsort(key, axis=-1)[:, :max_boxes]

        out = dict(sample)
        out["boxes"] = ops.take_along_axis(boxes, ops.expand_dims(order, -1), axis=1)
        out["labels"] = ops.take_along_axis(labels, order, axis=1)
        out["mask"] = ops.take_along_axis(mask, order, axis=1)
        return out

    @staticmethod
    def clip_and_filter(boxes, mask, height, width, min_side=2.0):
        """Clip boxes to the image and drop the ones that survive too small.

        Returns ``(boxes, mask)``. Degenerate boxes keep their coordinates but
        lose their mask entry, so no downstream op has to cope with a negative
        width.
        """
        upper = ops.convert_to_tensor([[[width, height, width, height]]], dtype="float32")
        boxes = ops.clip(boxes, 0.0, upper)
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        keep = ops.logical_and(widths > min_side, heights > min_side)
        return boxes, mask * ops.cast(keep, "float32")

    def compute_output_shape(self, input_shape):
        return dict(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "prob": self.prob,
                "pad_value": self.pad_value,
                "max_boxes": self.max_boxes,
                "data_format": self.data_format,
                "seed": self.seed,
            }
        )
        return config
