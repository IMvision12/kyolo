"""Composition of detection augmentations, in Ultralytics' order."""

from __future__ import annotations

import keras

from .base import DEFAULT_PAD_VALUE, DetectionAugmentation
from .geometric import RandomFlip, RandomPerspective
from .mix import CopyPaste, MixUp
from .mosaic import Mosaic
from .photometric import RandomHSV

__all__ = ["AugmentationPipeline", "MIXING_LAYERS"]

MIXING_LAYERS = (Mosaic, MixUp, CopyPaste)


@keras.saving.register_keras_serializable(package="kyolo")
class AugmentationPipeline(keras.layers.Layer):
    """Apply a list of detection augmentations in order.

    Build one with :meth:`from_hyperparameters` to get Ultralytics' YOLOv8+
    recipe and ordering::

        Mosaic -> CopyPaste -> RandomPerspective -> MixUp
               -> RandomHSV -> RandomFlip(vertical) -> RandomFlip(horizontal)

    The order matters: the image-combining stages run first on an oversized
    canvas, and ``RandomPerspective`` is what crops back to the training
    resolution. Photometric jitter and flips come last, on the final frame.

    Args:
        layers: the augmentations to apply, in order.
        max_boxes: trim the final box list to this many boxes, keeping the valid
            ones. Mosaic and mixup multiply the box count, so this is how the
            pipeline hands the loss a bounded tensor. ``None`` keeps them all.
    """

    def __init__(self, layers, max_boxes=None, **kwargs):
        super().__init__(**kwargs)
        self.augmentations = list(layers)
        self.max_boxes = max_boxes
        self._mosaic_closed = False

    @classmethod
    def from_hyperparameters(
        cls,
        image_size=640,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        max_boxes=None,
        pad_value=DEFAULT_PAD_VALUE,
        value_range=(0.0, 1.0),
        data_format=None,
        seed=None,
        **kwargs,
    ):
        """Build the standard pipeline from Ultralytics' hyperparameter names.

        The defaults are Ultralytics' ``default.yaml`` values, so
        ``AugmentationPipeline.from_hyperparameters(image_size=640)`` reproduces
        the stock YOLOv8-through-YOLO26 training augmentation. Setting a
        probability to ``0`` leaves that stage out entirely rather than running
        it as a no-op.

        Args:
            image_size: the resolution training runs at; the pipeline always
                returns images of this size.
            value_range: range of the incoming pixel values, needed by the HSV
                jitter. ``(0, 1)`` matches
                :class:`kyolo.preprocessing.YOLOPreprocessor`.
            seed: base seed; each stage derives its own from it, so the whole
                pipeline is reproducible.
        """
        counter = iter(range(1000))

        def next_seed():
            return None if seed is None else seed + next(counter)

        shared = {"pad_value": pad_value, "data_format": data_format}
        layers = []
        if mosaic > 0:
            layers.append(Mosaic(prob=mosaic, seed=next_seed(), **shared))
        if copy_paste > 0:
            layers.append(
                CopyPaste(prob=copy_paste, mode=copy_paste_mode, seed=next_seed(), **shared)
            )
        layers.append(
            RandomPerspective(
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                output_size=image_size,
                seed=next_seed(),
                **shared,
            )
        )
        if mixup > 0:
            layers.append(MixUp(prob=mixup, seed=next_seed(), **shared))
        if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
            layers.append(
                RandomHSV(
                    hgain=hsv_h,
                    sgain=hsv_s,
                    vgain=hsv_v,
                    value_range=value_range,
                    seed=next_seed(),
                    **shared,
                )
            )
        if flipud > 0:
            layers.append(RandomFlip(direction="vertical", prob=flipud, seed=next_seed(), **shared))
        if fliplr > 0:
            layers.append(
                RandomFlip(direction="horizontal", prob=fliplr, seed=next_seed(), **shared)
            )
        return cls(layers, max_boxes=max_boxes, **kwargs)

    def build(self, input_shape=None):
        """Build the stages; none of them hold shape-dependent state."""
        for layer in self.augmentations:
            layer.build(input_shape)
        self.built = True

    @property
    def mosaic_closed(self):
        """Whether the image-combining stages are currently switched off."""
        return self._mosaic_closed

    def close_mosaic(self):
        """Stop combining images, as Ultralytics does for the final epochs.

        Skips mosaic, mixup and copy-paste so the last stretch of training sees
        only real, un-stitched images. ``RandomPerspective`` then receives
        images already at the training resolution and, because it is configured
        by output size rather than by a crop border, needs no adjustment.

        Intended to be driven by
        :class:`kyolo.training.CloseMosaic`. Because the stages are skipped in
        Python, call this from a callback between epochs rather than from inside
        a compiled step.
        """
        self._mosaic_closed = True

    def open_mosaic(self):
        """Re-enable the image-combining stages."""
        self._mosaic_closed = False

    def active(self):
        if not self._mosaic_closed:
            return self.augmentations
        return [layer for layer in self.augmentations if not isinstance(layer, MIXING_LAYERS)]

    def call(self, inputs, training=True):
        if not training:
            return dict(inputs)

        sample = inputs
        for layer in self.active():
            sample = layer(sample, training=training)
        if self.max_boxes is not None:
            sample = DetectionAugmentation.trim(sample, self.max_boxes)
        return sample

    def compute_output_shape(self, input_shape):
        shape = input_shape
        for layer in self.active():
            shape = layer.compute_output_shape(shape)
        if self.max_boxes is not None:
            shape = dict(shape)
            for key in ("boxes", "labels", "mask"):
                dims = list(shape[key])
                if dims[1] is not None:
                    dims[1] = min(dims[1], self.max_boxes)
                shape[key] = tuple(dims)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layers": [keras.saving.serialize_keras_object(a) for a in self.augmentations],
                "max_boxes": self.max_boxes,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["layers"] = [keras.saving.deserialize_keras_object(a) for a in config["layers"]]
        return cls(**config)
