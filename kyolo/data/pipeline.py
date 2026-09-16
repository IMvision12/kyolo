"""Building a training input pipeline on Grain.

The division of labour is deliberate:

* **Grain**, in worker threads or processes, does the per-sample IO -- decode,
  letterbox, pad the target lists -- in plain numpy. Grain is used rather than
  ``tf.data`` so that a TensorFlow install is not a prerequisite for feeding a
  JAX or PyTorch model; its ``MapDataset`` is random-access, which also makes
  epoch reshuffling and rectangular batching straightforward.
* **Keras**, on the assembled batch, does the randomized augmentation
  (:mod:`kyolo.augmentation`) as vectorized ``keras.ops``, so it runs on the
  same backend and device as training.

The result is a :class:`keras.utils.PyDataset`, which ``model.fit`` consumes
directly and which knows its own length, so ``steps_per_epoch`` is unnecessary.
"""

from __future__ import annotations

import keras
import numpy as np

from .config import DataConfig
from .rect import RectangularPlan
from .source import YOLODataSource
from .transforms import DEFAULT_PAD_VALUE, LetterboxSample, PadTargets, RectangularShapes

__all__ = ["GrainDataLoader", "build_source"]


def require_grain():
    try:
        import grain
    except ImportError as exc:
        raise ImportError(
            "the kyolo data pipeline is built on Grain. Install it with "
            "`pip install kyolo[data]` or `pip install grain`."
        ) from exc
    return grain


def build_source(data, split="train", cache=None, nc=None):
    """Resolve ``data`` into a :class:`~kyolo.data.YOLODataSource`.

    Args:
        data: a :class:`~kyolo.data.DataConfig`, a path to a ``data.yaml``, or an
            already-built ``YOLODataSource`` (returned unchanged).
        split: which split to read when ``data`` is a config.
        cache: image cache mode; see :class:`~kyolo.data.YOLODataSource`.
        nc: number of classes; taken from the config when not given.
    """
    if isinstance(data, YOLODataSource):
        return data
    config = data if isinstance(data, DataConfig) else DataConfig.from_yaml(data)
    return YOLODataSource(
        config.image_paths(split), nc=nc if nc is not None else config.nc, cache=cache
    )


class GrainDataLoader(keras.utils.PyDataset):
    """A Keras-ready detection dataloader backed by Grain.

    ::

        from kyolo.data import GrainDataLoader

        train = GrainDataLoader("coco8.yaml", "train", batch_size=16, augment=True)
        val = GrainDataLoader("coco8.yaml", "val", batch_size=16, rect=True)
        detector.fit(train, validation_data=val, epochs=100)

    Yields ``(x, y)`` batches in the form
    :class:`kyolo.training.YOLODetector` trains on::

        x = {"images": (B, S, S, 3) float32}
        y = {"boxes": (B, M, 4) float32, "labels": (B, M) int32,
             "mask": (B, M) float32}

    ``boxes`` are xyxy in pixels of the letterboxed image and ``mask`` marks the
    real ones.

    Shuffling is re-randomized every epoch by rebuilding the Grain pipeline with
    a new seed in :meth:`on_epoch_end`. That is cheap -- a ``MapDataset`` is a
    lazy index transformation, so nothing is re-read -- and it keeps each epoch
    an exact permutation of the dataset rather than a window onto an infinite
    stream.

    Parallelism comes from :class:`keras.utils.PyDataset`: pass ``workers=8``
    for threaded loading (recommended, since decoding releases the GIL and the
    augmentation stays on one backend), or additionally
    ``use_multiprocessing=True`` to fan the numpy work out across processes.

    Args:
        data: a ``data.yaml`` path, a :class:`~kyolo.data.DataConfig`, or a
            :class:`~kyolo.data.YOLODataSource`.
        split: split to read.
        batch_size: images per batch.
        image_size: letterbox target side, and the resolution augmentation
            returns.
        max_boxes: target slots per image. Images with more objects keep the
            first ``max_boxes``.
        augment: ``None``/``False`` for no augmentation, ``True`` for
            Ultralytics' defaults, a dict of hyperparameter overrides passed to
            :meth:`~kyolo.augmentation.AugmentationPipeline.from_hyperparameters`,
            or a ready-made pipeline.
        shuffle: reshuffle every epoch. Defaults to ``True`` for the ``train``
            split, and to ``False`` otherwise or whenever ``rect`` is set.
        rect: rectangular batching -- one shape per batch instead of squares.
            Faster, but it fixes the order by aspect ratio, so it implies no
            shuffling; asking for both explicitly is an error. Because the shape
            then differs per batch, the model has to be built for variable
            input, e.g. ``yolov8n(nc=nc, input_shape=(None, None, 3))``.
        rect_pad: extra fraction of a stride added to each rectangular batch
            shape before rounding up. Ultralytics' ``0.5`` means the long side
            comes out one stride *above* ``image_size`` (672 for 640); pass
            ``0.0`` to keep every shape at or below ``image_size``.
        rect_stride: stride the rectangular shapes are rounded up to.
        cache: ``None``, ``"ram"`` or ``"disk"``.
        scaleup: let letterboxing upscale images smaller than ``image_size``.
        pad_value: letterbox pad fill, in 0-255 units.
        drop_remainder: drop a final short batch. Defaults to ``True`` when
            shuffling (training) and ``False`` otherwise, so validation covers
            every image.
        seed: base shuffling seed; epoch ``k`` uses ``seed + k``.
        nc: number of classes, for validating label files.
        keep_metadata: forward ``ratio``/``pad``/``orig_shape``/``index`` into
            ``x``, which validation needs to map predictions back to original
            image coordinates.
        value_range: range the output pixels are scaled to from ``0-255``.
        data_format: data format of the yielded images; the Keras default when
            not given.
        **kwargs: forwarded to :class:`keras.utils.PyDataset` (``workers``,
            ``use_multiprocessing``, ``max_queue_size``).
    """

    def __init__(
        self,
        data,
        split="train",
        batch_size=16,
        image_size=640,
        max_boxes=100,
        augment=None,
        shuffle=None,
        rect=False,
        rect_pad=0.5,
        rect_stride=32,
        cache=None,
        scaleup=True,
        pad_value=DEFAULT_PAD_VALUE,
        drop_remainder=None,
        seed=0,
        nc=None,
        keep_metadata=False,
        value_range=(0.0, 1.0),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        grain = require_grain()

        if shuffle is None:
            shuffle = split == "train" and not rect
        if drop_remainder is None:
            drop_remainder = bool(shuffle)
        if rect and shuffle:
            raise ValueError(
                "rect=True sorts the dataset by aspect ratio, which shuffling would "
                "undo. Use rect=True with shuffle=False (the default for val/test)."
            )

        source = build_source(data, split=split, cache=cache, nc=nc)

        plan = None
        if rect:
            plan = RectangularPlan(
                source.shapes,
                batch_size=batch_size,
                image_size=image_size,
                stride=rect_stride,
                pad=rect_pad,
            )
            source = YOLODataSource(
                [source.image_paths[i] for i in plan.order],
                nc=source.nc,
                label_paths=[source.label_paths[i] for i in plan.order],
                cache=source.cache,
            )

        letterbox = LetterboxSample(image_size=image_size, pad_value=pad_value, scaleup=scaleup)
        pad_targets = PadTargets(max_boxes=max_boxes, keep_metadata=keep_metadata)

        count = len(source)
        num_batches = count // batch_size if drop_remainder else -(-count // batch_size)
        if num_batches == 0:
            raise ValueError(
                f"split {split!r} has {count} image(s), which is fewer than "
                f"batch_size={batch_size} with drop_remainder=True. Lower the batch "
                "size or pass drop_remainder=False."
            )

        def dataset_factory(epoch):
            dataset = grain.MapDataset.source(source)
            if plan is not None:
                dataset = dataset.map_with_index(RectangularShapes(plan))
            if shuffle:
                dataset = dataset.shuffle(seed=seed + epoch)
            dataset = dataset.map(letterbox).map(pad_targets)
            return dataset.batch(batch_size, drop_remainder=drop_remainder)

        self.source = source
        self.augment = resolve_augment(
            augment,
            image_size=image_size,
            max_boxes=max_boxes,
            pad_value=pad_value,
            value_range=value_range,
            data_format=data_format,
            seed=seed,
        )
        self.value_range = tuple(float(v) for v in value_range)
        self.metadata_keys = ("ratio", "pad", "orig_shape", "index") if keep_metadata else ()
        self._factory = dataset_factory
        self._num_batches = num_batches
        self._epoch = 0
        self._dataset = dataset_factory(0)

    @property
    def nc(self):
        """Number of classes the labels are validated against, if known."""
        return self.source.nc

    def __len__(self):
        return self._num_batches

    def on_epoch_end(self):
        """Re-shuffle by rebuilding the pipeline with the next epoch's seed."""
        self._epoch += 1
        self._dataset = self._factory(self._epoch)

    def to_float(self, images):
        low, high = self.value_range
        return np.asarray(images, dtype="float32") * ((high - low) / 255.0) + low

    def __getitem__(self, index):
        if not 0 <= index < self._num_batches:
            raise IndexError(
                f"batch index {index} is out of range for {self._num_batches} batches."
            )
        batch = self._dataset[index]
        sample = {
            "images": self.to_float(batch["images"]),
            "boxes": batch["boxes"],
            "labels": batch["labels"],
            "mask": batch["mask"],
        }
        if self.augment is not None:
            sample = self.augment(sample, training=True)

        inputs = {"images": sample["images"]}
        for key in self.metadata_keys:
            if key in batch:
                inputs[key] = batch[key]
        targets = {
            "boxes": sample["boxes"],
            "labels": sample["labels"],
            "mask": sample["mask"],
        }
        return inputs, targets


def resolve_augment(augment, image_size, max_boxes, pad_value, value_range, data_format, seed):
    """Turn the ``augment`` argument into a pipeline (or ``None``)."""
    if augment is None or augment is False:
        return None

    from ..augmentation import AugmentationPipeline

    if augment is True:
        overrides = {}
    elif isinstance(augment, dict):
        overrides = augment
    else:
        return augment

    scale = (pad_value[0] if isinstance(pad_value, (tuple, list)) else pad_value) / 255.0
    return AugmentationPipeline.from_hyperparameters(
        image_size=image_size,
        max_boxes=max_boxes,
        pad_value=value_range[0] + scale * (value_range[1] - value_range[0]),
        value_range=value_range,
        data_format=data_format,
        seed=seed,
        **overrides,
    )
