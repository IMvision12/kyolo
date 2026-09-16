"""Detection augmentation layers in pure Keras 3.

Every layer here is a ``keras.layers.Layer`` written in ``keras.ops``, so the
same augmentation runs on TensorFlow, JAX and PyTorch, on CPU or GPU, and
composes with the rest of kyolo without a second image library.

They are *batched*: a layer takes and returns the sample dict
``{"images", "boxes", "labels", "mask"}`` with a leading batch axis, the same
format :class:`kyolo.training.YOLODetector` trains on. Working on batches is
what lets mosaic, mixup and copy-paste source their second image from the batch
itself instead of needing a dataset handle, and it keeps everything a single
vectorized tensor op.

    from kyolo.augmentation import AugmentationPipeline

    augment = AugmentationPipeline.from_hyperparameters(image_size=640)
    batch = augment(batch)

:meth:`~AugmentationPipeline.from_hyperparameters` defaults to Ultralytics'
``default.yaml`` values and stage order. See
:mod:`kyolo.data` for the Grain input pipeline that feeds it.
"""

from __future__ import annotations

from .base import DEFAULT_PAD_VALUE, DetectionAugmentation
from .geometric import RandomFlip, RandomPerspective
from .mix import CopyPaste, MixUp
from .mosaic import Mosaic
from .photometric import RandomHSV
from .pipeline import MIXING_LAYERS, AugmentationPipeline

__all__ = [
    "AugmentationPipeline",
    "CopyPaste",
    "DetectionAugmentation",
    "DEFAULT_PAD_VALUE",
    "MIXING_LAYERS",
    "MixUp",
    "Mosaic",
    "RandomFlip",
    "RandomHSV",
    "RandomPerspective",
]
