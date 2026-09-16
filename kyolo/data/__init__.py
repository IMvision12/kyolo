"""Detection data loading, built on Grain.

    from kyolo.data import GrainDataLoader

    train = GrainDataLoader("coco8.yaml", "train", batch_size=16, augment=True)
    val = GrainDataLoader("coco8.yaml", "val", batch_size=16, rect=True)
    detector.fit(train, validation_data=val, epochs=100)

The pipeline splits the work by what each tool is good at: Grain does
random-access, parallel, plain-numpy IO per sample (decode, letterbox, pad the
target lists), and :mod:`kyolo.augmentation` does the randomized augmentation on
the assembled batch in ``keras.ops``. Grain rather than ``tf.data`` keeps a
TensorFlow install from being a prerequisite for training a JAX or PyTorch
model, and its ``MapDataset`` random access is what makes exact per-epoch
reshuffling and rectangular batching simple.

The loader is a :class:`keras.utils.PyDataset`, so ``model.fit`` takes it as-is
and no ``steps_per_epoch`` is needed.

Datasets are described the Ultralytics way, by a ``data.yaml`` plus one
``.txt`` of normalized boxes per image; :func:`coco_to_yolo` and
:func:`voc_to_yolo` produce that layout from COCO JSON or VOC XML.

Grain is an optional dependency (``pip install kyolo[data]``); importing this
module without it is fine, and only :class:`GrainDataLoader` will complain.
"""

from __future__ import annotations

from .config import IMAGE_EXTENSIONS, DataConfig
from .convert import coco_to_yolo, voc_to_yolo
from .labels import (
    label_path_for,
    load_yolo_label,
    normalized_to_xyxy,
    write_yolo_label,
    xyxy_to_normalized,
)
from .pipeline import GrainDataLoader, build_source
from .rect import RectangularPlan
from .source import YOLODataSource
from .transforms import LetterboxSample, PadTargets, RectangularShapes

__all__ = [
    "DataConfig",
    "GrainDataLoader",
    "IMAGE_EXTENSIONS",
    "LetterboxSample",
    "PadTargets",
    "RectangularPlan",
    "RectangularShapes",
    "YOLODataSource",
    "build_source",
    "coco_to_yolo",
    "label_path_for",
    "load_yolo_label",
    "normalized_to_xyxy",
    "voc_to_yolo",
    "write_yolo_label",
    "xyxy_to_normalized",
]
