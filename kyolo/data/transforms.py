"""Per-sample transforms for the Grain pipeline.

These run in Grain's worker processes, where the job is IO and reshaping rather
than maths: decode, letterbox to a fixed size, and pad the target lists so the
samples can be stacked. They are deliberately plain numpy (and Pillow) so a
worker never has to import a Keras backend.

The heavy, randomized augmentation happens afterwards, on the assembled batch,
in :mod:`kyolo.augmentation`.

Each transform is a plain callable, so it goes straight into
``grain.MapDataset.map`` without subclassing anything.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LetterboxSample", "PadTargets", "RectangularShapes"]

DEFAULT_PAD_VALUE = 114


def resize(image, width, height):
    """Bilinear resize of a ``(H, W, 3)`` uint8 array."""
    from PIL import Image

    if (image.shape[1], image.shape[0]) == (width, height):
        return image
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.BILINEAR), dtype=image.dtype
    )


class LetterboxSample:
    """Resize preserving aspect ratio and pad to the target shape.

    The numpy counterpart of :class:`kyolo.layers.Letterbox`, using the same
    rounding, so a model trained through this pipeline and one served through
    :class:`kyolo.preprocessing.YOLOPreprocessor` see the same geometry.

    Boxes are scaled and shifted alongside the image, and the ``ratio`` / ``pad``
    that were applied are recorded on the sample so predictions can be mapped
    back to original image coordinates with :func:`kyolo.ops.scale_boxes`.

    A ``target_shape`` written onto the sample by :class:`RectangularShapes`
    overrides ``image_size``, which is what makes rectangular batching work.

    Args:
        image_size: target square side, or an explicit ``(height, width)``.
        pad_value: fill for the padded border, in 0-255 units.
        scaleup: allow upscaling of small images. ``False`` (Ultralytics'
            inference default) only ever shrinks, which is better for accuracy
            but leaves more padding.
    """

    def __init__(self, image_size=640, pad_value=DEFAULT_PAD_VALUE, scaleup=True):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(int(v) for v in image_size)
        self.pad_value = int(pad_value)
        self.scaleup = bool(scaleup)

    def __call__(self, sample):
        image = sample["image"]
        source_h, source_w = image.shape[:2]
        target_h, target_w = sample.get("target_shape", self.image_size)
        target_h, target_w = int(target_h), int(target_w)

        ratio = min(target_h / source_h, target_w / source_w)
        if not self.scaleup:
            ratio = min(ratio, 1.0)

        scaled_w = int(round(source_w * ratio))
        scaled_h = int(round(source_h * ratio))
        resized = resize(image, scaled_w, scaled_h)

        half_w = (target_w - scaled_w) / 2.0
        half_h = (target_h - scaled_h) / 2.0
        left = max(int(round(half_w - 0.1)), 0)
        top = max(int(round(half_h - 0.1)), 0)

        padded = np.full((target_h, target_w, 3), self.pad_value, dtype=image.dtype)
        padded[top : top + scaled_h, left : left + scaled_w] = resized

        boxes = np.asarray(sample["boxes"], dtype="float32").reshape(-1, 4) * ratio
        boxes += np.array([left, top, left, top], dtype="float32")

        out = dict(sample)
        out["image"] = padded
        out["boxes"] = boxes
        out["ratio"] = np.array([ratio, ratio], dtype="float32")
        out["pad"] = np.array([left, top], dtype="float32")
        out.pop("target_shape", None)
        return out


class PadTargets:
    """Pad the per-image target lists to a fixed length so samples can stack.

    Emits the training-batch key names, so once Grain batches these the result
    is the ``{"images", "boxes", "labels", "mask"}`` dict that
    :mod:`kyolo.augmentation` and :class:`kyolo.training.YOLODetector` expect.
    ``mask`` is the only thing distinguishing a real box from padding from here
    on.

    Images with more than ``max_boxes`` objects keep the first ``max_boxes``.

    Args:
        max_boxes: fixed number of target slots per image.
        keep_metadata: also pass through ``ratio``, ``pad``, ``orig_shape`` and
            ``index``. Needed to map predictions back to original coordinates
            during validation; dropped by default to keep train batches lean.
    """

    def __init__(self, max_boxes=100, keep_metadata=False):
        if max_boxes < 1:
            raise ValueError(f"max_boxes must be positive; got {max_boxes}.")
        self.max_boxes = int(max_boxes)
        self.keep_metadata = bool(keep_metadata)

    def __call__(self, sample):
        boxes = np.asarray(sample["boxes"], dtype="float32").reshape(-1, 4)
        labels = np.asarray(sample["labels"], dtype="int32").reshape(-1)
        count = min(len(labels), self.max_boxes)

        padded_boxes = np.zeros((self.max_boxes, 4), dtype="float32")
        padded_labels = np.zeros((self.max_boxes,), dtype="int32")
        mask = np.zeros((self.max_boxes,), dtype="float32")
        if count:
            padded_boxes[:count] = boxes[:count]
            padded_labels[:count] = labels[:count]
            mask[:count] = 1.0

        out = {
            "images": sample["image"],
            "boxes": padded_boxes,
            "labels": padded_labels,
            "mask": mask,
        }
        if self.keep_metadata:
            for key in ("ratio", "pad", "orig_shape", "index"):
                if key in sample:
                    out[key] = sample[key]
        return out


class RectangularShapes:
    """Stamp each sample with the target shape of the batch it belongs to.

    Rectangular batching sorts images by aspect ratio so that each batch can be
    letterboxed to a shape that fits its own images, instead of every image
    being padded to a square. Tall images then waste no width and wide images no
    height, which is a real speedup at validation time.

    Used via ``grain.MapDataset.map_with_index``, because the target shape is a
    property of the sample's *position* in the (already sorted) dataset rather
    than of the sample itself.

    Args:
        plan: the :class:`kyolo.data.RectangularPlan` describing the ordering
            and the per-batch shapes.
    """

    def __init__(self, plan):
        self.plan = plan

    def __call__(self, index, sample):
        out = dict(sample)
        out["target_shape"] = self.plan.shape_for(int(index))
        return out
