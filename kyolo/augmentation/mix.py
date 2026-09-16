"""Augmentations that blend or transplant content between two images."""

from __future__ import annotations

import keras
from keras import ops

from .base import DetectionAugmentation

__all__ = ["MixUp", "CopyPaste"]

_EPS = 1e-9


def intersection_over_area(source, target, target_mask):
    """``(B, M, M)`` intersection of each source box over its own area.

    Ultralytics' copy-paste keeps a candidate only when it barely overlaps every
    box already in the image, and measures that with intersection-over-area
    rather than IoU so that a small box inside a large one still counts as
    overlapping. Invalid target boxes are forced to zero overlap so padding
    cannot veto a candidate.
    """
    src = ops.expand_dims(source, 2)
    dst = ops.expand_dims(target, 1)

    inter_w = ops.maximum(
        ops.minimum(src[..., 2], dst[..., 2]) - ops.maximum(src[..., 0], dst[..., 0]), 0.0
    )
    inter_h = ops.maximum(
        ops.minimum(src[..., 3], dst[..., 3]) - ops.maximum(src[..., 1], dst[..., 1]), 0.0
    )
    inter = inter_w * inter_h

    area = ops.maximum(source[..., 2] - source[..., 0], 0.0) * ops.maximum(
        source[..., 3] - source[..., 1], 0.0
    )
    ioa = inter / (ops.expand_dims(area, -1) + _EPS)
    return ioa * ops.expand_dims(target_mask, 1)


@keras.saving.register_keras_serializable(package="kyolo")
class MixUp(DetectionAugmentation):
    """Blend two images and take the union of their boxes.

    Matches Ultralytics' ``MixUp``: the blend weight is drawn from
    ``Beta(alpha, alpha)`` with ``alpha=32``, which concentrates tightly around
    ``0.5`` (standard deviation about ``0.06``), so both images stay clearly
    visible. Boxes from both are kept at full confidence -- mixup for detection
    relies on the classification loss seeing two valid, partially transparent
    objects rather than on weighting the targets.

    The partner image is the next one in the batch, so a batch size of at least
    2 is required for this to do anything.

    Args:
        prob: per-sample probability of blending. Ultralytics defaults to ``0``
            (off) for the YOLOv8-and-later recipes.
        alpha: both shape parameters of the symmetric Beta distribution.
        max_boxes: optional cap on the doubled output box count.
    """

    def __init__(self, prob=0.15, alpha=32.0, max_boxes=None, **kwargs):
        super().__init__(prob=prob, max_boxes=max_boxes, **kwargs)
        if alpha <= 0:
            raise ValueError(f"alpha must be positive; got {alpha}.")
        self.alpha = float(alpha)

    def augment(self, sample):
        images = sample["images"]
        batch = ops.shape(images)[0]
        partner = self.gather_samples(sample, self.partner_indices(batch))

        weight = keras.random.beta(
            (batch,), self.alpha, self.alpha, dtype="float32", seed=self.seed_generator
        )
        gate = ops.reshape(self.should_apply(batch, rank=1), (-1,))
        weight = 1.0 - (1.0 - weight) * gate

        image_weight = ops.reshape(weight, (-1, 1, 1, 1))
        blended = images * image_weight + partner["images"] * (1.0 - image_weight)

        label_gate = ops.reshape(gate, (-1, 1))
        merged = self.concat_boxes(
            [
                sample,
                {
                    "boxes": partner["boxes"],
                    "labels": partner["labels"],
                    "mask": partner["mask"] * label_gate,
                },
            ]
        )

        out = dict(sample)
        out["images"] = blended
        out.update(merged)
        return out

    def compute_output_shape(self, input_shape):
        shape = dict(input_shape)
        for key in ("boxes", "labels", "mask"):
            dims = list(shape[key])
            if dims[1] is not None:
                dims[1] = min(2 * dims[1], self.max_boxes or 2 * dims[1])
            shape[key] = tuple(dims)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


@keras.saving.register_keras_serializable(package="kyolo")
class CopyPaste(DetectionAugmentation):
    """Transplant object crops from a second view into the image.

    Ultralytics' ``CopyPaste`` cuts objects along their *segmentation* masks.
    kyolo is detection-only, so there are no masks to cut along and this layer
    pastes the axis-aligned box crop instead. The selection rule is kept:
    a candidate is pasted only when its intersection-over-area against every box
    already present is below ``max_overlap``, so pasted objects do not bury
    existing ones.

    Args:
        prob: per-object probability of pasting a given candidate. Ultralytics
            defaults to ``0`` (off).
        mode: ``"flip"`` takes candidates from the image's own horizontal mirror
            (Ultralytics' default ``copy_paste_mode``); ``"mixup"`` takes them
            from another image in the batch.
        max_overlap: reject candidates whose intersection-over-area with any
            existing box reaches this value.
        max_boxes: optional cap on the doubled output box count.
    """

    def __init__(self, prob=0.1, mode="flip", max_overlap=0.3, max_boxes=None, **kwargs):
        super().__init__(prob=prob, max_boxes=max_boxes, **kwargs)
        if mode not in ("flip", "mixup"):
            raise ValueError(f"mode must be 'flip' or 'mixup'; got {mode!r}.")
        self.mode = mode
        self.max_overlap = float(max_overlap)

    def source(self, sample, width):
        """The view candidates are cut from."""
        if self.mode == "mixup":
            batch = ops.shape(sample["images"])[0]
            return self.gather_samples(sample, self.partner_indices(batch))

        boxes = sample["boxes"]
        mirrored = ops.stack(
            [
                width - boxes[..., 2],
                boxes[..., 1],
                width - boxes[..., 0],
                boxes[..., 3],
            ],
            axis=-1,
        )
        return {
            "images": ops.flip(sample["images"], axis=2),
            "boxes": mirrored,
            "labels": sample["labels"],
            "mask": sample["mask"],
        }

    def selection(self, sample, source):
        """Per-candidate ``{0., 1.}`` decision to paste."""
        batch, count = ops.shape(source["mask"])[0], source["mask"].shape[1]
        overlap = intersection_over_area(source["boxes"], sample["boxes"], sample["mask"])
        uncrowded = ops.all(overlap < self.max_overlap, axis=-1)

        drawn = self.uniform((batch, count)) < self.prob
        keep = ops.logical_and(uncrowded, drawn)
        return source["mask"] * ops.cast(keep, "float32")

    def paste_mask(self, boxes, selected, height, width):
        """``(B, H, W)`` mask of pixels covered by at least one chosen box.

        Built as a separable contraction over rows and columns rather than a
        dense ``(B, M, H, W)`` stack, which would be orders of magnitude larger
        than the image itself.
        """
        xs = ops.reshape(ops.arange(width, dtype="float32"), (1, 1, width))
        ys = ops.reshape(ops.arange(height, dtype="float32"), (1, 1, height))
        x1, y1 = boxes[..., 0:1], boxes[..., 1:2]
        x2, y2 = boxes[..., 2:3], boxes[..., 3:4]

        spans_x = ops.cast(ops.logical_and(xs >= x1, xs < x2), "float32")
        spans_y = ops.cast(ops.logical_and(ys >= y1, ys < y2), "float32")

        weighted_rows = spans_y * ops.expand_dims(selected, -1)
        coverage = ops.einsum("bjy,bjx->byx", weighted_rows, spans_x)
        return coverage > 0.0

    def augment(self, sample):
        images = sample["images"]
        height, width = self.static_hw(images)

        source = self.source(sample, float(width))
        selected = self.selection(sample, source)
        paste = self.paste_mask(source["boxes"], selected, height, width)

        merged = self.concat_boxes(
            [
                sample,
                {
                    "boxes": source["boxes"],
                    "labels": source["labels"],
                    "mask": selected,
                },
            ]
        )

        out = dict(sample)
        out["images"] = ops.where(ops.expand_dims(paste, -1), source["images"], images)
        out.update(merged)
        return out

    def compute_output_shape(self, input_shape):
        shape = dict(input_shape)
        for key in ("boxes", "labels", "mask"):
            dims = list(shape[key])
            if dims[1] is not None:
                dims[1] = min(2 * dims[1], self.max_boxes or 2 * dims[1])
            shape[key] = tuple(dims)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode, "max_overlap": self.max_overlap})
        return config
