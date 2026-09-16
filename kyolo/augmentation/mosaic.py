"""Four-image mosaic augmentation."""

from __future__ import annotations

import keras
from keras import ops

from .base import DetectionAugmentation

__all__ = ["Mosaic"]


@keras.saving.register_keras_serializable(package="kyolo")
class Mosaic(DetectionAugmentation):
    """Tile four images into one ``2S x 2S`` canvas around a random centre.

    This is Ultralytics' ``Mosaic`` (the ``n=4`` variant) with the same geometry:
    four ``S x S`` images are pasted so their corners meet at a centre drawn
    from ``U(S/2, 3S/2)``, each clipped to the canvas, and whatever the four
    do not cover stays ``pad_value`` grey. The output is deliberately *twice* the
    input resolution -- Ultralytics then crops back to ``S`` in the following
    affine step, and so does kyolo. Pair this layer with
    :class:`~kyolo.augmentation.RandomPerspective` with ``output_size=S``::

        Mosaic() -> RandomPerspective(output_size=S)

    Cropping afterwards (rather than shrinking the four images to quadrants) is
    what preserves object scale: the mosaic doubles the field of view instead of
    halving object size, which is the property that makes mosaic useful.

    The other three images come from elsewhere in the *batch* rather than from a
    dataset handle, which keeps the layer a pure tensor op. It therefore needs a
    batch size of at least 4 to actually mix four distinct images; smaller
    batches still work but will repeat images within the canvas.

    When the per-sample probability draw fails, the single input image is placed
    in the centre of the canvas instead. A centred crop back to ``S x S`` then
    returns it untouched, which is how ``prob`` stays meaningful even though
    every sample has to come out the same size.

    Note that the canvas is four times the area of the input, so this layer is
    the memory high-water mark of a pipeline -- a batch of 16 at ``S=640`` holds
    a ``(16, 1280, 1280, 3)`` float32 tensor. The following crop brings it
    straight back down.

    Args:
        prob: per-sample probability of building a mosaic.
        max_boxes: optional cap on the (up to four-fold) output box count.
    """

    def __init__(self, prob=1.0, max_boxes=None, **kwargs):
        super().__init__(prob=prob, max_boxes=max_boxes, **kwargs)

    def center(self, batch, height, width):
        """Random mosaic centre in canvas pixels, as integers."""
        center_x = ops.floor(self.uniform((batch,), width / 2.0, 3.0 * width / 2.0))
        center_y = ops.floor(self.uniform((batch,), height / 2.0, 3.0 * height / 2.0))
        return center_x, center_y

    def quadrant_offsets(self, center_x, center_y, height, width):
        """Paste offset of each quadrant, as ``(offset_x, offset_y)`` pairs.

        Reproduces the corner-meeting placement of Ultralytics' ``_mosaic4``:
        the left column sits at ``centre - S`` and the right column at
        ``centre``, likewise for rows.
        """
        left, right = center_x - width, center_x
        top, bottom = center_y - height, center_y
        return (
            (left, top),
            (right, top),
            (left, bottom),
            (right, bottom),
        )

    def mosaic_images(self, images, center_x, center_y, gate):
        """Assemble the canvas with a single gather.

        For every canvas pixel this works out which quadrant it falls in, which
        batch element that quadrant draws from, and where in that image to read.
        Turning the whole mosaic into one gather (instead of four warp-and-paste
        passes) keeps exactly one canvas-sized tensor alive.
        """
        height, width = self.static_hw(images)
        batch = ops.shape(images)[0]
        canvas_h, canvas_w = 2 * height, 2 * width

        xs = ops.reshape(ops.arange(canvas_w, dtype="float32"), (1, 1, canvas_w))
        ys = ops.reshape(ops.arange(canvas_h, dtype="float32"), (1, canvas_h, 1))
        cx = ops.reshape(center_x, (-1, 1, 1))
        cy = ops.reshape(center_y, (-1, 1, 1))

        on_right = ops.cast(xs >= cx, "float32")
        on_bottom = ops.cast(ys >= cy, "float32")
        quadrant = on_bottom * 2.0 + on_right

        source_x = xs - cx + width * (1.0 - on_right)
        source_y = ys - cy + height * (1.0 - on_bottom)

        single_x = ops.broadcast_to(xs - width / 2.0, ops.shape(source_x))
        single_y = ops.broadcast_to(ys - height / 2.0, ops.shape(source_y))

        use_mosaic = ops.cast(ops.reshape(gate, (-1, 1, 1)) > 0.5, "bool")
        quadrant = ops.where(use_mosaic, quadrant, ops.zeros_like(quadrant))
        source_x = ops.where(use_mosaic, source_x, single_x)
        source_y = ops.where(use_mosaic, source_y, single_y)

        inside = ops.logical_and(
            ops.logical_and(source_x >= 0.0, source_x < width),
            ops.logical_and(source_y >= 0.0, source_y < height),
        )

        index_x = ops.cast(ops.clip(source_x, 0.0, width - 1.0), "int32")
        index_y = ops.cast(ops.clip(source_y, 0.0, height - 1.0), "int32")
        quadrant = ops.cast(quadrant, "int32")

        sample_index = ops.reshape(ops.arange(batch, dtype="int32"), (-1, 1, 1))
        source_sample = ops.mod(sample_index + quadrant, batch)

        flat_index = (source_sample * height + index_y) * width + index_x
        channels = images.shape[-1]
        gathered = ops.take(
            ops.reshape(images, (-1, channels)), ops.reshape(flat_index, (-1,)), axis=0
        )
        canvas = ops.reshape(gathered, (-1, canvas_h, canvas_w, channels))
        return ops.where(ops.expand_dims(inside, -1), canvas, self.pad_value)

    def mosaic_boxes(self, sample, center_x, center_y, height, width, gate):
        """Offset each quadrant's boxes into the canvas and concatenate them."""
        batch = ops.shape(sample["images"])[0]
        offsets = self.quadrant_offsets(center_x, center_y, height, width)
        centered = ops.convert_to_tensor(
            [[[width / 2.0, height / 2.0, width / 2.0, height / 2.0]]], dtype="float32"
        )
        box_gate = ops.reshape(gate, (-1, 1, 1))
        label_gate = ops.reshape(gate, (-1, 1))

        pieces = []
        for quadrant, (offset_x, offset_y) in enumerate(offsets):
            partner = self.gather_samples(sample, self.partner_indices(batch, quadrant))
            shift = ops.stack([offset_x, offset_y, offset_x, offset_y], axis=-1)
            mosaic_boxes = partner["boxes"] + ops.expand_dims(shift, 1)

            if quadrant == 0:
                single_boxes = sample["boxes"] + centered
                single_labels = sample["labels"]
                single_mask = sample["mask"]
            else:
                single_boxes = ops.zeros_like(partner["boxes"])
                single_labels = ops.zeros_like(partner["labels"])
                single_mask = ops.zeros_like(partner["mask"])

            pieces.append(
                {
                    "boxes": ops.where(box_gate > 0.5, mosaic_boxes, single_boxes),
                    "labels": ops.where(label_gate > 0.5, partner["labels"], single_labels),
                    "mask": ops.where(label_gate > 0.5, partner["mask"], single_mask),
                }
            )

        return self.concat_boxes(pieces)

    def augment(self, sample):
        images = sample["images"]
        height, width = self.static_hw(images)
        batch = ops.shape(images)[0]

        center_x, center_y = self.center(batch, height, width)
        gate = self.should_apply(batch, rank=1)

        merged = self.mosaic_boxes(sample, center_x, center_y, height, width, gate)
        boxes, mask = self.clip_and_filter(
            merged["boxes"], merged["mask"], 2 * height, 2 * width, min_side=0.0
        )

        out = dict(sample)
        out["images"] = self.mosaic_images(images, center_x, center_y, gate)
        out["boxes"] = boxes
        out["labels"] = merged["labels"]
        out["mask"] = mask
        return out

    def compute_output_shape(self, input_shape):
        shape = dict(input_shape)
        images = list(shape["images"])
        spatial = (1, 2) if self.data_format == "channels_last" else (2, 3)
        for axis in spatial:
            if images[axis] is not None:
                images[axis] *= 2
        shape["images"] = tuple(images)

        for key in ("boxes", "labels", "mask"):
            dims = list(shape[key])
            if dims[1] is not None:
                dims[1] = min(4 * dims[1], self.max_boxes or 4 * dims[1])
            shape[key] = tuple(dims)
        return shape
