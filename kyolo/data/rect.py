"""Rectangular batching: one letterbox shape per batch instead of per dataset."""

from __future__ import annotations

import numpy as np

__all__ = ["RectangularPlan"]


class RectangularPlan:
    """Group images of similar aspect ratio so each batch gets its own shape.

    Padding every image to a square wastes compute on the padding: a 640x480
    photo letterboxed to 640x640 is a quarter grey. Sorting the dataset by
    aspect ratio and giving each batch the smallest shape that fits its own
    images removes most of that, which is why Ultralytics validates with
    ``rect=True``.

    There are two trade-offs. The order is fixed by aspect ratio, so this is
    incompatible with shuffling and therefore used for validation and inference
    rather than training; and since the shape varies per batch, the model must
    be built for variable input (e.g. ``input_shape=(None, None, 3)``).

    Note that with Ultralytics' ``pad=0.5`` the long side lands one stride
    *above* ``image_size`` -- 672 for a 640 request -- because the padded size
    is rounded up to a multiple of ``stride``. That is what Ultralytics does;
    pass ``pad=0.0`` to keep every shape at or below ``image_size``.

    Shapes come from image headers, so building a plan does not decode anything.

    Args:
        shapes: ``(N, 2)`` original ``(height, width)`` per image, in dataset
            order (see :attr:`kyolo.data.YOLODataSource.shapes`).
        batch_size: images per batch; the grouping follows it exactly.
        image_size: the long side to scale to.
        stride: both output dimensions are rounded up to a multiple of this,
            since the network downsamples by it.
        pad: extra fraction of a stride added before rounding up, giving a small
            border. Ultralytics uses ``0.5``.

    Attributes:
        order: the permutation to apply to the dataset, ascending by aspect
            ratio.
        batch_shapes: ``(num_batches, 2)`` target ``(height, width)``.
    """

    def __init__(self, shapes, batch_size, image_size=640, stride=32, pad=0.5):
        shapes = np.asarray(shapes, dtype="float64").reshape(-1, 2)
        if len(shapes) == 0:
            raise ValueError("`shapes` is empty; there is nothing to plan.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive; got {batch_size}.")

        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.stride = int(stride)
        self.pad = float(pad)

        aspect = shapes[:, 0] / shapes[:, 1]
        self.order = np.argsort(aspect, kind="stable")
        sorted_aspect = aspect[self.order]

        count = len(shapes)
        num_batches = -(-count // self.batch_size)
        batch_of = np.arange(count) // self.batch_size

        fractions = np.ones((num_batches, 2), dtype="float64")
        for batch in range(num_batches):
            in_batch = sorted_aspect[batch_of == batch]
            lowest, highest = in_batch.min(), in_batch.max()
            if highest < 1:
                fractions[batch] = [highest, 1.0]
            elif lowest > 1:
                fractions[batch] = [1.0, 1.0 / lowest]

        scaled = fractions * self.image_size / self.stride + self.pad
        self.batch_shapes = (np.ceil(scaled).astype("int32")) * self.stride

    def __len__(self):
        return len(self.batch_shapes)

    def __repr__(self):
        unique = np.unique(self.batch_shapes, axis=0)
        return (
            f"RectangularPlan(n={len(self.order)}, batches={len(self)}, "
            f"distinct_shapes={len(unique)})"
        )

    def shape_for(self, position):
        """Target ``(height, width)`` for the sample at ``position`` in the
        reordered dataset."""
        batch = int(position) // self.batch_size
        batch = min(batch, len(self.batch_shapes) - 1)
        height, width = self.batch_shapes[batch]
        return int(height), int(width)
