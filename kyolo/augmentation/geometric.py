"""Geometric augmentation: flips and the random perspective/affine warp.

Both transform the image *and* the boxes. ``RandomPerspective`` additionally
owns the crop back to the training resolution, which is what lets
:class:`~kyolo.augmentation.Mosaic` hand it an oversized canvas.
"""

from __future__ import annotations

import math

import keras
from keras import ops

from .base import DetectionAugmentation

__all__ = ["RandomFlip", "RandomPerspective"]

_EPS = 1e-9


@keras.saving.register_keras_serializable(package="kyolo")
class RandomFlip(DetectionAugmentation):
    """Flip images and boxes along one axis, per sample.

    Args:
        direction: ``"horizontal"`` or ``"vertical"``.
        prob: per-sample flip probability. Ultralytics' defaults are ``0.5``
            horizontal and ``0.0`` vertical.
    """

    def __init__(self, direction="horizontal", prob=0.5, **kwargs):
        super().__init__(prob=prob, **kwargs)
        if direction not in ("horizontal", "vertical"):
            raise ValueError(f"direction must be 'horizontal' or 'vertical'; got {direction!r}.")
        self.direction = direction

    def augment(self, sample):
        images = sample["images"]
        boxes = sample["boxes"]
        height, width = self.static_hw(images)
        batch = ops.shape(images)[0]

        axis = 2 if self.direction == "horizontal" else 1
        flipped_images = ops.flip(images, axis=axis)

        if self.direction == "horizontal":
            mirrored = ops.stack(
                [
                    width - boxes[..., 2],
                    boxes[..., 1],
                    width - boxes[..., 0],
                    boxes[..., 3],
                ],
                axis=-1,
            )
        else:
            mirrored = ops.stack(
                [
                    boxes[..., 0],
                    height - boxes[..., 3],
                    boxes[..., 2],
                    height - boxes[..., 1],
                ],
                axis=-1,
            )

        gate = self.should_apply(batch, rank=4)
        out = dict(sample)
        out["images"] = flipped_images * gate + images * (1.0 - gate)
        box_gate = ops.reshape(gate, (-1, 1, 1))
        out["boxes"] = mirrored * box_gate + boxes * (1.0 - box_gate)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"direction": self.direction})
        return config


@keras.saving.register_keras_serializable(package="kyolo")
class RandomPerspective(DetectionAugmentation):
    """Random rotation, scale, shear, translation and perspective warp.

    Reproduces Ultralytics' ``RandomPerspective``, including the matrix order
    ``M = T @ S @ R @ P @ C`` (centre, perspective, rotate+scale, shear,
    translate) and the surviving-box filter.

    The layer also resizes the *frame*: ``output_size`` fixes the returned
    resolution regardless of the input's. This replaces Ultralytics' ``border``
    bookkeeping and is what makes the mosaic pipeline work in one pass -- a
    ``2S x 2S`` mosaic and a plain ``S x S`` image both come out ``S x S``, so
    turning mosaic off (see ``close_mosaic``) needs no other change. Input
    content is centred in the output frame, so with all strengths at zero and
    ``output_size`` equal to half the input side the result is exactly the
    centre crop Ultralytics' ``border=-S//2`` produces.

    Args:
        degrees: rotation range in degrees, ``U(-degrees, degrees)``.
        translate: translation as a fraction of the output size; the content
            centre lands in ``0.5 +/- translate``.
        scale: scale jitter; the gain is ``U(1 - scale, 1 + scale)``.
        shear: shear range in degrees, applied on both axes.
        perspective: magnitude of the projective terms. Ultralytics defaults to
            ``0``, i.e. a pure affine warp.
        output_size: ``int`` or ``(height, width)`` for the returned frame.
            ``None`` keeps the input resolution.
        min_box_side: drop warped boxes thinner than this many pixels.
        min_area_ratio: drop warped boxes that keep less than this fraction of
            their (scale-corrected) original area, which is how boxes pushed
            out of frame are discarded.
        max_aspect_ratio: drop warped boxes more elongated than this.
        prob: probability of applying the random part. At ``0`` the layer still
            performs the deterministic centre crop to ``output_size``.
    """

    def __init__(
        self,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        output_size=None,
        min_box_side=2.0,
        min_area_ratio=0.1,
        max_aspect_ratio=100.0,
        prob=1.0,
        **kwargs,
    ):
        super().__init__(prob=prob, **kwargs)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if output_size is not None:
            output_size = tuple(int(v) for v in output_size)
            if len(output_size) != 2 or min(output_size) < 1:
                raise ValueError(
                    "output_size must be a positive int or an (height, width) "
                    f"pair; got {output_size!r}."
                )
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.shear = float(shear)
        self.perspective = float(perspective)
        self.output_size = output_size
        self.min_box_side = float(min_box_side)
        self.min_area_ratio = float(min_area_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)

    def resolved_output_size(self, height, width):
        return self.output_size if self.output_size is not None else (height, width)

    def matrix(self, batch, in_hw, out_hw):
        """Build the batched ``(B, 3, 3)`` warp and the ``(B,)`` scale gain."""
        in_h, in_w = in_hw
        out_h, out_w = out_hw
        gate = ops.reshape(self.should_apply(batch, rank=1), (-1,))
        zeros = ops.zeros((batch,), dtype="float32")
        ones = ops.ones((batch,), dtype="float32")

        def jitter(magnitude, identity=0.0):
            """A gated ``U(-magnitude, magnitude)`` draw around ``identity``."""
            if magnitude == 0.0:
                return ops.full((batch,), identity, dtype="float32")
            value = self.uniform((batch,), identity - magnitude, identity + magnitude)
            return identity + (value - identity) * gate

        def mat3(rows):
            """Stack a 3x3 grid of ``(B,)`` tensors into a ``(B, 3, 3)`` matrix."""
            return ops.stack([ops.stack(row, axis=-1) for row in rows], axis=-2)

        angle = jitter(self.degrees) * (math.pi / 180.0)
        gain = jitter(self.scale, identity=1.0)
        shear_x = ops.tan(jitter(self.shear) * (math.pi / 180.0))
        shear_y = ops.tan(jitter(self.shear) * (math.pi / 180.0))
        persp_x = jitter(self.perspective)
        persp_y = jitter(self.perspective)
        shift_x = jitter(self.translate, identity=0.5) * float(out_w)
        shift_y = jitter(self.translate, identity=0.5) * float(out_h)

        center = mat3(
            [
                [ones, zeros, ops.full((batch,), -in_w / 2.0, dtype="float32")],
                [zeros, ones, ops.full((batch,), -in_h / 2.0, dtype="float32")],
                [zeros, zeros, ones],
            ]
        )
        perspective = mat3(
            [
                [ones, zeros, zeros],
                [zeros, ones, zeros],
                [persp_x, persp_y, ones],
            ]
        )
        alpha = gain * ops.cos(angle)
        beta = gain * ops.sin(angle)
        rotate = mat3(
            [
                [alpha, beta, zeros],
                [-beta, alpha, zeros],
                [zeros, zeros, ones],
            ]
        )
        shear = mat3(
            [
                [ones, shear_x, zeros],
                [shear_y, ones, zeros],
                [zeros, zeros, ones],
            ]
        )
        translate = mat3(
            [
                [ones, zeros, shift_x],
                [zeros, ones, shift_y],
                [zeros, zeros, ones],
            ]
        )

        matrix = ops.matmul(perspective, center)
        matrix = ops.matmul(rotate, matrix)
        matrix = ops.matmul(shear, matrix)
        matrix = ops.matmul(translate, matrix)
        return matrix, gain

    def warp_images(self, images, matrix, in_hw, out_hw):
        """Warp and crop, keeping ``perspective_transform``'s same-size output.

        ``keras.ops.image.perspective_transform`` returns an image the size of
        its input, so the output frame is instead placed at an offset inside a
        canvas of the input's size and sliced out afterwards. When the requested
        output is larger than the input the canvas is padded first.
        """
        in_h, in_w = in_hw
        out_h, out_w = out_hw

        pad_x = max(0, -(-(out_w - in_w) // 2))
        pad_y = max(0, -(-(out_h - in_h) // 2))
        if pad_x or pad_y:
            images = ops.pad(
                images,
                [[0, 0], [pad_y, pad_y], [pad_x, pad_x], [0, 0]],
                mode="constant",
                constant_values=self.pad_value,
            )
        canvas_h = in_h + 2 * pad_y
        canvas_w = in_w + 2 * pad_x
        offset_x = (canvas_w - out_w) // 2
        offset_y = (canvas_h - out_h) // 2

        batch = ops.shape(images)[0]
        corners = ops.convert_to_tensor(
            [[[0.0, 0.0], [0.0, in_h], [in_w, 0.0], [in_w, in_h]]], dtype="float32"
        )
        corners = ops.broadcast_to(corners, (batch, 4, 2))
        canvas_offset = ops.convert_to_tensor([[[pad_x, pad_y]]], dtype="float32")
        frame_offset = ops.convert_to_tensor([[[offset_x, offset_y]]], dtype="float32")

        start_points = corners + canvas_offset
        homogeneous = ops.concatenate([corners, ops.ones_like(corners[..., :1])], axis=-1)
        projected = ops.matmul(homogeneous, ops.transpose(matrix, (0, 2, 1)))
        end_points = projected[..., :2] / (projected[..., 2:3] + _EPS) + frame_offset

        warped = ops.image.perspective_transform(
            images,
            start_points,
            end_points,
            interpolation="bilinear",
            fill_value=self.pad_value,
            data_format="channels_last",
        )
        return warped[:, offset_y : offset_y + out_h, offset_x : offset_x + out_w, :]

    def warp_boxes(self, boxes, mask, matrix, gain, out_hw):
        """Warp boxes, then drop the ones the warp destroyed."""
        out_h, out_w = out_hw
        count = boxes.shape[1]

        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        corners = ops.reshape(
            ops.stack(
                [
                    ops.stack([x1, y1], axis=-1),
                    ops.stack([x2, y2], axis=-1),
                    ops.stack([x1, y2], axis=-1),
                    ops.stack([x2, y1], axis=-1),
                ],
                axis=-2,
            ),
            (-1, count * 4, 2),
        )
        homogeneous = ops.concatenate([corners, ops.ones_like(corners[..., :1])], axis=-1)
        projected = ops.matmul(homogeneous, ops.transpose(matrix, (0, 2, 1)))
        mapped = ops.reshape(projected[..., :2] / (projected[..., 2:3] + _EPS), (-1, count, 4, 2))
        xs, ys = mapped[..., 0], mapped[..., 1]
        warped = ops.stack(
            [
                ops.min(xs, axis=-1),
                ops.min(ys, axis=-1),
                ops.max(xs, axis=-1),
                ops.max(ys, axis=-1),
            ],
            axis=-1,
        )

        upper = ops.convert_to_tensor([[[out_w, out_h, out_w, out_h]]], dtype="float32")
        clipped = ops.clip(warped, 0.0, upper)
        new_w = clipped[..., 2] - clipped[..., 0]
        new_h = clipped[..., 3] - clipped[..., 1]

        gain_col = ops.reshape(gain, (-1, 1))
        old_w = (boxes[..., 2] - boxes[..., 0]) * gain_col
        old_h = (boxes[..., 3] - boxes[..., 1]) * gain_col

        area_ratio = (new_w * new_h) / (old_w * old_h + _EPS)
        aspect = ops.maximum(new_w / (new_h + _EPS), new_h / (new_w + _EPS))
        keep = ops.logical_and(new_w > self.min_box_side, new_h > self.min_box_side)
        keep = ops.logical_and(keep, area_ratio > self.min_area_ratio)
        keep = ops.logical_and(keep, aspect < self.max_aspect_ratio)
        return clipped, mask * ops.cast(keep, "float32")

    def augment(self, sample):
        images = sample["images"]
        in_hw = self.static_hw(images)
        out_hw = self.resolved_output_size(*in_hw)
        batch = ops.shape(images)[0]

        matrix, gain = self.matrix(batch, in_hw, out_hw)
        boxes, mask = self.warp_boxes(sample["boxes"], sample["mask"], matrix, gain, out_hw)

        out = dict(sample)
        out["images"] = self.warp_images(images, matrix, in_hw, out_hw)
        out["boxes"] = boxes
        out["mask"] = mask
        return out

    def compute_output_shape(self, input_shape):
        shape = dict(input_shape)
        images = list(shape["images"])
        spatial = (1, 2) if self.data_format == "channels_last" else (2, 3)
        if self.output_size is not None:
            images[spatial[0]], images[spatial[1]] = self.output_size
        shape["images"] = tuple(images)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "degrees": self.degrees,
                "translate": self.translate,
                "scale": self.scale,
                "shear": self.shear,
                "perspective": self.perspective,
                "output_size": self.output_size,
                "min_box_side": self.min_box_side,
                "min_area_ratio": self.min_area_ratio,
                "max_aspect_ratio": self.max_aspect_ratio,
            }
        )
        return config
