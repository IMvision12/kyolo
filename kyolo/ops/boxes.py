"""Bounding-box geometry helpers in pure ``keras.ops``."""

from __future__ import annotations

import math

from keras import ops

__all__ = [
    "xywh2xyxy",
    "xyxy2xywh",
    "box_area",
    "bbox_iou",
    "pairwise_iou",
    "clip_boxes",
    "scale_boxes",
]


def xywh2xyxy(boxes):
    """(cx, cy, w, h) -> (x1, y1, x2, y2). Operates on the last axis."""
    cxcy = boxes[..., :2]
    wh = boxes[..., 2:4]
    x1y1 = cxcy - wh / 2.0
    x2y2 = cxcy + wh / 2.0
    return ops.concatenate([x1y1, x2y2], axis=-1)


def xyxy2xywh(boxes):
    """(x1, y1, x2, y2) -> (cx, cy, w, h). Operates on the last axis."""
    x1y1 = boxes[..., :2]
    x2y2 = boxes[..., 2:4]
    cxcy = (x1y1 + x2y2) / 2.0
    wh = x2y2 - x1y1
    return ops.concatenate([cxcy, wh], axis=-1)


def _split_box_columns(boxes):
    """Split ``(..., 4+)`` into the xyxy block and any trailing passthrough columns.

    Letting callers keep extra columns means a ``(B, N, 6)``
    ``[x1, y1, x2, y2, score, class]`` tensor straight out of
    ``YOLOPostprocessor`` can be fed in without being taken apart first.
    """
    boxes = ops.convert_to_tensor(boxes, dtype="float32")
    columns = boxes.shape[-1]
    if columns is None:
        raise ValueError(
            "the last axis of `boxes` must be statically known (4 for plain xyxy, "
            "or 6 for a detection tensor); got a dynamic last axis."
        )
    if columns < 4:
        raise ValueError(f"`boxes` needs at least 4 columns (x1, y1, x2, y2); got {columns}.")
    return boxes[..., :4], boxes[..., 4:] if columns > 4 else None


def _align_pair(values, rank, name):
    """Reshape a ``(2,)`` or ``(B, 2)`` parameter to broadcast over rank-``rank`` boxes."""
    values = ops.convert_to_tensor(values, dtype="float32")
    ndim = len(values.shape)
    if ndim == 1:
        return ops.reshape(values, (1,) * (rank - 1) + (2,))
    if ndim != 2:
        raise ValueError(f"`{name}` must have shape (2,) or (B, 2); got {tuple(values.shape)}.")
    if rank < 3:
        raise ValueError(
            f"a per-image `{name}` of shape {tuple(values.shape)} needs batched boxes "
            f"shaped (B, N, 4+), but `boxes` has rank {rank}."
        )

    return ops.reshape(values, (-1,) + (1,) * (rank - 2) + (2,))


def clip_boxes(boxes, shape):
    """Clip xyxy boxes so they lie inside an image.

    Args:
        boxes: ``(..., 4+)`` xyxy boxes. Columns past the first four (e.g. score
            and class) are passed through untouched.
        shape: image ``(height, width)``, or a ``(B, 2)`` batch of them when
            every image in the batch has its own size.

    Returns:
        A tensor shaped like ``boxes``.
    """
    xyxy, extra = _split_box_columns(boxes)
    hw = _align_pair(shape, len(xyxy.shape), "shape")
    height, width = hw[..., :1], hw[..., 1:]
    upper = ops.concatenate([width, height, width, height], axis=-1)

    xyxy = ops.minimum(ops.maximum(xyxy, 0.0), upper)
    return xyxy if extra is None else ops.concatenate([xyxy, extra], axis=-1)


def scale_boxes(boxes, ratio, pad, orig_shape=None, clip=True):
    """Map boxes from letterboxed model space back to original image pixels.

    This is the inverse of the ``YOLOPreprocessor`` / ``Letterbox`` transform,
    ``x_orig = (x_letterboxed - pad_x) / ratio_x``, so it takes the ``ratio`` and
    ``pad`` that the preprocessor returned alongside the image::

        batch = preprocessor(image)
        detections = postprocessor(model(batch["images"]))
        detections = scale_boxes(
            detections, batch["ratio"], batch["pad"], image.shape[:2]
        )

    Padding is removed before dividing by the ratio, matching Ultralytics'
    ``scale_boxes``. Rows that ``YOLOPostprocessor`` zero-padded stay all-zero
    when ``clip`` is ``True``, so the usual ``score > threshold`` filter keeps
    working afterwards.

    Args:
        boxes: ``(..., 4+)`` xyxy boxes in letterboxed coordinates. Columns past
            the first four are passed through, so a ``(B, N, 6)``
            ``[x1, y1, x2, y2, score, class]`` detection tensor works directly.
        ratio: ``(2,)`` or ``(B, 2)`` resize ratio ``(ratio_x, ratio_y)``.
        pad: ``(2,)`` or ``(B, 2)`` padding ``(pad_x, pad_y)`` in pixels, i.e. the
            padding added to the left and top, as returned by ``Letterbox``.
        orig_shape: original image ``(height, width)``, or a ``(B, 2)`` batch of
            them. Required when ``clip`` is ``True``.
        clip: clip the result to the original image bounds.

    Returns:
        A tensor shaped like ``boxes``, in original-image pixel coordinates.
    """
    xyxy, extra = _split_box_columns(boxes)
    rank = len(xyxy.shape)
    gain = _align_pair(ratio, rank, "ratio")
    offset = _align_pair(pad, rank, "pad")

    xyxy = (xyxy - ops.concatenate([offset, offset], axis=-1)) / ops.concatenate(
        [gain, gain], axis=-1
    )
    if clip:
        if orig_shape is None:
            raise ValueError("`orig_shape` is required when `clip=True`.")
        xyxy = clip_boxes(xyxy, orig_shape)
    return xyxy if extra is None else ops.concatenate([xyxy, extra], axis=-1)


def box_area(boxes):
    """Area of xyxy boxes; last axis is [x1, y1, x2, y2]."""
    w = ops.maximum(boxes[..., 2] - boxes[..., 0], 0.0)
    h = ops.maximum(boxes[..., 3] - boxes[..., 1], 0.0)
    return w * h


def bbox_iou(box1, box2, xywh=False, giou=False, diou=False, ciou=False, eps=1e-7):
    """IoU / GIoU / DIoU / CIoU between aligned box sets (elementwise on rows).

    ``box1`` and ``box2`` must broadcast to a common shape ``(..., 4)``. Returns
    a tensor of shape ``(...,)``. Used by the detection loss (CIoU) and anywhere
    an aligned overlap is needed.
    """
    if xywh:
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    b1x1, b1y1, b1x2, b1y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2x1, b2y1, b2x2, b2y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_w = ops.maximum(ops.minimum(b1x2, b2x2) - ops.maximum(b1x1, b2x1), 0.0)
    inter_h = ops.maximum(ops.minimum(b1y2, b2y2) - ops.maximum(b1y1, b2y1), 0.0)
    inter = inter_w * inter_h

    w1 = ops.maximum(b1x2 - b1x1, 0.0)
    h1 = ops.maximum(b1y2 - b1y1, 0.0)
    w2 = ops.maximum(b2x2 - b2x1, 0.0)
    h2 = ops.maximum(b2y2 - b2y1, 0.0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if not (giou or diou or ciou):
        return iou

    cw = ops.maximum(b1x2, b2x2) - ops.minimum(b1x1, b2x1)
    ch = ops.maximum(b1y2, b2y2) - ops.minimum(b1y1, b2y1)

    if ciou or diou:
        c2 = cw * cw + ch * ch + eps
        b1cx, b1cy = (b1x1 + b1x2) / 2.0, (b1y1 + b1y2) / 2.0
        b2cx, b2cy = (b2x1 + b2x2) / 2.0, (b2y1 + b2y2) / 2.0
        rho2 = (b2cx - b1cx) ** 2 + (b2cy - b1cy) ** 2
        if diou:
            return iou - rho2 / c2
        v = (4.0 / (math.pi**2)) * ops.square(
            ops.arctan(w2 / (h2 + eps)) - ops.arctan(w1 / (h1 + eps))
        )
        alpha = v / (v - iou + (1.0 + eps))
        alpha = ops.stop_gradient(alpha)
        return iou - (rho2 / c2 + v * alpha)

    c_area = cw * ch + eps
    return iou - (c_area - union) / c_area


def pairwise_iou(boxes1, boxes2, eps=1e-7):
    """IoU matrix between two xyxy box sets.

    Args:
        boxes1: ``(N, 4)`` xyxy.
        boxes2: ``(M, 4)`` xyxy.
    Returns:
        ``(N, M)`` IoU matrix.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    b1 = ops.expand_dims(boxes1, 1)
    b2 = ops.expand_dims(boxes2, 0)
    inter_w = ops.maximum(
        ops.minimum(b1[..., 2], b2[..., 2]) - ops.maximum(b1[..., 0], b2[..., 0]), 0.0
    )
    inter_h = ops.maximum(
        ops.minimum(b1[..., 3], b2[..., 3]) - ops.maximum(b1[..., 1], b2[..., 1]), 0.0
    )
    inter = inter_w * inter_h
    union = ops.expand_dims(area1, 1) + ops.expand_dims(area2, 0) - inter + eps
    return inter / union
