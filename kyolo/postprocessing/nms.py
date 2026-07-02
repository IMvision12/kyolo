"""Non-maximum suppression and NMS-free top-k selection - pure ``keras.ops``.

The greedy NMS is unrolled for a fixed ``max_detections`` iterations and is
fully vectorized across the batch, so it traces on TensorFlow, JAX and PyTorch
without any backend-native NMS op. Outputs are fixed-size ``(B, max_det, 6)``
tensors ``[x1, y1, x2, y2, score, class]`` zero-padded to ``max_det``.
"""

from __future__ import annotations

import keras
from keras import ops

from ..ops.boxes import xywh2xyxy

__all__ = ["batched_nms", "top_k_detections", "detections_to_list", "NonMaxSuppression"]


def _iou_one_to_many(a, b, eps=1e-7):
    """IoU between ``a`` (B,1,4) and ``b`` (B,N,4) -> (B,N)."""
    x1 = ops.maximum(a[..., 0], b[..., 0])
    y1 = ops.maximum(a[..., 1], b[..., 1])
    x2 = ops.minimum(a[..., 2], b[..., 2])
    y2 = ops.minimum(a[..., 3], b[..., 3])
    # Lower-bound-only clamp via maximum: ops.clip with a None bound is not
    # portable across backends (fails on TensorFlow and PyTorch).
    iw = ops.maximum(x2 - x1, 0.0)
    ih = ops.maximum(y2 - y1, 0.0)
    inter = iw * ih
    area_a = ops.maximum(a[..., 2] - a[..., 0], 0.0) * ops.maximum(a[..., 3] - a[..., 1], 0.0)
    area_b = ops.maximum(b[..., 2] - b[..., 0], 0.0) * ops.maximum(b[..., 3] - b[..., 1], 0.0)
    return inter / (area_a + area_b - inter + eps)


def batched_nms(boxes, scores, classes, iou_threshold=0.7, max_detections=300, conf_threshold=0.25):
    """Greedy class-aware NMS.

    Args:
        boxes: ``(B, N, 4)`` xyxy pixel boxes.
        scores: ``(B, N)`` per-anchor confidence (max class score).
        classes: ``(B, N)`` integer class ids.
        iou_threshold / max_detections / conf_threshold: NMS params.

    Returns:
        ``(B, max_detections, 6)`` = ``[x1, y1, x2, y2, score, class]`` padded.
    """
    boxes = ops.cast(boxes, "float32")
    scores = ops.cast(scores, "float32")
    classes_f = ops.cast(classes, "float32")
    n = ops.shape(boxes)[1]

    # class-aware coordinate offset so boxes of different classes never overlap
    max_coord = ops.max(boxes) + 1.0
    off = ops.expand_dims(classes_f * max_coord, -1)  # (B,N,1)
    off_boxes = boxes + off

    suppressed = scores <= conf_threshold  # (B,N) bool

    dets = []
    for _ in range(max_detections):
        cur = ops.where(suppressed, ops.full_like(scores, -1.0), scores)  # (B,N)
        best = ops.argmax(cur, axis=1)  # (B,)
        best_score = ops.max(cur, axis=1)  # (B,)
        valid = ops.cast(best_score > conf_threshold, "float32")  # (B,)

        idx = ops.expand_dims(best, -1)  # (B,1)
        idx4 = ops.broadcast_to(ops.expand_dims(idx, -1), (ops.shape(boxes)[0], 1, 4))
        best_box = ops.take_along_axis(boxes, idx4, axis=1)  # (B,1,4)
        best_off = ops.take_along_axis(off_boxes, idx4, axis=1)  # (B,1,4)
        best_cls = ops.take_along_axis(classes_f, idx, axis=1)  # (B,1)

        det = ops.concatenate(
            [
                ops.squeeze(best_box, 1),  # (B,4)
                ops.expand_dims(best_score, -1),  # (B,1)
                best_cls,  # (B,1)
            ],
            axis=-1,
        )  # (B,6)
        det = det * ops.expand_dims(valid, -1)
        dets.append(det)

        iou = _iou_one_to_many(best_off, off_boxes)  # (B,N)
        onehot = ops.cast(ops.one_hot(best, n), "bool")  # (B,N)
        suppress_now = ops.logical_or(onehot, iou > iou_threshold)
        valid_b = ops.cast(ops.expand_dims(valid, -1) > 0, "bool")
        suppressed = ops.logical_or(suppressed, ops.logical_and(suppress_now, valid_b))

    return ops.stack(dets, axis=1)  # (B, max_det, 6)


def top_k_detections(boxes, scores, max_detections=300, conf_threshold=0.0):
    """NMS-free selection for end-to-end models (YOLOv10 / YOLO26).

    Selects the top ``max_detections`` (anchor, class) pairs by score.

    Args:
        boxes: ``(B, A, 4)`` xyxy pixel boxes.
        scores: ``(B, A, nc)`` per-class probabilities.
        max_detections / conf_threshold.

    Returns:
        ``(B, max_detections, 6)`` = ``[x1, y1, x2, y2, score, class]``.
    """
    b = ops.shape(scores)[0]
    a = ops.shape(scores)[1]
    nc = ops.shape(scores)[2]
    flat = ops.reshape(scores, (b, a * nc))  # (B, A*nc)
    top_scores, top_idx = ops.top_k(flat, k=max_detections)
    anchor_idx = top_idx // nc  # (B, k)
    class_idx = top_idx % nc

    idx4 = ops.broadcast_to(ops.expand_dims(anchor_idx, -1), (b, max_detections, 4))
    sel_boxes = ops.take_along_axis(boxes, idx4, axis=1)  # (B,k,4)

    valid = ops.cast(top_scores > conf_threshold, "float32")
    det = ops.concatenate(
        [
            sel_boxes,
            ops.expand_dims(top_scores, -1),
            ops.expand_dims(ops.cast(class_idx, "float32"), -1),
        ],
        axis=-1,
    )
    return det * ops.expand_dims(valid, -1)


def detections_to_list(detections, score_threshold=1e-6):
    """Convert a padded ``(B, max_det, 6)`` tensor to a list of ``(n_i, 6)`` arrays.

    Rows with score below ``score_threshold`` (i.e. padding) are dropped. Runs
    eagerly (uses numpy); intended for post-inference consumption.
    """

    dets = ops.convert_to_numpy(detections)
    out = []
    for img in dets:
        keep = img[:, 4] > score_threshold
        out.append(img[keep])
    return out


@keras.saving.register_keras_serializable(package="kyolo")
class NonMaxSuppression(keras.layers.Layer):
    """NMS as a layer. Input ``(B, N, 4 + nc)`` (xywh + class probs)."""

    def __init__(self, nc=80, conf_threshold=0.25, iou_threshold=0.7, max_detections=300, **kwargs):
        super().__init__(**kwargs)
        self.nc = nc
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def call(self, inputs):
        boxes = xywh2xyxy(inputs[..., :4])
        cls = inputs[..., 4 : 4 + self.nc]
        scores = ops.max(cls, axis=-1)
        classes = ops.argmax(cls, axis=-1)
        return batched_nms(
            boxes, scores, classes, self.iou_threshold, self.max_detections, self.conf_threshold
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nc": self.nc,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_detections": self.max_detections,
            }
        )
        return config
