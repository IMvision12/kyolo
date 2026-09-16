"""Non-maximum suppression and NMS-free top-k selection - pure ``keras.ops``.

The greedy NMS keeps the ``pre_nms_topk`` highest-scoring candidates of every
image (sorted) and sweeps them in score order inside a single
``keras.ops.while_loop``: candidate ``i``, if still kept, suppresses every
lower-ranked box it overlaps. It is exact greedy NMS, vectorised across the
batch, needs no backend-native NMS op, and traces as one compact loop node
(rather than an unrolled graph) on TensorFlow, JAX and PyTorch. Outputs are
fixed-size ``(B, max_det, 6)`` tensors ``[x1, y1, x2, y2, score, class]``
zero-padded to ``max_det``.

Each iteration needs only candidate ``i``'s overlaps against the others, so that
single ``(B, K)`` row is computed inside the loop instead of materialising the
whole ``(B, K, K)`` matrix up front. Same arithmetic and the same total work,
but peak memory is ``O(B*K)`` rather than ``O(B*K^2)`` -- which is what makes a
``pre_nms_topk`` large enough to not cost recall affordable at all (a
``(K, K)`` bool matrix alone is 900 MB at ``K=30000``). The sweep also stops as
soon as every image has ``max_detections`` confirmed survivors, so ordinary
inference visits only a handful of candidates no matter how large ``K`` is.
"""

from __future__ import annotations

import keras
from keras import ops

from ..ops.boxes import xywh2xyxy

__all__ = ["batched_nms", "top_k_detections", "detections_to_list", "NonMaxSuppression"]


_PAD_SCORE = float("-inf")


def _box_areas(boxes):
    """Areas of xyxy ``boxes`` (B,K,4) -> (B,K)."""

    return ops.maximum(boxes[..., 2] - boxes[..., 0], 0.0) * ops.maximum(
        boxes[..., 3] - boxes[..., 1], 0.0
    )


def _iou_row(boxes, areas, i, eps=1e-7):
    """IoU of candidate ``i`` against every candidate: (B,K,4) -> (B,K).

    One row of the pairwise matrix. Computing rows on demand keeps peak memory
    at ``O(B*K)``; the arithmetic is elementwise and identical to building the
    full matrix, so the result is bit-for-bit the same.
    """
    b1 = ops.expand_dims(ops.take(boxes, i, axis=1), 1)
    area_i = ops.expand_dims(ops.take(areas, i, axis=1), 1)
    iw = ops.maximum(
        ops.minimum(b1[..., 2], boxes[..., 2]) - ops.maximum(b1[..., 0], boxes[..., 0]), 0.0
    )
    ih = ops.maximum(
        ops.minimum(b1[..., 3], boxes[..., 3]) - ops.maximum(b1[..., 1], boxes[..., 1]), 0.0
    )
    inter = iw * ih
    return inter / (area_i + areas - inter + eps)


def _gather_rows(x, idx):
    """``take_along_axis`` over axis 1 for ``x`` of shape (B,N,C) with ``idx`` (B,K)."""
    b = ops.shape(idx)[0]
    k = ops.shape(idx)[1]
    c = ops.shape(x)[2]
    idx = ops.broadcast_to(ops.expand_dims(idx, -1), (b, k, c))
    return ops.take_along_axis(x, idx, axis=1)


def batched_nms(
    boxes,
    scores,
    classes,
    iou_threshold=0.7,
    max_detections=300,
    conf_threshold=0.25,
    pre_nms_topk=30000,
):
    """Greedy class-aware NMS.

    Args:
        boxes: ``(B, N, 4)`` xyxy pixel boxes.
        scores: ``(B, N)`` per-anchor confidence (max class score).
        classes: ``(B, N)`` integer class ids.
        iou_threshold / max_detections / conf_threshold: NMS params.
        pre_nms_topk: at most this many of the highest-scoring boxes per image
            enter NMS (``max_detections`` at least; capped at ``N`` when ``N`` is
            statically known). The cap applies *before* the sweep, so anything
            ranked beyond it is dropped even if everything above it ends up
            suppressed -- the default matches Ultralytics' ``max_nms=30000``,
            which is effectively no cap at the usual anchor counts (8400 at
            640px). Cost is ``O(min(n_above_threshold, max_detections) * K)``, so
            a larger ``K`` makes each sweep step proportionally wider: measured
            ~1.6x total NMS time at a realistic ``conf_threshold=0.25`` and up to
            ~5x for low-threshold evaluation sweeps. Lower it only if you need
            that time back and accept losing detections ranked beyond it.

    Returns:
        ``(B, max_detections, 6)`` = ``[x1, y1, x2, y2, score, class]`` padded.
        Rows are sorted by descending score; padding rows are all zeros.

    Works with a dynamic ``N`` (unknown at trace time), which is the usual case
    inside a ``tf.function`` or an exported SavedModel; the result is identical
    to the statically-shaped call.
    """
    boxes = ops.cast(boxes, "float32")
    scores = ops.cast(scores, "float32")
    classes_f = ops.cast(classes, "float32")
    n_static = boxes.shape[1]
    k = max(int(pre_nms_topk), int(max_detections))
    if n_static is not None:
        k = min(k, n_static)
    else:
        scores = ops.pad(scores, [[0, 0], [0, k]], constant_values=_PAD_SCORE)
        boxes = ops.pad(boxes, [[0, 0], [0, k], [0, 0]])
        classes_f = ops.pad(classes_f, [[0, 0], [0, k]])

    scores, order = ops.top_k(scores, k=k)
    classes_f = ops.take_along_axis(classes_f, order, axis=1)
    boxes = _gather_rows(boxes, order)
    valid = scores > conf_threshold

    span = ops.max(boxes) - ops.min(boxes) + 1.0
    off_boxes = boxes + ops.expand_dims(classes_f * span, -1)
    areas = _box_areas(off_boxes)
    positions = ops.arange(k, dtype="int32")

    n_iter = ops.max(ops.sum(ops.cast(valid, "int32"), axis=1))
    zero = ops.convert_to_tensor(0, dtype="int32")
    n_kept = ops.zeros_like(ops.sum(ops.cast(valid, "int32"), axis=1))

    def cond(i, keep, n_kept):

        return ops.logical_and(i < n_iter, ops.min(n_kept) < max_detections)

    def body(i, keep, n_kept):

        kept_i = ops.take(keep, i, axis=1)

        row = ops.logical_and(
            _iou_row(off_boxes, areas, i) > iou_threshold,
            ops.expand_dims(positions > i, 0),
        )
        suppress = ops.logical_and(row, ops.expand_dims(kept_i, -1))
        return (
            i + 1,
            ops.logical_and(keep, ops.logical_not(suppress)),
            n_kept + ops.cast(kept_i, "int32"),
        )

    _, keep, _ = ops.while_loop(cond, body, (zero, valid, n_kept))

    kept_scores = ops.where(keep, scores, -1.0)
    if k < max_detections:
        pad = max_detections - k
        kept_scores = ops.pad(kept_scores, [[0, 0], [0, pad]], constant_values=-1.0)
        boxes = ops.pad(boxes, [[0, 0], [0, pad], [0, 0]])
        classes_f = ops.pad(classes_f, [[0, 0], [0, pad]])
    top_scores, idx = ops.top_k(kept_scores, k=max_detections)
    sel_boxes = _gather_rows(boxes, idx)
    sel_cls = ops.take_along_axis(classes_f, idx, axis=1)
    valid_out = ops.cast(top_scores >= 0.0, "float32")

    det = ops.concatenate(
        [
            sel_boxes,
            ops.expand_dims(top_scores, -1),
            ops.expand_dims(sel_cls, -1),
        ],
        axis=-1,
    )
    return det * ops.expand_dims(valid_out, -1)


def top_k_detections(boxes, scores, max_detections=300, conf_threshold=0.0):
    """NMS-free selection for end-to-end models (YOLOv10 / YOLO26 one2one heads).

    Selects the top ``max_detections`` (anchor, class) pairs by score. Only
    meaningful for a head trained with one-to-one assignment; the heads kyolo
    builds are one-to-many, so decode them with :func:`batched_nms` instead.

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
    flat = ops.reshape(scores, (b, a * nc))
    top_scores, top_idx = ops.top_k(flat, k=max_detections)
    anchor_idx = top_idx // nc
    class_idx = top_idx % nc

    idx4 = ops.broadcast_to(ops.expand_dims(anchor_idx, -1), (b, max_detections, 4))
    sel_boxes = ops.take_along_axis(boxes, idx4, axis=1)

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

    def __init__(
        self,
        nc=80,
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=300,
        pre_nms_topk=30000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nc = nc
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.pre_nms_topk = pre_nms_topk

    def call(self, inputs):
        boxes = xywh2xyxy(inputs[..., :4])
        cls = inputs[..., 4 : 4 + self.nc]
        scores = ops.max(cls, axis=-1)
        classes = ops.argmax(cls, axis=-1)
        return batched_nms(
            boxes,
            scores,
            classes,
            self.iou_threshold,
            self.max_detections,
            self.conf_threshold,
            self.pre_nms_topk,
        )

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.max_detections, 6)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nc": self.nc,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_detections": self.max_detections,
                "pre_nms_topk": self.pre_nms_topk,
            }
        )
        return config
