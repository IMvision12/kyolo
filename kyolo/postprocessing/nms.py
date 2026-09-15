"""Non-maximum suppression and NMS-free top-k selection - pure ``keras.ops``.

The greedy NMS keeps the ``pre_nms_topk`` highest-scoring candidates of every
image (sorted), computes their pairwise class-aware overlaps once, and then
sweeps them in score order inside a single ``keras.ops.while_loop``: candidate
``i``, if still kept, suppresses every lower-ranked box it overlaps. The loop
body is three cheap ``(B, K)`` ops and the loop stops as soon as the
above-threshold prefix has been visited, so the cost scales with the number of
real candidates rather than with ``max_detections`` x anchors. It is exact
greedy NMS, vectorised across the batch, needs no backend-native NMS op, and
traces as one compact loop node (rather than an unrolled graph) on TensorFlow,
JAX and PyTorch. Outputs are fixed-size ``(B, max_det, 6)`` tensors
``[x1, y1, x2, y2, score, class]`` zero-padded to ``max_det``.
"""

from __future__ import annotations

import keras
from keras import ops

from ..ops.boxes import xywh2xyxy

__all__ = ["batched_nms", "top_k_detections", "detections_to_list", "NonMaxSuppression"]

# Score given to candidates that only exist to pad a dynamically-shaped batch up
# to ``k`` columns. It must fail ``scores > conf_threshold`` for every sane
# threshold; ``-inf`` is only ever compared or replaced, never used in
# arithmetic, so it cannot produce a NaN.
_PAD_SCORE = float("-inf")


def _pairwise_iou(boxes, eps=1e-7):
    """IoU matrix of xyxy ``boxes`` (B,K,4) -> (B,K,K)."""
    b1 = ops.expand_dims(boxes, 2)  # (B,K,1,4)
    b2 = ops.expand_dims(boxes, 1)  # (B,1,K,4)
    # Lower-bound-only clamp via maximum: ops.clip with a None bound is not
    # portable across backends (fails on TensorFlow and PyTorch).
    iw = ops.maximum(ops.minimum(b1[..., 2], b2[..., 2]) - ops.maximum(b1[..., 0], b2[..., 0]), 0.0)
    ih = ops.maximum(ops.minimum(b1[..., 3], b2[..., 3]) - ops.maximum(b1[..., 1], b2[..., 1]), 0.0)
    inter = iw * ih  # (B,K,K)
    area = ops.maximum(boxes[..., 2] - boxes[..., 0], 0.0) * ops.maximum(
        boxes[..., 3] - boxes[..., 1], 0.0
    )  # (B,K)
    union = ops.expand_dims(area, 2) + ops.expand_dims(area, 1) - inter + eps
    return inter / union


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
    pre_nms_topk=1000,
):
    """Greedy class-aware NMS.

    Args:
        boxes: ``(B, N, 4)`` xyxy pixel boxes.
        scores: ``(B, N)`` per-anchor confidence (max class score).
        classes: ``(B, N)`` integer class ids.
        iou_threshold / max_detections / conf_threshold: NMS params.
        pre_nms_topk: at most this many of the highest-scoring boxes per image
            enter NMS (``max_detections`` at least; capped at ``N`` when ``N`` is
            statically known). Bounds the ``(B, K, K)`` overlap matrix and the
            sweep length. 1000 is plenty for ``max_detections=300`` at ordinary
            thresholds; raise it when you evaluate with a very low
            ``conf_threshold`` and crowded images.

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
        # N is unknown at trace time (a tf.function / SavedModel with a dynamic
        # input signature), so it cannot be clamped against. ``ops.top_k`` needs
        # a static ``k`` and fails outright when handed fewer than ``k`` columns
        # ("input must have at least k columns"), so pad by a static ``k``
        # instead: ``N + k >= k`` holds for every N. The padded candidates score
        # ``-inf`` so they fail the ``valid`` threshold below and are discarded
        # by the final compaction, and their boxes are zeros, so they have zero
        # area and can never suppress or be suppressed.
        scores = ops.pad(scores, [[0, 0], [0, k]], constant_values=_PAD_SCORE)
        boxes = ops.pad(boxes, [[0, 0], [0, k], [0, 0]])
        classes_f = ops.pad(classes_f, [[0, 0], [0, k]])

    # Highest-scoring K candidates per image, sorted, so the sweep visits them
    # in NMS order and the survivors come out already ranked.
    scores, order = ops.top_k(scores, k=k)  # (B,K) descending
    classes_f = ops.take_along_axis(classes_f, order, axis=1)
    boxes = _gather_rows(boxes, order)
    valid = scores > conf_threshold  # (B,K) bool, a prefix of every row

    # Class-aware coordinate offset so boxes of different classes never overlap.
    # The span (not just the max) keeps classes apart with negative coordinates.
    span = ops.max(boxes) - ops.min(boxes) + 1.0
    off_boxes = boxes + ops.expand_dims(classes_f * span, -1)

    # overlap[b, i, j]: candidate i suppresses candidate j if i is kept. Only
    # lower-ranked boxes (j > i) can be suppressed, never the box itself.
    positions = ops.arange(k, dtype="int32")
    later = ops.expand_dims(positions, 0) > ops.expand_dims(positions, 1)  # (K,K): j > i
    overlap = ops.logical_and(_pairwise_iou(off_boxes) > iou_threshold, ops.expand_dims(later, 0))

    # Only the above-threshold prefix needs to be swept: everything after it is
    # neither kept nor able to suppress anything.
    n_iter = ops.max(ops.sum(ops.cast(valid, "int32"), axis=1))

    def cond(i, keep):
        return i < n_iter

    def body(i, keep):
        row = ops.take(overlap, i, axis=1)  # (B,K): what candidate i would suppress
        kept_i = ops.take(keep, i, axis=1)  # (B,)
        suppress = ops.logical_and(row, ops.expand_dims(kept_i, -1))
        return i + 1, ops.logical_and(keep, ops.logical_not(suppress))

    _, keep = ops.while_loop(cond, body, (ops.convert_to_tensor(0, dtype="int32"), valid))

    # Compact the survivors (already in descending-score order) to the front.
    kept_scores = ops.where(keep, scores, -1.0)
    if k < max_detections:
        pad = max_detections - k
        kept_scores = ops.pad(kept_scores, [[0, 0], [0, pad]], constant_values=-1.0)
        boxes = ops.pad(boxes, [[0, 0], [0, pad], [0, 0]])
        classes_f = ops.pad(classes_f, [[0, 0], [0, pad]])
    top_scores, idx = ops.top_k(kept_scores, k=max_detections)  # (B,max_det)
    sel_boxes = _gather_rows(boxes, idx)  # (B,max_det,4)
    sel_cls = ops.take_along_axis(classes_f, idx, axis=1)  # (B,max_det)
    valid_out = ops.cast(top_scores >= 0.0, "float32")  # suppressed / padding rows carry -1

    det = ops.concatenate(
        [
            sel_boxes,
            ops.expand_dims(top_scores, -1),
            ops.expand_dims(sel_cls, -1),
        ],
        axis=-1,
    )  # (B, max_det, 6)
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

    def __init__(
        self,
        nc=80,
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=300,
        pre_nms_topk=1000,
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
        # Declared explicitly so symbolic (functional-model) use never has to
        # trace the data-dependent while_loop inside ``call``.
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
