"""Single-branch YOLO detection loss (Ultralytics ``v8DetectionLoss`` parity).

Consumes the raw head feature list and a target dict; returns the total loss
and its components. Written densely in ``keras.ops`` so it differentiates and
traces on every backend.

The criterion is the same three terms for every YOLO family, assembled by
:class:`~kyolo.losses.BboxLoss` and this class:

* **cls** -- BCE-with-logits against the assigner's soft (alignment-scaled)
  target scores. Not focal or varifocal loss: Ultralytics uses plain BCE, and
  the soft targets are what carry the quality signal.
* **box** -- ``1 - CIoU`` on the assigned anchors.
* **dist** -- a second regression term whose form follows the box
  parameterization. With a distributional box branch (``reg_max > 1``:
  v5u/v8/v9/v10/11/12) it is the Distribution Focal Loss. With a direct box
  branch (``reg_max == 1``: YOLO26) there are no bins to classify, so it is an
  L1 on the LTRB distances normalized by image size. It is reported as ``"dfl"``
  or ``"l1"`` accordingly, and always under the stable key ``"dist"``.

All three are normalized by ``target_scores_sum`` and weighted by the assigned
soft score, so they are batch-size invariant; the returned ``"loss"`` is then
scaled by the batch size, matching Ultralytics' ``loss.sum() * batch_size``.
The ``"box"`` / ``"cls"`` / ``"dist"`` entries stay un-scaled, for reporting.

The loss always runs in float32, so it is safe under ``mixed_float16`` /
``mixed_bfloat16`` regardless of the dtype the head emits.
"""

from __future__ import annotations

from keras import ops

from ..layers.common import resolve_data_format
from ..ops.anchors import dist2bbox, make_anchors
from ..ops.tal import TaskAlignedAssigner
from .bbox_loss import BboxLoss

__all__ = ["YOLODetectionLoss"]


def _bce_with_logits(logits, targets):
    """Numerically-stable elementwise binary cross-entropy from logits."""
    return ops.maximum(logits, 0.0) - logits * targets + ops.log1p(ops.exp(-ops.abs(logits)))


class YOLODetectionLoss:
    """Task-aligned detection loss for a single (one-to-many) head.

    Args:
        nc: number of classes.
        reg_max: box-branch bin count. ``1`` means a direct box branch, which
            selects the normalized-L1 distance term instead of DFL.
        strides: per-level strides matching the model outputs.
        box_gain / cls_gain / dfl_gain: loss-term weights. ``dfl_gain`` weights
            whichever distance term is active, DFL or L1 -- Ultralytics reuses
            the single ``dfl`` hyperparameter for both.
        tal_topk: TAL candidate count per GT.
        tal_topk2: optional tighter second top-k in the assigner; ``1`` gives
            one-to-one assignment. ``None`` disables it.
        class_weights: optional ``(nc,)`` per-class multiplier on the
            classification term, for imbalanced datasets.
        use_dfl: force the distance term off even when ``reg_max > 1``.
        data_format: layout of the head feature maps ("channels_last",
            "channels_first", or None for the global Keras config).
    """

    def __init__(
        self,
        nc=80,
        reg_max=16,
        strides=(8, 16, 32),
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        tal_topk=10,
        tal_topk2=None,
        class_weights=None,
        use_dfl=True,
        data_format=None,
    ):
        self.nc = nc
        self.reg_max = reg_max
        self.strides = tuple(strides)
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.tal_topk = tal_topk
        self.tal_topk2 = tal_topk2
        self.data_format = resolve_data_format(data_format)
        self.use_dfl = use_dfl and reg_max > 1
        self.no = 4 * reg_max + nc if self.use_dfl else 4 + nc
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=nc,
            alpha=0.5,
            beta=6.0,
            strides=self.strides,
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(reg_max if self.use_dfl else 1)
        self.proj = ops.arange(reg_max, dtype="float32")
        self.class_weights = (
            None
            if class_weights is None
            else ops.reshape(ops.cast(ops.convert_to_tensor(class_weights), "float32"), (1, 1, -1))
        )

    @property
    def dist_name(self):
        """Reporting name of the distance term: ``"dfl"`` or ``"l1"``."""
        return "dfl" if self.use_dfl else "l1"

    def __call__(self, feats, targets):
        return self.compute(feats, targets)

    def _dfl_decode(self, pred_dist):
        """(B, A, 4*reg_max) logits -> (B, A, 4) expected distances."""
        b = ops.shape(pred_dist)[0]
        a = ops.shape(pred_dist)[1]
        x = ops.reshape(pred_dist, (b, a, 4, self.reg_max))
        x = ops.softmax(x, axis=-1)

        proj = ops.cast(ops.reshape(self.proj, (1, 1, 1, self.reg_max)), x.dtype)
        return ops.sum(x * proj, axis=-1)

    def _flatten(self, feats):
        """Head feature maps -> ``((B, A, no)``, per-level ``(H, W)`` shapes)."""
        shapes = []
        flat = []
        for f in feats:
            if self.data_format == "channels_first":
                f = ops.transpose(f, (0, 2, 3, 1))
            h = f.shape[1] if f.shape[1] is not None else ops.shape(f)[1]
            w = f.shape[2] if f.shape[2] is not None else ops.shape(f)[2]
            shapes.append((h, w))
            b = ops.shape(f)[0]
            flat.append(ops.reshape(f, (b, h * w, self.no)))

        return ops.cast(ops.concatenate(flat, axis=1), "float32"), shapes

    def _image_size(self, shapes):
        """Input ``(height, width)`` in pixels, from the finest level's grid."""
        h, w = shapes[0]
        stride = float(self.strides[0])
        return ops.stack(
            [
                ops.cast(h, "float32") * stride,
                ops.cast(w, "float32") * stride,
            ]
        )

    def compute(self, feats, targets):
        gt_bboxes = ops.cast(targets["boxes"], "float32")
        gt_labels = ops.cast(targets["labels"], "int32")
        mask = ops.cast(targets["mask"], "float32")

        x, shapes = self._flatten(feats)
        pred_dist = x[..., : self.no - self.nc]
        pred_scores = x[..., self.no - self.nc :]
        imgsz = self._image_size(shapes)

        anchors, stride_t = make_anchors(shapes, self.strides)
        anchors = ops.cast(anchors, "float32")
        stride_t = ops.cast(stride_t, "float32")

        if self.use_dfl:
            dist = self._dfl_decode(pred_dist)
        else:
            dist = pred_dist
        anchors_b = ops.expand_dims(anchors, 0)
        pred_bboxes = dist2bbox(dist, anchors_b, xywh=False, axis=-1)

        stride_row = ops.reshape(stride_t, (1, -1, 1))
        pred_bboxes_px = pred_bboxes * stride_row
        anchors_px = anchors * stride_t

        _, target_bboxes_px, target_scores, fg_mask = self.assigner(
            ops.sigmoid(pred_scores),
            pred_bboxes_px,
            anchors_px,
            gt_labels,
            gt_bboxes,
            mask,
        )
        target_bboxes = target_bboxes_px / stride_row
        fg_mask = ops.cast(fg_mask > 0, "float32")

        target_scores_sum = ops.maximum(ops.sum(target_scores), 1.0)

        cls = _bce_with_logits(pred_scores, target_scores)
        if self.class_weights is not None:
            cls = cls * self.class_weights
        cls_loss = ops.sum(cls) / target_scores_sum

        box_loss, dist_loss = self.bbox_loss(
            pred_dist,
            pred_bboxes,
            anchors_b,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_row,
        )

        batch_size = ops.cast(ops.shape(pred_scores)[0], "float32")
        total = batch_size * (
            self.box_gain * box_loss + self.cls_gain * cls_loss + self.dfl_gain * dist_loss
        )

        losses = {
            "loss": total,
            "box": box_loss,
            "cls": cls_loss,
            "dist": dist_loss,
        }
        losses[self.dist_name] = dist_loss
        return losses
