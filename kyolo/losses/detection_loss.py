"""YOLO detection loss (v8-style: TAL assignment + CIoU + DFL + BCE).

Consumes the raw head feature list and a target dict; returns the total loss
and its components. Written densely in ``keras.ops`` so it differentiates and
traces on every backend.

The returned ``"loss"`` is scaled by the batch size, matching Ultralytics'
``loss.sum() * batch_size``; the ``"box"`` / ``"cls"`` / ``"dfl"`` entries are
the un-scaled per-term values used for reporting.
"""

from __future__ import annotations

from keras import ops

from ..layers.common import resolve_data_format
from ..ops.anchors import bbox2dist, dist2bbox, make_anchors
from ..ops.boxes import bbox_iou
from ..ops.tal import TaskAlignedAssigner

__all__ = ["YOLODetectionLoss"]


def _bce_with_logits(logits, targets):
    """Numerically-stable elementwise binary cross-entropy from logits."""
    return ops.maximum(logits, 0.0) - logits * targets + ops.log1p(ops.exp(-ops.abs(logits)))


class YOLODetectionLoss:
    """Task-aligned detection loss.

    Args:
        nc: number of classes.
        reg_max: DFL bin count (``use_dfl`` implied by ``reg_max > 1``).
        strides: per-level strides matching the model outputs.
        box_gain / cls_gain / dfl_gain: loss-term weights.
        tal_topk: TAL candidate count per GT.
        use_dfl: whether the box branch is DFL-distributional.
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
        use_dfl=True,
        data_format=None,
    ):
        self.nc = nc
        self.reg_max = reg_max
        self.strides = tuple(strides)
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.data_format = resolve_data_format(data_format)
        self.use_dfl = use_dfl and reg_max > 1
        self.no = 4 * reg_max + nc if self.use_dfl else 4 + nc
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=nc, alpha=0.5, beta=6.0)
        self.proj = ops.arange(reg_max, dtype="float32")

    def __call__(self, feats, targets):
        return self.compute(feats, targets)

    # ------------------------------------------------------------------ #
    def _dfl_decode(self, pred_dist):
        """(B, A, 4*reg_max) logits -> (B, A, 4) expected distances."""
        b = ops.shape(pred_dist)[0]
        a = ops.shape(pred_dist)[1]
        x = ops.reshape(pred_dist, (b, a, 4, self.reg_max))
        x = ops.softmax(x, axis=-1)
        proj = ops.reshape(self.proj, (1, 1, 1, self.reg_max))
        return ops.sum(x * proj, axis=-1)

    def _df_loss(self, pred_dist, target):
        """Distribution focal loss. pred_dist (B,A,4,reg_max), target (B,A,4)."""
        logp = ops.log_softmax(pred_dist, axis=-1)
        tl = ops.cast(ops.floor(target), "int32")
        tl = ops.clip(tl, 0, self.reg_max - 1)
        tr = ops.clip(tl + 1, 0, self.reg_max - 1)
        wl = ops.cast(tr, "float32") - target
        wr = 1.0 - wl
        oh_l = ops.cast(ops.one_hot(tl, self.reg_max), "float32")
        oh_r = ops.cast(ops.one_hot(tr, self.reg_max), "float32")
        ce_l = -ops.sum(logp * oh_l, axis=-1)
        ce_r = -ops.sum(logp * oh_r, axis=-1)
        return ops.mean(ce_l * wl + ce_r * wr, axis=-1)  # (B, A)

    def compute(self, feats, targets):
        gt_bboxes = ops.cast(targets["boxes"], "float32")  # (B,M,4) xyxy pixels
        gt_labels = ops.cast(targets["labels"], "int32")  # (B,M)
        mask = ops.cast(targets["mask"], "float32")  # (B,M)

        # --- flatten head outputs -> (B, A, no) ---
        shapes = []
        flat = []
        for f in feats:
            if self.data_format == "channels_first":
                f = ops.transpose(f, (0, 2, 3, 1))  # (B,C,H,W) -> (B,H,W,C)
            h = f.shape[1] if f.shape[1] is not None else ops.shape(f)[1]
            w = f.shape[2] if f.shape[2] is not None else ops.shape(f)[2]
            shapes.append((h, w))
            b = ops.shape(f)[0]
            flat.append(ops.reshape(f, (b, h * w, self.no)))
        x = ops.concatenate(flat, axis=1)  # (B, A, no)
        pred_dist = x[..., : self.no - self.nc]
        pred_scores = x[..., self.no - self.nc :]

        anchors, stride_t = make_anchors(shapes, self.strides)  # (A,2),(A,1) grid units
        anchors = ops.cast(anchors, "float32")
        stride_t = ops.cast(stride_t, "float32")

        if self.use_dfl:
            dist = self._dfl_decode(pred_dist)  # (B,A,4)
        else:
            dist = pred_dist  # already 4 distances
        anchors_b = ops.expand_dims(anchors, 0)  # (1,A,2)
        pred_bboxes = dist2bbox(dist, anchors_b, xywh=False, axis=-1)  # (B,A,4) grid units

        stride_row = ops.reshape(stride_t, (1, -1, 1))  # (1,A,1)
        pred_bboxes_px = pred_bboxes * stride_row
        anchors_px = anchors * stride_t  # (A,2)

        # The assigner detaches both predictions internally (it mirrors the
        # @torch.no_grad() Ultralytics assigner), so the returned targets are
        # constants w.r.t. the network outputs.
        target_labels, target_bboxes_px, target_scores, fg_mask = self.assigner(
            ops.sigmoid(pred_scores),
            pred_bboxes_px,
            anchors_px,
            gt_labels,
            gt_bboxes,
            mask,
        )
        target_bboxes = target_bboxes_px / stride_row  # back to grid units

        target_scores_sum = ops.maximum(ops.sum(target_scores), 1.0)

        # --- classification (BCE over all anchors) ---
        cls_loss = ops.sum(_bce_with_logits(pred_scores, target_scores)) / target_scores_sum

        # --- box (CIoU) weighted by the assigned soft score ---
        weight = ops.sum(target_scores, axis=-1)  # (B,A); 0 for background
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, ciou=True)  # (B,A)
        box_loss = ops.sum((1.0 - iou) * weight) / target_scores_sum

        # --- DFL ---
        if self.use_dfl:
            b = ops.shape(pred_dist)[0]
            a = ops.shape(pred_dist)[1]
            pd = ops.reshape(pred_dist, (b, a, 4, self.reg_max))
            tgt_dist = bbox2dist(anchors_b, target_bboxes, self.reg_max)  # (B,A,4)
            dfl = self._df_loss(pd, tgt_dist)  # (B,A)
            dfl_loss = ops.sum(dfl * weight) / target_scores_sum
        else:
            dfl_loss = ops.convert_to_tensor(0.0)

        # Ultralytics' ``v8DetectionLoss`` ends with ``loss.sum() * batch_size``.
        # Each term above is already normalized by ``target_scores_sum``, so the
        # components are batch-size invariant; this factor is what makes the
        # gradient magnitude (and therefore any learning rate copied from an
        # Ultralytics recipe) scale the same way. The reported components stay
        # un-scaled, mirroring the un-scaled ``loss.detach()`` returned there.
        batch_size = ops.cast(ops.shape(pred_scores)[0], "float32")
        total = batch_size * (
            self.box_gain * box_loss + self.cls_gain * cls_loss + self.dfl_gain * dfl_loss
        )
        return {
            "loss": total,
            "box": box_loss,
            "cls": cls_loss,
            "dfl": dfl_loss,
        }
