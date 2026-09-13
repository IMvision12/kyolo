"""Task-Aligned Assigner (TAL) in pure ``keras.ops``.

This is the label-assignment strategy used by the anchor-free YOLO detectors
(v5u/v8/v9/v10/11/12/26). It selects, for every ground-truth box, the ``topk``
anchors whose *alignment metric* ``score^alpha * iou^beta`` is highest, resolves
anchors claimed by several GTs in favour of the highest-IoU GT, and returns
soft classification targets scaled by the (normalized) alignment metric.

The implementation is fully dense (no boolean-index assignment) so it traces
cleanly on TensorFlow, JAX and PyTorch backends.
"""

from __future__ import annotations

from keras import ops

from .boxes import bbox_iou

__all__ = ["TaskAlignedAssigner"]


class TaskAlignedAssigner:
    """Callable assigner. See module docstring.

    Args:
        topk: candidate anchors kept per GT.
        num_classes: number of classes ``nc``.
        alpha: classification exponent in the alignment metric.
        beta: localization (IoU) exponent in the alignment metric.
        eps: numerical epsilon.
    """

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        self.topk = topk
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Assign targets.

        Args:
            pd_scores: ``(B, A, nc)`` predicted class probabilities (post-sigmoid).
            pd_bboxes: ``(B, A, 4)`` predicted boxes, xyxy, in the same units as
                ``anc_points`` (feature-grid units).
            anc_points: ``(A, 2)`` anchor centres (x, y) in grid units.
            gt_labels: ``(B, M)`` integer class ids (padding may be anything).
            gt_bboxes: ``(B, M, 4)`` GT boxes, xyxy, grid units.
            mask_gt: ``(B, M)`` 1.0 for real GTs, 0.0 for padding.

        Returns:
            ``(target_labels, target_bboxes, target_scores, fg_mask)``:
              * ``target_labels``: ``(B, A)`` int.
              * ``target_bboxes``: ``(B, A, 4)`` xyxy grid units.
              * ``target_scores``: ``(B, A, nc)`` soft one-hot * alignment.
              * ``fg_mask``: ``(B, A)`` float, 1.0 for positive anchors.

        The assignment is a pure target-construction step: like Ultralytics'
        ``@torch.no_grad()`` assigner, no gradient flows from the returned
        targets back into the predictions (the alignment metric depends on the
        predicted scores, so without this the BCE targets would be
        differentiable w.r.t. the class logits they supervise).
        """
        pd_scores = ops.stop_gradient(pd_scores)
        pd_bboxes = ops.stop_gradient(pd_bboxes)
        mask_gt_bn = ops.expand_dims(ops.cast(mask_gt, "float32"), -1)  # (B,M,1)

        # ---- candidates whose centre falls inside each GT box -------------
        in_gts = self._in_gt_mask(anc_points, gt_bboxes)  # (B,M,A)

        # ---- alignment metric + overlaps ----------------------------------
        align_metric, overlaps = self._box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes
        )  # (B,M,A), (B,M,A)
        valid = in_gts * ops.cast(mask_gt_bn, "float32")  # (B,M,A) via broadcast
        align_metric = align_metric * valid
        overlaps = overlaps * valid

        # ---- top-k candidates per GT --------------------------------------
        mask_topk = self._topk_mask(align_metric)  # (B,M,A)
        mask_pos = mask_topk * valid  # (B,M,A)

        # ---- resolve anchors matched to several GTs (keep max IoU) --------
        target_gt_idx, fg_mask, mask_pos = self._resolve_conflicts(mask_pos, overlaps)

        # ---- gather targets ------------------------------------------------
        target_labels, target_bboxes, target_scores = self._gather_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # ---- normalize soft scores by the alignment metric ----------------
        align_metric = align_metric * mask_pos
        pos_align = ops.max(align_metric, axis=-1, keepdims=True)  # (B,M,1)
        pos_overlap = ops.max(overlaps * mask_pos, axis=-1, keepdims=True)  # (B,M,1)
        norm = align_metric * pos_overlap / (pos_align + self.eps)  # (B,M,A)
        norm = ops.max(norm, axis=1)  # (B,A)
        target_scores = target_scores * ops.expand_dims(norm, -1)

        return target_labels, target_bboxes, target_scores, fg_mask

    # ------------------------------------------------------------------ #
    def _in_gt_mask(self, anc_points, gt_bboxes, eps=1e-9):
        # anc_points (A,2) -> (1,1,A,2)
        xy = ops.reshape(anc_points, (1, 1, -1, 2))
        lt = ops.expand_dims(gt_bboxes[..., :2], 2)  # (B,M,1,2)
        rb = ops.expand_dims(gt_bboxes[..., 2:4], 2)  # (B,M,1,2)
        deltas = ops.concatenate([xy - lt, rb - xy], axis=-1)  # (B,M,A,4)
        return ops.cast(ops.min(deltas, axis=-1) > eps, "float32")  # (B,M,A)

    def _box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        # class scores of each GT's class: (B,M,A)
        onehot = ops.one_hot(ops.cast(ops.maximum(gt_labels, 0), "int32"), self.nc)
        onehot = ops.cast(onehot, pd_scores.dtype)  # (B,M,nc)
        bbox_scores = ops.matmul(onehot, ops.transpose(pd_scores, (0, 2, 1)))  # (B,M,A)

        gt = ops.expand_dims(gt_bboxes, 2)  # (B,M,1,4)
        pd = ops.expand_dims(pd_bboxes, 1)  # (B,1,A,4)
        overlaps = bbox_iou(gt, pd, xywh=False, ciou=True)  # (B,M,A)
        overlaps = ops.clip(overlaps, 0.0, 1.0)

        align_metric = ops.power(bbox_scores, self.alpha) * ops.power(overlaps, self.beta)
        return align_metric, overlaps

    def _topk_mask(self, metrics):
        # metrics (B,M,A); pick topk over anchors
        _, idx = ops.top_k(metrics, k=self.topk)  # (B,M,topk)
        a = ops.shape(metrics)[-1]
        onehot = ops.one_hot(idx, a)  # (B,M,topk,A)
        mask = ops.sum(onehot, axis=2)  # (B,M,A)
        mask = ops.cast(mask > 0, "float32")
        # drop candidates with (near) zero metric
        mask = mask * ops.cast(metrics > self.eps, "float32")
        return mask

    def _resolve_conflicts(self, mask_pos, overlaps):
        fg = ops.sum(mask_pos, axis=1)  # (B,A)
        m = ops.shape(mask_pos)[1]
        # anchors claimed by >1 GT -> keep the GT with highest overlap
        max_idx = ops.argmax(overlaps, axis=1)  # (B,A)
        is_max = ops.transpose(ops.one_hot(max_idx, m), (0, 2, 1))  # (B,M,A)
        multi = ops.cast(ops.expand_dims(fg, 1) > 1, "float32")  # (B,1,A)
        mask_pos = multi * is_max + (1.0 - multi) * mask_pos
        fg = ops.sum(mask_pos, axis=1)  # (B,A)
        target_gt_idx = ops.argmax(mask_pos, axis=1)  # (B,A)
        return target_gt_idx, fg, mask_pos

    def _gather_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        idx = ops.cast(target_gt_idx, "int32")  # (B,A)
        target_labels = ops.take_along_axis(gt_labels, idx, axis=1)  # (B,A)
        target_labels = ops.cast(target_labels, "int32")

        idx_b = ops.expand_dims(idx, -1)  # (B,A,1)
        idx_b = ops.broadcast_to(idx_b, (ops.shape(idx)[0], ops.shape(idx)[1], 4))
        target_bboxes = ops.take_along_axis(gt_bboxes, idx_b, axis=1)  # (B,A,4)

        onehot = ops.one_hot(ops.maximum(target_labels, 0), self.nc)  # (B,A,nc)
        onehot = ops.cast(onehot, "float32")
        target_scores = onehot * ops.expand_dims(ops.cast(fg_mask > 0, "float32"), -1)
        return target_labels, target_bboxes, target_scores
