"""Task-Aligned Assigner (TAL) in pure ``keras.ops``.

This is the label-assignment strategy used by the anchor-free YOLO detectors
(v5u/v8/v9/v10/11/12/26). It selects, for every ground-truth box, the ``topk``
anchors whose *alignment metric* ``score^alpha * iou^beta`` is highest, resolves
anchors claimed by several GTs in favour of the highest-IoU GT, and returns
soft classification targets scaled by the (normalized) alignment metric.

Two refinements from current Ultralytics are included, both stride-aware:

* **Small-target assignment (STAL).** A GT side shorter than the finest stride
  can fall between anchor centres and collect no candidates at all, so the loss
  never supervises it. Before the "is the anchor centre inside the box?" test,
  any GT side below ``strides[0]`` is grown (about the box centre) to
  ``strides[1]``. Only candidate selection sees the grown box; the regression
  target stays the original one.
* **``topk2``.** An optional second, tighter top-k applied *after* multi-GT
  conflict resolution, keeping only the ``topk2`` best-aligned anchors per GT.
  ``topk2=1`` is what makes one-to-one (NMS-free) assignment one-to-one. It
  defaults to ``topk``, which skips the step entirely.

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
        strides: per-level strides, ascending, in the same units as the boxes
            passed to ``__call__`` (pixels). Drives the small-target expansion:
            sides below ``strides[0]`` are grown to ``strides[1]``.
        topk2: optional tighter top-k applied after conflict resolution.
            ``None`` (default) means "same as ``topk``", which is a no-op.
        eps: numerical epsilon.
    """

    def __init__(
        self,
        topk=10,
        num_classes=80,
        alpha=0.5,
        beta=6.0,
        strides=(8, 16, 32),
        topk2=None,
        eps=1e-9,
    ):
        self.topk = topk
        self.nc = num_classes
        self.alpha = alpha
        self.beta = beta
        self.strides = tuple(strides) if strides else (8, 16, 32)
        self.topk2 = topk2 or topk
        self.eps = eps

        self.small_side = float(self.strides[0])
        self.expanded_side = float(self.strides[1] if len(self.strides) > 1 else self.strides[0])

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

        gt_bboxes = ops.cast(gt_bboxes, "float32")
        mask_gt_bn = ops.expand_dims(ops.cast(mask_gt, "float32"), -1)

        in_gts = self._in_gt_mask(anc_points, gt_bboxes, mask_gt_bn)

        align_metric, overlaps = self._box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        valid = in_gts * ops.cast(mask_gt_bn, "float32")
        align_metric = align_metric * valid
        overlaps = overlaps * valid

        mask_topk = self._topk_mask(align_metric)
        mask_pos = mask_topk * valid

        target_gt_idx, fg_mask, mask_pos = self._resolve_conflicts(mask_pos, overlaps, align_metric)

        target_labels, target_bboxes, target_scores = self._gather_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric = align_metric * mask_pos
        pos_align = ops.max(align_metric, axis=-1, keepdims=True)
        pos_overlap = ops.max(overlaps * mask_pos, axis=-1, keepdims=True)
        norm = align_metric * pos_overlap / (pos_align + self.eps)
        norm = ops.max(norm, axis=1)
        target_scores = target_scores * ops.expand_dims(norm, -1)

        return target_labels, target_bboxes, target_scores, fg_mask

    def _expand_small_boxes(self, gt_bboxes, mask_gt_bn):
        """Grow sub-stride GT sides to one stride, about the box centre (STAL).

        Each side is tested and grown independently, so a thin-but-long box only
        has its short side expanded. Padded GT rows are excluded via
        ``mask_gt_bn``: their boxes are all-zero, so every side is "small", and
        without the gate they would become real ``expanded_side``-wide boxes at
        the origin and capture anchors that belong to no object.
        """
        cxcy = (gt_bboxes[..., :2] + gt_bboxes[..., 2:4]) / 2.0
        wh = gt_bboxes[..., 2:4] - gt_bboxes[..., :2]
        small = ops.cast(wh < self.small_side, "float32") * mask_gt_bn
        wh = small * self.expanded_side + (1.0 - small) * wh
        half = wh / 2.0
        return ops.concatenate([cxcy - half, cxcy + half], axis=-1)

    def _in_gt_mask(self, anc_points, gt_bboxes, mask_gt_bn, eps=1e-9):
        gt_bboxes = self._expand_small_boxes(gt_bboxes, mask_gt_bn)

        xy = ops.reshape(anc_points, (1, 1, -1, 2))
        lt = ops.expand_dims(gt_bboxes[..., :2], 2)
        rb = ops.expand_dims(gt_bboxes[..., 2:4], 2)
        deltas = ops.concatenate([xy - lt, rb - xy], axis=-1)
        return ops.cast(ops.min(deltas, axis=-1) > eps, "float32")

    def _box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):

        onehot = ops.one_hot(ops.cast(ops.maximum(gt_labels, 0), "int32"), self.nc)
        onehot = ops.cast(onehot, pd_scores.dtype)
        bbox_scores = ops.matmul(onehot, ops.transpose(pd_scores, (0, 2, 1)))

        gt = ops.expand_dims(gt_bboxes, 2)
        pd = ops.expand_dims(pd_bboxes, 1)
        overlaps = bbox_iou(gt, pd, xywh=False, ciou=True)
        overlaps = ops.clip(overlaps, 0.0, 1.0)

        align_metric = ops.power(bbox_scores, self.alpha) * ops.power(overlaps, self.beta)
        return align_metric, overlaps

    def _topk_mask(self, metrics):
        _, idx = ops.top_k(metrics, k=self.topk)
        a = ops.shape(metrics)[-1]
        onehot = ops.one_hot(idx, a)
        mask = ops.sum(onehot, axis=2)
        return ops.cast(mask > 0, "float32")

    def _resolve_conflicts(self, mask_pos, overlaps, align_metric):
        fg = ops.sum(mask_pos, axis=1)
        m = ops.shape(mask_pos)[1]

        max_idx = ops.argmax(overlaps, axis=1)
        is_max = ops.transpose(ops.one_hot(max_idx, m), (0, 2, 1))
        multi = ops.cast(ops.expand_dims(fg, 1) > 1, "float32")
        mask_pos = multi * is_max + (1.0 - multi) * mask_pos

        if self.topk2 != self.topk:
            masked_metric = align_metric * mask_pos
            _, idx = ops.top_k(masked_metric, k=self.topk2)
            a = ops.shape(mask_pos)[-1]
            keep = ops.cast(ops.sum(ops.one_hot(idx, a), axis=2) > 0, "float32")

            mask_pos = mask_pos * keep

        fg = ops.sum(mask_pos, axis=1)
        target_gt_idx = ops.argmax(mask_pos, axis=1)
        return target_gt_idx, fg, mask_pos

    def _gather_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        idx = ops.cast(target_gt_idx, "int32")
        target_labels = ops.take_along_axis(gt_labels, idx, axis=1)
        target_labels = ops.cast(target_labels, "int32")

        idx_b = ops.expand_dims(idx, -1)
        idx_b = ops.broadcast_to(idx_b, (ops.shape(idx)[0], ops.shape(idx)[1], 4))
        target_bboxes = ops.take_along_axis(gt_bboxes, idx_b, axis=1)

        onehot = ops.one_hot(ops.maximum(target_labels, 0), self.nc)
        onehot = ops.cast(onehot, "float32")
        target_scores = onehot * ops.expand_dims(ops.cast(fg_mask > 0, "float32"), -1)
        return target_labels, target_bboxes, target_scores
