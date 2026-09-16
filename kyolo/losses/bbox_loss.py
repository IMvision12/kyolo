"""Box-regression terms: CIoU plus a parameterization-dependent distance.

With a distributional box branch (``reg_max > 1``) the distance term is
:class:`~kyolo.losses.DistributionFocalLoss`. With a direct box branch
(``reg_max == 1``, YOLO26) it is an L1 on LTRB distances normalized by image
size.
"""

from __future__ import annotations

from keras import ops

from ..ops.anchors import bbox2dist
from ..ops.boxes import bbox_iou
from .dfl_loss import DistributionFocalLoss

__all__ = ["BboxLoss"]


class BboxLoss:
    """The two box-regression terms: CIoU plus DFL or normalized L1.

    Args:
        reg_max: DFL bin count of the box branch. ``> 1`` selects the DFL
            distance term, ``== 1`` the normalized-L1 one.
    """

    def __init__(self, reg_max=16):
        self.reg_max = reg_max
        self.dfl = DistributionFocalLoss(reg_max) if reg_max > 1 else None

    @property
    def use_dfl(self):
        return self.dfl is not None

    def __call__(
        self,
        pred_dist,
        pred_bboxes,
        anchors,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        """Compute ``(box_loss, dist_loss)``.

        Args:
            pred_dist: ``(B, A, 4*reg_max)`` bin logits, or ``(B, A, 4)`` raw
                LTRB distances when ``reg_max == 1``.
            pred_bboxes: ``(B, A, 4)`` decoded xyxy predictions, stride units.
            anchors: broadcastable ``(1, A, 2)`` anchor centres, stride units.
            target_bboxes: ``(B, A, 4)`` assigned xyxy targets, stride units.
            target_scores: ``(B, A, nc)`` soft target scores from the assigner.
            target_scores_sum: scalar normalizer.
            fg_mask: ``(B, A)`` 1.0 on assigned (foreground) anchors.
            imgsz: ``(2,)`` image ``(height, width)`` in pixels.
            stride: ``(1, A, 1)`` per-anchor stride.
        """
        weight = ops.sum(target_scores, axis=-1) * fg_mask

        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, ciou=True)
        box_loss = ops.sum((1.0 - iou) * weight) / target_scores_sum

        if self.dfl is not None:
            b = ops.shape(pred_dist)[0]
            a = ops.shape(pred_dist)[1]
            pred_bins = ops.reshape(pred_dist, (b, a, 4, self.reg_max))
            target_ltrb = bbox2dist(anchors, target_bboxes, self.reg_max)
            dist = self.dfl(pred_bins, target_ltrb)
        else:
            dist = self.normalized_l1(pred_dist, anchors, target_bboxes, imgsz, stride)

        dist_loss = ops.sum(dist * weight) / target_scores_sum
        return box_loss, dist_loss

    def normalized_l1(self, pred_dist, anchors, target_bboxes, imgsz, stride):
        """Mean absolute LTRB error in image-fraction units -> ``(B, A)``.

        Distances arrive in stride units, where the same absolute error counts
        three times more on P5 than on P3 simply because the cells are larger.
        Multiplying by the stride puts both sides in pixels, and dividing by the
        image size makes the term scale-free, so one gain works for every level
        and every input resolution.
        """
        target_ltrb = bbox2dist(anchors, target_bboxes)

        height, width = imgsz[0], imgsz[1]
        norm = ops.reshape(ops.stack([width, height, width, height]), (1, 1, 4))
        pred_frac = pred_dist * stride / norm
        target_frac = target_ltrb * stride / norm
        return ops.mean(ops.abs(pred_frac - target_frac), axis=-1)
