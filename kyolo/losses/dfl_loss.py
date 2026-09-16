"""Distribution Focal Loss (DFL).

The box branch predicts, per side, a distribution over ``reg_max`` integer
distances rather than one number. A continuous target ``t`` is supervised by
splitting it across bins ``floor(t)`` and ``floor(t) + 1`` in proportion to
how close it is to each.

References:
    https://arxiv.org/abs/2006.04388
"""

from __future__ import annotations

from keras import ops

__all__ = ["DistributionFocalLoss"]


class DistributionFocalLoss:
    """Cross-entropy onto the two bins a target falls between.

    Args:
        reg_max: number of bins per side. Must be at least 2.
    """

    def __init__(self, reg_max=16):
        if reg_max < 2:
            raise ValueError(
                f"DistributionFocalLoss needs reg_max >= 2 (got {reg_max}); a direct box "
                "branch has no bins to classify. Use BboxLoss, which switches to a "
                "normalized L1 at reg_max == 1."
            )
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """``pred_dist`` ``(B, A, 4, reg_max)`` logits, ``target`` ``(B, A, 4)`` -> ``(B, A)``."""
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
        return ops.mean(ce_l * wl + ce_r * wr, axis=-1)
