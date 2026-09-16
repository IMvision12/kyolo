"""End-to-end (NMS-free) detection loss (Ultralytics ``E2ELoss`` parity).

Wraps two :class:`~kyolo.losses.YOLODetectionLoss` branches -- one-to-many and
one-to-one -- and shifts weight from the former to the latter over training
(ProgLoss). Used by YOLO26 once a one-to-one head exists.
"""

from __future__ import annotations

from .detection_loss import YOLODetectionLoss

__all__ = ["E2EDetectionLoss"]


class E2EDetectionLoss:
    """End-to-end (NMS-free) criterion: a one-to-many and a one-to-one branch.

    An NMS-free head needs a branch that emits exactly one prediction per
    object, but one-to-one assignment alone is a weak training signal -- far too
    few positives early on. So both branches are supervised at once and the
    weight is shifted: the one-to-many branch does the early feature learning
    and decays from ``o2m_gain`` to ``final_o2m_gain``, while the one-to-one
    branch it hands off to takes ``1 - o2m`` and ends up dominating. This is
    Ultralytics' progressive loss balancing (ProgLoss), used by YOLO26.

    Expects predictions as ``{"one2many": feats, "one2one": feats}``. The
    reported components come from the one-to-one branch, since that is the one
    that will actually be decoded at inference.

    Call :meth:`update` once per epoch to advance the schedule;
    :class:`kyolo.training.ProgressiveLossSchedule` does that from ``fit()``.

    Note:
        No kyolo model currently builds a one-to-one head, so this criterion has
        no in-package producer yet -- it is driven by whatever supplies the two
        branches. See the one2one head item in the project TODO.

    Args:
        epochs: total training epochs, the horizon the decay is spread over.
        one2many_topk: TAL ``topk`` for the one-to-many branch.
        one2one_topk / one2one_topk2: TAL ``topk`` / ``topk2`` for the
            one-to-one branch. ``topk2=1`` is what makes it one-to-one.
        o2m_gain: initial weight of the one-to-many branch.
        final_o2m_gain: weight it decays to.
        progressive: set ``False`` to hold the weights fixed at 1.0 each, i.e.
            plain summation of the two branches (Ultralytics' older
            ``E2EDetectLoss``, for which ``one2one_topk=1, one2one_topk2=None``
            is the matching assignment).
        **loss_kwargs: forwarded to both :class:`YOLODetectionLoss` branches
            (``nc``, ``reg_max``, ``strides``, gains, ``data_format``, ...).
    """

    def __init__(
        self,
        epochs=100,
        one2many_topk=10,
        one2one_topk=7,
        one2one_topk2=1,
        o2m_gain=0.8,
        final_o2m_gain=0.1,
        progressive=True,
        **loss_kwargs,
    ):
        self.one2many = YOLODetectionLoss(tal_topk=one2many_topk, **loss_kwargs)
        self.one2one = YOLODetectionLoss(
            tal_topk=one2one_topk, tal_topk2=one2one_topk2, **loss_kwargs
        )
        self.epochs = epochs
        self.progressive = progressive
        self.initial_o2m_gain = o2m_gain
        self.final_o2m_gain = final_o2m_gain
        self.updates = 0
        self.o2m_gain = o2m_gain if progressive else 1.0
        self.o2o_gain = (1.0 - o2m_gain) if progressive else 1.0

    @property
    def dist_name(self):
        return self.one2one.dist_name

    def __call__(self, preds, targets):
        return self.compute(preds, targets)

    def compute(self, preds, targets):
        if not isinstance(preds, dict) or not {"one2many", "one2one"} <= set(preds):
            raise ValueError(
                "E2EDetectionLoss expects predictions as "
                "{'one2many': feats, 'one2one': feats}; got "
                f"{type(preds).__name__}"
                + (f" with keys {sorted(preds)}" if isinstance(preds, dict) else "")
                + "."
            )
        o2m = self.one2many(preds["one2many"], targets)
        o2o = self.one2one(preds["one2one"], targets)
        total = o2m["loss"] * self.o2m_gain + o2o["loss"] * self.o2o_gain
        losses = {
            "loss": total,
            "box": o2o["box"],
            "cls": o2o["cls"],
            "dist": o2o["dist"],
        }
        losses[self.dist_name] = o2o["dist"]
        return losses

    def decay(self, updates):
        """One-to-many weight after ``updates`` epochs: linear, then floored."""
        remaining = max(1.0 - updates / max(self.epochs - 1, 1), 0.0)
        return remaining * (self.initial_o2m_gain - self.final_o2m_gain) + self.final_o2m_gain

    def update(self):
        """Advance the schedule one epoch."""
        if not self.progressive:
            return
        self.updates += 1
        self.o2m_gain = self.decay(self.updates)
        self.o2o_gain = max(1.0 - self.o2m_gain, 0.0)
