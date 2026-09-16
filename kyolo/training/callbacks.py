"""Training callbacks."""

from __future__ import annotations

import keras

__all__ = ["ProgressiveLossSchedule"]


class ProgressiveLossSchedule(keras.callbacks.Callback):
    """Advance a progressive loss schedule once per epoch.

    :class:`kyolo.losses.E2EDetectionLoss` shifts weight from its one-to-many
    branch to its one-to-one branch over training, but only when something calls
    ``update()``. Adding this callback to ``fit()`` is that something::

        detector.fit(ds, epochs=100, callbacks=[ProgressiveLossSchedule()])

    The loss is found on the model (``model.loss_fn``) unless one is passed
    explicitly, which is what a criterion held somewhere else needs.

    Args:
        loss: the loss object to update. ``None`` (default) takes
            ``model.loss_fn``.
    """

    def __init__(self, loss=None):
        super().__init__()
        self._loss = loss

    def _target(self):
        loss = self._loss if self._loss is not None else getattr(self.model, "loss_fn", None)
        if loss is None or not hasattr(loss, "update"):
            raise ValueError(
                "ProgressiveLossSchedule needs a loss with an update() method "
                "(e.g. kyolo.losses.E2EDetectionLoss); got "
                f"{type(loss).__name__}. Pass one explicitly with "
                "ProgressiveLossSchedule(loss=...)."
            )
        return loss

    def on_train_begin(self, logs=None):

        self._target()

    def on_epoch_end(self, epoch, logs=None):
        loss = self._target()
        loss.update()
        if logs is not None and hasattr(loss, "o2m_gain"):
            logs["o2m_gain"] = loss.o2m_gain
