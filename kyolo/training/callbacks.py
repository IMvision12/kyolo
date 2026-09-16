"""Training callbacks."""

from __future__ import annotations

import keras

__all__ = ["CloseMosaic", "ProgressiveLossSchedule"]


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

    def target(self):
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

        self.target()

    def on_epoch_end(self, epoch, logs=None):
        loss = self.target()
        loss.update()
        if logs is not None and hasattr(loss, "o2m_gain"):
            logs["o2m_gain"] = loss.o2m_gain


class CloseMosaic(keras.callbacks.Callback):
    """Turn off the image-combining augmentations for the last epochs.

    Mosaic, mixup and copy-paste stitch several images together, which is a
    strong regularizer but means the model spends training on collages it will
    never see at inference. Ultralytics' ``close_mosaic`` switches them off for
    the final epochs so training finishes on real images; it is worth a
    measurable amount of mAP and is on by default there.

    Pass the training dataloader (or its pipeline directly)::

        train = GrainDataLoader("coco8.yaml", "train", augment=True)
        detector.fit(train, epochs=100, callbacks=[CloseMosaic(train)])

    The decision is re-evaluated at the start of every epoch, so this behaves
    correctly when training resumes partway through.

    Args:
        target: a :class:`kyolo.data.GrainDataLoader`, or an
            :class:`kyolo.augmentation.AugmentationPipeline`.
        close_epochs: how many final epochs to train without image mixing.
        verbose: log the switch when it happens.
    """

    def __init__(self, target, close_epochs=10, verbose=1):
        super().__init__()
        if close_epochs < 0:
            raise ValueError(f"close_epochs must be non-negative; got {close_epochs}.")
        self.target = target
        self.close_epochs = int(close_epochs)
        self.verbose = verbose

    def pipeline(self):
        pipeline = getattr(self.target, "augment", self.target)
        if pipeline is None or not hasattr(pipeline, "close_mosaic"):
            raise ValueError(
                "CloseMosaic needs an AugmentationPipeline, or a dataloader with "
                "one on its `augment` attribute; got "
                f"{type(pipeline).__name__}. Was the dataloader built with "
                "augment=True?"
            )
        return pipeline

    def on_train_begin(self, logs=None):
        self.pipeline()
        if self.close_epochs and not (self.params or {}).get("epochs"):
            raise ValueError(
                "CloseMosaic needs to know the total number of epochs to work "
                "out when to stop mixing, but fit() did not report one. Pass "
                "`epochs=` to fit(), or CloseMosaic(..., close_epochs=0) to "
                "disable the schedule."
            )

    def on_epoch_begin(self, epoch, logs=None):
        pipeline = self.pipeline()
        total = (self.params or {}).get("epochs")
        should_close = bool(self.close_epochs and total and epoch >= total - self.close_epochs)

        if should_close and not pipeline.mosaic_closed:
            pipeline.close_mosaic()
            if self.verbose:
                print(f"\nEpoch {epoch + 1}: closing mosaic / mixup / copy-paste.")
        elif not should_close and pipeline.mosaic_closed:
            pipeline.open_mosaic()
