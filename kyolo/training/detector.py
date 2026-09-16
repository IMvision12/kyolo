"""``YOLODetector`` - a trainable / fine-tunable wrapper around a raw model.

Wraps a feature-outputting kyolo model together with
:class:`kyolo.losses.YOLODetectionLoss` and plugs into ``keras.Model.fit`` via
``compute_loss`` (so training works on TensorFlow, JAX and PyTorch backends
without any backend-specific gradient code).

Batches may be supplied either as a single dict
``{"images", "boxes", "labels", "mask"}`` or as an ``(x, y)`` pair with
``x = {"images": ...}`` (or a bare image tensor) and
``y = {"boxes", "labels", "mask"}``.
"""

from __future__ import annotations

import keras

from ..losses import E2EDetectionLoss, YOLODetectionLoss
from ..postprocessing.postprocessor import YOLOPostprocessor

__all__ = ["YOLODetector"]


@keras.saving.register_keras_serializable(package="kyolo")
class YOLODetector(keras.Model):
    def __init__(
        self,
        model,
        loss=None,
        nc=None,
        reg_max=None,
        strides=None,
        end_to_end=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.body = model
        self.nc = nc if nc is not None else getattr(model, "nc", 80)
        self.reg_max = reg_max if reg_max is not None else getattr(model, "reg_max", 16)
        self.strides = strides if strides is not None else getattr(model, "strides", (8, 16, 32))

        self.end_to_end = (
            end_to_end if end_to_end is not None else getattr(model, "end_to_end", False)
        )
        self.loss_fn = loss or self._default_loss(model)

        self._box = keras.metrics.Mean(name="box_loss")
        self._cls = keras.metrics.Mean(name="cls_loss")
        self._dist = keras.metrics.Mean(name=f"{self.loss_fn.dist_name}_loss")

        self._postprocessor = YOLOPostprocessor(
            nc=self.nc,
            reg_max=self.reg_max,
            strides=self.strides,
            end_to_end=self.end_to_end,
            data_format=getattr(model, "data_format", None),
        )

    def _default_loss(self, model):
        """Build the criterion the wrapped model asks for.

        Loss settings that follow from the architecture (the gains, the TAL
        top-k, and for end-to-end models the two branches' assignment) travel
        with the model as ``model.loss_config``, set by ``finalize_detector``.
        Reading them here is what makes ``YOLODetector(yolo26n())`` come out
        with YOLO26's criterion without the caller knowing any of the numbers.
        """
        config = dict(getattr(model, "loss_config", None) or {})
        config.update(
            nc=self.nc,
            reg_max=self.reg_max,
            strides=self.strides,
            data_format=getattr(model, "data_format", None),
        )
        if self.end_to_end:
            return E2EDetectionLoss(**config)
        return YOLODetectionLoss(**config)

    def _images(self, x):
        if isinstance(x, dict):
            return x["images"]
        return x

    def _targets(self, x, y):
        src = y if y is not None else x
        return {
            "boxes": src["boxes"],
            "labels": src["labels"],
            "mask": src["mask"],
        }

    def call(self, inputs, training=False):
        return self.body(self._images(inputs), training=training)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        targets = self._targets(x, y)
        losses = self.loss_fn(y_pred, targets)
        self._box.update_state(losses["box"])
        self._cls.update_state(losses["cls"])
        self._dist.update_state(losses["dist"])
        return losses["loss"]

    def detect(self, images, conf_threshold=0.25, iou_threshold=0.7, max_detections=300):
        """Run the model + post-processing and return ``(B, max_det, 6)`` detections.

        ``images`` must already be preprocessed (letterboxed / scaled). Use
        :class:`kyolo.preprocessing.YOLOPreprocessor` first. Works before and
        after training; the thresholds apply to this call only.
        """
        pp = self._postprocessor
        pp.conf_threshold = conf_threshold
        pp.iou_threshold = iou_threshold
        pp.max_detections = max_detections
        feats = self.body(images, training=False)
        return pp(feats)

    def get_config(self):

        return {
            "model": keras.saving.serialize_keras_object(self.body),
            "nc": self.nc,
            "reg_max": self.reg_max,
            "strides": list(self.strides),
            "end_to_end": self.end_to_end,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        config["strides"] = tuple(config["strides"])
        return cls(**config)
