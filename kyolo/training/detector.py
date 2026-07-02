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

from ..losses.detection_loss import YOLODetectionLoss

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
        # NMS-free models (YOLOv10 / YOLO26) self-report via ``model.end_to_end``.
        self.end_to_end = (
            end_to_end if end_to_end is not None else getattr(model, "end_to_end", False)
        )
        self.loss_fn = loss or YOLODetectionLoss(
            nc=self.nc,
            reg_max=self.reg_max,
            strides=self.strides,
            data_format=getattr(model, "data_format", None),
        )

        self._box = keras.metrics.Mean(name="box_loss")
        self._cls = keras.metrics.Mean(name="cls_loss")
        self._dfl = keras.metrics.Mean(name="dfl_loss")
        self._postprocessor = None

    # ------------------------------------------------------------------ #
    @property
    def metrics(self):
        return [self._box, self._cls, self._dfl]

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

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        targets = self._targets(x, y)
        losses = self.loss_fn(y_pred, targets)
        self._box.update_state(losses["box"])
        self._cls.update_state(losses["cls"])
        self._dfl.update_state(losses["dfl"])
        return losses["loss"]

    # ------------------------------------------------------------------ #
    def detect(self, images, conf_threshold=0.25, iou_threshold=0.7, max_detections=300):
        """Run the model + post-processing and return ``(B, max_det, 6)`` detections.

        ``images`` must already be preprocessed (letterboxed / scaled). Use
        :class:`kyolo.preprocessing.YOLOPreprocessor` first.
        """
        from ..postprocessing.postprocessor import YOLOPostprocessor

        if self._postprocessor is None:
            self._postprocessor = YOLOPostprocessor(
                nc=self.nc,
                reg_max=self.reg_max,
                strides=self.strides,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                end_to_end=self.end_to_end,
                data_format=getattr(self.body, "data_format", None),
            )
        feats = self.body(images, training=False)
        return self._postprocessor(feats)

    def get_config(self):
        # The wrapped body is a functional/subclassed model; serialise it.
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
