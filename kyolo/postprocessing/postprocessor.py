"""Decode raw head outputs into detections (DFL decode + NMS / top-k)."""

from __future__ import annotations

import keras
from keras import ops

from ..layers.dfl import DFL
from ..ops.anchors import decode_raw_predictions
from ..ops.boxes import xywh2xyxy
from .nms import batched_nms, top_k_detections

__all__ = ["YOLOPostprocessor"]


@keras.saving.register_keras_serializable(package="kyolo")
class YOLOPostprocessor(keras.layers.Layer):
    """Turn a model's raw ``[P3, P4, P5]`` outputs into detections.

    Args:
        nc: number of classes.
        reg_max: DFL bins (``>1`` -> DFL integral; ``1`` -> direct distances).
        strides: per-level strides.
        conf_threshold / iou_threshold / max_detections: NMS params.
        end_to_end: if ``True`` skip NMS and do NMS-free top-k selection
            (YOLOv10 / YOLO26). If ``False`` run greedy class-aware NMS.
        data_format: layout of the input feature maps.

    Call returns ``(B, max_detections, 6)`` = ``[x1, y1, x2, y2, score, class]``.
    """

    def __init__(
        self,
        nc=80,
        reg_max=16,
        strides=(8, 16, 32),
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=300,
        end_to_end=False,
        data_format="channels_last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nc = nc
        self.reg_max = reg_max
        self.strides = tuple(strides)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.end_to_end = end_to_end
        self.data_format = data_format
        self.dfl = DFL(reg_max) if reg_max > 1 else None
        if self.dfl is not None:
            self.dfl.build((None, 4 * reg_max, None))

    def call(self, feats):
        boxes_xywh, scores = decode_raw_predictions(
            feats, self.strides, self.reg_max, self.nc, self.dfl, self.data_format
        )  # (B,A,4) pixels, (B,A,nc)
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        if self.end_to_end:
            return top_k_detections(boxes_xyxy, scores, self.max_detections, self.conf_threshold)
        classes = ops.argmax(scores, axis=-1)
        best = ops.max(scores, axis=-1)
        return batched_nms(
            boxes_xyxy,
            best,
            classes,
            self.iou_threshold,
            self.max_detections,
            self.conf_threshold,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nc": self.nc,
                "reg_max": self.reg_max,
                "strides": self.strides,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "max_detections": self.max_detections,
                "end_to_end": self.end_to_end,
                "data_format": self.data_format,
            }
        )
        return config
