"""Detection losses."""

from .bbox_loss import BboxLoss
from .detection_loss import YOLODetectionLoss
from .dfl_loss import DistributionFocalLoss
from .e2e_detection_loss import E2EDetectionLoss

__all__ = [
    "YOLODetectionLoss",
    "E2EDetectionLoss",
    "BboxLoss",
    "DistributionFocalLoss",
]
