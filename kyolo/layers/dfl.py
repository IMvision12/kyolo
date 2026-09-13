"""Distribution Focal Loss (DFL) integral layer.

DFL turns the ``4 * reg_max`` distributional box predictions produced by the
anchor-free YOLO heads (v5u/v8/v9/v10/11/12) into four continuous distances
(left, top, right, bottom) by taking the expectation of a softmax over
``reg_max`` bins. YOLO26 drops it entirely (its head regresses the four
distances directly, ``reg_max = 1``).
"""

from __future__ import annotations

import keras
from keras import ops

__all__ = ["DFL"]


@keras.saving.register_keras_serializable(package="kyolo")
class DFL(keras.layers.Layer):
    """Integral of a discrete distribution over ``reg_max`` bins.

    Input shape:  ``(batch, 4 * reg_max, anchors)``
    Output shape: ``(batch, 4, anchors)``

    The bin centres are the fixed integers ``[0, 1, ..., reg_max - 1]``. The
    official implementation stores them as the weights of a frozen 1x1 conv;
    here they are recomputed as a constant in ``call``, which is numerically
    identical and leaves the layer stateless (no variables to save, load or
    track), so it can live inside layers that are created lazily.
    """

    def __init__(self, reg_max=16, **kwargs):
        super().__init__(**kwargs)
        self.reg_max = reg_max

    def call(self, x):
        b = ops.shape(x)[0]
        a = ops.shape(x)[2]
        # (b, 4, reg_max, a)
        x = ops.reshape(x, (b, 4, self.reg_max, a))
        # softmax over the reg_max bins
        x = ops.softmax(x, axis=2)
        # expectation: sum(prob * bin_centre) over the bins
        bins = ops.cast(ops.arange(self.reg_max), x.dtype)
        bins = ops.reshape(bins, (1, 1, self.reg_max, 1))
        return ops.sum(x * bins, axis=2)  # (b, 4, a)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({"reg_max": self.reg_max})
        return config
