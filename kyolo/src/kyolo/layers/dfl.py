"""Distribution Focal Loss (DFL) integral layer.

DFL turns the ``4 * reg_max`` distributional box predictions produced by the
anchor-free YOLO heads (v6/v8/v9/v11/v12) into four continuous distances
(left, top, right, bottom) by taking the expectation of a softmax over
``reg_max`` bins. YOLOv10 keeps DFL; YOLO26 drops it entirely (its head
regresses the four distances directly).
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
    here they are a non-trainable constant, which is numerically identical and
    keeps the layer backend-agnostic.
    """

    def __init__(self, reg_max=16, **kwargs):
        super().__init__(**kwargs)
        self.reg_max = reg_max

    def build(self, input_shape):
        # Frozen bin centres [0 .. reg_max-1].
        self.bins = self.add_weight(
            name="bins",
            shape=(self.reg_max,),
            initializer=keras.initializers.Constant(list(range(self.reg_max))),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        b = ops.shape(x)[0]
        a = ops.shape(x)[2]
        # (b, 4, reg_max, a)
        x = ops.reshape(x, (b, 4, self.reg_max, a))
        # softmax over the reg_max bins
        x = ops.softmax(x, axis=2)
        # expectation: sum(prob * bin_centre) over the bins
        weights = ops.reshape(self.bins, (1, 1, self.reg_max, 1))
        x = ops.sum(x * weights, axis=2)  # (b, 4, a)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"reg_max": self.reg_max})
        return config
