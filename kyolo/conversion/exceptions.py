"""Typed errors raised during PyTorch -> Keras weight conversion."""

from __future__ import annotations

__all__ = [
    "WeightConversionError",
    "WeightMappingError",
    "WeightShapeMismatchError",
    "WeightCountMismatchError",
]


class WeightConversionError(Exception):
    """Base class for all weight-conversion errors."""


class WeightMappingError(WeightConversionError):
    """A Keras variable could not be matched to a PyTorch parameter."""

    def __init__(self, keras_name, torch_name):
        self.keras_name = keras_name
        self.torch_name = torch_name
        super().__init__(
            f"No PyTorch parameter matched Keras variable '{keras_name}' "
            f"(tried '{torch_name}'). Adjust the name mapping or use method='order'."
        )


class WeightShapeMismatchError(WeightConversionError):
    """A matched pair of tensors have incompatible shapes."""

    def __init__(self, keras_name, keras_shape, torch_name, torch_shape):
        self.keras_name = keras_name
        self.torch_name = torch_name
        super().__init__(
            f"Shape mismatch: Keras '{keras_name}' {tuple(keras_shape)} vs "
            f"PyTorch '{torch_name}' {tuple(torch_shape)}."
        )


class WeightCountMismatchError(WeightConversionError):
    """The Keras model and the checkpoint have a different number of tensors."""

    def __init__(self, keras_count, torch_count):
        self.keras_count = keras_count
        self.torch_count = torch_count
        super().__init__(
            f"Parameter count mismatch: Keras model has {keras_count} variables "
            f"but the checkpoint has {torch_count} tensors. Check the variant, "
            f"`nc`, and the deploy/train setting."
        )
