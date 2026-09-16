"""PyTorch -> Keras 3 weight conversion for :mod:`kyolo`.

The official YOLO weights are AGPL-3.0 and are **not** redistributed by this
project. kyolo does not download, cache, or auto-load them; it only provides the
transfer engine so you can convert a ``.pt`` checkpoint you have obtained
yourself into Keras ``.weights.h5`` format. Run the per-model converter, e.g.::

    python -m kyolo.models.yolov8.convert_yolov8_torch_to_keras \
        --weights yolov8n.pt --output yolov8n.weights.h5 --variant n

Public API
----------
* :func:`load_torch_state_dict` -- read a ``.pt`` into ordered numpy arrays.
* :func:`transfer_torch_to_keras` -- name-based transfer (the default; use this).
* :func:`transfer_by_order` -- positional transfer. Does not fit the families
  kyolo ships, whose CSP blocks are built in a different order than the
  reference registers them; see :mod:`kyolo.conversion.convert`.
* :func:`convert_weights` -- high-level: load, transfer, save.

See :mod:`kyolo.conversion.convert` for details and the important caveat that a
clean transfer must still be validated against the reference outputs.
"""

from __future__ import annotations

from .convert import (
    convert_weights,
    load_torch_state_dict,
    transfer_by_order,
    transfer_torch_to_keras,
)
from .exceptions import (
    WeightConversionError,
    WeightCountMismatchError,
    WeightMappingError,
    WeightShapeMismatchError,
)
from .mappings import DEFAULT_MAPPING, NAME_MAPPINGS, get_mapping

__all__ = [
    "transfer_torch_to_keras",
    "transfer_by_order",
    "load_torch_state_dict",
    "convert_weights",
    "DEFAULT_MAPPING",
    "NAME_MAPPINGS",
    "get_mapping",
    "WeightConversionError",
    "WeightMappingError",
    "WeightShapeMismatchError",
    "WeightCountMismatchError",
]
