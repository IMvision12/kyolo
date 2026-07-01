"""PyTorch -> Keras 3 weight conversion for :mod:`kyolo`.

The official YOLO weights are AGPL-3.0 and are **not** redistributed by this
project. These utilities let a user convert a ``.pt`` checkpoint they have
obtained themselves into Keras ``.weights.h5`` format.

Public API
----------
* :func:`load_torch_state_dict` -- read a ``.pt`` into ordered numpy arrays.
* :func:`transfer_by_order` -- robust positional transfer (recommended).
* :func:`transfer_torch_to_keras` -- best-effort name-based transfer.
* :func:`convert_weights` -- high-level: load, transfer, save.
* :func:`load_pretrained` -- download + convert + cache + load in one call
  (this is what the model factories use for ``convert_weights=True``).
* :func:`download_file` -- fetch and cache a remote URL.

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
from .file_downloader import DEFAULT_CACHE, download_file, validate_url
from .mappings import DEFAULT_MAPPING, NAME_MAPPINGS, get_mapping
from .pretrained import default_cache_dir, load_pretrained

__all__ = [
    "transfer_torch_to_keras",
    "transfer_by_order",
    "load_torch_state_dict",
    "convert_weights",
    "load_pretrained",
    "default_cache_dir",
    "download_file",
    "validate_url",
    "DEFAULT_CACHE",
    "DEFAULT_MAPPING",
    "NAME_MAPPINGS",
    "get_mapping",
    "WeightConversionError",
    "WeightMappingError",
    "WeightShapeMismatchError",
    "WeightCountMismatchError",
]
