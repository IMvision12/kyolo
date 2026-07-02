"""Torch-safe layer naming.

kyolo names its layers with dotted paths (``model.0.conv``, ``backbone.stem``)
that mirror the official PyTorch module hierarchy, which keeps the weight
conversion in :mod:`kyolo.conversion` simple. The PyTorch Keras backend, though,
registers every variable in a ``torch.nn.ParameterDict`` keyed by
``variable.path`` and torch forbids ``.`` in parameter names. Building any model
on the torch backend therefore fails with ``parameter name can't contain "."``.

To stay backend-agnostic we hand Keras a torch-safe name (``.`` -> ``-``) while
keeping the dotted name as the logical reference. Hyphens never appear literally
in the reference names, so the mapping is a clean bijection and
:func:`kyolo.conversion.convert.transfer_torch_to_keras` reverses it (``-`` ->
``.``) when it derives PyTorch keys from Keras variable paths. The order-based
transfer does not depend on names at all, so it is unaffected.

Usage: builder modules import ``layers`` from here instead of from ``keras`` and
otherwise call ``layers.Conv2D(...)`` exactly as before; names are sanitized
transparently.
"""

from __future__ import annotations

import keras

__all__ = ["kname", "unkname", "layers"]


def kname(name):
    """Convert a dotted reference name to a torch-safe Keras layer name."""
    return name.replace(".", "-") if isinstance(name, str) else name


def unkname(name):
    """Reverse :func:`kname`: recover the dotted reference name."""
    return name.replace("-", ".") if isinstance(name, str) else name


class _SanitizedLayers:
    """Proxy for ``keras.layers`` that sanitizes the ``name`` of every layer.

    Attribute access is forwarded to ``keras.layers``; callables are wrapped so
    a ``name=`` keyword is passed through :func:`kname` before construction.
    Everything else (non-callable attributes) is returned untouched.
    """

    def __getattr__(self, attr):
        obj = getattr(keras.layers, attr)
        if not callable(obj):
            return obj

        def _construct(*args, **kwargs):
            if "name" in kwargs:
                kwargs["name"] = kname(kwargs["name"])
            return obj(*args, **kwargs)

        return _construct


layers = _SanitizedLayers()
