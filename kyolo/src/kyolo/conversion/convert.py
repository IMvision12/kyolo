"""PyTorch -> Keras 3 weight conversion for the YOLO family.

This module moves parameters out of an official (Ultralytics / WongKinYiu)
``.pt`` checkpoint and into an equivalent :mod:`kyolo` Keras 3 model. Nothing
here depends on a particular Keras backend: reading/writing variables goes
through :meth:`keras.Variable.assign`, and tensors are shuttled around as plain
NumPy arrays.

Two transfer strategies are provided:

``transfer_by_order`` (method ``"order"``, RECOMMENDED)
    Walks the Keras variables and the Torch ``state_dict`` **in build order**
    and zips them together positionally. Because the :mod:`kyolo` builders
    create their sub-layers in the exact same order the PyTorch modules are
    registered, this lines up without relying on fragile string matching. It
    only needs the *counts* and *shapes* to agree, which it validates loudly.

``transfer_torch_to_keras`` (method ``"name"``, best-effort)
    Derives a candidate Torch key for every Keras variable from its dotted
    ``path`` (e.g. ``model.0.conv/kernel`` -> ``model.0.conv.weight``) and looks
    it up directly. This is convenient when the two graphs are structurally
    identical, but the derived names can drift from a specific checkpoint's
    layout (the detection head in particular), so it may need per-model mapping
    tuning. Prefer ``"order"`` unless you have a reason not to.

Layout conventions handled here
-------------------------------
* Conv2D kernel: Keras is ``HWIO`` ``(kh, kw, in/groups, out)`` while PyTorch is
  ``OIHW`` ``(out, in/groups, kh, kw)`` -> ``np.transpose(w, (2, 3, 1, 0))``.
  Grouped / depth-wise convolutions use the same transpose.
* Dense/Linear kernel: Keras ``(in, out)`` vs Torch ``(out, in)`` -> transpose.
* BatchNormalization: Keras ``gamma, beta, moving_mean, moving_variance`` map to
  Torch ``weight, bias, running_mean, running_var``.
* Conv/Dense bias: copied verbatim.

.. important::
   A successful, no-error transfer does **not** by itself guarantee a
   bit-exact model. Padding conventions, activation choices and the exact
   assembly of the detection head must be validated by running the converted
   model against the PyTorch reference on the same input and comparing outputs.
   Treat the reports returned here as a first, necessary check -- not proof of
   correctness.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mappings import DEFAULT_MAPPING

__all__ = [
    "load_torch_state_dict",
    "transfer_by_order",
    "transfer_torch_to_keras",
    "convert_weights",
]


# Maps a Keras variable leaf name -> the corresponding PyTorch parameter suffix.
_SUFFIX_MAP = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
    "bias": "bias",
}


# --------------------------------------------------------------------------- #
# Loading a torch checkpoint into plain numpy arrays
# --------------------------------------------------------------------------- #
def _require_torch():
    """Import torch lazily, raising a friendly error if it is unavailable."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - trivial
        raise ImportError(
            "PyTorch is required to read '.pt' checkpoints but is not installed. "
            "Install it with `pip install torch` or `pip install kyolo[convert]`. "
            "torch is only needed for weight conversion, not for running kyolo."
        ) from exc
    return torch


def _looks_like_state_dict(obj, torch) -> bool:
    """True if ``obj`` is a mapping whose values are (mostly) tensors."""
    if not isinstance(obj, dict) or not obj:
        return False
    return any(torch.is_tensor(v) for v in obj.values())


def _to_state_dict(obj, torch) -> Dict[str, "object"]:
    """Reduce an arbitrary loaded checkpoint object to a ``name -> tensor`` map.

    Handles, in order of preference: a raw ``state_dict``; an ``nn.Module``; an
    Ultralytics-style checkpoint ``{"model": nn.Module, "ema": ..., ...}``; and
    the generic ``{"state_dict": ...}`` container.
    """
    # A live nn.Module (Ultralytics stores the model object itself).
    if not isinstance(obj, dict) and hasattr(obj, "state_dict"):
        try:
            return obj.float().state_dict()
        except Exception:  # pragma: no cover - some modules dislike .float()
            return obj.state_dict()

    if isinstance(obj, dict):
        # Ultralytics checkpoint: prefer the trained model, fall back to EMA.
        for key in ("model", "ema"):
            if obj.get(key) is not None:
                return _to_state_dict(obj[key], torch)
        if obj.get("state_dict") is not None:
            return _to_state_dict(obj["state_dict"], torch)
        if _looks_like_state_dict(obj, torch):
            return obj

    raise ValueError(
        "Could not interpret the checkpoint as a state_dict. Expected a raw "
        "state_dict, an nn.Module, or a dict containing a 'model'/'ema'/"
        f"'state_dict' entry, but got: {type(obj)!r}."
    )


def load_torch_state_dict(path: str) -> "OrderedDict[str, np.ndarray]":
    """Load a ``.pt`` file into an ordered ``name -> numpy.ndarray`` mapping.

    Insertion order is preserved (this is what the ``"order"`` transfer relies
    on). ``num_batches_tracked`` buffers and any non-tensor entries are dropped,
    and every tensor is moved to CPU and upcast to ``float32``.

    Args:
        path: Path to the ``.pt`` checkpoint.

    Returns:
        An ``OrderedDict`` of NumPy arrays in the checkpoint's parameter order.
    """
    torch = _require_torch()

    # weights_only=False is required to unpickle Ultralytics' nn.Module
    # checkpoints; older torch releases lack the argument entirely.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    state = _to_state_dict(ckpt, torch)

    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for name, value in state.items():
        if name.endswith("num_batches_tracked"):
            continue
        if not torch.is_tensor(value):
            continue
        out[name] = value.detach().cpu().float().numpy()
    if not out:
        raise ValueError(f"No tensor parameters were found in checkpoint: {path!r}")
    return out


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _leaf(path: str) -> str:
    """Return the trailing variable name of a Keras ``variable.path``."""
    return path.rsplit("/", 1)[-1]


def _var_kind(var) -> str:
    """Classify a Keras variable so we know how (if at all) to transpose it."""
    name = _leaf(var.path)
    ndim = len(var.shape)
    if name.endswith("kernel"):
        if ndim == 4:
            return "conv_kernel"
        if ndim == 2:
            return "dense_kernel"
        return "other"
    if name == "bias":
        return "bias"
    if name == "gamma":
        return "bn_gamma"
    if name == "beta":
        return "bn_beta"
    if name == "moving_mean":
        return "bn_mean"
    if name == "moving_variance":
        return "bn_var"
    return "other"


def _transpose_for_kind(arr: np.ndarray, kind: str) -> np.ndarray:
    """Apply the PyTorch->Keras axis permutation implied by ``kind``."""
    if kind == "conv_kernel":
        # OIHW -> HWIO. Works for plain, grouped and depth-wise convolutions.
        return np.transpose(arr, (2, 3, 1, 0))
    if kind == "dense_kernel":
        return np.transpose(arr, (1, 0))
    return arr


def _assign(var, arr: np.ndarray) -> None:
    """Assign ``arr`` into a Keras variable, matching its dtype."""
    var.assign(arr.astype(var.dtype))


# --------------------------------------------------------------------------- #
# Order-based transfer (robust, recommended)
# --------------------------------------------------------------------------- #
def transfer_by_order(
    keras_model,
    torch_state: "Dict[str, np.ndarray]",
    verbose: bool = True,
    strict_shapes: bool = True,
) -> Dict[str, int]:
    """Positionally zip Keras variables against the Torch parameters.

    Both sides are walked in build order: ``keras_model.weights`` yields
    variables layer-by-layer (trainable then non-trainable within each layer),
    which matches the ``weight, bias, running_mean, running_var`` grouping of a
    PyTorch ``state_dict``. The correct transpose is chosen from each Keras
    variable's kind; only element counts and post-transpose shapes need to line
    up, both of which are validated.

    Args:
        keras_model: The target Keras model (already built).
        torch_state: Ordered ``name -> ndarray`` mapping from
            :func:`load_torch_state_dict`.
        verbose: Print a short progress summary.
        strict_shapes: Raise on any per-tensor shape mismatch. When ``False``,
            mismatches are skipped and counted instead.

    Returns:
        ``{"transferred": int, "skipped": int, "total": int}``.

    Raises:
        ValueError: If the number of Keras variables and Torch tensors differ,
            or (when ``strict_shapes``) on a shape mismatch.
    """
    keras_vars = [(v, _var_kind(v)) for v in keras_model.weights]
    torch_items: List[Tuple[str, np.ndarray]] = list(torch_state.items())

    if len(keras_vars) != len(torch_items):
        raise ValueError(
            "Parameter count mismatch between the Keras model and the Torch "
            f"checkpoint: Keras has {len(keras_vars)} variables but the "
            f"checkpoint has {len(torch_items)} tensors. This usually means the "
            "architecture/config does not match the weights (wrong variant, "
            "wrong `nc`, or a deploy/train mismatch)."
        )

    transferred = 0
    skipped = 0
    for idx, ((var, kind), (tk, arr)) in enumerate(zip(keras_vars, torch_items)):
        converted = _transpose_for_kind(arr, kind)
        if tuple(converted.shape) != tuple(var.shape):
            message = (
                f"[{idx}] shape mismatch: Keras '{var.path}' expects "
                f"{tuple(var.shape)} (kind={kind}) but Torch '{tk}' is "
                f"{tuple(arr.shape)} -> {tuple(converted.shape)} after transpose."
            )
            if strict_shapes:
                raise ValueError(message)
            if verbose:
                print(f"  skip: {message}")
            skipped += 1
            continue
        _assign(var, converted)
        transferred += 1

    total = len(keras_vars)
    if verbose:
        print(
            f"[order] transferred {transferred}/{total} variables"
            + (f" ({skipped} skipped)" if skipped else "")
        )
    return {"transferred": transferred, "skipped": skipped, "total": total}


# --------------------------------------------------------------------------- #
# Name-based transfer (best-effort)
# --------------------------------------------------------------------------- #
def transfer_torch_to_keras(
    keras_model,
    torch_state: "Dict[str, np.ndarray]",
    name_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    strict: bool = False,
) -> Dict[str, object]:
    """Name-driven transfer: derive each Torch key from the Keras variable path.

    For a variable whose path is ``model.0.conv/kernel`` this builds the
    candidate key ``model.0.conv.weight`` (via the leaf-suffix rules
    ``kernel->weight``, ``gamma->weight``, ``beta->bias``,
    ``moving_mean->running_mean``, ``moving_variance->running_var``,
    ``bias->bias``) and then applies every ``old -> new`` substring replacement
    in ``name_mapping``, in order. If the resulting key exists in
    ``torch_state`` the value is transposed to the Keras layout and assigned;
    otherwise the variable is recorded as a miss.

    This path is best-effort. The dotted :mod:`kyolo` naming mirrors the
    reference modules, but individual checkpoints (especially detection heads)
    can diverge and need a per-model ``name_mapping``. When in doubt use
    :func:`transfer_by_order`.

    Args:
        keras_model: The target Keras model (already built).
        torch_state: ``name -> ndarray`` mapping from
            :func:`load_torch_state_dict`.
        name_mapping: Ordered substring replacements applied to each derived
            Torch key. Defaults to :data:`kyolo.conversion.mappings.DEFAULT_MAPPING`.
        verbose: Print a short summary (and the first few misses).
        strict: Raise on the first miss / shape mismatch instead of recording it.

    Returns:
        ``{"transferred", "skipped", "total", "misses"}`` where ``misses`` is a
        list of ``(keras_path, torch_key, reason)`` tuples.
    """
    if name_mapping is None:
        name_mapping = DEFAULT_MAPPING

    transferred = 0
    misses: List[Tuple[str, str, str]] = []
    total = 0

    for var in keras_model.weights:
        total += 1
        path = var.path
        if "/" in path:
            layer_path, leaf = path.rsplit("/", 1)
        else:
            layer_path, leaf = path, _leaf(path)
        suffix = _SUFFIX_MAP.get(leaf, leaf)
        torch_key = f"{layer_path}.{suffix}"
        for old, new in name_mapping.items():
            torch_key = torch_key.replace(old, new)

        if torch_key not in torch_state:
            if strict:
                raise KeyError(
                    f"No Torch parameter matched Keras variable '{path}' "
                    f"(tried '{torch_key}'). Adjust the name_mapping or use "
                    "method='order'."
                )
            misses.append((path, torch_key, "missing"))
            continue

        arr = torch_state[torch_key]
        converted = _transpose_for_kind(arr, _var_kind(var))
        if tuple(converted.shape) != tuple(var.shape):
            reason = f"shape {tuple(arr.shape)}->{tuple(converted.shape)} != {tuple(var.shape)}"
            if strict:
                raise ValueError(
                    f"Shape mismatch for '{path}' <- '{torch_key}': {reason}."
                )
            misses.append((path, torch_key, reason))
            continue

        _assign(var, converted)
        transferred += 1

    if verbose:
        print(f"[name] transferred {transferred}/{total} variables ({len(misses)} misses)")
        for path, torch_key, reason in misses[:10]:
            print(f"  miss: {path} <- {torch_key} ({reason})")
        if len(misses) > 10:
            print(f"  ... and {len(misses) - 10} more")

    return {
        "transferred": transferred,
        "skipped": len(misses),
        "total": total,
        "misses": misses,
    }


# --------------------------------------------------------------------------- #
# High-level driver
# --------------------------------------------------------------------------- #
def convert_weights(
    model,
    torch_weights_path: str,
    output_path: Optional[str] = None,
    method: str = "order",
    name_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Load a ``.pt`` file, transfer it into ``model`` and save ``.weights.h5``.

    Args:
        model: A built Keras model to receive the weights.
        torch_weights_path: Path to the source ``.pt`` checkpoint.
        output_path: Where to write the Keras weights. Defaults to
            ``<stem>.weights.h5`` next to the checkpoint's basename. Pass
            ``False`` to skip saving.
        method: ``"order"`` (recommended) or ``"name"``.
        name_mapping: Optional substring mapping for the ``"name"`` method.
        verbose: Print progress and a final summary.

    Returns:
        The transfer report from the chosen method.
    """
    torch_state = load_torch_state_dict(torch_weights_path)

    if method == "order":
        report = transfer_by_order(model, torch_state, verbose=verbose)
    elif method == "name":
        report = transfer_torch_to_keras(
            model, torch_state, name_mapping=name_mapping, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'order' or 'name'.")

    if output_path is not False:
        if output_path is None:
            stem = os.path.splitext(os.path.basename(torch_weights_path))[0]
            output_path = f"{stem}.weights.h5"
        model.save_weights(output_path)
        if verbose:
            print(f"Saved Keras weights to: {output_path}")

    if verbose:
        print(
            "Reminder: a clean transfer is necessary but not sufficient. "
            "Validate the converted model's outputs against the PyTorch "
            "reference before trusting it."
        )
    return report
