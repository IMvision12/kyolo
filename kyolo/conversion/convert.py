"""PyTorch -> Keras 3 weight conversion for the YOLO family.

This module moves parameters out of an official (Ultralytics / WongKinYiu)
``.pt`` checkpoint and into an equivalent :mod:`kyolo` Keras 3 model. Nothing
here depends on a particular Keras backend: reading/writing variables goes
through :meth:`keras.Variable.assign`, and tensors are shuttled around as plain
NumPy arrays.

Two transfer strategies are provided:

``transfer_torch_to_keras`` (method ``"name"``, the default -- use this)
    Derives a candidate Torch key for every Keras variable from its ``path``
    (e.g. ``model-0-conv/kernel`` -> ``model.0.conv.weight``) and looks it up
    directly. Keras layer names are hyphenated for torch-backend safety (see
    :mod:`kyolo.layers.knames`), so the separators are mapped back to ``.``.
    The :mod:`kyolo` layer names mirror the reference module paths, so matching
    is exact for the official checkpoints. Individual checkpoints (the detection
    head in particular) can still diverge and need a per-model ``name_mapping``.

``transfer_by_order`` (method ``"order"``, rarely usable)
    Walks the Keras variables and the Torch ``state_dict`` positionally and zips
    them together, needing only the counts and post-transpose shapes to agree.

    **This does not work for any of the families kyolo ships**, because the two
    build orders disagree inside every CSP block: kyolo emits ``cv1`` ->
    ``m.{i}`` -> ``cv2`` while the reference ``C2f.__init__`` registers ``cv1``
    -> ``cv2`` -> ``m.{i}``. Confirmed on a real ``yolov8n`` at ``model.2``::

        kyolo:       cv1.conv, cv1.bn, m.0.cv1, m.0.cv2, cv2.conv, cv2.bn
        ultralytics: cv1.conv, cv1.bn, cv2.conv, cv2.bn, m.0.cv1, m.0.cv2

    Same pattern in ``c3`` and ``c3k2``, so it affects v5, v8, v10, 11, 12 and
    26. It normally fails loudly on a shape mismatch, but a bottleneck's
    ``cv1``/``cv2`` have identical shapes at equal channel counts, so a silent
    swap is possible. Reach for it only when you have checked that the two
    orders line up for your specific checkpoint.

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

   What the reports *do* guarantee is completeness: :func:`convert_weights`
   refuses to save unless every Keras variable was filled and every Torch tensor
   was read, because a part-random checkpoint still loads cleanly and only shows
   up as bad predictions much later.
"""

from __future__ import annotations

import os
import warnings
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
# ``scale`` is kyolo's leaf for the YOLO12 A2C2f residual gamma (a bare Torch
# ``nn.Parameter`` named ``gamma``); no other kyolo layer emits a ``scale`` leaf.
_SUFFIX_MAP = {
    "kernel": "weight",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
    "bias": "bias",
    "scale": "gamma",
}

# Torch tensors that kyolo intentionally does not carry as model weights, so they
# must be dropped before an order-based (positional) transfer or the counts will
# not line up. The anchor-free Ultralytics heads store the fixed DFL projection
# ``[0, 1, ..., reg_max-1]`` as a frozen ``...dfl.conv.weight`` conv; kyolo folds
# that integral into post-processing instead, so it has no matching variable.
_NON_TRANSFERABLE_SUFFIXES = ("dfl.conv.weight",)


def _drop_non_transferable(torch_state: "Dict[str, np.ndarray]") -> "OrderedDict[str, np.ndarray]":
    """Return ``torch_state`` without buffers kyolo does not hold (e.g. DFL)."""
    return OrderedDict(
        (k, v)
        for k, v in torch_state.items()
        if not any(k.endswith(sfx) for sfx in _NON_TRANSFERABLE_SUFFIXES)
    )


def _unclaimed_torch_keys(torch_state, claimed) -> List[str]:
    """Torch tensors that no Keras variable took.

    Without this a *partially* overlapping checkpoint is indistinguishable from a
    correct one: counting only the Keras side cannot tell you that half the
    checkpoint went unread. Buffers kyolo deliberately does not hold (the frozen
    DFL projection) are not reported, since they are never claimed by design.
    """
    return [
        key
        for key in torch_state
        if key not in claimed and not any(key.endswith(s) for s in _NON_TRANSFERABLE_SUFFIXES)
    ]


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
            "Install it with `pip install torch` or `pip install kyolo[conversion]`. "
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
# Order-based transfer (positional; does not fit the shipped families)
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

    .. warning::
       *Within* a layer the grouping matches, but the layer order does not for
       any family kyolo ships: the CSP builders emit ``cv1`` -> ``m.{i}`` ->
       ``cv2`` where the reference registers ``cv1`` -> ``cv2`` -> ``m.{i}``.
       Use :func:`transfer_torch_to_keras` (``method="name"``) instead unless you
       have verified the orders agree for your checkpoint. See the module
       docstring.

    Args:
        keras_model: The target Keras model (already built).
        torch_state: Ordered ``name -> ndarray`` mapping from
            :func:`load_torch_state_dict`.
        verbose: Print a short progress summary.
        strict_shapes: Raise on any per-tensor shape mismatch. When ``False``,
            mismatches are skipped and counted instead.

    Returns:
        ``{"transferred", "skipped", "total", "misses", "unclaimed"}`` -- see
        :func:`transfer_torch_to_keras` for the shape of the last two.

    Raises:
        ValueError: If the number of Keras variables and Torch tensors differ,
            or (when ``strict_shapes``) on a shape mismatch.
    """
    keras_vars = [(v, _var_kind(v)) for v in keras_model.weights]
    torch_items: List[Tuple[str, np.ndarray]] = list(_drop_non_transferable(torch_state).items())

    if len(keras_vars) != len(torch_items):
        raise ValueError(
            "Parameter count mismatch between the Keras model and the Torch "
            f"checkpoint: Keras has {len(keras_vars)} variables but the "
            f"checkpoint has {len(torch_items)} tensors. This usually means the "
            "architecture/config does not match the weights (wrong variant, "
            "wrong `nc`, or a deploy/train mismatch)."
        )

    transferred = 0
    misses: List[Tuple[str, str, str]] = []
    claimed = set()
    for idx, ((var, kind), (tk, arr)) in enumerate(zip(keras_vars, torch_items)):
        converted = _transpose_for_kind(arr, kind)
        if tuple(converted.shape) != tuple(var.shape):
            reason = (
                f"shape mismatch: Keras expects {tuple(var.shape)} (kind={kind}) "
                f"but Torch is {tuple(arr.shape)} -> {tuple(converted.shape)} after transpose"
            )
            if strict_shapes:
                raise ValueError(f"[{idx}] {reason.replace('Keras', f'Keras {var.path!r}')}.")
            if verbose:
                print(f"  skip: [{idx}] {var.path} <- {tk} ({reason})")
            misses.append((var.path, tk, reason))
            continue
        _assign(var, converted)
        claimed.add(tk)
        transferred += 1

    total = len(keras_vars)
    unclaimed = _unclaimed_torch_keys(torch_state, claimed)
    if verbose:
        print(
            f"[order] transferred {transferred}/{total} variables"
            + (f" ({len(misses)} skipped)" if misses else "")
        )
    return {
        "transferred": transferred,
        "skipped": len(misses),
        "total": total,
        "misses": misses,
        "unclaimed": unclaimed,
    }


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

    For a variable whose path is ``model-0-conv/kernel`` this recovers the
    dotted layer path (``-`` -> ``.``) and builds the candidate key
    ``model.0.conv.weight`` (via the leaf-suffix rules ``kernel->weight``,
    ``gamma->weight``, ``beta->bias``, ``moving_mean->running_mean``,
    ``moving_variance->running_var``, ``bias->bias``) and then applies every
    ``old -> new`` substring replacement in ``name_mapping``, in order. If the
    resulting key exists in ``torch_state`` the value is transposed to the Keras
    layout and assigned; otherwise the variable is recorded as a miss.

    This is the recommended path. The dotted :mod:`kyolo` naming mirrors the
    reference modules, so matching is exact for the official checkpoints, but
    individual checkpoints (especially detection heads) can diverge and need a
    per-model ``name_mapping``. :func:`transfer_by_order` is *not* the fallback
    to reach for: the build orders disagree inside every CSP block (see the
    module docstring).

    Args:
        keras_model: The target Keras model (already built).
        torch_state: ``name -> ndarray`` mapping from
            :func:`load_torch_state_dict`.
        name_mapping: Ordered substring replacements applied to each derived
            Torch key. Defaults to :data:`kyolo.conversion.mappings.DEFAULT_MAPPING`.
        verbose: Print a short summary (and the first few misses).
        strict: Raise on the first miss / shape mismatch instead of recording it.

    Returns:
        ``{"transferred", "skipped", "total", "misses", "unclaimed"}`` where
        ``misses`` is a list of ``(keras_path, torch_key, reason)`` tuples for
        Keras variables that were not filled, and ``unclaimed`` lists Torch keys
        that no Keras variable took (excluding buffers kyolo never holds).
    """
    if name_mapping is None:
        name_mapping = DEFAULT_MAPPING

    transferred = 0
    misses: List[Tuple[str, str, str]] = []
    claimed = set()
    total = 0

    for var in keras_model.weights:
        total += 1
        path = var.path
        if "/" in path:
            layer_path, leaf = path.rsplit("/", 1)
        else:
            layer_path, leaf = path, _leaf(path)
        # Keras layer names are torch-safe (dots -> hyphens, see
        # kyolo.layers.knames); undo that to recover the dotted PyTorch key.
        layer_path = layer_path.replace("-", ".")
        suffix = _SUFFIX_MAP.get(leaf, leaf)
        torch_key = f"{layer_path}.{suffix}"
        for old, new in name_mapping.items():
            torch_key = torch_key.replace(old, new)

        if torch_key not in torch_state:
            if strict:
                raise KeyError(
                    f"No Torch parameter matched Keras variable '{path}' "
                    f"(tried '{torch_key}'). Adjust the name_mapping to suit "
                    "the checkpoint's layout."
                )
            misses.append((path, torch_key, "missing"))
            continue

        arr = torch_state[torch_key]
        converted = _transpose_for_kind(arr, _var_kind(var))
        if tuple(converted.shape) != tuple(var.shape):
            reason = f"shape {tuple(arr.shape)}->{tuple(converted.shape)} != {tuple(var.shape)}"
            if strict:
                raise ValueError(f"Shape mismatch for '{path}' <- '{torch_key}': {reason}.")
            misses.append((path, torch_key, reason))
            continue

        _assign(var, converted)
        claimed.add(torch_key)
        transferred += 1

    unclaimed = _unclaimed_torch_keys(torch_state, claimed)
    if verbose:
        print(f"[name] transferred {transferred}/{total} variables ({len(misses)} misses)")
        for path, torch_key, reason in misses[:10]:
            print(f"  miss: {path} <- {torch_key} ({reason})")
        if len(misses) > 10:
            print(f"  ... and {len(misses) - 10} more")
        if unclaimed:
            print(f"  {len(unclaimed)} Torch tensor(s) were never claimed, e.g.:")
            for key in unclaimed[:10]:
                print(f"    unused: {key}")
            if len(unclaimed) > 10:
                print(f"    ... and {len(unclaimed) - 10} more")

    return {
        "transferred": transferred,
        "skipped": len(misses),
        "total": total,
        "misses": misses,
        "unclaimed": unclaimed,
    }


# --------------------------------------------------------------------------- #
# High-level driver
# --------------------------------------------------------------------------- #
def _sample(items, render, limit=10):
    """Render up to ``limit`` items as indented lines, noting how many remain."""
    lines = [f"    {render(item)}" for item in items[:limit]]
    if len(items) > limit:
        lines.append(f"    ... and {len(items) - limit} more")
    return lines


def _incomplete_transfer_error(report, method) -> Optional[str]:
    """Describe an incomplete transfer, or return ``None`` if it was complete.

    Both directions matter. Unfilled Keras variables keep their random
    initialisation, and unread Torch tensors mean part of the checkpoint silently
    did not apply -- neither is visible from the transferred count alone.
    """
    total = report["total"]
    transferred = report["transferred"]
    misses = report.get("misses", [])
    unclaimed = report.get("unclaimed", [])
    if not misses and not unclaimed:
        return None

    if transferred == 0:
        lines = [
            f"Weight conversion matched nothing: 0 of {total} Keras variables were "
            f"filled with method={method!r}, so the model is still entirely randomly "
            "initialised. Saving this checkpoint would produce a file that loads "
            "cleanly and predicts noise."
        ]
    else:
        lines = [
            f"Weight conversion was incomplete with method={method!r}: "
            f"{transferred} of {total} Keras variables were filled."
        ]

    if misses:
        lines.append("")
        lines.append(f"{len(misses)} Keras variable(s) kept their random initialisation:")
        lines += _sample(misses, lambda m: f"{m[0]} <- {m[1]} ({m[2]})")
    if unclaimed:
        lines.append("")
        lines.append(f"{len(unclaimed)} Torch tensor(s) in the checkpoint were never used:")
        lines += _sample(unclaimed, str)

    lines.append("")
    lines.append(
        "This usually means the checkpoint does not correspond to the model that "
        "was built: wrong family or variant, wrong `nc`, a deploy/train mismatch, "
        "or a `name_mapping` that no longer fits the checkpoint's layout."
    )
    lines.append(
        "To fine-tune on a different class count, convert with the checkpoint's own "
        "`nc` and then load with `kyolo.models.load_pretrained_weights`, which is "
        "built to re-initialise just the classification branch."
    )
    lines.append(
        "If a partial transfer really is what you want (e.g. while bringing up a new "
        "family), pass `allow_partial=True` to downgrade this error to a warning."
    )
    return "\n".join(lines)


def convert_weights(
    model,
    torch_weights_path: str,
    output_path: Optional[str] = None,
    method: str = "name",
    name_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    allow_partial: bool = False,
) -> Dict[str, object]:
    """Load a ``.pt`` file, transfer it into ``model`` and save ``.weights.h5``.

    An incomplete transfer raises **before** anything is written, so a
    partially-converted checkpoint never reaches disk. Both directions are
    checked: Keras variables no Torch tensor filled (they would keep their random
    initialisation) and Torch tensors no Keras variable claimed (part of the
    checkpoint silently did not apply).

    Args:
        model: A built Keras model to receive the weights.
        torch_weights_path: Path to the source ``.pt`` checkpoint.
        output_path: Where to write the Keras weights. Defaults to
            ``<stem>.weights.h5`` next to the checkpoint's basename. Pass
            ``False`` to skip saving.
        method: ``"name"`` (default, recommended for the official checkpoints)
            or ``"order"``. kyolo layer names mirror the Ultralytics modules, so
            name matching is exact; the positional ``"order"`` transfer instead
            needs the Keras build order to match the Torch registration order,
            which does not hold for the CSP blocks (kyolo builds ``cv1, m..., cv2``
            while Ultralytics registers ``cv1, cv2, m...``), so prefer ``"name"``.
        name_mapping: Optional substring mapping for the ``"name"`` method.
        verbose: Print progress and a final summary.
        allow_partial: Downgrade an incomplete transfer from an error to a
            warning and save anyway. Only useful while bringing up a new family;
            the resulting checkpoint is part random.

    Returns:
        The transfer report from the chosen method, with ``"misses"`` and
        ``"unclaimed"`` describing anything that did not line up.

    Raises:
        ValueError: If the transfer was incomplete and ``allow_partial`` is
            ``False``, or if ``method`` is unknown.
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

    # Gate the save: a checkpoint full of random initialisation loads without
    # complaint later, so this is the last place the mismatch can be caught.
    problem = _incomplete_transfer_error(report, method)
    if problem is not None:
        if not allow_partial:
            raise ValueError(f"{problem}\n\nSource checkpoint: {torch_weights_path!r}")
        warnings.warn(
            f"Saving a partially-converted checkpoint (allow_partial=True).\n{problem}",
            RuntimeWarning,
            stacklevel=2,
        )

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
