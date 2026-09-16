"""Shared model-assembly utilities for every kyolo detector.

Model files here follow a common recipe:

    inputs = image_input(input_shape, data_format)
    feats  = build_backbone_neck(inputs, ...)     # list [P3, P4, P5]
    model  = finalize_detector(inputs, feats, nc, reg_max, ...)

``finalize_detector`` attaches the shared decoupled head and returns a plain
``keras.Model`` whose output is the list of three raw feature maps
``[P3, P4, P5]`` (each ``B x H x W x (4*reg_max + nc)``). Keeping the model a
standard functional graph means it is trainable, fine-tunable and serialisable
with no special machinery - training targets/loss live in
:mod:`kyolo.losses` and post-processing in :mod:`kyolo.postprocessing`.

``load_pretrained_weights`` loads a converted checkpoint into such a model and
tolerates a different class count, which is what fine-tuning COCO weights on a
custom dataset needs.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
import zipfile

import keras
import numpy as np
from keras import ops

from ..heads.detect import detect_head
from ..layers.common import resolve_data_format
from ..layers.knames import kname, layers

__all__ = [
    "make_divisible",
    "scale_channels",
    "scale_depth",
    "image_input",
    "finalize_detector",
    "load_pretrained_weights",
    "DEFAULT_LOSS_CONFIG",
]


DEFAULT_LOSS_CONFIG = {
    "box_gain": 7.5,
    "cls_gain": 0.5,
    "dfl_gain": 1.5,
    "tal_topk": 10,
}


def make_divisible(x, divisor=8):
    """Round ``x`` up to the nearest multiple of ``divisor``."""
    return int(math.ceil(x / divisor) * divisor)


def scale_channels(channels, width, divisor=8, max_channels=None):
    """Scale a base channel count by ``width`` (kept divisible by ``divisor``)."""
    if max_channels is not None:
        channels = min(channels, max_channels)
    return make_divisible(channels * width, divisor)


def scale_depth(n, depth):
    """Scale a repeat count by ``depth`` (minimum 1)."""
    return max(round(n * depth), 1)


def image_input(input_shape=(640, 640, 3), data_format=None, name="images"):
    """Create the image ``Input`` tensor for the given shape / data format."""
    data_format = resolve_data_format(data_format)
    if input_shape is None:
        input_shape = (640, 640, 3)
    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3-tuple, got {input_shape}")

    if data_format == "channels_first":
        if input_shape[0] in (1, 3):
            c, h, w = input_shape
        else:
            h, w, c = input_shape
        shape = (c, h, w)
    else:
        if input_shape[-1] in (1, 3):
            h, w, c = input_shape
        else:
            c, h, w = input_shape
        shape = (h, w, c)
    return layers.Input(shape=shape, name=name)


def finalize_detector(
    inputs,
    feats,
    nc=80,
    reg_max=16,
    cls_dw=False,
    head_act=True,
    data_format=None,
    name="kyolo",
    head_name="head",
    strides=(8, 16, 32),
    end_to_end=False,
    backbone_end=None,
    loss_config=None,
):
    """Attach the shared head and return the assembled ``keras.Model``.

    The returned model carries a few attributes (``nc``, ``reg_max``,
    ``strides``, ``num_levels``, ``end_to_end``, ``head_prefix``,
    ``backbone_end``, ``loss_config``) that the loss / post-processor,
    :class:`kyolo.training.YOLODetector`, :func:`load_pretrained_weights` and
    :func:`kyolo.training.freeze_backbone` read. ``head_name`` is the layer-name
    prefix for the head (kept stable across variants so a single weight mapping
    works for a whole family). ``end_to_end`` marks a head trained with
    one-to-one assignment that must be decoded NMS-free; the heads built here
    are all one-to-many, so it stays ``False``. ``backbone_end`` is the index of
    the last backbone stage (``model.<backbone_end>``), i.e. everything before
    the neck's first upsample. ``loss_config`` overrides entries of
    :data:`DEFAULT_LOSS_CONFIG` for this family.
    """
    data_format = resolve_data_format(data_format)
    outputs = detect_head(
        feats,
        nc=nc,
        reg_max=reg_max,
        cls_dw=cls_dw,
        act=head_act,
        data_format=data_format,
        strides=strides,
        name=head_name,
    )
    model = keras.Model(inputs=inputs, outputs=outputs, name=kname(name))

    model.nc = nc
    model.reg_max = reg_max
    model.strides = tuple(strides)
    model.num_levels = len(outputs)
    model.data_format = data_format
    model.end_to_end = end_to_end
    model.head_prefix = kname(head_name)
    model.backbone_end = backbone_end
    model.loss_config = {**DEFAULT_LOSS_CONFIG, **(loss_config or {})}
    return model


def _class_branch_layers(model):
    """The head's classification-branch layers (``<head>.cv3.*``) that hold weights.

    These are the only layers whose weight shapes depend on ``nc`` (the final
    1x1 conv has ``nc`` outputs and the branch width is ``max(ch, min(nc, 100))``).
    """
    prefix = getattr(model, "head_prefix", None)
    if not prefix:
        return []
    return [l for l in model.layers if l.name.startswith(f"{prefix}-cv3-") and l.weights]


def _weights_h5_path(path, tmpdir):
    """Return a ``.weights.h5`` path for ``path``, extracting it from a ``.keras`` zip."""
    if not path.endswith(".keras"):
        return path
    with zipfile.ZipFile(path) as archive:
        return archive.extract("model.weights.h5", tmpdir)


def load_pretrained_weights(model, weights, verbose=True):
    """Load a converted Keras checkpoint into ``model``, tolerating a different ``nc``.

    A strict ``model.load_weights`` is tried first. If it fails, the mismatch is
    allowed only in the head's classification branch, whose weight shapes depend
    on the class count: every other layer is loaded strictly (so a checkpoint
    for the wrong variant still raises), and class-branch layers whose shapes
    happen to match are filled in while the rest keep their fresh
    initialisation and are re-learned during fine-tuning. This is the same
    "transfer everything that fits" recipe Ultralytics uses (``intersect_dicts``).

    Args:
        model: a kyolo detector built with the target ``nc``.
        weights: path to a ``.weights.h5`` (or ``.keras``) file produced by the
            per-model converter or by ``model.save_weights``.
        verbose: print a one-line summary when the class branch is re-initialised.

    Returns:
        ``{"reinitialized": [layer names left at their initial weights]}``.

    Raises:
        FileNotFoundError: if ``weights`` does not exist.
        ValueError: if the checkpoint does not match the architecture anywhere
            outside the classification branch.
    """
    if not os.path.exists(weights):
        raise FileNotFoundError(f"weights file not found: {weights!r}")

    try:
        model.load_weights(weights)
        return {"reinitialized": []}
    except ValueError as strict_error:
        cls_layers = _class_branch_layers(model)
        if not cls_layers:
            raise
        first_error = strict_error

    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping nested container")
        warnings.filterwarnings("ignore", message="A total of .* objects could not be loaded")
        h5_path = _weights_h5_path(weights, tmpdir)

        try:
            model.load_weights(h5_path, objects_to_skip=cls_layers)
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint {weights!r} does not match the {model.name} architecture "
                "(mismatches outside the head's classification branch, so this is not "
                "just a different class count). Check the variant, data layout and "
                f"deploy setting.\n\nOriginal error:\n{first_error}"
            ) from exc

        before = {l.name: [ops.convert_to_numpy(w) for w in l.weights] for l in cls_layers}
        model.load_weights(h5_path, skip_mismatch=True)

    reinitialized = [
        l.name
        for l in cls_layers
        if all(
            np.array_equal(a, ops.convert_to_numpy(b)) for a, b in zip(before[l.name], l.weights)
        )
    ]
    if verbose:
        n_weighted = sum(1 for l in model.layers if l.weights)
        print(
            f"Loaded {n_weighted - len(reinitialized)}/{n_weighted} weighted layers from "
            f"{os.path.basename(weights)}; {len(reinitialized)} classification-branch layers "
            f"({model.head_prefix}-cv3-*) keep their random init because nc={model.nc} "
            "differs from the checkpoint."
        )
    return {"reinitialized": reinitialized}
