"""Tests for the PyTorch -> Keras weight conversion driver.

The focus is the safety gate in ``convert_weights``: an incomplete transfer must
fail *before* anything is written, because a part-random ``.weights.h5`` loads
cleanly later and only shows up as bad predictions.

Skipped gracefully when no Keras backend or no torch is available (torch is only
needed to write the ``.pt`` fixtures, not to run kyolo).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

try:
    import torch

    from kyolo.conversion.convert import _SUFFIX_MAP, convert_weights
    from kyolo.models import yolov8n

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"keras backend / torch / kyolo.conversion unavailable: {_IMPORT_ERROR}",
)


@pytest.fixture(scope="module")
def model():
    """A small but real detector, so variable paths match a genuine conversion."""
    return yolov8n(nc=80, input_shape=(64, 64, 3), deploy=True)


def _torch_key(var):
    """Reproduce the Torch key that the name-based transfer derives for ``var``."""
    layer_path, leaf = var.path.rsplit("/", 1)
    return f"{layer_path.replace('-', '.')}.{_SUFFIX_MAP.get(leaf, leaf)}"


def _to_torch_layout(array, path):
    """Invert the Keras<-Torch transpose so the fixture has Torch-side shapes."""
    if path.endswith("kernel") and array.ndim == 4:
        return np.transpose(array, (3, 2, 0, 1))  # HWIO -> OIHW
    if path.endswith("kernel") and array.ndim == 2:
        return np.transpose(array, (1, 0))
    return array


def _complete_state(model):
    """A complete, correctly-named Torch state dict for ``model``."""
    return {
        _torch_key(v): _to_torch_layout(np.asarray(v).astype("float32") + 1.0, v.path)
        for v in model.weights
    }


def _write_pt(state, directory):
    path = os.path.join(str(directory), "checkpoint.pt")
    torch.save({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}, path)
    return path


def _convert(model, state, directory, **kwargs):
    """Run a conversion, returning the report and whether a file was written."""
    source = _write_pt(state, directory)
    output = os.path.join(str(directory), "converted.weights.h5")
    report = convert_weights(model, source, output_path=output, verbose=False, **kwargs)
    return report, os.path.exists(output)


def test_complete_conversion_saves(model, tmp_path):
    """The happy path: every variable filled, every tensor read, file written."""
    report, saved = _convert(model, _complete_state(model), tmp_path)
    assert saved
    assert report["skipped"] == 0
    assert report["unclaimed"] == []
    assert report["transferred"] == report["total"] == len(model.weights)


def test_zero_match_conversion_raises_without_writing(model, tmp_path):
    """The headline bug: a checkpoint matching *nothing* used to save happily.

    The result loads without complaint and predicts noise, and the only previous
    signal was a 'transferred 0/297' line scrolling past among other prints.
    """
    renamed = {f"backbone.{i}.weight": v for i, v in enumerate(_complete_state(model).values())}
    with pytest.raises(ValueError, match="matched nothing"):
        _convert(model, renamed, tmp_path)
    assert not os.path.exists(os.path.join(str(tmp_path), "converted.weights.h5"))


def test_partial_conversion_raises_without_writing(model, tmp_path):
    """A few unmatched variables are enough to refuse the save."""
    state = _complete_state(model)
    for key in list(state)[:5]:
        del state[key]
    with pytest.raises(ValueError, match="incomplete"):
        _convert(model, state, tmp_path)
    assert not os.path.exists(os.path.join(str(tmp_path), "converted.weights.h5"))


def test_unclaimed_torch_tensors_are_reported(model, tmp_path):
    """The reverse check: every Keras variable filled, but the .pt had leftovers.

    Without this a partially-overlapping checkpoint is indistinguishable from a
    correct one, since the Keras-side count is perfect.
    """
    state = _complete_state(model)
    state["model.99.unexpected.weight"] = np.zeros((4, 4), "float32")
    state["model.98.alsounexpected.bias"] = np.zeros((4,), "float32")
    with pytest.raises(ValueError, match="never used"):
        _convert(model, state, tmp_path)
    assert not os.path.exists(os.path.join(str(tmp_path), "converted.weights.h5"))


def test_shape_mismatch_raises_without_writing(model, tmp_path):
    state = _complete_state(model)
    state[next(iter(state))] = np.zeros((7, 7, 7, 7), "float32")
    with pytest.raises(ValueError, match="incomplete"):
        _convert(model, state, tmp_path)
    assert not os.path.exists(os.path.join(str(tmp_path), "converted.weights.h5"))


def test_dfl_projection_does_not_trip_the_gate(model, tmp_path):
    """Real Ultralytics checkpoints carry a DFL buffer kyolo never holds.

    kyolo folds the DFL integral into post-processing, so no Keras variable
    claims ``...dfl.conv.weight``. It must not be counted as unclaimed or every
    genuine conversion would now fail.
    """
    state = _complete_state(model)
    state["model.22.dfl.conv.weight"] = np.arange(16, dtype="float32").reshape(1, 16, 1, 1)
    report, saved = _convert(model, state, tmp_path)
    assert saved
    assert report["unclaimed"] == []


def test_allow_partial_warns_and_saves(model, tmp_path):
    """The escape hatch stays available, but it is loud."""
    state = _complete_state(model)
    for key in list(state)[:5]:
        del state[key]
    with pytest.warns(RuntimeWarning, match="partially-converted"):
        report, saved = _convert(model, state, tmp_path, allow_partial=True)
    assert saved
    assert report["skipped"] == 5


def test_error_message_names_the_offending_variables(model, tmp_path):
    """The message has to be actionable, not just a count."""
    state = _complete_state(model)
    dropped = list(state)[0]
    del state[dropped]
    with pytest.raises(ValueError) as excinfo:
        _convert(model, state, tmp_path)
    message = str(excinfo.value)
    assert dropped in message  # the specific torch key that was missing
    assert "load_pretrained_weights" in message  # the fine-tuning route
    assert "allow_partial" in message  # how to override
    assert "checkpoint.pt" in message  # which file was being converted
