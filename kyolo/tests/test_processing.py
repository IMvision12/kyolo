"""Tests for the pre- and post-processing helpers.

The module is skipped gracefully when no Keras backend (or the corresponding
kyolo subpackage) is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if keras / a backend / the kyolo subpackages are missing.
try:
    import keras

    from kyolo.postprocessing import YOLOPostprocessor
    from kyolo.preprocessing import YOLOPreprocessor

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    keras = None
    YOLOPostprocessor = None
    YOLOPreprocessor = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"keras backend / kyolo processing unavailable: {_IMPORT_ERROR}",
)

NC = 80
REG_MAX = 16
CHANNELS = 4 * REG_MAX + NC  # 144
MAX_DETECTIONS = 300


def test_preprocessor_letterbox_shapes():
    """A (480, 640, 3) uint8 image is letterboxed to (1, 640, 640, 3)."""
    image = np.random.randint(0, 256, size=(480, 640, 3), dtype="uint8")

    preprocessor = YOLOPreprocessor(image_size=640, normalize=True, letterbox=True)
    out = preprocessor(image)

    assert set(out.keys()) >= {"images", "ratio", "pad"}
    assert tuple(out["images"].shape) == (1, 640, 640, 3)
    assert tuple(out["ratio"].shape) == (1, 2)
    assert tuple(out["pad"].shape) == (1, 2)


def test_postprocessor_output_shape():
    """Raw feats for a 256 input postprocess to (1, max_detections, 6)."""
    # Feature maps for a 256x256 input at strides (8, 16, 32).
    feats = [
        keras.ops.zeros((1, 32, 32, CHANNELS)),
        keras.ops.zeros((1, 16, 16, CHANNELS)),
        keras.ops.zeros((1, 8, 8, CHANNELS)),
    ]

    postprocessor = YOLOPostprocessor(
        nc=NC,
        reg_max=REG_MAX,
        strides=(8, 16, 32),
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_detections=MAX_DETECTIONS,
        end_to_end=False,
    )
    detections = postprocessor(feats)

    shape = tuple(detections.shape)
    assert shape[0] == 1
    assert shape[1] == MAX_DETECTIONS
    assert shape[2] == 6
