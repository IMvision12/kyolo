"""Build + forward-pass smoke tests for a representative subset of models.

The whole module is skipped gracefully when no Keras backend (TensorFlow / JAX /
PyTorch) is importable, so the test suite still collects cleanly in a
backend-less environment. A small 256x256 input keeps every build fast.
"""

from __future__ import annotations

import pytest

# Skip the entire module if keras (or a backend, or kyolo.models) is unavailable.
try:
    import keras

    import kyolo.models as models

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    keras = None
    models = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"keras backend / kyolo.models unavailable: {_IMPORT_ERROR}",
)

# A representative small variant factory from every family.
MODEL_NAMES = [
    "yolov5n",
    "yolov6n",
    "yolov8n",
    "yolov9t",
    "yolov10n",
    "yolo11n",
    "yolo12n",
    "yolo26n",
    "yolov7_tiny",
]

NC = 80
REG_MAX = 16
INPUT_SIZE = 256
STRIDES = (8, 16, 32)
EXPECTED_CHANNELS = 4 * REG_MAX + NC  # 4 * 16 + 80 == 144


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_and_forward(name):
    """Each model builds and returns 3 raw feature maps with 144 channels."""
    factory = getattr(models, name)
    model = factory(nc=NC, input_shape=(INPUT_SIZE, INPUT_SIZE, 3), deploy=True)

    x = keras.ops.zeros((1, INPUT_SIZE, INPUT_SIZE, 3))
    outputs = model(x)

    assert isinstance(outputs, list), f"{name}: expected a list of feature maps"
    assert len(outputs) == 3, f"{name}: expected 3 feature maps, got {len(outputs)}"

    for feat, stride in zip(outputs, STRIDES):
        shape = tuple(feat.shape)
        assert shape[0] == 1, f"{name}: batch dim should be 1, got {shape}"
        assert shape[-1] == EXPECTED_CHANNELS, (
            f"{name}: channel dim should be {EXPECTED_CHANNELS}, got {shape[-1]}"
        )
        assert shape[1] == INPUT_SIZE // stride, f"{name}: bad H for stride {stride}: {shape}"
        assert shape[2] == INPUT_SIZE // stride, f"{name}: bad W for stride {stride}: {shape}"
