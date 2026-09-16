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
    from keras import ops

    from kyolo.models import yolov8n
    from kyolo.postprocessing import YOLOPostprocessor, batched_nms
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


def _np(x):
    return ops.convert_to_numpy(x)


requires_tensorflow = pytest.mark.skipif(
    keras is None or keras.backend.backend() != "tensorflow",
    reason="dynamic (None) shapes and SavedModel export are TensorFlow-specific; "
    "JAX and PyTorch always see static shapes",
)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocessor_letterbox_shapes():
    """A (480, 640, 3) uint8 image is letterboxed to (1, 640, 640, 3)."""
    image = np.random.randint(0, 256, size=(480, 640, 3), dtype="uint8")

    preprocessor = YOLOPreprocessor(image_size=640, normalize=True, letterbox=True)
    out = preprocessor(image)

    assert set(out.keys()) >= {"images", "ratio", "pad"}
    assert tuple(out["images"].shape) == (1, 640, 640, 3)
    assert tuple(out["ratio"].shape) == (1, 2)
    assert tuple(out["pad"].shape) == (1, 2)


def test_preprocessor_scales_uint8_and_0_255_floats_to_unit_range():
    pre = YOLOPreprocessor(image_size=64)  # square input -> no padding involved
    uint8_img = np.full((64, 64, 3), 200, dtype="uint8")
    float_img = np.full((64, 64, 3), 200.0, dtype="float32")
    np.testing.assert_allclose(_np(pre(uint8_img)["images"]), 200 / 255.0, rtol=1e-5)
    np.testing.assert_allclose(_np(pre(float_img)["images"]), 200 / 255.0, rtol=1e-5)


def test_preprocessor_auto_range_is_per_image():
    """A [0, 1] image and a 0-255 image in one batch are scaled independently."""
    pre = YOLOPreprocessor(image_size=64)
    batch = np.stack(
        [np.full((64, 64, 3), 0.5, dtype="float32"), np.full((64, 64, 3), 200.0, dtype="float32")]
    )
    images = _np(pre(batch)["images"])
    np.testing.assert_allclose(images[0], 0.5, rtol=1e-5)
    np.testing.assert_allclose(images[1], 200 / 255.0, rtol=1e-5)


def test_preprocessor_explicit_input_range_overrides_heuristic():
    dark_0_255 = np.full((64, 64, 3), 0.5, dtype="float32")  # darker than 1/255
    bright_unit = np.full((64, 64, 3), 200.0, dtype="float32")
    forced_255 = YOLOPreprocessor(image_size=64, input_range=(0, 255))
    forced_unit = YOLOPreprocessor(image_size=64, input_range=(0, 1))
    np.testing.assert_allclose(_np(forced_255(dark_0_255)["images"]), 0.5 / 255.0, rtol=1e-5)
    np.testing.assert_allclose(_np(forced_unit(bright_unit)["images"]), 200.0, rtol=1e-5)
    with pytest.raises(ValueError):
        YOLOPreprocessor(image_size=64, input_range=(0, 100))


def test_preprocessor_normalize_false_leaves_pixels_and_pad_untouched():
    image = np.full((32, 64, 3), 200.0, dtype="float32")  # 2:1 -> letterbox pads top/bottom
    raw = _np(YOLOPreprocessor(image_size=64, normalize=False)(image)["images"])[0]
    unit = _np(YOLOPreprocessor(image_size=64, normalize=True)(image)["images"])[0]
    # image rows keep their raw value, pad rows carry the raw 114 gray
    assert raw[32, 0, 0] == pytest.approx(200.0)
    assert raw[0, 0, 0] == pytest.approx(114.0)
    assert unit[32, 0, 0] == pytest.approx(200 / 255.0, rel=1e-5)
    assert unit[0, 0, 0] == pytest.approx(114 / 255.0, rel=1e-5)


def test_preprocessor_mean_std_only_when_normalizing():
    image = np.full((64, 64, 3), 255.0, dtype="float32")
    with_norm = YOLOPreprocessor(image_size=64, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(_np(with_norm(image)["images"]), 1.0, rtol=1e-5)
    without = YOLOPreprocessor(
        image_size=64, normalize=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )
    np.testing.assert_allclose(_np(without(image)["images"]), 255.0, rtol=1e-5)


def test_preprocessor_round_trips_through_config():
    pre = YOLOPreprocessor(image_size=64, input_range=(0, 255), normalize=False)
    clone = YOLOPreprocessor.from_config(pre.get_config())
    assert clone.input_range == (0, 255) and clone.normalize is False


def test_preprocessor_letterbox_geometry_matches_reference():
    """``ratio`` / ``pad`` reproduce Ultralytics' LetterBox for odd aspect ratios."""
    for h, w in [(480, 640), (1080, 810), (333, 500), (427, 640), (721, 1281)]:
        out = YOLOPreprocessor(image_size=640)(np.zeros((h, w, 3), "float32"))
        # reference: r = min(new/h, new/w); dw/dh = (new - round(side * r)) / 2
        r = min(640 / h, 640 / w)
        dw = (640 - int(round(w * r))) / 2
        dh = (640 - int(round(h * r))) / 2
        np.testing.assert_allclose(_np(out["ratio"])[0], [r, r], rtol=1e-6)
        np.testing.assert_allclose(_np(out["pad"])[0], [dw, dh], rtol=1e-6)
        assert tuple(out["images"].shape) == (1, 640, 640, 3)


def test_preprocessor_compute_output_shape():
    """Declared statically so symbolic use never traces ``call``."""
    pre = YOLOPreprocessor(image_size=64)
    shapes = pre.compute_output_shape((None, 40, 50, 3))
    assert shapes["images"] == (None, 64, 64, 3)
    assert shapes["ratio"] == (None, 2) and shapes["pad"] == (None, 2)
    # a single (H, W, C) image still comes back batched
    assert pre.compute_output_shape((40, 50, 3))["images"] == (1, 64, 64, 3)
    # channels_first swaps the layout
    first = YOLOPreprocessor(image_size=64, data_format="channels_first")
    assert first.compute_output_shape((None, 40, 50, 3))["images"] == (None, 3, 64, 64)
    # auto padding is input-dependent, so the spatial dims are unknown
    auto = YOLOPreprocessor(image_size=64, auto=True)
    assert auto.compute_output_shape((None, 40, 50, 3))["images"] == (None, None, None, 3)


def test_preprocessor_in_functional_model():
    """The preprocessor can be attached symbolically and stay traceable.

    ``ops.image.resize`` requires a static Python ``size``, so the letterbox
    geometry must be computed from the static shape; deriving it from
    ``ops.shape`` made the layer unusable in any graph.
    """
    inputs = keras.Input(shape=(40, 50, 3))
    pre = YOLOPreprocessor(image_size=64)
    model = keras.Model(inputs, pre(inputs)["images"])
    assert tuple(model.output_shape) == (None, 64, 64, 3)
    out = model(np.random.rand(2, 40, 50, 3).astype("float32"))
    assert tuple(out.shape) == (2, 64, 64, 3)


@requires_tensorflow
def test_preprocessor_traceable_in_graph_mode():
    """It survives ``tf.function`` (static and dynamic batch) and ``tf.data``."""
    import tensorflow as tf

    pre = YOLOPreprocessor(image_size=64)
    image = np.random.rand(2, 40, 50, 3).astype("float32")

    static = tf.function(
        lambda x: pre(x), input_signature=[tf.TensorSpec([2, 40, 50, 3], tf.float32)]
    )
    np.testing.assert_allclose(_np(static(image)["images"]), _np(pre(image)["images"]), atol=1e-5)

    # only the batch dimension may stay dynamic
    dynamic = tf.function(
        lambda x: pre(x), input_signature=[tf.TensorSpec([None, 40, 50, 3], tf.float32)]
    )
    for batch in (1, 4):
        out = dynamic(np.random.rand(batch, 40, 50, 3).astype("float32"))
        assert tuple(out["images"].shape) == (batch, 64, 64, 3)
        assert tuple(out["ratio"].shape) == (batch, 2)

    # and it works inside a tf.data pipeline
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(4, 40, 50, 3).astype("float32"))
    dataset = dataset.map(lambda x: pre(x)["images"])
    assert tuple(next(iter(dataset)).shape) == (1, 64, 64, 3)


@requires_tensorflow
def test_letterbox_rejects_unknown_spatial_dims():
    """A dynamic H/W cannot work, so it must fail loudly rather than mid-graph."""
    import tensorflow as tf

    from kyolo.layers.letterbox import Letterbox

    layer = Letterbox(64)
    with pytest.raises(ValueError, match="statically-known spatial dimensions"):
        tf.function(
            lambda x: layer(x),
            input_signature=[tf.TensorSpec([1, None, None, 3], tf.float32)],
        ).get_concrete_function()


@requires_tensorflow
def test_end_to_end_pipeline_exports_to_saved_model():
    """preprocess -> model -> postprocess exports and reloads.

    This is the payoff for making the letterbox and NMS graph-safe: the whole
    pipeline can now be captured in a SavedModel.
    """
    import tempfile

    import tensorflow as tf

    detector = yolov8n(nc=NC, input_shape=(64, 64, 3))
    pre = YOLOPreprocessor(image_size=64)
    post = YOLOPostprocessor(
        nc=NC, reg_max=detector.reg_max, strides=detector.strides, max_detections=10
    )
    inputs = keras.Input(shape=(40, 50, 3))
    full = keras.Model(inputs, post(detector(pre(inputs)["images"])))
    assert tuple(full.output_shape) == (None, 10, 6)

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/exported"
        full.export(path)
        # keep a reference: the loaded object owns the variables `serve` closes over
        reloaded = tf.saved_model.load(path)
        served = reloaded.serve(np.random.rand(1, 40, 50, 3).astype("float32"))
        assert tuple(served.shape) == (1, 10, 6)
        assert np.isfinite(_np(served)).all()


# --------------------------------------------------------------------------- #
# Postprocessing / NMS
# --------------------------------------------------------------------------- #
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


def _reference_nms(boxes, scores, classes, iou_threshold, conf_threshold, max_detections):
    """Plain-Python greedy class-aware NMS; returns kept indices in score order."""

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
        return inter / (area(a) + area(b) - inter + 1e-7)

    keep = []
    for i in np.argsort(-scores, kind="stable"):
        if scores[i] <= conf_threshold:
            break
        if all(classes[j] != classes[i] or iou(boxes[i], boxes[j]) <= iou_threshold for j in keep):
            keep.append(int(i))
            if len(keep) == max_detections:
                break
    return keep


def _clustered_boxes(rng, batch, n, clusters=40, num_classes=3):
    """Boxes jittered around cluster centres so plenty of them overlap."""
    centres = rng.uniform(50, 600, (batch, clusters, 2))
    boxes = np.zeros((batch, n, 4), np.float32)
    classes = np.zeros((batch, n), np.int32)
    for b in range(batch):
        for k in range(n):
            c = centres[b, k % clusters] + rng.normal(0, 6, 2)
            w, h = rng.uniform(30, 80, 2)
            boxes[b, k] = (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)
            classes[b, k] = (k % clusters) % num_classes
    scores = rng.uniform(0, 1, (batch, n)).astype(np.float32)
    return boxes, scores, classes


@pytest.mark.parametrize("max_detections", [60, 500])
def test_batched_nms_matches_reference(max_detections):
    """The while_loop sweep is exact greedy NMS (also when max_det exceeds N)."""
    rng = np.random.default_rng(0)
    boxes, scores, classes = _clustered_boxes(rng, batch=2, n=400)
    out = _np(batched_nms(boxes, scores, classes, 0.5, max_detections, 0.25))
    assert out.shape == (2, max_detections, 6)
    for b in range(2):
        keep = _reference_nms(boxes[b], scores[b], classes[b], 0.5, 0.25, max_detections)
        got = out[b][out[b][:, 4] > 0]
        expected = np.concatenate(
            [boxes[b][keep], scores[b][keep, None], classes[b][keep, None].astype(np.float32)],
            axis=1,
        )
        assert len(got) == len(keep)
        np.testing.assert_allclose(got, expected, atol=1e-4)
        # padding rows are all zero
        assert not out[b][len(keep) :].any()


def test_batched_nms_pre_nms_topk_limits_candidates():
    """With a cap, NMS runs over exactly the top-K scoring boxes of each image."""
    rng = np.random.default_rng(2)
    boxes, scores, classes = _clustered_boxes(rng, batch=2, n=400)
    out = _np(batched_nms(boxes, scores, classes, 0.5, 30, 0.0, pre_nms_topk=100))
    for b in range(2):
        top = np.argsort(-scores[b], kind="stable")[:100]
        keep = _reference_nms(boxes[b][top], scores[b][top], classes[b][top], 0.5, 0.0, 30)
        got = out[b][out[b][:, 4] > 0]
        np.testing.assert_allclose(got[:, :4], boxes[b][top][keep], atol=1e-4)


def test_batched_nms_handles_negative_coordinates_and_empty_images():
    rng = np.random.default_rng(1)
    boxes, scores, classes = _clustered_boxes(rng, batch=2, n=100)
    boxes -= 400.0  # push most coordinates negative
    scores[1] = 0.0  # second image has nothing above threshold
    out = _np(batched_nms(boxes, scores, classes, 0.5, 30, 0.25))
    keep = _reference_nms(boxes[0], scores[0], classes[0], 0.5, 0.25, 30)
    got = out[0][out[0][:, 4] > 0]
    np.testing.assert_allclose(got[:, :4], boxes[0][keep], atol=1e-4)
    assert not out[1].any()


@requires_tensorflow
@pytest.mark.parametrize("n", [5, 120, 999, 1000, 3000])
def test_batched_nms_with_dynamic_anchor_count(n):
    """NMS traces with an unknown ``N`` and matches the statically-shaped result.

    ``ops.top_k`` needs a static ``k`` and raises "input must have at least k
    columns" when handed a narrower tensor, so a dynamic ``N`` -- the normal case
    inside a ``tf.function`` or an exported SavedModel -- has to be padded up to
    ``k`` rather than clamped against.
    """
    import tensorflow as tf

    rng = np.random.default_rng(3)
    boxes, scores, classes = _clustered_boxes(rng, batch=2, n=n)

    traced = tf.function(
        lambda b, s, c: batched_nms(b, s, c, 0.5, 50, 0.25, 1000),
        input_signature=[
            tf.TensorSpec([None, None, 4], tf.float32),
            tf.TensorSpec([None, None], tf.float32),
            tf.TensorSpec([None, None], tf.int32),
        ],
    )
    dynamic = _np(traced(boxes, scores, classes))
    static = _np(batched_nms(boxes, scores, classes, 0.5, 50, 0.25, 1000))

    assert np.isfinite(dynamic).all()
    np.testing.assert_allclose(dynamic, static, atol=1e-4)


@requires_tensorflow
def test_postprocessor_with_dynamic_feature_shapes():
    """The full decode + NMS pipeline traces with unknown feature-map sizes."""
    import tensorflow as tf

    post = YOLOPostprocessor(
        nc=NC, reg_max=REG_MAX, strides=(8, 16, 32), conf_threshold=0.05, max_detections=50
    )
    spec = [tf.TensorSpec([None, None, None, CHANNELS], tf.float32) for _ in range(3)]
    traced = tf.function(lambda feats: post(feats), input_signature=[spec])

    rng = np.random.default_rng(4)
    for size in (64, 160):
        feats = [
            rng.normal(0, 1, (1, size // s, size // s, CHANNELS)).astype("float32")
            for s in (8, 16, 32)
        ]
        dynamic = _np(traced(feats))
        static = _np(post([ops.convert_to_tensor(f) for f in feats]))
        assert dynamic.shape == (1, 50, 6)
        np.testing.assert_allclose(dynamic, static, atol=1e-4)


def test_postprocessor_in_functional_model():
    """The postprocessor can be attached symbolically (no NMS tracing needed)."""
    model = yolov8n(nc=NC, input_shape=(128, 128, 3))
    post = YOLOPostprocessor(nc=NC, reg_max=model.reg_max, strides=model.strides, max_detections=50)
    full = keras.Model(model.inputs, post(model.outputs))
    assert tuple(full.output_shape) == (None, 50, 6)
    out = full(np.zeros((1, 128, 128, 3), np.float32))
    assert tuple(out.shape) == (1, 50, 6)
