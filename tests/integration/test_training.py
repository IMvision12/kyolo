"""Training, fine-tuning and loss-target tests for ``YOLODetector``.

Everything runs on a tiny 128x128 nano model with synthetic boxes so the whole
module stays fast on CPU. The module is skipped gracefully when no Keras backend
is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import keras
    from keras import ops

    from kyolo.losses import YOLODetectionLoss
    from kyolo.models import load_pretrained_weights, yolo26n, yolov8n, yolov8s
    from kyolo.ops.anchors import dist2bbox, make_anchors
    from kyolo.training import YOLODetector, freeze_backbone

    _IMPORT_ERROR = None
except Exception as exc:
    keras = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"keras backend / kyolo unavailable: {_IMPORT_ERROR}",
)

SIZE = 128
NC = 3
BATCH = 2
MAX_BOXES = 5


def synthetic_batch(rng, nc=NC):
    """One ``(images, targets)`` batch of random images with random xyxy boxes."""
    images = rng.random((BATCH, SIZE, SIZE, 3), dtype=np.float32)
    boxes = np.zeros((BATCH, MAX_BOXES, 4), np.float32)
    labels = np.zeros((BATCH, MAX_BOXES), np.int32)
    mask = np.zeros((BATCH, MAX_BOXES), np.float32)
    for b in range(BATCH):
        for i in range(int(rng.integers(1, MAX_BOXES + 1))):
            cx, cy = rng.uniform(0.25, 0.75, 2) * SIZE
            w, h = rng.uniform(0.1, 0.4, 2) * SIZE
            boxes[b, i] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            labels[b, i] = rng.integers(nc)
            mask[b, i] = 1.0
    return images, {"boxes": boxes, "labels": labels, "mask": mask}


def generator(seed=0):
    rng = np.random.default_rng(seed)
    while True:
        yield synthetic_batch(rng)


@pytest.fixture(scope="module")
def trained_detector():
    """A detector that has been through ``fit()`` (so its state is built)."""
    detector = YOLODetector(yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3)))
    detector.compile(optimizer=keras.optimizers.Adam(1e-3))
    history = detector.fit(generator(), epochs=2, steps_per_epoch=2, verbose=0)
    return detector, history


def test_detect_works_after_fit(trained_detector):
    """``detect()`` must not try to add state to an already-built model."""
    detector, _ = trained_detector
    images, _ = synthetic_batch(np.random.default_rng(1))
    dets = detector.detect(images, conf_threshold=0.1, iou_threshold=0.5, max_detections=40)
    assert tuple(dets.shape) == (BATCH, 40, 6)

    dets2 = detector.detect(images, conf_threshold=0.9, max_detections=10)
    assert tuple(dets2.shape) == (BATCH, 10, 6)
    assert detector._postprocessor.conf_threshold == 0.9


def test_loss_is_reported_and_reset(trained_detector):
    """``loss`` is logged next to the component losses and its tracker resets per epoch."""
    detector, history = trained_detector
    assert set(history.history) == {"loss", "box_loss", "cls_loss", "dfl_loss"}

    assert int(ops.convert_to_numpy(detector._loss_tracker.count)) == 2 * BATCH
    result = detector.evaluate(generator(seed=3), steps=1, return_dict=True, verbose=0)
    assert "loss" in result and np.isfinite(result["loss"])


def test_assigner_targets_carry_no_gradient():
    """TAL targets are constants w.r.t. the predictions (Ultralytics runs it under no_grad)."""
    loss = YOLODetectionLoss(nc=NC, reg_max=16, strides=(8, 16, 32))
    rng = np.random.default_rng(0)
    _, targets = synthetic_batch(rng)
    shapes = [(SIZE // s, SIZE // s) for s in loss.strides]
    feats = [rng.normal(size=(BATCH, h, w, loss.no)).astype("float32") for h, w in shapes]

    def target_score_sum(fs):
        flat = ops.concatenate([ops.reshape(f, (BATCH, -1, loss.no)) for f in fs], axis=1)
        pred_dist, pred_scores = flat[..., : loss.no - NC], flat[..., loss.no - NC :]
        anchors, stride_t = make_anchors(shapes, loss.strides)
        dist = loss.dfl_decode(pred_dist)
        boxes = dist2bbox(dist, ops.expand_dims(anchors, 0), xywh=False)
        boxes = boxes * ops.reshape(stride_t, (1, -1, 1))
        _, _, target_scores, _ = loss.assigner(
            ops.sigmoid(pred_scores),
            boxes,
            anchors * stride_t,
            targets["labels"],
            targets["boxes"],
            targets["mask"],
        )
        return ops.sum(target_scores)

    backend = keras.backend.backend()
    if backend == "jax":
        import jax

        grads = jax.grad(target_score_sum)([ops.convert_to_tensor(f) for f in feats])
    elif backend == "tensorflow":
        import tensorflow as tf

        variables = [tf.Variable(f) for f in feats]
        with tf.GradientTape() as tape:
            value = target_score_sum(variables)
        grads = [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(tape.gradient(value, variables), variables)
        ]
    elif backend == "torch":
        import torch

        tensors = [torch.tensor(f, requires_grad=True) for f in feats]
        value = target_score_sum(tensors)
        if not value.requires_grad:
            grads = [torch.zeros_like(t) for t in tensors]
        else:
            grads = torch.autograd.grad(value, tensors, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(t) for g, t in zip(grads, tensors)]
    else:
        pytest.skip(f"no autodiff helper for backend {backend!r}")

    assert max(float(ops.max(ops.abs(g))) for g in grads) == 0.0


def test_loss_finite_on_background_only_batch():
    loss = YOLODetectionLoss(nc=NC, reg_max=16, strides=(8, 16, 32))
    feats = [
        ops.zeros((BATCH, SIZE // s, SIZE // s, loss.no), dtype="float32") for s in loss.strides
    ]
    targets = {
        "boxes": np.zeros((BATCH, MAX_BOXES, 4), np.float32),
        "labels": np.zeros((BATCH, MAX_BOXES), np.int32),
        "mask": np.zeros((BATCH, MAX_BOXES), np.float32),
    }
    out = loss(feats, targets)
    assert np.isfinite(float(ops.convert_to_numpy(out["loss"])))


def test_load_pretrained_weights_with_different_nc(tmp_path):
    """COCO-style weights load into a model with another class count."""
    source = yolov8n(nc=80, input_shape=(SIZE, SIZE, 3))
    path = str(tmp_path / "yolov8n.weights.h5")
    source.save_weights(path)

    target = yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3), weights=path)

    for name in ("model-0-conv", "model-12-cv1-conv", f"{target.head_prefix}-cv2-0-2"):
        got = ops.convert_to_numpy(target.get_layer(name).weights[0])
        want = ops.convert_to_numpy(source.get_layer(name).weights[0])
        np.testing.assert_allclose(got, want)

    for i, stride in enumerate(target.strides):
        bias = ops.convert_to_numpy(target.get_layer(f"{target.head_prefix}-cv3-{i}-2").bias)
        np.testing.assert_allclose(bias, np.log(5.0 / NC / (640.0 / stride) ** 2), rtol=1e-5)

    detector = YOLODetector(target)
    detector.compile(optimizer=keras.optimizers.Adam(1e-3))
    detector.fit(generator(), epochs=1, steps_per_epoch=1, verbose=0)


def test_load_pretrained_weights_reports_reinitialized_layers(tmp_path):
    source = yolov8n(nc=80, input_shape=(SIZE, SIZE, 3))
    path = str(tmp_path / "yolov8n.weights.h5")
    source.save_weights(path)
    target = yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3))
    report = load_pretrained_weights(target, path, verbose=False)
    assert report["reinitialized"]
    assert all(n.startswith(f"{target.head_prefix}-cv3-") for n in report["reinitialized"])

    same = yolov8n(nc=80, input_shape=(SIZE, SIZE, 3))
    assert load_pretrained_weights(same, path, verbose=False)["reinitialized"] == []


def test_load_pretrained_weights_rejects_wrong_variant(tmp_path):
    source = yolov8n(nc=80, input_shape=(SIZE, SIZE, 3))
    path = str(tmp_path / "yolov8n.weights.h5")
    source.save_weights(path)
    with pytest.raises(ValueError, match="does not match"):
        yolov8s(nc=NC, input_shape=(SIZE, SIZE, 3), weights=path)
    with pytest.raises(FileNotFoundError):
        yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3), weights=str(tmp_path / "missing.weights.h5"))


def test_freeze_backbone():
    model = yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3))
    assert model.backbone_end == 9
    frozen = freeze_backbone(model)
    assert frozen, "no layers were frozen"
    frozen_names = {layer.name for layer in frozen}
    assert "model-0-conv" in frozen_names and "model-9-cv2-conv" in frozen_names

    assert all(layer.trainable for layer in model.layers if layer.name.startswith("model-12-"))
    assert all(layer.trainable for layer in model.layers if layer.name.startswith("model-22-"))
    assert not any(layer.trainable for layer in frozen)


def test_yolo26_detector_trains_and_detects():
    """The DFL-free head (reg_max=1) goes through fit() and detect() as well."""
    model = yolo26n(nc=NC, input_shape=(SIZE, SIZE, 3))
    assert model.end_to_end is False
    detector = YOLODetector(model)
    detector.compile(optimizer=keras.optimizers.Adam(1e-3))
    detector.fit(generator(), epochs=1, steps_per_epoch=1, verbose=0)
    images, _ = synthetic_batch(np.random.default_rng(2))
    assert tuple(detector.detect(images).shape) == (BATCH, 300, 6)


@pytest.mark.parametrize("factory,reg_max", [(yolov8n, 16), (yolo26n, 1)])
def test_head_bias_init_matches_reference(factory, reg_max):
    """The head reproduces Ultralytics' ``Detect.bias_init``."""
    model = factory(nc=NC, input_shape=(SIZE, SIZE, 3))
    assert model.reg_max == reg_max
    for i, stride in enumerate(model.strides):
        box_bias = ops.convert_to_numpy(model.get_layer(f"{model.head_prefix}-cv2-{i}-2").bias)
        cls_bias = ops.convert_to_numpy(model.get_layer(f"{model.head_prefix}-cv3-{i}-2").bias)

        np.testing.assert_allclose(box_bias, 1.0, rtol=1e-6)

        np.testing.assert_allclose(cls_bias, np.log(5.0 / NC / (640.0 / stride) ** 2), rtol=1e-5)

    first = ops.convert_to_numpy(model.get_layer(f"{model.head_prefix}-cv3-0-2").bias)
    assert float(first[0]) < -np.log(99.0)


def test_dfl_free_head_predicts_non_degenerate_boxes():
    """Regression guard: a zero box bias makes reg_max=1 heads untrainable.

    With ``reg_max == 1`` there is no DFL softmax -- the four box channels *are*
    the predicted distances -- so a zero output bias yields zero-area boxes, an
    exactly-zero IoU, an exactly-zero ``iou ** beta`` alignment metric, no
    weighted positives, and therefore zero box/DFL gradient forever.
    """
    model = yolo26n(nc=NC, input_shape=(SIZE, SIZE, 3))
    loss = YOLODetectionLoss(nc=NC, reg_max=1, strides=model.strides)
    assert loss.use_dfl is False

    rng = np.random.default_rng(0)
    images, targets = synthetic_batch(rng)
    feats = model(ops.convert_to_tensor(images))

    flat, shapes = [], []
    for f in feats:
        h, w = f.shape[1], f.shape[2]
        shapes.append((h, w))
        flat.append(ops.reshape(f, (BATCH, h * w, loss.no)))
    raw = ops.concatenate(flat, axis=1)
    anchors, stride_t = make_anchors(shapes, model.strides)
    boxes = dist2bbox(
        raw[..., : loss.no - NC],
        ops.expand_dims(ops.cast(anchors, "float32"), 0),
        xywh=False,
        axis=-1,
    )
    b = ops.convert_to_numpy(boxes)
    areas = np.maximum(b[..., 2] - b[..., 0], 0.0) * np.maximum(b[..., 3] - b[..., 1], 0.0)
    assert areas.mean() > 0.0, "reg_max=1 head predicts degenerate zero-area boxes"

    out = loss.compute(feats, targets)
    assert float(ops.convert_to_numpy(out["box"])) > 0.0


def test_loss_runs_under_mixed_precision():
    """The loss computes in float32 even when the head emits float16."""
    original = keras.mixed_precision.global_policy()
    try:
        keras.mixed_precision.set_global_policy("mixed_float16")
        model = yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3))
        loss = YOLODetectionLoss(nc=NC, reg_max=16, strides=model.strides)
        images, targets = synthetic_batch(np.random.default_rng(0))
        feats = model(ops.convert_to_tensor(images))
        assert keras.backend.standardize_dtype(feats[0].dtype) == "float16"
        out = loss.compute(feats, targets)
        for key in ("loss", "box", "cls", "dfl"):
            value = ops.convert_to_numpy(out[key])
            assert keras.backend.standardize_dtype(out[key].dtype) == "float32"
            assert np.isfinite(value).all(), f"{key} is not finite under mixed_float16"
    finally:
        keras.mixed_precision.set_global_policy(original)
