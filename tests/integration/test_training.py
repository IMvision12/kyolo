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
except Exception as exc:  # pragma: no cover - depends on environment
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


def _synthetic_batch(rng, nc=NC):
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


def _generator(seed=0):
    rng = np.random.default_rng(seed)
    while True:
        yield _synthetic_batch(rng)


@pytest.fixture(scope="module")
def trained_detector():
    """A detector that has been through ``fit()`` (so its state is built)."""
    detector = YOLODetector(yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3)))
    detector.compile(optimizer=keras.optimizers.Adam(1e-3))
    history = detector.fit(_generator(), epochs=2, steps_per_epoch=2, verbose=0)
    return detector, history


def test_detect_works_after_fit(trained_detector):
    """``detect()`` must not try to add state to an already-built model."""
    detector, _ = trained_detector
    images, _ = _synthetic_batch(np.random.default_rng(1))
    dets = detector.detect(images, conf_threshold=0.1, iou_threshold=0.5, max_detections=40)
    assert tuple(dets.shape) == (BATCH, 40, 6)
    # the thresholds apply per call, not just on the first one
    dets2 = detector.detect(images, conf_threshold=0.9, max_detections=10)
    assert tuple(dets2.shape) == (BATCH, 10, 6)
    assert detector._postprocessor.conf_threshold == 0.9


def test_loss_is_reported_and_reset(trained_detector):
    """``loss`` is logged next to the component losses and its tracker resets per epoch."""
    detector, history = trained_detector
    assert set(history.history) == {"loss", "box_loss", "cls_loss", "dfl_loss"}
    # Mean.count accumulates the sample weight (= batch size) per step; after
    # 2 epochs of 2 steps it must hold only the last epoch: 2 steps * BATCH.
    assert int(ops.convert_to_numpy(detector._loss_tracker.count)) == 2 * BATCH
    result = detector.evaluate(_generator(seed=3), steps=1, return_dict=True, verbose=0)
    assert "loss" in result and np.isfinite(result["loss"])


def test_assigner_targets_carry_no_gradient():
    """TAL targets are constants w.r.t. the predictions (Ultralytics runs it under no_grad)."""
    loss = YOLODetectionLoss(nc=NC, reg_max=16, strides=(8, 16, 32))
    rng = np.random.default_rng(0)
    _, targets = _synthetic_batch(rng)
    shapes = [(SIZE // s, SIZE // s) for s in loss.strides]
    feats = [rng.normal(size=(BATCH, h, w, loss.no)).astype("float32") for h, w in shapes]

    def target_score_sum(fs):
        flat = ops.concatenate([ops.reshape(f, (BATCH, -1, loss.no)) for f in fs], axis=1)
        pred_dist, pred_scores = flat[..., : loss.no - NC], flat[..., loss.no - NC :]
        anchors, stride_t = make_anchors(shapes, loss.strides)
        dist = loss._dfl_decode(pred_dist)
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
            # fully detached from the inputs: no gradient path at all
            grads = [torch.zeros_like(t) for t in tensors]
        else:
            grads = torch.autograd.grad(value, tensors, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(t) for g, t in zip(grads, tensors)]
    else:  # pragma: no cover
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
    # backbone / neck / box branch transferred ...
    for name in ("model-0-conv", "model-12-cv1-conv", f"{target.head_prefix}-cv2-0-2"):
        got = ops.convert_to_numpy(target.get_layer(name).weights[0])
        want = ops.convert_to_numpy(source.get_layer(name).weights[0])
        np.testing.assert_allclose(got, want)
    # ... while the class branch kept its (deterministic) init: the class
    # output conv bias is the prior -log((1 - 0.01) / 0.01).
    bias = ops.convert_to_numpy(target.get_layer(f"{target.head_prefix}-cv3-0-2").bias)
    np.testing.assert_allclose(bias, -np.log(99.0), rtol=1e-5)
    # the fresh model still trains
    detector = YOLODetector(target)
    detector.compile(optimizer=keras.optimizers.Adam(1e-3))
    detector.fit(_generator(), epochs=1, steps_per_epoch=1, verbose=0)


def test_load_pretrained_weights_reports_reinitialized_layers(tmp_path):
    source = yolov8n(nc=80, input_shape=(SIZE, SIZE, 3))
    path = str(tmp_path / "yolov8n.weights.h5")
    source.save_weights(path)
    target = yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3))
    report = load_pretrained_weights(target, path, verbose=False)
    assert report["reinitialized"]
    assert all(n.startswith(f"{target.head_prefix}-cv3-") for n in report["reinitialized"])
    # same nc -> exact load, nothing re-initialised
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
    # neck and head stay trainable
    assert all(layer.trainable for layer in model.layers if layer.name.startswith("model-12-"))
    assert all(layer.trainable for layer in model.layers if layer.name.startswith("model-22-"))
    assert not any(layer.trainable for layer in frozen)


def test_yolo26_detector_trains_and_detects():
    """The DFL-free head (reg_max=1) goes through fit() and detect() as well."""
    model = yolo26n(nc=NC, input_shape=(SIZE, SIZE, 3))
    assert model.end_to_end is False  # kyolo builds the one-to-many head -> NMS decode
    detector = YOLODetector(model)
    detector.compile(optimizer=keras.optimizers.Adam(1e-3))
    detector.fit(_generator(), epochs=1, steps_per_epoch=1, verbose=0)
    images, _ = _synthetic_batch(np.random.default_rng(2))
    assert tuple(detector.detect(images).shape) == (BATCH, 300, 6)
