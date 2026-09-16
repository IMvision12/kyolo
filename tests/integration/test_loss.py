"""Numerical tests for the loss pieces: ``BboxLoss``, the assigner, ``E2EDetectionLoss``.

These pin the arithmetic against hand-computed expectations and against
structural properties (resolution invariance, one-to-one assignment), so they
run on any backend and need no reference implementation. The diff against the
real Ultralytics criterion lives in ``test_loss_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import keras
    from keras import ops

    from kyolo.losses import BboxLoss, DistributionFocalLoss, E2EDetectionLoss, YOLODetectionLoss
    from kyolo.models import yolo26n, yolov8n
    from kyolo.ops.anchors import bbox2dist
    from kyolo.ops.tal import TaskAlignedAssigner
    from kyolo.training import ProgressiveLossSchedule, YOLODetector

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
STRIDES = (8, 16, 32)


def make_feats(rng, no, size=SIZE, batch=BATCH):
    return [rng.normal(size=(batch, size // s, size // s, no)).astype("float32") for s in STRIDES]


def make_targets(rng, nc=NC, batch=BATCH, size=SIZE, side=(0.1, 0.4)):
    boxes = np.zeros((batch, MAX_BOXES, 4), np.float32)
    labels = np.zeros((batch, MAX_BOXES), np.int32)
    mask = np.zeros((batch, MAX_BOXES), np.float32)
    for b in range(batch):
        for i in range(MAX_BOXES):
            cx, cy = rng.uniform(0.3, 0.7, 2) * size
            w, h = rng.uniform(*side, 2) * size
            boxes[b, i] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            labels[b, i] = rng.integers(nc)
            mask[b, i] = 1.0
    return {"boxes": boxes, "labels": labels, "mask": mask}


def one_positive(pred_left, target_left=0.0, imgsz=(64.0, 64.0), stride=8.0, anchor=4.0):
    """A single-anchor, single-positive ``BboxLoss`` call in stride units.

    The anchor sits at ``anchor`` and the target box spans ``[target_left, 0,
    2*anchor, 2*anchor]``, so the target LTRB is ``(anchor - target_left,
    anchor, anchor, anchor)``. Only the left distance is perturbed, which makes
    the expected L1 easy to write down by hand.
    """
    target = np.array([[[target_left, 0.0, 2 * anchor, 2 * anchor]]], "float32")
    pred_ltrb = np.array([[[pred_left, anchor, anchor, anchor]]], "float32")
    return {
        "pred_dist": ops.convert_to_tensor(pred_ltrb),
        "pred_bboxes": ops.convert_to_tensor(target),
        "anchors": ops.convert_to_tensor(np.full((1, 1, 2), anchor, "float32")),
        "target_bboxes": ops.convert_to_tensor(target),
        "target_scores": ops.convert_to_tensor(np.ones((1, 1, 1), "float32")),
        "target_scores_sum": ops.convert_to_tensor(1.0),
        "fg_mask": ops.convert_to_tensor(np.ones((1, 1), "float32")),
        "imgsz": ops.convert_to_tensor(np.array(imgsz, "float32")),
        "stride": ops.convert_to_tensor(np.full((1, 1, 1), stride, "float32")),
    }


def test_bbox_loss_selects_distance_term_by_reg_max():
    assert BboxLoss(16).use_dfl is True
    assert BboxLoss(1).use_dfl is False
    assert isinstance(BboxLoss(16).dfl, DistributionFocalLoss)
    assert BboxLoss(1).dfl is None
    with pytest.raises(ValueError, match="reg_max >= 2"):
        DistributionFocalLoss(1)


def test_normalized_l1_is_a_fraction_of_image_size():
    """The reg_max=1 distance term is the mean |LTRB error| over image size.

    One stride unit of error on the left edge at stride 8 is 8 px; over a 64 px
    image that is 1/8 of the width, averaged over the four sides -> 0.03125.
    """
    box_loss, dist_loss = BboxLoss(1)(**one_positive(pred_left=5.0))
    assert float(ops.convert_to_numpy(dist_loss)) == pytest.approx(0.125 / 4, rel=1e-6)

    assert float(ops.convert_to_numpy(box_loss)) < 1e-6


def test_normalized_l1_is_resolution_invariant():
    """The same error *as a fraction of the image* costs the same at any size.

    This is the whole point of dividing by the image size rather than leaving
    the term in stride units, where a P5 error would count 4x a P3 one.
    """
    small = BboxLoss(1)(**one_positive(pred_left=5.0, imgsz=(64.0, 64.0), stride=8.0))[1]

    large = BboxLoss(1)(**one_positive(pred_left=5.0, imgsz=(128.0, 128.0), stride=16.0))[1]
    assert float(ops.convert_to_numpy(small)) == pytest.approx(
        float(ops.convert_to_numpy(large)), rel=1e-6
    )


def test_normalized_l1_keeps_the_sign_of_the_error():
    """Over- and under-shooting by the same amount cost the same.

    ``bbox2dist`` must not be clipped to ``[0, ...)`` on this path: a target
    distance can legitimately be negative for a background anchor, and clipping
    would also collapse over-shoots.
    """
    over = BboxLoss(1)(**one_positive(pred_left=5.0))[1]
    under = BboxLoss(1)(**one_positive(pred_left=3.0))[1]
    assert float(ops.convert_to_numpy(over)) == pytest.approx(
        float(ops.convert_to_numpy(under)), rel=1e-6
    )


def test_bbox_loss_ignores_background_rows():
    """Background rows cannot contribute, even carrying absurd targets.

    Ultralytics selects foreground rows by boolean indexing; kyolo weights a
    dense tensor by a zero mask instead, which is only equivalent if no
    background row is non-finite (``0 * nan == nan``).
    """
    kwargs = one_positive(pred_left=5.0)
    reference = float(ops.convert_to_numpy(BboxLoss(1)(**kwargs)[1]))

    def pad(name, extra):
        value = ops.convert_to_numpy(kwargs[name])
        return ops.convert_to_tensor(np.concatenate([value, extra], axis=1))

    padded = dict(kwargs)
    padded["pred_dist"] = pad("pred_dist", np.array([[[1e6, -1e6, 1e6, -1e6]]], "float32"))
    padded["pred_bboxes"] = pad("pred_bboxes", np.zeros((1, 1, 4), "float32"))
    padded["target_bboxes"] = pad("target_bboxes", np.zeros((1, 1, 4), "float32"))
    padded["anchors"] = pad("anchors", np.zeros((1, 1, 2), "float32"))
    padded["target_scores"] = pad("target_scores", np.zeros((1, 1, 1), "float32"))
    padded["fg_mask"] = ops.convert_to_tensor(np.array([[1.0, 0.0]], "float32"))
    padded["stride"] = pad("stride", np.full((1, 1, 1), 8.0, "float32"))

    box_loss, dist_loss = BboxLoss(1)(**padded)
    assert np.isfinite(float(ops.convert_to_numpy(box_loss)))
    assert float(ops.convert_to_numpy(dist_loss)) == pytest.approx(reference, rel=1e-6)


def test_bbox2dist_rejects_unusable_reg_max():
    """``clip(dist, 0, reg_max - 1 - 0.01)`` is nonsense at reg_max=1."""
    anchors = ops.zeros((1, 1, 2))
    boxes = ops.zeros((1, 1, 4))
    with pytest.raises(ValueError, match="no DFL bin range"):
        bbox2dist(anchors, boxes, 1)

    out = bbox2dist(ops.convert_to_tensor(np.zeros((1, 1, 2), "float32")), boxes)
    assert tuple(out.shape) == (1, 1, 4)


@pytest.mark.parametrize("reg_max,name", [(16, "dfl"), (1, "l1")])
def test_distance_term_is_reported_under_its_own_name(reg_max, name):
    loss = YOLODetectionLoss(nc=NC, reg_max=reg_max, strides=STRIDES)
    assert loss.dist_name == name
    rng = np.random.default_rng(0)
    out = loss(make_feats(rng, loss.no), make_targets(rng))
    assert name in out and "dist" in out
    assert float(ops.convert_to_numpy(out[name])) == float(ops.convert_to_numpy(out["dist"]))

    assert ("l1" if reg_max > 1 else "dfl") not in out


def test_dfl_free_loss_has_a_live_distance_term():
    """At reg_max=1 the third term used to be a hard-coded 0.0."""
    rng = np.random.default_rng(0)
    loss = YOLODetectionLoss(nc=NC, reg_max=1, strides=STRIDES)
    out = loss(make_feats(rng, loss.no), make_targets(rng))
    dist = float(ops.convert_to_numpy(out["dist"]))
    assert dist > 0.0

    total = float(ops.convert_to_numpy(out["loss"]))
    expected = BATCH * (
        loss.box_gain * float(ops.convert_to_numpy(out["box"]))
        + loss.cls_gain * float(ops.convert_to_numpy(out["cls"]))
        + loss.dfl_gain * dist
    )
    assert total == pytest.approx(expected, rel=1e-5)


def test_class_weights_scale_the_classification_term():
    rng = np.random.default_rng(0)
    feats, targets = make_feats(rng, 4 * 16 + NC), make_targets(rng)
    plain = YOLODetectionLoss(nc=NC, reg_max=16, strides=STRIDES)(feats, targets)
    weighted = YOLODetectionLoss(nc=NC, reg_max=16, strides=STRIDES, class_weights=[2.0] * NC)(
        feats, targets
    )
    assert float(ops.convert_to_numpy(weighted["cls"])) == pytest.approx(
        2.0 * float(ops.convert_to_numpy(plain["cls"])), rel=1e-5
    )

    assert float(ops.convert_to_numpy(weighted["box"])) == pytest.approx(
        float(ops.convert_to_numpy(plain["box"])), rel=1e-6
    )


@pytest.mark.parametrize("factory,name", [(yolov8n, "dfl_loss"), (yolo26n, "l1_loss")])
def test_detector_names_the_metric_after_the_active_term(factory, name):
    model = factory(nc=NC, input_shape=(SIZE, SIZE, 3))
    detector = YOLODetector(model)
    assert {m.name for m in (detector._box, detector._cls, detector._dist)} == {
        "box_loss",
        "cls_loss",
        name,
    }


def test_detector_builds_the_loss_from_the_model_config():
    """``model.loss_config`` is what makes the right criterion automatic."""
    model = yolov8n(nc=NC, input_shape=(SIZE, SIZE, 3))
    assert model.loss_config["tal_topk"] == 10
    assert model.loss_config["box_gain"] == 7.5
    detector = YOLODetector(model)
    assert detector.loss_fn.tal_topk == 10
    assert detector.loss_fn.box_gain == 7.5
    assert detector.loss_fn.nc == NC

    model.loss_config = {**model.loss_config, "box_gain": 1.25, "tal_topk": 4}
    assert YOLODetector(model).loss_fn.box_gain == 1.25
    assert YOLODetector(model).loss_fn.assigner.topk == 4


def assigner_case(seed=0, side=3.0, n_gt=4, batch=2):
    """Anchors in pixels plus predictions that overlap the GT boxes."""
    rng = np.random.default_rng(seed)
    from kyolo.ops.anchors import make_anchors

    shapes = [(SIZE // s, SIZE // s) for s in STRIDES]
    anchors, stride_t = make_anchors(shapes, STRIDES)
    anchors = (ops.convert_to_numpy(anchors) * ops.convert_to_numpy(stride_t)).astype("float32")
    a = anchors.shape[0]

    boxes = np.zeros((batch, n_gt, 4), "float32")
    labels = np.zeros((batch, n_gt), "int32")
    for b in range(batch):
        for g in range(n_gt):
            cx, cy = rng.uniform(side + 2, SIZE - side - 2, 2)
            boxes[b, g] = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
            labels[b, g] = rng.integers(NC)
    mask = np.ones((batch, n_gt), "float32")

    pred = np.zeros((batch, a, 4), "float32")
    for b in range(batch):
        pred[b] = boxes[b][rng.integers(0, n_gt, size=a)] + rng.normal(0.0, 1.0, size=(a, 4))
    scores = rng.uniform(0.0, 1.0, size=(batch, a, NC)).astype("float32")
    return anchors, scores.astype("float32"), pred, boxes, labels, mask


def assign(tal, case):
    anchors, scores, pred, boxes, labels, mask = case
    return tal(
        ops.convert_to_tensor(scores),
        ops.convert_to_tensor(pred),
        ops.convert_to_tensor(anchors),
        ops.convert_to_tensor(labels),
        ops.convert_to_tensor(boxes),
        ops.convert_to_tensor(mask),
    )


def test_small_target_expansion_finds_positives_sub_stride_boxes_would_miss():
    """A 3 px box is smaller than the 8 px anchor spacing, so it can fall between
    anchor centres and never be supervised. Growing it to one stride for
    candidate selection is Ultralytics' STAL."""
    case = assigner_case(side=3.0)
    args = {"topk": 10, "num_classes": NC, "alpha": 0.5, "beta": 6.0}

    *_, without = assign(TaskAlignedAssigner(strides=(1, 1, 1), **args), case)
    *_, with_stal = assign(TaskAlignedAssigner(strides=STRIDES, **args), case)
    n_without = int(ops.convert_to_numpy(ops.sum(ops.cast(without > 0, "int32"))))
    n_with = int(ops.convert_to_numpy(ops.sum(ops.cast(with_stal > 0, "int32"))))
    assert n_with > n_without, (n_with, n_without)

    big = assigner_case(side=40.0)
    *_, big_without = assign(TaskAlignedAssigner(strides=(1, 1, 1), **args), big)
    *_, big_with = assign(TaskAlignedAssigner(strides=STRIDES, **args), big)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(big_without) > 0, ops.convert_to_numpy(big_with) > 0
    )


def test_expansion_does_not_resurrect_padded_ground_truth():
    """Padded rows are all-zero, so every side is "small"; without the mask gate
    they would become real stride-sized boxes at the origin."""
    anchors, scores, pred, boxes, labels, mask = assigner_case(side=20.0, n_gt=3)
    mask[:, 1:] = 0.0
    boxes[:, 1:] = 0.0
    tal = TaskAlignedAssigner(topk=10, num_classes=NC, alpha=0.5, beta=6.0, strides=STRIDES)
    _, target_bboxes, _, fg = assign(tal, (anchors, scores, pred, boxes, labels, mask))
    assigned = ops.convert_to_numpy(target_bboxes)[ops.convert_to_numpy(fg) > 0]

    assert len(assigned) > 0
    assert not np.any(np.all(np.isclose(assigned, 0.0), axis=-1))


def test_topk2_makes_the_assignment_one_to_one():
    case = assigner_case(side=20.0, n_gt=4)
    args = {"topk": 7, "num_classes": NC, "alpha": 0.5, "beta": 6.0, "strides": STRIDES}
    *_, many = assign(TaskAlignedAssigner(**args), case)
    *_, one = assign(TaskAlignedAssigner(topk2=1, **args), case)
    n_many = int(ops.convert_to_numpy(ops.sum(ops.cast(many > 0, "int32"))))
    n_one = int(ops.convert_to_numpy(ops.sum(ops.cast(one > 0, "int32"))))

    assert n_one <= 8
    assert n_one < n_many

    assert TaskAlignedAssigner(topk=7, strides=STRIDES).topk2 == 7


def e2e_inputs(reg_max=1, seed=0):
    rng = np.random.default_rng(seed)
    no = 4 * reg_max + NC if reg_max > 1 else 4 + NC
    preds = {"one2many": make_feats(rng, no), "one2one": make_feats(rng, no)}
    return preds, make_targets(rng)


def test_e2e_loss_weights_the_two_branches_progressively():
    preds, targets = e2e_inputs()
    loss = E2EDetectionLoss(nc=NC, reg_max=1, strides=STRIDES, epochs=10)
    assert loss.o2m_gain == pytest.approx(0.8)
    assert loss.o2o_gain == pytest.approx(0.2)

    out = loss(preds, targets)
    o2m = loss.one2many(preds["one2many"], targets)
    o2o = loss.one2one(preds["one2one"], targets)
    expected = 0.8 * float(ops.convert_to_numpy(o2m["loss"])) + 0.2 * float(
        ops.convert_to_numpy(o2o["loss"])
    )
    assert float(ops.convert_to_numpy(out["loss"])) == pytest.approx(expected, rel=1e-5)

    assert float(ops.convert_to_numpy(out["box"])) == pytest.approx(
        float(ops.convert_to_numpy(o2o["box"])), rel=1e-6
    )
    assert "l1" in out

    for _ in range(9):
        loss.update()
    assert loss.o2m_gain == pytest.approx(0.1)
    assert loss.o2o_gain == pytest.approx(0.9)

    loss.update()
    assert loss.o2m_gain == pytest.approx(0.1)


def test_e2e_loss_branches_use_one_to_many_and_one_to_one_assignment():
    loss = E2EDetectionLoss(nc=NC, reg_max=1, strides=STRIDES)
    assert loss.one2many.assigner.topk == 10
    assert loss.one2many.assigner.topk2 == 10
    assert loss.one2one.assigner.topk == 7
    assert loss.one2one.assigner.topk2 == 1


def test_e2e_loss_can_sum_the_branches_without_a_schedule():
    preds, targets = e2e_inputs()
    loss = E2EDetectionLoss(nc=NC, reg_max=1, strides=STRIDES, progressive=False)
    assert (loss.o2m_gain, loss.o2o_gain) == (1.0, 1.0)
    loss.update()
    assert (loss.o2m_gain, loss.o2o_gain) == (1.0, 1.0)
    out = loss(preds, targets)
    expected = float(
        ops.convert_to_numpy(loss.one2many(preds["one2many"], targets)["loss"])
    ) + float(ops.convert_to_numpy(loss.one2one(preds["one2one"], targets)["loss"]))
    assert float(ops.convert_to_numpy(out["loss"])) == pytest.approx(expected, rel=1e-5)


def test_e2e_loss_requires_both_branches():
    loss = E2EDetectionLoss(nc=NC, reg_max=1, strides=STRIDES)
    rng = np.random.default_rng(0)
    feats, targets = make_feats(rng, 4 + NC), make_targets(rng)
    with pytest.raises(ValueError, match="one2many"):
        loss(feats, targets)
    with pytest.raises(ValueError, match="one2many"):
        loss({"one2one": feats}, targets)


def test_progressive_schedule_callback_advances_once_per_epoch():
    loss = E2EDetectionLoss(nc=NC, reg_max=1, strides=STRIDES, epochs=5)
    callback = ProgressiveLossSchedule(loss=loss)
    logs = {}
    callback.on_epoch_end(0, logs)
    assert loss.updates == 1
    assert logs["o2m_gain"] == pytest.approx(loss.o2m_gain)
    assert loss.o2m_gain < 0.8

    plain = ProgressiveLossSchedule(loss=YOLODetectionLoss(nc=NC, reg_max=16, strides=STRIDES))
    with pytest.raises(ValueError, match="update"):
        plain.on_train_begin()
