"""Numerical parity against the real Ultralytics criterion.

Diffs kyolo's ``YOLODetectionLoss`` / ``TaskAlignedAssigner`` against
``ultralytics.utils.loss.v8DetectionLoss`` and
``ultralytics.utils.tal.TaskAlignedAssigner`` on identical inputs. This is what
backs the claim that kyolo trains the same criterion rather than merely a
similar one, so it compares the assignment (``fg_mask``, ``target_bboxes``,
``target_scores``) as well as the four scalars.

Skipped unless ultralytics is installed *and* the Keras backend is torch, so
both sides run the same kernels on the same device and the residual is float32
rounding rather than a backend difference. Cross-backend agreement is a separate
concern, covered by the rest of the suite.

Run with::

    pip install ultralytics
    KERAS_BACKEND=torch pytest tests/integration/test_loss_parity.py

Tolerances are *relative*: the classification term sums ``B * A * nc`` elements
and lands around 400, where one float32 ulp is already ~3e-5, so an absolute
1e-5 bound is below the representable noise floor.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

keras = pytest.importorskip("keras")
ultralytics = pytest.importorskip("ultralytics")
torch = pytest.importorskip("torch")

from keras import ops

from kyolo.losses import YOLODetectionLoss
from kyolo.ops.anchors import make_anchors
from kyolo.ops.tal import TaskAlignedAssigner

pytestmark = pytest.mark.skipif(
    keras.backend.backend() != "torch",
    reason=f"parity runs on the torch backend (current: {keras.backend.backend()})",
)

NC = 4
IMG = 64
STRIDES = (8, 16, 32)
BATCH = 3
N_GT = 5
RTOL = 1e-5


def cpu():
    """A fresh "run on the cpu" scope.

    Keras' torch backend allocates on cuda when one is visible, while the
    ultralytics reference below stays on the cpu. Comparing across devices would
    fold cpu-vs-gpu kernel differences into the residual, and ``fg_mask`` is an
    exact comparison that a flipped top-k tie would break, so both sides are
    pinned to the cpu. ``keras.device`` scopes are single-use, hence a factory
    rather than one shared object.
    """
    return keras.device("cpu")


def head_outputs(reg_max, seed, tiny=False):
    """Random head feature maps plus equal-count GT, in kyolo's conventions."""
    rng = np.random.default_rng(seed)
    no = 4 * reg_max + NC if reg_max > 1 else 4 + NC
    feats = [
        rng.normal(0.0, 1.0, size=(BATCH, IMG // s, IMG // s, no)).astype("float32")
        for s in STRIDES
    ]
    boxes = np.zeros((BATCH, N_GT, 4), "float32")
    labels = np.zeros((BATCH, N_GT), "int32")
    for b in range(BATCH):
        for g in range(N_GT):
            side = rng.uniform(2.0, 6.0) if tiny else rng.uniform(8.0, 28.0)
            cx, cy = rng.uniform(side / 2 + 1, IMG - side / 2 - 1, 2)
            boxes[b, g] = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
            labels[b, g] = rng.integers(0, NC)

    return feats, boxes, labels, np.ones((BATCH, N_GT), "float32")


def fake_model(reg_max):
    """The minimum surface ``v8DetectionLoss.__init__`` touches."""
    detect = SimpleNamespace(
        nc=NC, reg_max=reg_max, stride=torch.tensor(STRIDES, dtype=torch.float)
    )
    return SimpleNamespace(
        args=SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=100),
        model=[detect],
        parameters=lambda: iter([torch.zeros(1)]),
    )


def ultralytics_loss(feats, boxes, labels, mask, reg_max):
    """``(total, [box, cls, dist])`` with the gains already applied."""
    from ultralytics.utils.loss import v8DetectionLoss

    criterion = v8DetectionLoss(fake_model(reg_max))
    no = 4 * reg_max + NC if reg_max > 1 else 4 + NC

    torch_feats = [torch.from_numpy(f).permute(0, 3, 1, 2).contiguous() for f in feats]
    flat = torch.cat([f.view(BATCH, no, -1) for f in torch_feats], 2)
    pred_boxes, pred_scores = flat.split((no - NC, NC), 1)

    batch_idx, cls, bboxes = [], [], []
    for b in range(BATCH):
        for g in range(N_GT):
            if mask[b, g] == 0:
                continue
            x1, y1, x2, y2 = boxes[b, g]
            batch_idx.append(b)
            cls.append(labels[b, g])
            bboxes.append(
                [(x1 + x2) / 2 / IMG, (y1 + y2) / 2 / IMG, (x2 - x1) / IMG, (y2 - y1) / IMG]
            )
    batch = {
        "batch_idx": torch.tensor(batch_idx, dtype=torch.float),
        "cls": torch.tensor(cls, dtype=torch.float).view(-1, 1),
        "bboxes": torch.tensor(bboxes, dtype=torch.float),
    }
    total, components = criterion.loss(
        {"boxes": pred_boxes, "scores": pred_scores, "feats": torch_feats}, batch
    )
    return float(total.sum()), [float(v) for v in components]


def kyolo_loss(feats, boxes, labels, mask, reg_max):
    criterion = YOLODetectionLoss(
        nc=NC, reg_max=reg_max, strides=STRIDES, data_format="channels_last"
    )
    with cpu():
        out = criterion(
            [ops.convert_to_tensor(f) for f in feats],
            {
                "boxes": ops.convert_to_tensor(boxes),
                "labels": ops.convert_to_tensor(labels),
                "mask": ops.convert_to_tensor(mask),
            },
        )
    value = {k: float(ops.convert_to_numpy(v)) for k, v in out.items()}

    return value["loss"], [
        value["box"] * criterion.box_gain,
        value["cls"] * criterion.cls_gain,
        value["dist"] * criterion.dfl_gain,
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("reg_max", [16, 1], ids=["dfl", "l1"])
def test_loss_matches_ultralytics(reg_max, seed):
    """Every term, and the total, to within float32 rounding."""
    inputs = head_outputs(reg_max, seed)
    u_total, u_parts = ultralytics_loss(*inputs, reg_max)
    k_total, k_parts = kyolo_loss(*inputs, reg_max)
    for name, want, got in zip(("box", "cls", "dist"), u_parts, k_parts):
        assert got == pytest.approx(want, rel=RTOL), name
    assert k_total == pytest.approx(u_total, rel=RTOL)


@pytest.mark.parametrize("reg_max", [16, 1], ids=["dfl", "l1"])
def test_loss_matches_ultralytics_on_sub_stride_boxes(reg_max):
    """Boxes small enough to trigger the stride-aware expansion."""
    inputs = head_outputs(reg_max, seed=0, tiny=True)
    u_total, _ = ultralytics_loss(*inputs, reg_max)
    k_total, _ = kyolo_loss(*inputs, reg_max)
    assert k_total == pytest.approx(u_total, rel=RTOL)


def test_normalized_l1_is_not_the_dfl_path():
    """Guard against the two reg_max paths accidentally agreeing.

    The parity above would also pass if both implementations silently returned
    zero for the third term, which is exactly the bug this work fixed.
    """
    inputs = head_outputs(1, seed=0)
    _, parts = ultralytics_loss(*inputs, 1)
    assert parts[2] > 0.0


def assigner_inputs(seed, tiny):
    rng = np.random.default_rng(seed)
    shapes = [(IMG // s, IMG // s) for s in STRIDES]
    with cpu():
        anchors, stride_t = make_anchors(shapes, STRIDES)
    anchors = (ops.convert_to_numpy(anchors) * ops.convert_to_numpy(stride_t)).astype("float32")
    n_anchors = anchors.shape[0]

    boxes = np.zeros((BATCH, N_GT, 4), "float32")
    labels = np.zeros((BATCH, N_GT), "int32")
    for b in range(BATCH):
        for g in range(N_GT):
            side = rng.uniform(2.0, 6.0) if tiny else rng.uniform(8.0, 28.0)
            cx, cy = rng.uniform(side / 2 + 1, IMG - side / 2 - 1, 2)
            boxes[b, g] = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
            labels[b, g] = rng.integers(0, NC)
    mask = np.ones((BATCH, N_GT), "float32")

    pred = np.zeros((BATCH, n_anchors, 4), "float32")
    for b in range(BATCH):
        pred[b] = boxes[b][rng.integers(0, N_GT, size=n_anchors)] + rng.normal(
            0.0, 1.5, size=(n_anchors, 4)
        )
    scores = rng.uniform(0.0, 1.0, size=(BATCH, n_anchors, NC)).astype("float32")
    return anchors, scores, pred, boxes, labels, mask


@pytest.mark.parametrize("topk,topk2", [(10, None), (7, 1), (10, 3)])
@pytest.mark.parametrize("tiny", [False, True], ids=["normal", "tiny"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_assigner_matches_ultralytics(seed, tiny, topk, topk2):
    """Identical assignment: the same anchors, targets and soft scores."""
    from ultralytics.utils.tal import TaskAlignedAssigner as UltraTAL

    anchors, scores, pred, boxes, labels, mask = assigner_inputs(seed, tiny)
    common = {"topk": topk, "num_classes": NC, "alpha": 0.5, "beta": 6.0}

    _, u_boxes, u_scores, u_fg, _ = UltraTAL(stride=list(STRIDES), topk2=topk2, **common)(
        torch.from_numpy(scores),
        torch.from_numpy(pred),
        torch.from_numpy(anchors),
        torch.from_numpy(labels.astype("float32")).unsqueeze(-1),
        torch.from_numpy(boxes),
        torch.from_numpy(mask).unsqueeze(-1).bool(),
    )
    with cpu():
        _, k_boxes, k_scores, k_fg = TaskAlignedAssigner(strides=STRIDES, topk2=topk2, **common)(
            ops.convert_to_tensor(scores),
            ops.convert_to_tensor(pred),
            ops.convert_to_tensor(anchors),
            ops.convert_to_tensor(labels),
            ops.convert_to_tensor(boxes),
            ops.convert_to_tensor(mask),
        )

    foreground = u_fg.numpy()

    np.testing.assert_array_equal(ops.convert_to_numpy(k_fg) > 0, foreground)
    assert foreground.sum() > 0, "degenerate case: nothing assigned"
    np.testing.assert_allclose(
        ops.convert_to_numpy(k_scores), u_scores.numpy(), rtol=1e-5, atol=1e-6
    )

    np.testing.assert_allclose(
        ops.convert_to_numpy(k_boxes) * foreground[..., None],
        u_boxes.numpy() * foreground[..., None],
        rtol=1e-6,
        atol=1e-6,
    )
