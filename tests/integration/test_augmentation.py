"""Tests for the pure-Keras detection augmentation layers.

The geometric layers are checked against an *image-derived* ground truth rather
than against a reimplementation of their own maths: a box is drawn into the
image as a bright rectangle, and after augmentation the bright pixels have to
still line up with the box the layer reports. That catches the failure mode that
actually matters -- image and labels drifting apart.

The module is skipped gracefully when no Keras backend is importable.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import keras
    from keras import ops

    from kyolo.augmentation import (
        AugmentationPipeline,
        CopyPaste,
        DetectionAugmentation,
        MixUp,
        Mosaic,
        RandomFlip,
        RandomHSV,
        RandomPerspective,
    )

    _IMPORT_ERROR = None
except Exception as exc:
    keras = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"keras backend / kyolo.augmentation unavailable: {_IMPORT_ERROR}",
)

SIDE = 64
BATCH = 4
SLOTS = 5


def as_numpy(x):
    return ops.convert_to_numpy(x)


def sample(batch=BATCH, slots=SLOTS, side=SIDE, boxes_per_image=3, seed=0):
    """A random batch in the training sample format."""
    rng = np.random.default_rng(seed)
    images = rng.random((batch, side, side, 3)).astype("float32")
    boxes = np.zeros((batch, slots, 4), "float32")
    labels = np.zeros((batch, slots), "int32")
    mask = np.zeros((batch, slots), "float32")
    for b in range(batch):
        for i in range(boxes_per_image):
            x1, y1 = rng.uniform(2, side / 2, 2)
            w, h = rng.uniform(10, side / 3, 2)
            boxes[b, i] = (x1, y1, x1 + w, y1 + h)
            labels[b, i] = i
            mask[b, i] = 1.0
    return {"images": images, "boxes": boxes, "labels": labels, "mask": mask}


def marker_sample(box=(10.0, 14.0, 34.0, 44.0), side=SIDE):
    """One image that is black except for a white rectangle equal to its box."""
    images = np.zeros((1, side, side, 3), "float32")
    x1, y1, x2, y2 = (int(v) for v in box)
    images[0, y1:y2, x1:x2, :] = 1.0
    return {
        "images": images,
        "boxes": np.array([[list(box)]], "float32"),
        "labels": np.zeros((1, 1), "int32"),
        "mask": np.ones((1, 1), "float32"),
    }


def bright_extent(images, index=0, threshold=0.5):
    """Bounding box of the bright pixels, as ``(x1, y1, x2, y2)``."""
    plane = as_numpy(images)[index, :, :, 0]
    rows, cols = np.where(plane > threshold)
    assert rows.size, "expected some bright pixels"
    return np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], "float32")


def valid_boxes(out, index=0):
    keep = as_numpy(out["mask"])[index] > 0.5
    return as_numpy(out["boxes"])[index][keep]


def test_layers_preserve_the_sample_contract():
    """Every layer returns the four sample keys with the expected dtypes."""
    layers = [
        RandomHSV(),
        RandomFlip(prob=1.0),
        RandomPerspective(),
        Mosaic(),
        MixUp(prob=1.0),
        CopyPaste(prob=1.0),
    ]
    for layer in layers:
        out = layer(sample())
        assert set(out) >= {"images", "boxes", "labels", "mask"}, type(layer).__name__
        assert keras.backend.standardize_dtype(out["images"].dtype) == "float32"
        assert keras.backend.standardize_dtype(out["labels"].dtype) == "int32"
        assert as_numpy(out["images"]).shape[0] == BATCH


def test_training_false_is_a_passthrough():
    """Augmentation is a no-op outside training, including the resizing ones."""
    batch = sample()
    for layer in (RandomHSV(), RandomFlip(prob=1.0), Mosaic(), RandomPerspective()):
        out = layer(batch, training=False)
        np.testing.assert_array_equal(as_numpy(out["images"]), batch["images"])
        np.testing.assert_array_equal(as_numpy(out["boxes"]), batch["boxes"])


def test_sample_dict_is_validated():
    with pytest.raises(TypeError, match="dict"):
        RandomHSV()(np.zeros((1, 8, 8, 3), "float32"))
    with pytest.raises(ValueError, match="missing"):
        RandomHSV()({"images": np.zeros((1, 8, 8, 3), "float32")})
    with pytest.raises(ValueError, match="rank 4"):
        RandomHSV()(
            {
                "images": np.zeros((8, 8, 3), "float32"),
                "boxes": np.zeros((1, 1, 4), "float32"),
                "labels": np.zeros((1, 1), "int32"),
                "mask": np.zeros((1, 1), "float32"),
            }
        )


def test_random_hsv_changes_colour_but_not_geometry():
    batch = sample()
    out = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5, seed=0)(batch)
    np.testing.assert_array_equal(as_numpy(out["boxes"]), batch["boxes"])
    np.testing.assert_array_equal(as_numpy(out["mask"]), batch["mask"])
    assert not np.allclose(as_numpy(out["images"]), batch["images"])
    assert as_numpy(out["images"]).min() >= 0.0
    assert as_numpy(out["images"]).max() <= 1.0


def test_random_hsv_with_zero_gains_is_identity():
    batch = sample()
    out = RandomHSV(hgain=0.0, sgain=0.0, vgain=0.0)(batch)
    np.testing.assert_array_equal(as_numpy(out["images"]), batch["images"])


def test_random_hsv_respects_the_value_range():
    """0-255 inputs must come back as 0-255, not squashed into the unit range."""
    batch = sample()
    batch["images"] = batch["images"] * 255.0
    out = RandomHSV(value_range=(0.0, 255.0), seed=0)(batch)
    assert as_numpy(out["images"]).max() > 20.0
    assert as_numpy(out["images"]).max() <= 255.0


def test_random_hsv_rejects_a_degenerate_value_range():
    with pytest.raises(ValueError, match="increasing"):
        RandomHSV(value_range=(1.0, 1.0))


@pytest.mark.parametrize("direction", ["horizontal", "vertical"])
def test_random_flip_keeps_boxes_on_their_content(direction):
    out = RandomFlip(direction=direction, prob=1.0)(marker_sample())
    np.testing.assert_allclose(valid_boxes(out)[0], bright_extent(out["images"]), atol=1e-4)


def test_random_flip_mirrors_the_expected_axis():
    batch = marker_sample(box=(10.0, 14.0, 34.0, 44.0))
    horizontal = valid_boxes(RandomFlip(prob=1.0)(batch))[0]
    np.testing.assert_allclose(horizontal, [SIDE - 34, 14, SIDE - 10, 44], atol=1e-4)

    vertical = valid_boxes(RandomFlip(direction="vertical", prob=1.0)(batch))[0]
    np.testing.assert_allclose(vertical, [10, SIDE - 44, 34, SIDE - 14], atol=1e-4)


def test_random_flip_with_zero_probability_is_identity():
    batch = sample()
    out = RandomFlip(prob=0.0)(batch)
    np.testing.assert_array_equal(as_numpy(out["images"]), batch["images"])
    np.testing.assert_array_equal(as_numpy(out["boxes"]), batch["boxes"])


def test_random_flip_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="horizontal"):
        RandomFlip(direction="diagonal")


def still_transform(**kwargs):
    """A RandomPerspective with every strength at zero."""
    return RandomPerspective(
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, **kwargs
    )


def test_perspective_with_zero_strength_is_exactly_identity():
    batch = sample()
    out = still_transform(output_size=SIDE)(batch)
    np.testing.assert_array_equal(as_numpy(out["images"]), batch["images"])
    np.testing.assert_allclose(as_numpy(out["boxes"]), batch["boxes"], atol=1e-3)
    np.testing.assert_array_equal(as_numpy(out["mask"]), batch["mask"])


def test_perspective_sets_the_output_resolution():
    for size, expected in ((32, 32), (SIDE, SIDE), (96, 96)):
        out = RandomPerspective(output_size=size)(sample())
        assert as_numpy(out["images"]).shape[1:3] == (expected, expected)
        assert RandomPerspective(output_size=size).compute_output_shape(
            {
                "images": (BATCH, SIDE, SIDE, 3),
                "boxes": (BATCH, SLOTS, 4),
                "labels": (BATCH, SLOTS),
                "mask": (BATCH, SLOTS),
            }
        )["images"] == (BATCH, expected, expected, 3)


def test_perspective_keeps_boxes_on_their_content_under_rotation():
    """A rotated marker still has to be covered by the reported box."""
    layer = RandomPerspective(degrees=25.0, scale=0.3, translate=0.1, seed=5)
    out = layer(marker_sample())
    box = valid_boxes(out)[0]
    bright = bright_extent(out["images"])
    assert box[0] <= bright[0] + 1.5 and box[1] <= bright[1] + 1.5
    assert box[2] >= bright[2] - 1.5 and box[3] >= bright[3] - 1.5


def test_perspective_drops_boxes_that_leave_the_frame():
    """A hard crop away from the object must remove its label, not keep a stub."""
    batch = marker_sample(box=(2.0, 2.0, 18.0, 18.0))
    out = RandomPerspective(
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        output_size=8,
    )(batch)
    assert float(ops.sum(out["mask"])) == 0.0


def test_perspective_pads_when_asked_to_grow_the_frame():
    """Upscaling the frame fills the new border with pad_value, not zeros."""
    batch = sample(batch=1)
    out = still_transform(output_size=128, pad_value=0.25)(batch)
    corner = as_numpy(out["images"])[0, 0, 0, 0]
    assert corner == pytest.approx(0.25, abs=1e-5)


def test_mosaic_doubles_the_canvas_and_the_box_budget():
    out = Mosaic(seed=1)(sample())
    assert as_numpy(out["images"]).shape[1:3] == (2 * SIDE, 2 * SIDE)
    assert as_numpy(out["boxes"]).shape[1] == 4 * SLOTS
    assert float(ops.sum(out["mask"])) > 0


def test_mosaic_boxes_track_the_image_they_came_from():
    """Each mosaic box must sit on pixels from the image its label names.

    Every image is a constant "tag" value and carries one full-frame box, so a
    box whose interior does not hold its own tag means the box bookkeeping and
    the pixel gather disagree.
    """
    images = np.zeros((BATCH, SIDE, SIDE, 3), "float32")
    boxes = np.zeros((BATCH, 1, 4), "float32")
    labels = np.zeros((BATCH, 1), "int32")
    for b in range(BATCH):
        images[b] = (b + 1) / (BATCH + 1)
        boxes[b, 0] = (0.0, 0.0, SIDE, SIDE)
        labels[b, 0] = b
    batch = {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "mask": np.ones((BATCH, 1), "float32"),
    }

    out = Mosaic(prob=1.0, seed=7)(batch)
    canvas = as_numpy(out["images"])
    out_boxes = as_numpy(out["boxes"])
    out_labels = as_numpy(out["labels"])
    out_mask = as_numpy(out["mask"])

    checked = 0
    for b in range(BATCH):
        for j in range(out_boxes.shape[1]):
            if out_mask[b, j] < 0.5:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in out_boxes[b, j])
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            interior = canvas[b, y1 + 1 : y2 - 1, x1 + 1 : x2 - 1, 0]
            expected = (out_labels[b, j] + 1) / (BATCH + 1)
            np.testing.assert_allclose(interior, expected, atol=1e-5)
            checked += 1
    assert checked >= BATCH


def test_mosaic_off_then_centre_crop_returns_the_original():
    """``prob=0`` centres the image so the following crop undoes the canvas."""
    batch = sample()
    staged = Mosaic(prob=0.0)(batch)
    out = still_transform(output_size=SIDE)(staged)

    np.testing.assert_array_equal(as_numpy(out["images"]), batch["images"])
    kept = as_numpy(out["mask"])[:, :SLOTS][..., None]
    np.testing.assert_allclose(
        as_numpy(out["boxes"])[:, :SLOTS] * kept, batch["boxes"] * kept, atol=1e-3
    )
    assert float(ops.sum(out["mask"])) == float(batch["mask"].sum())


def test_mosaic_leaves_grey_where_no_image_reaches():
    """The random centre always leaves an uncovered strip, filled with pad_value."""
    out = Mosaic(prob=1.0, pad_value=0.6, seed=2)(sample())
    assert np.isclose(as_numpy(out["images"]), 0.6).any()


def test_mixup_blends_near_half_and_unions_the_boxes():
    batch = sample()
    out = MixUp(prob=1.0, alpha=32.0, seed=0)(batch)
    assert as_numpy(out["boxes"]).shape[1] == 2 * SLOTS
    assert float(ops.sum(out["mask"])) == pytest.approx(2 * batch["mask"].sum())

    blended = as_numpy(out["images"])
    assert not np.allclose(blended, batch["images"])
    assert abs(float(blended.mean()) - float(batch["images"].mean())) < 0.05


def test_mixup_with_zero_probability_keeps_only_its_own_boxes():
    batch = sample()
    out = MixUp(prob=0.0)(batch)
    np.testing.assert_allclose(as_numpy(out["images"]), batch["images"], atol=1e-6)
    assert float(ops.sum(out["mask"])) == pytest.approx(batch["mask"].sum())


def test_mixup_rejects_a_nonpositive_alpha():
    with pytest.raises(ValueError, match="positive"):
        MixUp(alpha=0.0)


def test_copy_paste_transplants_pixels_and_labels():
    """The mirrored marker gets pasted, and arrives with a matching box."""
    batch = marker_sample(box=(4.0, 6.0, 20.0, 26.0))
    out = CopyPaste(prob=1.0, mode="flip", seed=0)(batch)

    assert float(ops.sum(out["mask"])) == 2.0
    pasted = valid_boxes(out)[1]
    np.testing.assert_allclose(pasted, [SIDE - 20, 6, SIDE - 4, 26], atol=1e-4)

    plane = as_numpy(out["images"])[0, :, :, 0]
    x1, y1, x2, y2 = (int(v) for v in pasted)
    assert plane[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1].min() > 0.5


def test_copy_paste_rejects_candidates_that_would_bury_existing_boxes():
    """A candidate overlapping an existing box is not pasted."""
    box = [20.0, 20.0, 44.0, 44.0]
    batch = {
        "images": np.ones((1, SIDE, SIDE, 3), "float32"),
        "boxes": np.array([[box]], "float32"),
        "labels": np.zeros((1, 1), "int32"),
        "mask": np.ones((1, 1), "float32"),
    }
    out = CopyPaste(prob=1.0, mode="flip", max_overlap=0.01, seed=0)(batch)
    assert float(ops.sum(out["mask"])) == 1.0


def test_copy_paste_with_zero_probability_adds_nothing():
    batch = sample()
    out = CopyPaste(prob=0.0)(batch)
    np.testing.assert_array_equal(as_numpy(out["images"]), batch["images"])
    assert float(ops.sum(out["mask"])) == pytest.approx(batch["mask"].sum())


def test_copy_paste_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="flip"):
        CopyPaste(mode="paste")


def test_trim_keeps_valid_boxes_and_their_order():
    batch = sample(slots=8, boxes_per_image=6)
    trimmed = DetectionAugmentation.trim(batch, 4)
    assert as_numpy(trimmed["boxes"]).shape[1] == 4
    np.testing.assert_array_equal(as_numpy(trimmed["mask"]), np.ones((BATCH, 4), "float32"))
    np.testing.assert_allclose(as_numpy(trimmed["boxes"]), batch["boxes"][:, :4], atol=1e-6)


def test_trim_is_a_noop_when_there_is_room():
    batch = sample()
    assert DetectionAugmentation.trim(batch, SLOTS + 10) is batch


def test_pipeline_uses_the_ultralytics_stage_order():
    pipeline = AugmentationPipeline.from_hyperparameters(
        image_size=SIDE, mosaic=1.0, copy_paste=0.5, mixup=0.5
    )
    assert [type(a).__name__ for a in pipeline.augmentations] == [
        "Mosaic",
        "CopyPaste",
        "RandomPerspective",
        "MixUp",
        "RandomHSV",
        "RandomFlip",
    ]


def test_pipeline_omits_stages_with_zero_probability():
    pipeline = AugmentationPipeline.from_hyperparameters(
        image_size=SIDE, mosaic=0.0, mixup=0.0, copy_paste=0.0, fliplr=0.0
    )
    assert [type(a).__name__ for a in pipeline.augmentations] == [
        "RandomPerspective",
        "RandomHSV",
    ]


def test_pipeline_returns_the_training_resolution_and_bounded_boxes():
    pipeline = AugmentationPipeline.from_hyperparameters(
        image_size=SIDE, mixup=1.0, max_boxes=SLOTS, seed=0
    )
    out = pipeline(sample())
    assert as_numpy(out["images"]).shape == (BATCH, SIDE, SIDE, 3)
    assert as_numpy(out["boxes"]).shape == (BATCH, SLOTS, 4)


def test_close_mosaic_removes_only_the_mixing_stages():
    pipeline = AugmentationPipeline.from_hyperparameters(
        image_size=SIDE, mosaic=1.0, mixup=0.5, copy_paste=0.5
    )
    assert not pipeline.mosaic_closed

    pipeline.close_mosaic()
    assert pipeline.mosaic_closed
    assert [type(a).__name__ for a in pipeline.active()] == [
        "RandomPerspective",
        "RandomHSV",
        "RandomFlip",
    ]
    out = pipeline(sample())
    assert as_numpy(out["images"]).shape == (BATCH, SIDE, SIDE, 3)

    pipeline.open_mosaic()
    assert not pipeline.mosaic_closed


def test_closed_pipeline_still_returns_the_training_resolution():
    """The perspective stage is configured by output size, so nothing else
    has to change when the mosaic canvas disappears."""
    pipeline = AugmentationPipeline.from_hyperparameters(image_size=SIDE, mosaic=1.0)
    pipeline.close_mosaic()
    out = pipeline(sample())
    assert as_numpy(out["images"]).shape[1:3] == (SIDE, SIDE)


def test_seeding_makes_the_pipeline_reproducible():
    first = AugmentationPipeline.from_hyperparameters(image_size=SIDE, seed=11)(sample())
    second = AugmentationPipeline.from_hyperparameters(image_size=SIDE, seed=11)(sample())
    np.testing.assert_allclose(as_numpy(first["images"]), as_numpy(second["images"]), atol=1e-6)
    np.testing.assert_allclose(as_numpy(first["boxes"]), as_numpy(second["boxes"]), atol=1e-6)


def test_channels_first_is_handled():
    batch = sample()
    batch["images"] = np.transpose(batch["images"], (0, 3, 1, 2))
    out = RandomFlip(prob=1.0, data_format="channels_first")(batch)
    assert as_numpy(out["images"]).shape == (BATCH, 3, SIDE, SIDE)

    mosaic = Mosaic(prob=1.0, data_format="channels_first")(batch)
    assert as_numpy(mosaic["images"]).shape == (BATCH, 3, 2 * SIDE, 2 * SIDE)


@pytest.mark.parametrize(
    "layer",
    [
        RandomHSV(hgain=0.02),
        RandomFlip(direction="vertical", prob=0.25),
        RandomPerspective(degrees=7.0, output_size=32),
        Mosaic(prob=0.5),
        MixUp(prob=0.5, alpha=8.0),
        CopyPaste(prob=0.5, mode="mixup"),
    ],
)
def test_layers_round_trip_through_their_config(layer):
    restored = type(layer).from_config(layer.get_config())
    assert type(restored) is type(layer)
    assert restored.get_config() == layer.get_config()


def test_pipeline_round_trips_through_its_config():
    pipeline = AugmentationPipeline.from_hyperparameters(image_size=SIDE, mixup=0.5, max_boxes=17)
    restored = AugmentationPipeline.from_config(pipeline.get_config())
    assert restored.max_boxes == 17
    assert [type(a).__name__ for a in restored.augmentations] == [
        type(a).__name__ for a in pipeline.augmentations
    ]


def test_dynamic_spatial_shape_is_rejected_with_a_useful_message():
    """The layers crop and pad by computed amounts, so they need static H/W."""
    layer = RandomPerspective()
    with pytest.raises(ValueError, match="statically-known"):
        layer.static_hw(keras.KerasTensor((None, None, None, 3)))
