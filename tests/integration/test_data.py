"""Tests for the Grain-backed data pipeline.

Everything runs against a small dataset written to a temporary directory, so
these exercise the real file layout (``images/`` and ``labels/`` trees, a
``data.yaml``) rather than mocks.

Grain and Pillow are optional dependencies, so the module skips gracefully when
either is missing.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

try:
    import grain
    import keras
    from PIL import Image

    from kyolo.augmentation import AugmentationPipeline
    from kyolo.data import (
        DataConfig,
        GrainDataLoader,
        LetterboxSample,
        PadTargets,
        RectangularPlan,
        YOLODataSource,
        build_source,
        coco_to_yolo,
        label_path_for,
        load_yolo_label,
        normalized_to_xyxy,
        voc_to_yolo,
        write_yolo_label,
        xyxy_to_normalized,
    )
    from kyolo.layers import Letterbox

    _IMPORT_ERROR = None
except Exception as exc:
    grain = None
    keras = None
    _IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"grain / pillow / kyolo.data unavailable: {_IMPORT_ERROR}",
)

NAMES = ["cat", "dog", "bird"]
SHAPES = [(480, 640), (640, 480), (500, 500), (300, 900), (720, 1280), (256, 256)]


def write_split(root, split, shapes, seed=0):
    """Write images plus YOLO labels in the ``images/`` / ``labels/`` layout."""
    images_dir = root / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for index, (height, width) in enumerate(shapes):
        path = images_dir / f"{split}_{index}.png"
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype="uint8")).save(path)
        count = 1 + index % 3
        write_yolo_label(
            label_path_for(path),
            [(index + j) % len(NAMES) for j in range(count)],
            [(0.3 + 0.1 * j, 0.4, 0.2, 0.25) for j in range(count)],
        )
        paths.append(path)
    return paths


@pytest.fixture
def dataset(tmp_path):
    """A two-split dataset on disk, with its ``data.yaml``."""
    write_split(tmp_path, "train", SHAPES)
    write_split(tmp_path, "val", SHAPES[:4], seed=1)
    config = DataConfig.from_directories(
        train="images/train", val="images/val", names=NAMES, root=tmp_path
    )
    yaml_path = config.to_yaml(tmp_path / "data.yaml")
    return {"root": tmp_path, "yaml": yaml_path, "config": config}


def test_config_round_trips_through_yaml(dataset):
    reloaded = DataConfig.from_yaml(dataset["yaml"])
    assert reloaded.names == NAMES
    assert reloaded.nc == 3
    assert len(reloaded.image_paths("train")) == len(SHAPES)
    assert len(reloaded.image_paths("val")) == 4


def test_config_accepts_names_as_a_mapping_or_a_list(tmp_path):
    yaml = pytest.importorskip("yaml")
    (tmp_path / "images").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), "uint8")).save(tmp_path / "images" / "a.png")

    as_mapping = tmp_path / "mapping.yaml"
    as_mapping.write_text(
        yaml.safe_dump({"train": "images", "names": {1: "dog", 0: "cat"}}), encoding="utf-8"
    )
    assert DataConfig.from_yaml(as_mapping).names == ["cat", "dog"]

    as_list = tmp_path / "list.yaml"
    as_list.write_text(
        yaml.safe_dump({"train": "images", "nc": 2, "names": ["cat", "dog"]}), encoding="utf-8"
    )
    assert DataConfig.from_yaml(as_list).names == ["cat", "dog"]


def test_config_rejects_inconsistent_class_counts(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"train": "images", "nc": 5, "names": ["cat", "dog"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must agree"):
        DataConfig.from_yaml(path)


def test_config_rejects_non_contiguous_name_indices(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "gap.yaml"
    path.write_text(
        yaml.safe_dump({"train": "images", "names": {0: "cat", 2: "dog"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contiguous"):
        DataConfig.from_yaml(path)


def test_config_reports_an_unknown_split(dataset):
    with pytest.raises(KeyError, match="test"):
        dataset["config"].image_paths("test")


def test_config_reads_a_txt_image_list(dataset, tmp_path):
    listing = tmp_path / "train_list.txt"
    paths = dataset["config"].image_paths("train")
    listing.write_text("\n".join(str(p) for p in paths[:3]) + "\n", encoding="utf-8")

    config = DataConfig.from_directories(train=str(listing), names=NAMES, root=tmp_path)
    assert len(config.image_paths("train")) == 3


def test_label_path_follows_the_images_to_labels_convention():
    assert label_path_for("/d/images/train/a.jpg").as_posix().endswith("/d/labels/train/a.txt")
    assert label_path_for("/d/plain/a.jpg").as_posix().endswith("/d/plain/a.txt")


def test_labels_round_trip(tmp_path):
    path = tmp_path / "a.txt"
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.2, 0.05, 0.05]], "float32")
    write_yolo_label(path, [2, 0], boxes)

    classes, loaded = load_yolo_label(path)
    np.testing.assert_array_equal(classes, [2, 0])
    np.testing.assert_allclose(loaded, boxes, atol=1e-6)


def test_missing_label_file_means_no_objects(tmp_path):
    classes, boxes = load_yolo_label(tmp_path / "absent.txt")
    assert classes.shape == (0,)
    assert boxes.shape == (0, 4)


def test_polygon_labels_are_reduced_to_their_enclosing_box(tmp_path):
    path = tmp_path / "poly.txt"
    path.write_text("1 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8\n", encoding="utf-8")
    classes, boxes = load_yolo_label(path)
    np.testing.assert_array_equal(classes, [1])
    np.testing.assert_allclose(boxes[0], [0.4, 0.5, 0.4, 0.6], atol=1e-6)


def test_degenerate_labels_are_dropped(tmp_path):
    path = tmp_path / "zero.txt"
    path.write_text("0 0.5 0.5 0.0 0.3\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    classes, boxes = load_yolo_label(path)
    np.testing.assert_array_equal(classes, [1])
    assert boxes.shape == (1, 4)


def test_malformed_label_reports_the_line(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.txt:1"):
        load_yolo_label(path)


def test_out_of_range_class_is_rejected(tmp_path):
    path = tmp_path / "oob.txt"
    path.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_yolo_label(path, nc=3)


def test_box_conversions_are_inverse():
    xyxy = np.array([[10.0, 20.0, 60.0, 80.0]], "float32")
    normalized = xyxy_to_normalized(xyxy, 200, 100)
    np.testing.assert_allclose(normalized_to_xyxy(normalized, 200, 100), xyxy, atol=1e-4)


def test_source_returns_pixel_boxes_for_the_original_image(dataset):
    source = YOLODataSource(dataset["config"].image_paths("train"), nc=3)
    assert len(source) == len(SHAPES)

    item = source[0]
    height, width = item["orig_shape"]
    assert item["image"].shape == (height, width, 3)
    assert item["image"].dtype == np.uint8
    assert item["boxes"].shape[1] == 4
    assert item["boxes"][:, 2].max() <= width + 1e-3
    assert item["boxes"][:, 3].max() <= height + 1e-3
    assert "YOLODataSource" in repr(source)


def test_source_reads_shapes_without_decoding(dataset):
    source = YOLODataSource(dataset["config"].image_paths("train"))
    np.testing.assert_array_equal(source.shapes, np.array(sorted_shapes(dataset)))


def sorted_shapes(dataset):
    """The dataset's (h, w) shapes in the sorted path order the source uses."""
    paths = dataset["config"].image_paths("train")
    order = [int(p.stem.split("_")[-1]) for p in paths]
    return [SHAPES[i] for i in order]


@pytest.mark.parametrize("cache", [None, "ram", "disk"])
def test_source_cache_modes_agree(dataset, cache):
    paths = dataset["config"].image_paths("train")
    plain = YOLODataSource(paths)[0]["image"]
    cached = YOLODataSource(paths, cache=cache)
    np.testing.assert_array_equal(cached[0]["image"], plain)
    np.testing.assert_array_equal(cached[0]["image"], plain)


def test_source_rejects_an_unknown_cache_mode(dataset):
    with pytest.raises(ValueError, match="cache must be"):
        YOLODataSource(dataset["config"].image_paths("train"), cache="gpu")


def test_source_survives_pickling_without_its_ram_cache(dataset):
    """Grain ships the source to worker processes, so it has to pickle."""
    source = YOLODataSource(dataset["config"].image_paths("train"), cache="ram")
    _ = source[0]
    restored = pickle.loads(pickle.dumps(source))
    assert len(restored) == len(source)
    assert restored._ram == {}
    np.testing.assert_array_equal(restored[0]["image"], source[0]["image"])


def test_source_rejects_mismatched_label_paths(dataset):
    paths = dataset["config"].image_paths("train")
    with pytest.raises(ValueError, match="parallel"):
        YOLODataSource(paths, label_paths=paths[:2])


def test_letterbox_matches_the_keras_layer_geometry(dataset):
    """The numpy loader and the serving layer must agree on ratio and padding.

    Interpolation differs (Pillow filters when downscaling, the Keras layer does
    not), so only the geometry is compared -- but the geometry is what decides
    where boxes land, and a mismatch would silently shift every label at serve
    time.
    """
    source = YOLODataSource(dataset["config"].image_paths("train"))
    transform = LetterboxSample(image_size=64)

    for index in range(len(source)):
        item = source[index]
        out = transform(item)
        _, ratio, pad = Letterbox(new_shape=64, scaleup=True)(item["image"].astype("float32"))
        np.testing.assert_allclose(out["ratio"], keras.ops.convert_to_numpy(ratio), atol=1e-6)
        np.testing.assert_allclose(out["pad"], keras.ops.convert_to_numpy(pad), atol=1e-6)
        assert out["image"].shape == (64, 64, 3)


def test_letterbox_moves_boxes_with_the_image(dataset):
    """A box covering the whole image must cover the whole unpadded region."""
    image = np.full((100, 200, 3), 7, dtype="uint8")
    sample = {
        "image": image,
        "boxes": np.array([[0.0, 0.0, 200.0, 100.0]], "float32"),
        "labels": np.zeros((1,), "int32"),
    }
    out = LetterboxSample(image_size=64, pad_value=0)(sample)

    ratio = out["ratio"][0]
    left, top = out["pad"]
    np.testing.assert_allclose(
        out["boxes"][0], [left, top, left + 200 * ratio, top + 100 * ratio], atol=1e-4
    )
    content = out["image"][int(top) : int(top) + int(round(100 * ratio))]
    assert content.min() > 0


def test_letterbox_pad_value_fills_the_border():
    sample = {
        "image": np.full((100, 200, 3), 255, dtype="uint8"),
        "boxes": np.zeros((0, 4), "float32"),
        "labels": np.zeros((0,), "int32"),
    }
    out = LetterboxSample(image_size=64, pad_value=114)(sample)
    assert out["image"][0, 0, 0] == 114


def test_pad_targets_pads_and_masks():
    sample = {
        "image": np.zeros((8, 8, 3), "uint8"),
        "boxes": np.array([[1.0, 2.0, 3.0, 4.0]], "float32"),
        "labels": np.array([2], "int32"),
    }
    out = PadTargets(max_boxes=4)(sample)
    assert out["boxes"].shape == (4, 4)
    np.testing.assert_array_equal(out["mask"], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["labels"], [2, 0, 0, 0])
    assert "images" in out


def test_pad_targets_truncates_crowded_images():
    sample = {
        "image": np.zeros((8, 8, 3), "uint8"),
        "boxes": np.tile(np.array([[1.0, 2.0, 3.0, 4.0]], "float32"), (9, 1)),
        "labels": np.zeros((9,), "int32"),
    }
    out = PadTargets(max_boxes=4)(sample)
    assert out["boxes"].shape == (4, 4)
    assert out["mask"].sum() == 4


def test_pad_targets_rejects_a_nonpositive_budget():
    with pytest.raises(ValueError, match="positive"):
        PadTargets(max_boxes=0)


def test_rectangular_plan_sorts_by_aspect_and_snaps_to_stride():
    plan = RectangularPlan(np.array(SHAPES), batch_size=2, image_size=64, stride=32)
    aspect = np.array([h / w for h, w in SHAPES])
    assert list(plan.order) == list(np.argsort(aspect, kind="stable"))
    assert (plan.batch_shapes % 32 == 0).all()
    assert len(plan.batch_shapes) == 3

    widest = SHAPES[plan.order[0]]
    height, width = plan.shape_for(0)
    assert width >= height, f"a wide image {widest} should get a wide batch shape"


def test_rectangular_plan_pad_controls_overshoot():
    """Ultralytics' pad=0.5 lands one stride above image_size; pad=0 does not."""
    shapes = np.array(SHAPES)
    default = RectangularPlan(shapes, batch_size=2, image_size=64, stride=32, pad=0.5)
    assert default.batch_shapes.max() == 96

    tight = RectangularPlan(shapes, batch_size=2, image_size=64, stride=32, pad=0.0)
    assert tight.batch_shapes.max() == 64


def test_rect_pad_flows_through_the_dataloader(dataset):
    loader = GrainDataLoader(
        dataset["yaml"], "val", batch_size=2, image_size=64, rect=True, rect_pad=0.0
    )
    for index in range(len(loader)):
        height, width = np.asarray(loader[index][0]["images"]).shape[1:3]
        assert max(height, width) <= 64


def test_rectangular_plan_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        RectangularPlan(np.zeros((0, 2)), batch_size=2)


def test_dataloader_yields_the_training_contract(dataset):
    loader = GrainDataLoader(dataset["yaml"], "train", batch_size=2, image_size=64, max_boxes=8)
    assert len(loader) == len(SHAPES) // 2

    inputs, targets = loader[0]
    assert inputs["images"].shape == (2, 64, 64, 3)
    assert keras.backend.standardize_dtype(inputs["images"].dtype) == "float32"
    assert targets["boxes"].shape == (2, 8, 4)
    assert targets["labels"].shape == (2, 8)
    assert targets["mask"].shape == (2, 8)
    assert 0.0 <= np.asarray(inputs["images"]).min()
    assert np.asarray(inputs["images"]).max() <= 1.0
    assert np.asarray(targets["mask"]).sum() > 0


def test_dataloader_boxes_stay_inside_the_letterboxed_image(dataset):
    loader = GrainDataLoader(dataset["yaml"], "train", batch_size=2, image_size=64, max_boxes=8)
    for index in range(len(loader)):
        _, targets = loader[index]
        boxes = np.asarray(targets["boxes"])
        mask = np.asarray(targets["mask"]) > 0.5
        assert boxes[mask].min() >= -1e-3
        assert boxes[mask].max() <= 64 + 1e-3


def test_dataloader_reshuffles_between_epochs(dataset):
    loader = GrainDataLoader(
        dataset["yaml"], "train", batch_size=2, image_size=32, shuffle=True, seed=5
    )
    before = np.asarray(loader[0][1]["labels"]).copy()
    orders = {before.tobytes()}
    for _ in range(6):
        loader.on_epoch_end()
        orders.add(np.asarray(loader[0][1]["labels"]).tobytes())
    assert len(orders) > 1, "batch 0 never changed across epochs"


def test_dataloader_is_deterministic_for_a_given_seed(dataset):
    def first_batch():
        loader = GrainDataLoader(
            dataset["yaml"], "train", batch_size=2, image_size=32, shuffle=True, seed=99
        )
        return np.asarray(loader[0][1]["boxes"])

    np.testing.assert_allclose(first_batch(), first_batch(), atol=1e-6)


def test_dataloader_covers_every_validation_image(dataset):
    loader = GrainDataLoader(
        dataset["yaml"], "val", batch_size=3, image_size=32, keep_metadata=True
    )
    seen = []
    for index in range(len(loader)):
        inputs, _ = loader[index]
        seen.extend(int(i) for i in np.asarray(inputs["index"]))
    assert sorted(seen) == list(range(4))


def test_validation_split_does_not_shuffle_or_drop(dataset):
    loader = GrainDataLoader(dataset["yaml"], "val", batch_size=3, image_size=32)
    assert len(loader) == 2


def test_dataloader_rejects_an_out_of_range_batch(dataset):
    loader = GrainDataLoader(dataset["yaml"], "val", batch_size=3, image_size=32)
    with pytest.raises(IndexError, match="out of range"):
        loader[len(loader)]


def test_rectangular_batches_get_their_own_shapes(dataset):
    loader = GrainDataLoader(dataset["yaml"], "train", batch_size=2, image_size=64, rect=True)
    shapes = {tuple(np.asarray(loader[i][0]["images"]).shape[1:3]) for i in range(len(loader))}
    assert len(shapes) > 1, f"rect batching produced a single shape: {shapes}"
    for height, width in shapes:
        assert height % 32 == 0 and width % 32 == 0


def test_rect_and_shuffle_are_mutually_exclusive(dataset):
    with pytest.raises(ValueError, match="rect=True"):
        GrainDataLoader(dataset["yaml"], "train", rect=True, shuffle=True)


def test_dataloader_reports_an_impossible_batch_size(dataset):
    with pytest.raises(ValueError, match="fewer than"):
        GrainDataLoader(dataset["yaml"], "train", batch_size=64, drop_remainder=True)


def test_augment_true_builds_the_default_pipeline(dataset):
    loader = GrainDataLoader(
        dataset["yaml"], "train", batch_size=4, image_size=64, max_boxes=8, augment=True
    )
    assert isinstance(loader.augment, AugmentationPipeline)
    inputs, targets = loader[0]
    assert inputs["images"].shape == (4, 64, 64, 3)
    assert targets["boxes"].shape == (4, 8, 4)


def test_augment_accepts_hyperparameter_overrides(dataset):
    loader = GrainDataLoader(
        dataset["yaml"],
        "train",
        batch_size=4,
        image_size=64,
        augment={
            "mosaic": 0.0,
            "mixup": 0.0,
            "fliplr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
        },
    )
    assert [type(a).__name__ for a in loader.augment.augmentations] == ["RandomPerspective"]


def test_augment_accepts_a_ready_made_pipeline(dataset):
    pipeline = AugmentationPipeline.from_hyperparameters(image_size=64, mosaic=0.0)
    loader = GrainDataLoader(
        dataset["yaml"], "train", batch_size=2, image_size=64, augment=pipeline
    )
    assert loader.augment is pipeline


def test_augmentation_is_off_by_default(dataset):
    loader = GrainDataLoader(dataset["yaml"], "train", batch_size=2, image_size=64)
    assert loader.augment is None


def test_build_source_passes_through_a_source(dataset):
    source = YOLODataSource(dataset["config"].image_paths("train"))
    assert build_source(source) is source


def test_value_range_scales_the_output(dataset):
    loader = GrainDataLoader(
        dataset["yaml"], "val", batch_size=2, image_size=32, value_range=(0.0, 255.0)
    )
    images = np.asarray(loader[0][0]["images"])
    assert images.max() > 2.0


def test_coco_conversion_maps_ids_densely_and_skips_crowds(tmp_path):
    images_dir = tmp_path / "images" / "train"
    images_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((100, 200, 3), "uint8")).save(images_dir / "a.png")

    annotations = tmp_path / "instances.json"
    annotations.write_text(
        """{
          "images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 100}],
          "categories": [{"id": 3, "name": "car"}, {"id": 1, "name": "person"}],
          "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 40], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [0, 0, 20, 20], "iscrowd": 1}
          ]
        }""",
        encoding="utf-8",
    )

    config = coco_to_yolo(annotations, images_dir)
    assert config.names == ["person", "car"]

    classes, boxes = load_yolo_label(tmp_path / "labels" / "train" / "a.txt")
    np.testing.assert_array_equal(classes, [0])
    np.testing.assert_allclose(boxes[0], [0.175, 0.4, 0.25, 0.4], atol=1e-6)


def test_coco_conversion_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        coco_to_yolo(tmp_path / "nope.json", tmp_path)


def test_voc_conversion_excludes_difficult_objects(tmp_path):
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "Annotations"
    images_dir.mkdir()
    annotations_dir.mkdir()
    Image.fromarray(np.zeros((100, 200, 3), "uint8")).save(images_dir / "b.png")
    (annotations_dir / "b.xml").write_text(
        """<annotation>
             <filename>b.png</filename>
             <size><width>200</width><height>100</height></size>
             <object><name>dog</name><difficult>0</difficult>
               <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>60</ymax></bndbox>
             </object>
             <object><name>cat</name><difficult>1</difficult>
               <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
             </object>
           </annotation>""",
        encoding="utf-8",
    )

    config = voc_to_yolo(annotations_dir, images_dir, names=NAMES)
    assert config.names == NAMES

    classes, boxes = load_yolo_label(tmp_path / "labels" / "b.txt")
    np.testing.assert_array_equal(classes, [1])
    np.testing.assert_allclose(boxes[0], [0.175, 0.35, 0.25, 0.5], atol=1e-6)


def test_voc_conversion_can_include_difficult_objects(tmp_path):
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "Annotations"
    images_dir.mkdir()
    annotations_dir.mkdir()
    Image.fromarray(np.zeros((100, 200, 3), "uint8")).save(images_dir / "b.png")
    (annotations_dir / "b.xml").write_text(
        """<annotation>
             <filename>b.png</filename>
             <size><width>200</width><height>100</height></size>
             <object><name>cat</name><difficult>1</difficult>
               <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
             </object>
           </annotation>""",
        encoding="utf-8",
    )
    voc_to_yolo(annotations_dir, images_dir, names=NAMES, include_difficult=True)
    classes, _ = load_yolo_label(tmp_path / "labels" / "b.txt")
    np.testing.assert_array_equal(classes, [0])


def test_detector_trains_from_the_dataloader(dataset):
    """The end-to-end path: Grain -> augmentation -> YOLODetector.fit.

    Also covers the ``CloseMosaic`` callback, since the schedule only means
    anything when a real ``fit`` drives it.
    """
    from kyolo.models import yolov8n
    from kyolo.training import CloseMosaic, YOLODetector

    loader = GrainDataLoader(
        dataset["yaml"],
        "train",
        batch_size=2,
        image_size=64,
        max_boxes=8,
        augment=True,
        nc=len(NAMES),
        seed=0,
    )
    detector = YOLODetector(yolov8n(nc=len(NAMES), input_shape=(64, 64, 3)))
    detector.compile(optimizer=keras.optimizers.SGD(1e-3))

    history = detector.fit(
        loader,
        epochs=2,
        steps_per_epoch=1,
        callbacks=[CloseMosaic(loader, close_epochs=1, verbose=0)],
        verbose=0,
    )
    assert np.isfinite(history.history["loss"]).all()
    assert {"box_loss", "cls_loss", "dfl_loss"} <= set(history.history)
    assert loader.augment.mosaic_closed, "CloseMosaic should have fired on the last epoch"


def test_close_mosaic_needs_an_augmented_loader(dataset):
    from kyolo.training import CloseMosaic

    loader = GrainDataLoader(dataset["yaml"], "train", batch_size=2, image_size=32)
    with pytest.raises(ValueError, match="augment=True"):
        CloseMosaic(loader).pipeline()


def test_voc_conversion_reports_classes_missing_from_names(tmp_path):
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "Annotations"
    images_dir.mkdir()
    annotations_dir.mkdir()
    (annotations_dir / "c.xml").write_text(
        """<annotation>
             <filename>c.png</filename>
             <size><width>20</width><height>20</height></size>
             <object><name>tractor</name>
               <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox>
             </object>
           </annotation>""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tractor"):
        voc_to_yolo(annotations_dir, images_dir, names=NAMES)
