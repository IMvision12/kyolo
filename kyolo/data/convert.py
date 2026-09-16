"""Converting COCO and Pascal VOC annotations into YOLO label files.

Both functions write the one-``.txt``-per-image layout that
:class:`kyolo.data.YOLODataSource` reads, and return a
:class:`kyolo.data.DataConfig` describing the result, so a converted dataset can
go straight into :class:`kyolo.data.GrainDataLoader`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import DataConfig
from .labels import write_yolo_label

__all__ = ["coco_to_yolo", "voc_to_yolo"]


def normalize(x_min, y_min, box_w, box_h, width, height):
    """Pixel ``(x, y, w, h)`` -> clipped normalized ``(cx, cy, w, h)``, or None."""
    if width <= 0 or height <= 0 or box_w <= 0 or box_h <= 0:
        return None

    x_min = max(0.0, min(float(x_min), width))
    y_min = max(0.0, min(float(y_min), height))
    x_max = max(0.0, min(x_min + float(box_w), width))
    y_max = max(0.0, min(y_min + float(box_h), height))
    if x_max <= x_min or y_max <= y_min:
        return None

    return (
        (x_min + x_max) / 2.0 / width,
        (y_min + y_max) / 2.0 / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    )


def coco_to_yolo(annotations, images_dir, labels_dir=None, skip_crowd=True, write_config=None):
    """Convert a COCO ``instances_*.json`` into YOLO label files.

    Category ids are remapped to a contiguous ``0..nc-1`` range in ascending id
    order. For official COCO that reproduces the canonical 80-class ordering,
    which is what the pretrained weights expect; COCO's own ids run to 90 with
    gaps, and training needs them dense.

    Args:
        annotations: path to the COCO JSON.
        images_dir: directory holding the images the JSON refers to.
        labels_dir: where to write labels. Defaults to the Ultralytics sibling
            layout, i.e. ``images_dir`` with its last ``images`` component
            renamed ``labels``.
        skip_crowd: skip ``iscrowd=1`` annotations, which mark unsegmented
            crowds rather than individual objects.
        write_config: optional path to also write a ``data.yaml`` to.

    Returns:
        A :class:`~kyolo.data.DataConfig` for the converted dataset.
    """
    annotations = Path(annotations)
    images_dir = Path(images_dir)
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO annotation file {annotations} does not exist.")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"images_dir {images_dir} is not a directory.")
    labels_dir = Path(labels_dir) if labels_dir else sibling_labels_dir(images_dir)

    with open(annotations, encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = sorted(payload.get("categories", []), key=lambda c: int(c["id"]))
    if not categories:
        raise ValueError(f"{annotations} has no `categories`; cannot assign class indices.")
    class_of = {int(c["id"]): i for i, c in enumerate(categories)}
    names = [str(c["name"]) for c in categories]

    images = {int(i["id"]): i for i in payload.get("images", [])}
    if not images:
        raise ValueError(f"{annotations} has no `images`.")

    grouped = {image_id: [] for image_id in images}
    for annotation in payload.get("annotations", []):
        if skip_crowd and annotation.get("iscrowd", 0):
            continue
        image_id = int(annotation["image_id"])
        info = images.get(image_id)
        if info is None:
            continue
        box = normalize(
            *annotation["bbox"], width=float(info["width"]), height=float(info["height"])
        )
        if box is not None:
            grouped[image_id].append((class_of[int(annotation["category_id"])], box))

    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_id, info in images.items():
        entries = grouped[image_id]
        target = labels_dir / Path(str(info["file_name"])).with_suffix(".txt").name
        write_yolo_label(
            target,
            [c for c, _ in entries],
            [b for _, b in entries],
        )

    config = DataConfig.from_directories(train=str(images_dir), names=names, root=images_dir.parent)
    if write_config:
        config.to_yaml(write_config)
    return config


def voc_to_yolo(
    annotations_dir,
    images_dir,
    labels_dir=None,
    names=None,
    include_difficult=False,
    write_config=None,
):
    """Convert Pascal VOC XML annotations into YOLO label files.

    Args:
        annotations_dir: directory of VOC ``.xml`` files.
        images_dir: directory holding the corresponding images.
        labels_dir: where to write labels. Defaults to the sibling layout.
        names: class names in the order their indices should take. ``None``
            collects every name in the annotations and sorts them
            alphabetically, so pass this explicitly when the ordering has to
            match an existing model.
        include_difficult: keep objects VOC flags as ``difficult``. These are
            excluded from the standard VOC training protocol.
        write_config: optional path to also write a ``data.yaml`` to.

    Returns:
        A :class:`~kyolo.data.DataConfig` for the converted dataset.
    """
    import xml.etree.ElementTree as ElementTree

    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)
    if not annotations_dir.is_dir():
        raise NotADirectoryError(f"annotations_dir {annotations_dir} is not a directory.")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"images_dir {images_dir} is not a directory.")
    labels_dir = Path(labels_dir) if labels_dir else sibling_labels_dir(images_dir)

    xml_files = sorted(annotations_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"no .xml annotations found under {annotations_dir}.")

    parsed = []
    discovered = set()
    for xml_file in xml_files:
        root = ElementTree.parse(xml_file).getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"{xml_file}: missing <size>, so boxes cannot be normalized.")
        width = float(size.findtext("width", "0"))
        height = float(size.findtext("height", "0"))

        objects = []
        for obj in root.findall("object"):
            if not include_difficult and obj.findtext("difficult", "0").strip() == "1":
                continue
            name = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")
            if not name or box is None:
                continue
            x_min = float(box.findtext("xmin", "0"))
            y_min = float(box.findtext("ymin", "0"))
            x_max = float(box.findtext("xmax", "0"))
            y_max = float(box.findtext("ymax", "0"))
            normalized = normalize(x_min, y_min, x_max - x_min, y_max - y_min, width, height)
            if normalized is not None:
                objects.append((name, normalized))
                discovered.add(name)

        parsed.append((xml_file, root.findtext("filename"), objects))

    class_names = list(names) if names is not None else sorted(discovered)
    index_of = {name: i for i, name in enumerate(class_names)}
    unknown = discovered - set(index_of)
    if unknown:
        raise ValueError(f"annotations contain classes missing from `names`: {sorted(unknown)}.")

    labels_dir.mkdir(parents=True, exist_ok=True)
    for xml_file, filename, objects in parsed:
        stem = Path(filename).stem if filename else xml_file.stem
        write_yolo_label(
            labels_dir / f"{stem}.txt",
            [index_of[name] for name, _ in objects],
            [box for _, box in objects],
        )

    config = DataConfig.from_directories(
        train=str(images_dir), names=class_names, root=images_dir.parent
    )
    if write_config:
        config.to_yaml(write_config)
    return config


def sibling_labels_dir(images_dir):
    """``.../images/train`` -> ``.../labels/train``, else a ``labels`` sibling."""
    parts = list(images_dir.parts)
    for position in range(len(parts) - 1, -1, -1):
        if parts[position] == "images":
            parts[position] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"
