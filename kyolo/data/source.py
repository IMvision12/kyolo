"""A random-access dataset source for Grain.

Grain pipelines are built on ``__len__`` / ``__getitem__`` (its
``RandomAccessDataSource`` protocol), which is a good fit for detection data:
one image plus one label file per index, no sequential reading, and free
reordering for shuffling and rectangular batching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import label_path_for, load_yolo_label, normalized_to_xyxy

__all__ = ["YOLODataSource"]

CACHE_MODES = (None, "ram", "disk")


def load_image(path):
    """Decode an image to ``(H, W, 3)`` uint8 RGB."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "reading images needs Pillow. Install it with `pip install pillow`."
        ) from exc

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype="uint8")


def read_shape(path):
    """Read ``(height, width)`` from the file header, without decoding."""
    from PIL import Image

    with Image.open(path) as handle:
        width, height = handle.size
    return height, width


class YOLODataSource:
    """Images plus YOLO ``.txt`` labels, addressed by index.

    Each item is a dict of plain numpy arrays::

        {
            "image":      (H, W, 3) uint8,
            "boxes":      (N, 4)    float32   # xyxy, in pixels
            "labels":     (N,)      int32
            "orig_shape": (2,)      int32     # (height, width)
            "index":      ()        int32
        }

    ``N`` varies per image; :class:`kyolo.data.PadTargets` fixes it before
    batching.

    Args:
        image_paths: the images, in the order they should be indexed.
        nc: number of classes. When given, out-of-range class indices in a label
            file are rejected rather than silently trained on.
        label_paths: explicit label files, parallel to ``image_paths``. Defaults
            to the Ultralytics ``images/`` -> ``labels/`` convention.
        cache: ``None`` to decode every time, ``"ram"`` to hold decoded images in
            memory, or ``"disk"`` to write a decoded ``.npy`` beside each image
            on first read. ``"disk"`` survives across runs and across worker
            processes; ``"ram"`` is per-process, so with Grain multiprocessing
            each worker caches only the shard it reads.
    """

    def __init__(self, image_paths, nc=None, label_paths=None, cache=None):
        if cache not in CACHE_MODES:
            raise ValueError(f"cache must be one of {CACHE_MODES}; got {cache!r}.")
        self.image_paths = [Path(p) for p in image_paths]
        if not self.image_paths:
            raise ValueError("`image_paths` is empty; there is nothing to read.")
        if label_paths is None:
            self.label_paths = [label_path_for(p) for p in self.image_paths]
        else:
            self.label_paths = [Path(p) for p in label_paths]
            if len(self.label_paths) != len(self.image_paths):
                raise ValueError(
                    f"got {len(self.image_paths)} images but "
                    f"{len(self.label_paths)} label paths; they must be parallel."
                )
        self.nc = nc
        self.cache = cache
        self._ram = {}
        self._shapes = None

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return (
            f"YOLODataSource(n={len(self)}, nc={self.nc}, cache={self.cache!r}, "
            f"first={self.image_paths[0].name!r})"
        )

    def __getstate__(self):
        """Drop the in-memory cache so Grain can ship this to worker processes."""
        state = self.__dict__.copy()
        state["_ram"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def shapes(self):
        """``(N, 2)`` array of original ``(height, width)`` per image.

        Read from image headers, so this does not decode anything. Used by
        rectangular batching to group images of similar aspect ratio.
        """
        if self._shapes is None:
            self._shapes = np.array([read_shape(p) for p in self.image_paths], dtype="int32")
        return self._shapes

    def image(self, index):
        if self.cache == "ram":
            if index not in self._ram:
                self._ram[index] = load_image(self.image_paths[index])
            return self._ram[index]

        if self.cache == "disk":
            cached = self.image_paths[index].with_suffix(".npy")
            if cached.is_file():
                return np.load(cached)
            image = load_image(self.image_paths[index])
            np.save(cached, image)
            return image

        return load_image(self.image_paths[index])

    def __getitem__(self, index):
        index = int(index)
        image = self.image(index)
        height, width = image.shape[:2]

        classes, boxes = load_yolo_label(self.label_paths[index], nc=self.nc)
        return {
            "image": image,
            "boxes": normalized_to_xyxy(boxes, width, height),
            "labels": classes,
            "orig_shape": np.array([height, width], dtype="int32"),
            "index": np.int32(index),
        }
