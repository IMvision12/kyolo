"""Dataset configuration, in Ultralytics' ``data.yaml`` format."""

from __future__ import annotations

import dataclasses
from pathlib import Path

__all__ = ["DataConfig", "IMAGE_EXTENSIONS"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

SPLIT_KEYS = ("train", "val", "test")


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "reading a dataset YAML needs PyYAML. Install it with "
            "`pip install kyolo[data]` or `pip install pyyaml`."
        ) from exc
    return yaml


def normalize_names(names, nc):
    """Accept ``names`` as a list, a tuple or an index->name mapping."""
    if names is None:
        if nc is None:
            raise ValueError("a dataset config needs either `names` or `nc`; neither was given.")
        return [f"class_{i}" for i in range(int(nc))]

    if isinstance(names, dict):
        try:
            indices = sorted(int(k) for k in names)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`names` keys must be class indices; got {list(names)}.") from exc
        if indices != list(range(len(indices))):
            raise ValueError(
                "`names` keys must be a contiguous 0-based range; got "
                f"{indices[:8]}{'...' if len(indices) > 8 else ''}."
            )
        ordered = [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    else:
        ordered = [str(n) for n in names]

    if nc is not None and int(nc) != len(ordered):
        raise ValueError(f"`nc` is {nc} but `names` has {len(ordered)} entries; they must agree.")
    return ordered


@dataclasses.dataclass
class DataConfig:
    """A resolved dataset description.

    Args:
        root: directory the split entries are relative to.
        splits: mapping of split name to the entries that define it. Each entry
            is a directory of images, a ``.txt`` file listing image paths, or a
            single image.
        names: class names, ordered by class index.
    """

    root: Path
    splits: dict
    names: list

    @property
    def nc(self):
        """Number of classes."""
        return len(self.names)

    @classmethod
    def from_yaml(cls, path):
        """Read an Ultralytics-style ``data.yaml``.

        Understands the documented layout::

            path: ../datasets/coco8
            train: images/train
            val: images/val
            names:
              0: person
              1: bicycle

        ``path`` is resolved relative to the YAML file itself (so a config can
        be moved around with its dataset), the split entries relative to
        ``path``, and ``names`` may equally be a plain list with a matching
        ``nc``, as older configs write it.
        """
        yaml = require_yaml()
        path = Path(path).expanduser().resolve()
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping; got {type(raw).__name__}.")

        root = path.parent
        if raw.get("path"):
            root = (root / str(raw["path"])).resolve()

        splits = {}
        for key in SPLIT_KEYS:
            entry = raw.get(key)
            if entry:
                splits[key] = [str(e) for e in (entry if isinstance(entry, list) else [entry])]
        if not splits:
            raise ValueError(
                f"{path} defines none of {SPLIT_KEYS}; at least one split is required."
            )

        return cls(
            root=root,
            splits=splits,
            names=normalize_names(raw.get("names"), raw.get("nc")),
        )

    @classmethod
    def from_directories(cls, train=None, val=None, test=None, names=None, nc=None, root=None):
        """Build a config from directories directly, with no YAML involved."""
        given = {"train": train, "val": val, "test": test}
        splits = {k: [str(v)] for k, v in given.items() if v is not None}
        if not splits:
            raise ValueError("at least one of `train`, `val` or `test` must be given.")
        return cls(
            root=Path(root).expanduser().resolve() if root else Path.cwd(),
            splits=splits,
            names=normalize_names(names, nc),
        )

    def image_paths(self, split="train"):
        """Every image path in ``split``, sorted for a reproducible order.

        Each entry resolves as a directory of images (searched recursively), a
        ``.txt`` file listing image paths one per line, or a single image.
        """
        if split not in self.splits:
            raise KeyError(
                f"split {split!r} is not defined; this config has {sorted(self.splits)}."
            )

        paths = []
        for entry in self.splits[split]:
            target = Path(entry)
            if not target.is_absolute():
                target = self.root / target

            if target.is_dir():
                paths.extend(
                    p for p in sorted(target.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS
                )
            elif target.suffix.lower() == ".txt":
                if not target.is_file():
                    raise FileNotFoundError(
                        f"image list {target} for split {split!r} does not exist."
                    )
                with open(target, encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        listed = Path(line)
                        paths.append(
                            listed if listed.is_absolute() else (target.parent / listed).resolve()
                        )
            elif target.is_file() and target.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(target)
            else:
                raise FileNotFoundError(
                    f"split entry {target} for split {split!r} is neither a directory, "
                    "a `.txt` image list, nor an image file."
                )

        if not paths:
            raise FileNotFoundError(
                f"no images found for split {split!r} under {self.root} "
                f"(entries: {self.splits[split]}). Supported extensions: "
                f"{', '.join(IMAGE_EXTENSIONS)}."
            )
        return sorted(set(paths))

    def to_yaml(self, path):
        """Write this config back out as a ``data.yaml``."""
        yaml = require_yaml()
        payload = {
            "path": str(self.root),
            **{k: v[0] if len(v) == 1 else v for k, v in self.splits.items()},
            "names": dict(enumerate(self.names)),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        return path
