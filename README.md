# kyolo — YOLO in pure Keras 3

`kyolo` is a pure [Keras 3](https://keras.io) implementation of the YOLO
object-detection family (YOLOv5, v6, v7, v8, v9, v10, YOLO11, YOLO12, YOLO26).
Every layer, loss, and pre/post-processing step is written entirely in
`keras.ops`, so the exact same code runs unchanged on the **TensorFlow**,
**JAX**, and **PyTorch** backends. It ships with best-effort utilities for
converting the official PyTorch checkpoints into Keras `.weights.h5` files.

## Supported models

| Family   | Variants                     | Detection head                         |
| -------- | ---------------------------- | -------------------------------------- |
| YOLOv5   | `n` `s` `m` `l` `x`          | Classic anchor-based (NMS)             |
| YOLOv6   | `n` `s` `m` `l`              | Anchor-free, DFL (NMS)                 |
| YOLOv7   | `yolov7` `yolov7-tiny` `yolov7-x` | Classic anchor-based (NMS)        |
| YOLOv8   | `n` `s` `m` `l` `x`          | Anchor-free, DFL (NMS)                 |
| YOLOv9   | `t` `s` `m` `c` `e`          | Anchor-free, DFL (NMS)                 |
| YOLOv10  | `n` `s` `m` `b` `l` `x`      | Anchor-free, DFL, **end-to-end (NMS-free)** |
| YOLO11   | `n` `s` `m` `l` `x`          | Anchor-free, DFL (NMS)                 |
| YOLO12   | `n` `s` `m` `l` `x`          | Anchor-free, DFL (NMS)                 |
| YOLO26   | `n` `s` `m` `l` `x`          | Anchor-free, DFL, **end-to-end (NMS-free)** |

Regardless of family, a model's forward pass returns a **list of 3 raw feature
maps** `[P3, P4, P5]`, each of shape `(B, Hi, Wi, 4 * reg_max + nc)` in
channels-last layout, with `reg_max = 16` and strides `(8, 16, 32)`. The
anchor-free families (v6/v8/v9/11/12/26) decode boxes with a Distribution Focal
Loss (DFL) regression head; the end-to-end families (v10, v26) are trained to be
NMS-free but a standard NMS postprocessor is still available for the rest.

## Installation

`kyolo` needs Keras 3 plus **exactly one** backend. Install the package in
editable mode with the backend extra you want:

```bash
# pick ONE backend
pip install -e ".[tensorflow]"
pip install -e ".[jax]"
pip install -e ".[torch]"

# optional extras
pip install -e ".[viz]"      # matplotlib + pillow, for the examples
pip install -e ".[conversion]"  # torch + ultralytics, for weight conversion
pip install -e ".[all]"      # tensorflow + convert + viz + dev
```

Select the active backend with the `KERAS_BACKEND` environment variable before
importing anything:

```bash
export KERAS_BACKEND=tensorflow   # or "jax" / "torch"
```

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # must be set before `import keras`
```

## Quickstart: inference

```python
import keras

from kyolo.models import yolov8n            # every variant is a factory: yolov5n, yolo11s, ...
from kyolo.preprocessing import YOLOPreprocessor
from kyolo.postprocessing import YOLOPostprocessor
from kyolo.utils import visualize_detections, COCO_CLASS_NAMES

# 1. Build a model (deploy=True fuses reparameterizable branches for inference).
model = yolov8n(nc=80, input_shape=(640, 640, 3), deploy=True)
# model.load_weights("yolov8n.weights.h5")  # a converted checkpoint (see below)

# 2. Preprocess: letterbox to a square and (optionally) normalize to [0, 1].
preprocessor = YOLOPreprocessor(image_size=640, normalize=True, letterbox=True)
batch = preprocessor.from_files("assets/bird.png")
# batch == {"images": (B, 640, 640, 3), "ratio": (B, 2), "pad": (B, 2)}

# 3. Forward pass -> list of 3 raw feature maps [P3, P4, P5].
raw_feats = model(batch["images"])

# 4. Postprocess -> (B, max_detections, 6) = [x1, y1, x2, y2, score, class_id].
postprocessor = YOLOPostprocessor(
    nc=80, reg_max=16, strides=(8, 16, 32),
    conf_threshold=0.25, iou_threshold=0.7, max_detections=300,
)
detections = postprocessor(raw_feats)

# 5. Visualize the first image in the batch.
image = keras.ops.convert_to_numpy(batch["images"])[0]
dets = keras.ops.convert_to_numpy(detections)[0]
visualize_detections(image, dets, class_names=COCO_CLASS_NAMES,
                     save_path="assets/inference_result.png")
```

A runnable version lives in [`examples/inference.py`](examples/inference.py).

## Training / fine-tuning

`kyolo.training.YOLODetector` is a `keras.Model` subclass that plugs into
`.fit()`. Datasets yield `(images, targets)` tuples where `targets` is
`{"boxes", "labels", "mask"}` (boxes are xyxy pixels, labels are ints, mask
marks real vs. padded boxes). `detect.detect(...)` runs the full decode + NMS
pipeline in one call:

```python
import keras

from kyolo.models import yolov8n
from kyolo.losses import YOLODetectionLoss
from kyolo.training import YOLODetector

nc = 80
model = yolov8n(nc=nc)

# Fine-tuning: load converted weights onto the bare model BEFORE wrapping it.
# model.load_weights("yolov8n.weights.h5")   # AGPL-3.0 converted checkpoint

# Freeze the backbone so only the neck + detection head train:
for layer in model.layers:
    if layer.name.startswith("backbone"):
        layer.trainable = False

loss = YOLODetectionLoss(nc=nc, reg_max=16, strides=(8, 16, 32))
detector = YOLODetector(model, loss=loss, nc=nc, reg_max=16)
detector.compile(optimizer=keras.optimizers.Adam(1e-3))

# `dataset` yields (images, {"boxes", "labels", "mask"}) tuples.
detector.fit(dataset, epochs=1, steps_per_epoch=2)

# Convenience: run the full detect -> NMS pipeline in one call.
detections = detector.detect(images, conf=0.25, iou=0.7)
```

A runnable synthetic-data version lives in [`examples/train.py`](examples/train.py).

## Weight conversion

The official YOLO checkpoints are **AGPL-3.0 licensed** and are **not**
redistributed with this project. To use pretrained weights you must convert an
official `.pt` file yourself. Each model ships a co-located converter:

```bash
# per-model converter (co-located under kyolo/models/<name>/)
python -m kyolo.models.yolov8.convert_yolov8_torch_to_keras \
    --weights yolov8n.pt --output yolov8n.weights.h5 --variant n

# or the unified CLI (installed as `kyolo-convert`)
kyolo-convert --model yolov8n --weights yolov8n.pt --output yolov8n.weights.h5
```

Conversion requires the `conversion` extra (`pip install -e ".[conversion]"`,
which pulls in `torch` and `ultralytics`). The programmatic entry point is
`from kyolo.conversion import convert_weights`.

> Conversion is **best-effort**: layer-name and tensor-layout mappings are
> maintained by hand, so always validate a converted model's outputs against the
> reference PyTorch implementation before trusting it.

## Package layout

The repo follows the [KerasFormers](https://github.com/IMvision12/KerasFormers)
layout: a flat top-level package with one self-contained folder per model.

```
kyolo/                       # flat package (no src/)
├── __init__.py
├── version.py               # single source of truth for the version → releases
├── models/
│   ├── __init__.py          # per-variant factories (yolov8n, yolo11s, ...)
│   ├── base.py              # shared assembly (image_input, finalize_detector)
│   └── yolov8/              # one folder per model (v5, v6, v7, v8, v9, v10, 11, 12, 26)
│       ├── __init__.py
│       ├── config.py                          # scale variants
│       ├── yolov8_model.py                     # architecture
│       └── convert_yolov8_torch_to_keras.py    # co-located weight converter
├── heads/                   # detection head
├── layers/                  # conv/bn, CSP/ELAN/GELAN/attention blocks, DFL, letterbox
├── ops/                     # anchors, box math, task-aligned assigner (pure keras.ops)
├── losses/                  # YOLODetectionLoss
├── preprocessing/           # YOLOPreprocessor
├── postprocessing/          # YOLOPostprocessor + pure-ops NMS
├── training/                # YOLODetector (keras.Model subclass)
├── conversion/              # convert_weights, name mappings, exceptions, CLI
└── utils/                   # COCO metadata + visualization

.github/workflows/           # release.yml (auto-release) + test_code.yml (CI)
docs/                        # one page per model
tests/integration/           # build + processing smoke tests
examples/                    # inference.py, train.py
```

## Releasing

Releases are automated. Bump `__version__` in
[`kyolo/version.py`](kyolo/version.py) and merge to `main`:
[`.github/workflows/release.yml`](.github/workflows/release.yml) detects the
change, creates the `v<version>` GitHub release, and publishes the wheel + sdist
to PyPI (via Trusted Publishing — no API token). The package version is read
dynamically from `kyolo.__version__`, so `version.py` is the single source of
truth.

## License

- The **code** in this repository is licensed under the
  [Apache License 2.0](LICENSE).
- The official YOLO **weights** are licensed under
  [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  They are not shipped here; anything you convert from them inherits AGPL-3.0,
  and using such weights subjects your project to the AGPL-3.0 requirements.
