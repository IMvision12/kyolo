# kyolo - YOLO in pure Keras 3

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

Each variant is a **factory function** named `<family><variant>`, imported
directly from `kyolo.models` (or `kyolo`):

```python
from kyolo.models import yolov5n, yolov8m, yolo11s, yolov9c, yolov7_tiny
model = yolov8m(nc=80)          # -> keras.Model
```

`kyolo.models.MODEL_NAMES` lists all 43 factories. (YOLOv7 uses `yolov7`,
`yolov7_tiny`, `yolov7_x`.)

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
pip install -e ".[conversion]"  # torch + ultralytics, for weight conversion
```

Visualization (`matplotlib` + `pillow`, used by the examples) ships with the
base install, so a plain `pip install kyolo` is enough to run them.

Select the active backend with the `KERAS_BACKEND` environment variable before
importing anything:

```bash
export KERAS_BACKEND=tensorflow   # or "jax" / "torch"
```

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # must be set before `import keras`
```

## Data format (channels_last / channels_first)

Every model, loss and pre/post-processor takes a `data_format` argument. Leaving
it as the default (`None`) follows the global Keras setting
`keras.config.image_data_format()`, so one call switches the whole library:

```python
import keras
from kyolo.models import yolov8n

keras.config.set_image_data_format("channels_first")
model = yolov8n(nc=80, input_shape=(3, 640, 640))   # (C, H, W) inputs, (B, C, H, W) feats

# ...or override per call, ignoring the global setting:
model = yolov8n(nc=80, input_shape=(640, 640, 3), data_format="channels_last")
```

`input_shape` is `(H, W, C)` for channels_last and `(C, H, W)` for
channels_first. `YOLOPreprocessor` always accepts channels_last `(H, W, C)`
images and returns them in the requested layout, so the raw image loading path
is unchanged.

## Quickstart: inference

```python
import keras

from kyolo.models import yolov8n            # every variant is a factory: yolov5n, yolo11s, ...
from kyolo.preprocessing import YOLOPreprocessor
from kyolo.postprocessing import YOLOPostprocessor
from kyolo.utils import visualize_detections, COCO_CLASS_NAMES

# 1. Build a model (deploy=True fuses reparameterizable branches for inference).
model = yolov8n(nc=80, input_shape=(640, 640, 3), deploy=True)
# model = yolov8n(convert_weights=True)      # auto-download + convert official COCO weights
# model = yolov8n(weights="yolov8n.weights.h5")  # or load a converted checkpoint (see below)

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

# Fine-tuning: build with the pretrained COCO backbone, then swap the head for
# your class count by fine-tuning. To start from converted weights, build the
# nc=80 model with convert_weights=True (or weights="yolov8n.weights.h5").
# model = yolov8n(convert_weights=True)

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
redistributed with this project. kyolo does not ship weights; it downloads and
converts them from source on demand.

### From the factory (recommended)

Every variant factory takes `convert_weights` and `weights` arguments:

```python
from kyolo.models import yolov8n

# Auto-download the official COCO ".pt", convert it, cache and load it.
# Needs the "conversion" extra (torch + ultralytics) and nc=80.
model = yolov8n(convert_weights=True)

# Load your own already-converted Keras weights.
model = yolov8n(weights="yolov8n.weights.h5")

# Convert a PyTorch checkpoint you supply (path or http URL) on load.
model = yolov8n(nc=80, weights="yolov8n.pt")
```

`convert_weights=True` caches the converted `.weights.h5` under `~/.cache/kyolo`
(override with `cache_dir=` or the `KYOLO_CACHE` env var), so subsequent calls
load instantly. Auto-download covers the ultralytics families (v5, v8, v9, v10,
11, 12, 26); for YOLOv6 / YOLOv7 fetch the `.pt` yourself and pass `weights=`.

### From the command line

Each model also ships a co-located converter, plus a unified CLI:

```bash
# per-model converter (co-located under kyolo/models/<name>/)
python -m kyolo.models.yolov8.convert_yolov8_torch_to_keras \
    --weights yolov8n.pt --output yolov8n.weights.h5 --variant n

# or the unified CLI (installed as `kyolo-convert`)
kyolo-convert --model yolov8n --weights yolov8n.pt --output yolov8n.weights.h5
```

Conversion requires the `conversion` extra (`pip install -e ".[conversion]"`,
which pulls in `torch` and `ultralytics`). The programmatic entry points are
`from kyolo.conversion import convert_weights, load_pretrained`.

> Conversion is **best-effort**: layer-name and tensor-layout mappings are
> maintained by hand, so always validate a converted model's outputs against the
> reference PyTorch implementation before trusting it.

## License

- The **code** in this repository is licensed under the
  [Apache License 2.0](LICENSE).
- The official YOLO **weights** are licensed under
  [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  They are not shipped here; anything you convert from them inherits AGPL-3.0,
  and using such weights subjects your project to the AGPL-3.0 requirements.
