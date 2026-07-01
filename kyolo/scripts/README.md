# Weight conversion scripts

`scripts/convert.py` converts an official YOLO **PyTorch `.pt` checkpoint** into
a kyolo **Keras `.weights.h5`** file.

> **Weights are not included.** The official YOLO weights are licensed
> **AGPL-3.0** and are *not* redistributed by this project. You must download
> the `.pt` file yourself (e.g. from Ultralytics / the original authors) before
> converting it. Converting weights does not change their license.

## Requirements

PyTorch is only needed to *read* the source checkpoint and is an optional
dependency:

```bash
pip install "kyolo[convert]"   # installs torch (+ ultralytics)
```

You also need one Keras backend installed (`tensorflow`, `jax`, or `torch`).

## Usage

```bash
python scripts/convert.py --model yolov8n --weights yolov8n.pt --output yolov8n.weights.h5 --method order
```

Equivalent console script (installed with the package):

```bash
kyolo-convert --model yolov8n --weights yolov8n.pt --output yolov8n.weights.h5 --method order
```

### Options

| Flag        | Default                     | Description                                        |
|-------------|-----------------------------|----------------------------------------------------|
| `--model`   | *(required)*                | Model name (see supported names below).            |
| `--weights` | *(required)*                | Path to the source `.pt` checkpoint.               |
| `--output`  | `<weights-stem>.weights.h5` | Output path for the Keras weights.                 |
| `--method`  | `order`                     | `order` (robust, recommended) or `name`.           |
| `--nc`      | `80`                        | Number of classes (80 = COCO).                     |
| `--imgsz`   | `640`                       | Square input size used to build the model.         |
| `--quiet`   | off                         | Suppress progress output.                          |

### Methods

- **`order`** (recommended): zips the Keras variables and the Torch parameters
  in build order and transfers them positionally. It does not depend on exact
  name matching and validates that counts and shapes agree.
- **`name`**: derives a Torch key from each Keras variable's dotted path. This
  is best-effort and may need per-model mapping tuning (see
  `src/kyolo/conversion/mappings.py`).

> A clean, error-free transfer is **necessary but not sufficient** for a
> bit-exact model. Always validate the converted model's outputs against the
> PyTorch reference on the same input before trusting it.

## Supported model names

- **YOLOv5**: `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`
- **YOLOv6**: `yolov6n`, `yolov6s`, `yolov6m`, `yolov6l`
- **YOLOv7**: `yolov7`, `yolov7-tiny`, `yolov7-x`
- **YOLOv8**: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- **YOLOv9**: `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov9e`
- **YOLOv10**: `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, `yolov10x`
- **YOLO11**: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- **YOLO12**: `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`
- **YOLO26**: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`
