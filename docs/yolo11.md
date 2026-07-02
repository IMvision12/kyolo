# YOLO11

C3k2 backbone + C2PSA attention, lightweight DFL head.

**Variant factories:** `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolo11n

# random init
model = yolo11n(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolo11n(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolo11n(weights="yolo11n.weights.h5")  # already-converted Keras weights
model = yolo11n(nc=80, weights="yolo11n.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLO11 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. The official checkpoint auto-downloads via ultralytics. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolo11.convert_yolo11_torch_to_keras \
    --weights yolo11n.pt --output yolo11n.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
