# YOLOv8

CSP (C2f) backbone + PAN-FPN neck, anchor-free DFL head.

**Variant factories:** `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov8n

# random init
model = yolov8n(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolov8n(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolov8n(weights="yolov8n.weights.h5")  # already-converted Keras weights
model = yolov8n(nc=80, weights="yolov8n.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLOv8 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. The official checkpoint auto-downloads via ultralytics. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolov8.convert_yolov8_torch_to_keras \
    --weights yolov8n.pt --output yolov8n.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
