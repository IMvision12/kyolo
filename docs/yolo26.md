# YOLO26

C3k2 + attention backbone. End-to-end NMS-free (2025).

**Variant factories:** `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`
**Post-processing:** NMS-free (end_to_end=True)

## Build

```python
from kyolo.models import yolo26n

# random init
model = yolo26n(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolo26n(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolo26n(weights="yolo26n.weights.h5")  # already-converted Keras weights
model = yolo26n(nc=80, weights="yolo26n.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLO26 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. The official checkpoint auto-downloads via ultralytics. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolo26.convert_yolo26_torch_to_keras \
    --weights yolo26n.pt --output yolo26n.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
