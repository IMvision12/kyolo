# YOLO12

Area-attention (A2C2f) backbone + PAN-FPN neck.

**Variant factories:** `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolo12n

# random init
model = yolo12n(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolo12n(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolo12n(weights="yolo12n.weights.h5")  # already-converted Keras weights
model = yolo12n(nc=80, weights="yolo12n.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLO12 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. The official checkpoint auto-downloads via ultralytics. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolo12.convert_yolo12_torch_to_keras \
    --weights yolo12n.pt --output yolo12n.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
