# YOLOv9

GELAN (RepNCSPELAN4 / ADown / SPPELAN) backbone + neck.

**Variant factories:** `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov9e`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov9t

# random init
model = yolov9t(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolov9t(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolov9t(weights="yolov9t.weights.h5")  # already-converted Keras weights
model = yolov9t(nc=80, weights="yolov9t.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLOv9 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. The official checkpoint auto-downloads via ultralytics. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolov9.convert_yolov9_torch_to_keras \
    --weights yolov9t.pt --output yolov9t.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
