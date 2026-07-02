# YOLOv6

EfficientRep backbone + Rep-PAN neck (reparameterizable).

**Variant factories:** `yolov6n`, `yolov6s`, `yolov6m`, `yolov6l`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov6n

# random init
model = yolov6n(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolov6n(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolov6n(weights="yolov6n.weights.h5")  # already-converted Keras weights
model = yolov6n(nc=80, weights="yolov6n.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLOv6 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. Auto-download is not available for this family; pass weights=<path.pt>. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolov6.convert_yolov6_torch_to_keras \
    --weights yolov6n.pt --output yolov6n.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
