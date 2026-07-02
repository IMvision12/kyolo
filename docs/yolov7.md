# YOLOv7

ELAN backbone + SPPCSPC + ELAN-PAN neck.

**Variant factories:** `yolov7`, `yolov7_tiny`, `yolov7_x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov7

# random init
model = yolov7(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolov7(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolov7(weights="yolov7.weights.h5")  # already-converted Keras weights
model = yolov7(nc=80, weights="yolov7.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLOv7 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. Auto-download is not available for this family; pass weights=<path.pt>. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolov7.convert_yolov7_torch_to_keras \
    --weights yolov7.pt --output yolov7.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
