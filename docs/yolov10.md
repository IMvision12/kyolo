# YOLOv10

C2f/C2fCIB + SCDown + PSA. End-to-end NMS-free.

**Variant factories:** `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, `yolov10x`
**Post-processing:** NMS-free (end_to_end=True)

## Build

```python
from kyolo.models import yolov10n

# random init
model = yolov10n(nc=80, input_shape=(640, 640, 3), deploy=True)

# auto-download + convert + load the official COCO weights
model = yolov10n(convert_weights=True)          # needs torch + ultralytics

# or load your own converted / PyTorch checkpoint
model = yolov10n(weights="yolov10n.weights.h5")  # already-converted Keras weights
model = yolov10n(nc=80, weights="yolov10n.pt")   # convert a PyTorch file on load
```

feats = model(images) returns a list of 3 raw feature maps [P3, P4, P5].

## Weight conversion

The official YOLOv10 weights are AGPL-3.0 and are not shipped with kyolo;
they are downloaded from source on demand. The official checkpoint auto-downloads via ultralytics. You can also run the
co-located converter directly:

```bash
python -m kyolo.models.yolov10.convert_yolov10_torch_to_keras \
    --weights yolov10n.pt --output yolov10n.weights.h5
```

A clean transfer is necessary but not sufficient: validate the outputs against
the PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
