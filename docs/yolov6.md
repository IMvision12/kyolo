# YOLOv6

EfficientRep backbone + Rep-PAN neck (reparameterizable).

**Variant factories:** `yolov6n`, `yolov6s`, `yolov6m`, `yolov6l`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov6n
model = yolov6n(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLOv6 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolov6.convert_yolov6_torch_to_keras \
    --weights yolov6n.pt --output yolov6n.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
