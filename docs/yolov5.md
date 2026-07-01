# YOLOv5

CSP (C3) backbone + PAN-FPN neck, anchor-free DFL head.

**Variant factories:** `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov5n
model = yolov5n(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLOv5 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolov5.convert_yolov5_torch_to_keras \
    --weights yolov5n.pt --output yolov5n.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
