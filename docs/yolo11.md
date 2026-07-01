# YOLO11

C3k2 backbone + C2PSA attention, lightweight DFL head.

**Variant factories:** `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolo11n
model = yolo11n(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLO11 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolo11.convert_yolo11_torch_to_keras \
    --weights yolo11n.pt --output yolo11n.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
