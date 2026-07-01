# YOLOv7

ELAN backbone + SPPCSPC + ELAN-PAN neck.

**Variant factories:** `yolov7`, `yolov7_tiny`, `yolov7_x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov7
model = yolov7(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLOv7 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolov7.convert_yolov7_torch_to_keras \
    --weights yolov7.pt --output yolov7.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
