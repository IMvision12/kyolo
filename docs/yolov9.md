# YOLOv9

GELAN (RepNCSPELAN4 / ADown / SPPELAN) backbone + neck.

**Variant factories:** `yolov9t`, `yolov9s`, `yolov9m`, `yolov9c`, `yolov9e`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolov9t
model = yolov9t(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLOv9 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolov9.convert_yolov9_torch_to_keras \
    --weights yolov9t.pt --output yolov9t.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
