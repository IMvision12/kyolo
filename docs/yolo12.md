# YOLO12

Area-attention (A2C2f) backbone + PAN-FPN neck.

**Variant factories:** `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`
**Post-processing:** greedy NMS

## Build

```python
from kyolo.models import yolo12n
model = yolo12n(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLO12 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolo12.convert_yolo12_torch_to_keras \
    --weights yolo12n.pt --output yolo12n.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
