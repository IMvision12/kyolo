# YOLO26

C3k2 + attention backbone. End-to-end NMS-free (2025).

**Variant factories:** `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`
**Post-processing:** NMS-free (`end_to_end=True`)

## Build

```python
from kyolo.models import yolo26n
model = yolo26n(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLO26 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolo26.convert_yolo26_torch_to_keras \
    --weights yolo26n.pt --output yolo26n.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
