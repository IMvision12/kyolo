# YOLOv10

C2f/C2fCIB + SCDown + PSA. End-to-end NMS-free.

**Variant factories:** `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l`, `yolov10x`
**Post-processing:** NMS-free (`end_to_end=True`)

## Build

```python
from kyolo.models import yolov10n
model = yolov10n(nc=80, input_shape=(640, 640, 3), deploy=True)
feats = model(images)          # list of 3 raw feature maps [P3, P4, P5]
```

## Weight conversion

The official YOLOv10 weights are **AGPL-3.0** and are not shipped with
kyolo. Convert your own checkpoint with the co-located converter:

```bash
python -m kyolo.models.yolov10.convert_yolov10_torch_to_keras \
    --weights yolov10n.pt --output yolov10n.weights.h5
```

A clean transfer is necessary but not sufficient — validate outputs against the
PyTorch reference. See [`conversion`](../kyolo/conversion) for details.
