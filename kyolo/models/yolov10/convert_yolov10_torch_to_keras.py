"""Convert an official YOLOv10 PyTorch checkpoint into kyolo Keras 3 weights.

The official YOLO weights are AGPL-3.0 licensed and are **not** redistributed
by kyolo — download your own ``.pt`` and convert it here. A clean transfer is
necessary but not sufficient: validate the converted model's outputs against
the reference before trusting them (see :mod:`kyolo.conversion`).

Usage::

    python -m kyolo.models.yolov10.convert_yolov10_torch_to_keras \
        --weights yolov10n.pt --output yolov10n.weights.h5 --variant 'n'
"""

from __future__ import annotations

from kyolo.conversion import convert_weights

from .yolov10_model import build_yolov10


def convert(
    weights_path,
    variant="n",
    nc=80,
    imgsz=640,
    output=None,
    method="order",
    verbose=True,
):
    """Build a YOLOv10 model and transfer weights from ``weights_path``."""
    model = build_yolov10(variant=variant, nc=nc, input_shape=(imgsz, imgsz, 3), deploy=True)
    return convert_weights(model, weights_path, output_path=output, method=method, verbose=verbose)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Convert YOLOv10 .pt -> kyolo .weights.h5")
    p.add_argument("--weights", required=True, help="Path to the source .pt checkpoint")
    p.add_argument("--variant", default="n", help="Model variant")
    p.add_argument("--nc", type=int, default=80, help="Number of classes")
    p.add_argument("--imgsz", type=int, default=640, help="Build image size")
    p.add_argument("--output", default=None, help="Output .weights.h5 path")
    p.add_argument("--method", default="order", choices=["order", "name"])
    args = p.parse_args()
    convert(
        args.weights,
        variant=args.variant,
        nc=args.nc,
        imgsz=args.imgsz,
        output=args.output,
        method=args.method,
    )


if __name__ == "__main__":
    main()
