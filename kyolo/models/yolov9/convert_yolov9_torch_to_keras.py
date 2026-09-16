"""Convert an official YOLOv9 PyTorch checkpoint into kyolo Keras 3 weights.

The official YOLO weights are AGPL-3.0 licensed and are **not** redistributed
by kyolo - download your own ``.pt`` and convert it here. A clean transfer is
necessary but not sufficient: validate the converted model's outputs against
the reference before trusting them (see :mod:`kyolo.conversion`).

Usage::

    python -m kyolo.models.yolov9.convert_yolov9_torch_to_keras \
        --weights yolov9c.pt --output yolov9c.weights.h5 --variant 'c'
"""

from __future__ import annotations

from kyolo.conversion import convert_weights

from .yolov9_model import build_yolov9


def convert(
    weights_path,
    variant="c",
    nc=80,
    imgsz=640,
    output=None,
    method="name",
    verbose=True,
    allow_partial=False,
):
    """Build a YOLOv9 model and transfer weights from ``weights_path``."""
    model = build_yolov9(variant=variant, nc=nc, input_shape=(imgsz, imgsz, 3), deploy=False)
    return convert_weights(
        model,
        weights_path,
        output_path=output,
        method=method,
        verbose=verbose,
        allow_partial=allow_partial,
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description="Convert YOLOv9 .pt -> kyolo .weights.h5")
    p.add_argument("--weights", required=True, help="Path to the source .pt checkpoint")
    p.add_argument("--variant", default="c", help="Model variant")
    p.add_argument("--nc", type=int, default=80, help="Number of classes")
    p.add_argument("--imgsz", type=int, default=640, help="Build image size")
    p.add_argument("--output", default=None, help="Output .weights.h5 path")
    p.add_argument("--method", default="name", choices=["order", "name"])
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Save even if the transfer was incomplete (the checkpoint will be "
        "part randomly initialized). Off by default.",
    )
    args = p.parse_args()
    convert(
        args.weights,
        variant=args.variant,
        nc=args.nc,
        imgsz=args.imgsz,
        output=args.output,
        method=args.method,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
