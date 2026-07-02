"""Command-line entry point for PyTorch -> Keras weight conversion.

Exposed as the ``kyolo-convert`` console script (see ``pyproject.toml``). Builds
the requested kyolo model in deploy (inference) form, then transfers weights
from a user-supplied ``.pt`` file. Each model also ships an equivalent
co-located converter at ``kyolo/models/<name>/convert_<name>_torch_to_keras.py``.

Example
-------
    kyolo-convert --model yolov8n --weights yolov8n.pt \
        --output yolov8n.weights.h5 --method order
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .convert import convert_weights
from .mappings import get_mapping


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kyolo-convert",
        description=(
            "Convert an official YOLO PyTorch (.pt) checkpoint into kyolo Keras "
            "weights (.weights.h5). The weights themselves are AGPL-3.0 and must "
            "be obtained by you; this tool only converts a file you already have."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g. yolov8n, yolo11s, yolov9c, yolov5m.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the source PyTorch .pt checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the Keras weights (default: <weights-stem>.weights.h5).",
    )
    parser.add_argument(
        "--method",
        choices=("order", "name"),
        default="name",
        help="Transfer strategy. 'name' (default) matches kyolo layer names to "
        "the Ultralytics modules and is exact for the official checkpoints; "
        "'order' is a positional fallback.",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=80,
        help="Number of classes the model was trained on (default: 80 = COCO).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input image size used to build the model (default: 640).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, build the model and run the conversion."""
    args = _build_parser().parse_args(argv)

    # Imported here (not at module load) so that `kyolo.conversion` stays
    # importable even while `kyolo.models` is under development.
    import kyolo.models as models

    key = args.model.replace("-", "_")
    factory = getattr(models, key, None)
    if factory is None or key not in models.MODEL_NAMES:
        raise SystemExit(
            f"Unknown model {args.model!r}. Available: {', '.join(models.MODEL_NAMES)}"
        )

    verbose = not args.quiet
    if verbose:
        print(f"Building {args.model} (nc={args.nc}, imgsz={args.imgsz}, deploy=True) ...")
    model = factory(
        nc=args.nc,
        input_shape=(args.imgsz, args.imgsz, 3),
        deploy=True,
    )

    name_mapping = get_mapping(args.model) if args.method == "name" else None
    report = convert_weights(
        model,
        args.weights,
        output_path=args.output,
        method=args.method,
        name_mapping=name_mapping,
        verbose=verbose,
    )

    transferred = report.get("transferred", 0)
    total = report.get("total", 0)
    if verbose:
        print(f"Done: {transferred}/{total} variables transferred.")
    # Non-zero exit if nothing (or not everything) transferred, to aid scripting.
    return 0 if transferred == total and total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
