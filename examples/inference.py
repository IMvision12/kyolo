"""End-to-end object-detection inference demo for :mod:`kyolo`.

This script wires together the full inference pipeline:

    build model -> preprocess -> forward pass -> postprocess -> rescale -> draw

Because the official YOLO checkpoints are AGPL-3.0 licensed they are **not**
shipped with this project. Running this demo without a converted checkpoint uses
randomly initialized weights, so the detections will be meaningless -- it still
exercises the whole pipeline and produces a rendered image. To get real
predictions, convert an official checkpoint yourself first with the per-model
converter (``python -m kyolo.models.<family>.convert_<family>_torch_to_keras``)
and pass the resulting ``.weights.h5`` with ``--weights``.

Example
-------
    python examples/inference.py --model yolov8l \\
        --weights yolov8l.weights.h5 --image assets/samples/zidane.jpg

A backend must be installed (``pip install kyolo[tensorflow]`` or jax / torch).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_IMAGE = os.path.join(_REPO_ROOT, "assets", "samples", "bus.jpg")
_DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "assets", "inference_result.png")


def parse_args():
    """Parse command-line arguments for the inference demo."""
    parser = argparse.ArgumentParser(description="kyolo inference demo")
    parser.add_argument(
        "--model",
        default="yolov8n",
        help="Variant factory name in kyolo.models (e.g. yolov8n, yolo11s).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a converted .weights.h5 checkpoint. Omit to use random "
        "weights (pipeline demo only).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=f"Path to an input image. Defaults to {_DEFAULT_IMAGE} if present, "
        "otherwise a random image is generated.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    return parser.parse_args()


def _load_image_array(path):
    """Load an RGB image as a ``(H, W, 3)`` uint8 numpy array (lazy PIL import)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to read image files. It ships with kyolo, so "
            "reinstall it with `pip install pillow` (or `pip install kyolo`)."
        ) from exc
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype="uint8")


def main():
    """Run the full inference pipeline and save a visualization."""
    args = parse_args()

    import keras

    import kyolo.models as models
    from kyolo.models import load_pretrained_weights
    from kyolo.ops import scale_boxes
    from kyolo.postprocessing import YOLOPostprocessor
    from kyolo.preprocessing import YOLOPreprocessor
    from kyolo.utils import COCO_CLASS_NAMES, visualize_detections

    nc = len(COCO_CLASS_NAMES)

    print(f"Building '{args.model}' (nc={nc}, imgsz={args.imgsz}) ...")
    factory = getattr(models, args.model.replace("-", "_"))
    model = factory(
        nc=nc,
        input_shape=(args.imgsz, args.imgsz, 3),
        deploy=True,
    )

    if args.weights is not None:
        print(f"Loading weights from '{args.weights}' ...")
        load_pretrained_weights(model, args.weights)
    else:
        print(
            "\n[!] No --weights given: using RANDOM weights, so detections are "
            "meaningless.\n"
            "    The official YOLO weights are AGPL-3.0 and are NOT shipped with "
            "kyolo.\n"
            "    Convert an official checkpoint yourself first with the per-model "
            "converter\n"
            "    (see the README's 'Weight conversion' section), then re-run with "
            "--weights <file>.weights.h5\n"
        )

    image_path = args.image or _DEFAULT_IMAGE
    preprocessor = YOLOPreprocessor(image_size=args.imgsz, normalize=True, letterbox=True)

    if os.path.exists(image_path):
        print(f"Preprocessing image '{image_path}' ...")
        image = _load_image_array(image_path)
    else:
        print(f"Image '{image_path}' not found; using a random image instead.")
        image = np.random.randint(0, 256, size=(480, 640, 3), dtype="uint8")
    batch = preprocessor(image)

    print("Running forward pass ...")
    raw_feats = model(batch["images"])

    postprocessor = YOLOPostprocessor(
        nc=nc,
        reg_max=model.reg_max,
        strides=model.strides,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_detections=300,
    )
    detections = postprocessor(raw_feats)

    detections = scale_boxes(detections, batch["ratio"], batch["pad"], orig_shape=image.shape[:2])

    det0 = keras.ops.convert_to_numpy(detections)[0]
    num_kept = int((det0[:, 4] > 0).sum())
    print(f"Kept {num_kept} detection(s) above conf={args.conf}.")

    os.makedirs(os.path.dirname(_DEFAULT_OUTPUT), exist_ok=True)
    visualize_detections(
        image,
        det0,
        class_names=COCO_CLASS_NAMES,
        score_threshold=args.conf,
        save_path=_DEFAULT_OUTPUT,
        title=f"{args.model} detections",
    )
    print(f"Saved visualization to '{_DEFAULT_OUTPUT}'.")


if __name__ == "__main__":
    main()
