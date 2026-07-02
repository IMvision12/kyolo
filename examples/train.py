"""Minimal, real training example for :mod:`kyolo`.

This trains a YOLO detector on a tiny *synthetic* dataset (random images with
random boxes/labels) so the whole training loop -- data batch -> loss ->
gradients -> optimizer step -- runs end to end without any external data.

The dataset is a plain Python generator yielding ``(x, y)`` tuples (the form
``keras.Model.fit`` requires from a generator)::

    x = (B, S, S, 3) float32                 # images
    y = {
        "boxes":  (B, M, 4)    float32        # xyxy in pixels
        "labels": (B, M)       int32
        "mask":   (B, M)       float32/bool   # 1 for real (non-padded) boxes
    }

(:class:`kyolo.training.YOLODetector` also accepts a single combined
``{"images", "boxes", "labels", "mask"}`` dict when you drive the training loop
yourself, but ``fit`` with a generator needs the ``(x, y)`` split shown here.)

A backend must be installed to actually run this (``pip install kyolo[tensorflow]``
or the jax / torch extra). ``--help`` works without one.

Example
-------
    python examples/train.py --model yolov8n --nc 80 --epochs 1
"""

from __future__ import annotations

import argparse

import numpy as np

# Fixed image size for the synthetic demo (small keeps it fast on CPU).
IMAGE_SIZE = 256
BATCH_SIZE = 2
MAX_BOXES = 8  # M: every sample is padded to this many boxes.


def parse_args():
    """Parse command-line arguments for the training demo."""
    parser = argparse.ArgumentParser(description="kyolo minimal training demo")
    parser.add_argument("--model", default="yolov8n", help="Variant factory name in kyolo.models.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--nc", type=int, default=80, help="Number of classes.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional converted checkpoint to fine-tune from (AGPL-3.0).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone layers so only the neck + head train.",
    )
    return parser.parse_args()


def _make_batch(rng, num_classes):
    """Build one synthetic dict batch of random images / boxes / labels."""
    images = rng.random((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")
    boxes = np.zeros((BATCH_SIZE, MAX_BOXES, 4), dtype="float32")
    labels = np.zeros((BATCH_SIZE, MAX_BOXES), dtype="int32")
    mask = np.zeros((BATCH_SIZE, MAX_BOXES), dtype="bool")

    for b in range(BATCH_SIZE):
        n = int(rng.integers(1, MAX_BOXES + 1))  # 1..MAX_BOXES real boxes
        for i in range(n):
            cx, cy = rng.uniform(0.2, 0.8, size=2) * IMAGE_SIZE
            w = rng.uniform(0.05, 0.35) * IMAGE_SIZE
            h = rng.uniform(0.05, 0.35) * IMAGE_SIZE
            x1 = float(np.clip(cx - w / 2, 0, IMAGE_SIZE))
            y1 = float(np.clip(cy - h / 2, 0, IMAGE_SIZE))
            x2 = float(np.clip(cx + w / 2, 0, IMAGE_SIZE))
            y2 = float(np.clip(cy + h / 2, 0, IMAGE_SIZE))
            boxes[b, i] = (x1, y1, x2, y2)
            labels[b, i] = int(rng.integers(0, num_classes))
            mask[b, i] = True

    targets = {"boxes": boxes, "labels": labels, "mask": mask.astype("float32")}
    return images, targets


def synthetic_dataset(num_classes, seed=0):
    """Infinite generator of synthetic ``(images, targets)`` batches."""
    rng = np.random.default_rng(seed)
    while True:
        yield _make_batch(rng, num_classes)


def build_detector(model_name, nc, weights=None, freeze_backbone=False):
    """Build a :class:`YOLODetector`, optionally fine-tuning from a checkpoint.

    Fine-tuning notes:
      * Load converted weights onto the *bare* model BEFORE wrapping it in the
        detector. The official weights are AGPL-3.0 -- convert them yourself with
        the per-model converter (see the README); they are not shipped here.
      * Freezing the backbone (``trainable = False``) trains only the neck and
        detection head, which is the usual recipe for small datasets.
    """
    import keras  # noqa: F401  (ensures a backend is importable)

    import kyolo.models as models
    from kyolo.losses import YOLODetectionLoss
    from kyolo.training import YOLODetector

    # deploy=False keeps reparameterizable branches un-fused for training.
    factory = getattr(models, model_name.replace("-", "_"))
    model = factory(nc=nc, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), deploy=False)

    # --- Fine-tuning: load weights before wrapping ------------------------
    if weights is not None:
        # e.g. model.load_weights("yolov8n.weights.h5")
        model.load_weights(weights)

    # --- Optionally freeze the backbone -----------------------------------
    # The exact prefix depends on how the architecture names its sub-layers;
    # Ultralytics-style models name backbone stages "backbone..." / "model.0"..
    if freeze_backbone:
        backbone_prefixes = ("backbone", "stem", "model.0", "model.1", "model.2")
        frozen = 0
        for layer in model.layers:
            if layer.name.startswith(backbone_prefixes):
                layer.trainable = False
                frozen += 1
        print(f"Froze {frozen} backbone layer(s).")

    loss = YOLODetectionLoss(nc=nc, reg_max=16, strides=(8, 16, 32))
    return YOLODetector(model, loss=loss, nc=nc, reg_max=16)


def main():
    """Build a detector and run a couple of training steps on synthetic data."""
    args = parse_args()

    import keras

    detector = build_detector(
        args.model,
        nc=args.nc,
        weights=args.weights,
        freeze_backbone=args.freeze_backbone,
    )
    detector.compile(optimizer=keras.optimizers.Adam(1e-3))

    dataset = synthetic_dataset(num_classes=args.nc)
    detector.fit(dataset, epochs=args.epochs, steps_per_epoch=2)

    print("Training demo finished.")


if __name__ == "__main__":
    main()
