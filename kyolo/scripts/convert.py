#!/usr/bin/env python
"""Runnable wrapper around ``kyolo.conversion.cli.main``.

Convert an official YOLO PyTorch ``.pt`` checkpoint (which you must obtain
yourself -- the weights are AGPL-3.0 and are not redistributed here) into kyolo
Keras ``.weights.h5`` format.

Usage:
    python scripts/convert.py --model yolov8n --weights yolov8n.pt \
        --output yolov8n.weights.h5 --method order

Run ``python scripts/convert.py --help`` for all options, and see
``scripts/README.md`` for the list of supported model names.
"""

from __future__ import annotations

from kyolo.conversion.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
