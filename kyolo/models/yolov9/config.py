"""Per-variant layer specs for YOLOv9 (GELAN).

Each variant is a full topology, not a width-scaled copy of another: the
Ultralytics YOLOv9 ``.yaml`` files list absolute channel counts and mix
different blocks (``ELAN1`` vs ``RepNCSPELAN4``, ``AConv`` vs ``ADown``, and the
``e`` variant's dual-branch programmable-gradient path with ``CBLinear`` /
``CBFuse``). We therefore mirror each ``.yaml`` directly as a list of layer
specs, interpreted by :func:`kyolo.models.yolov9.yolov9_model.build_yolov9`.

A spec entry is ``(from, op, args)``:

* ``from``  - ``-1`` for the previous layer, an ``int`` index into prior
  outputs, or a ``list`` of those (gathered, with ``-1`` meaning "previous").
* ``op``    - block name (see the interpreter's dispatch table).
* ``args``  - positional block args, matching the Ultralytics module signature.

The trailing ``Detect`` entry carries the indices of the three feature maps fed
to the detection head; its layer index becomes the head name (``model.<idx>``).
"""

from __future__ import annotations

# Backbone + neck + Detect, mirroring ultralytics/cfg/models/v9/yolov9<v>.yaml.
YOLOV9_SPECS = {
    "t": [
        (-1, "Conv", [16, 3, 2]),  # 0
        (-1, "Conv", [32, 3, 2]),  # 1
        (-1, "ELAN1", [32, 32, 16]),  # 2
        (-1, "AConv", [64]),  # 3
        (-1, "RepNCSPELAN4", [64, 64, 32, 3]),  # 4
        (-1, "AConv", [96]),  # 5
        (-1, "RepNCSPELAN4", [96, 96, 48, 3]),  # 6
        (-1, "AConv", [128]),  # 7
        (-1, "RepNCSPELAN4", [128, 128, 64, 3]),  # 8
        (-1, "SPPELAN", [128, 64]),  # 9
        (-1, "Upsample", []),  # 10
        ([-1, 6], "Concat", []),  # 11
        (-1, "RepNCSPELAN4", [96, 96, 48, 3]),  # 12
        (-1, "Upsample", []),  # 13
        ([-1, 4], "Concat", []),  # 14
        (-1, "RepNCSPELAN4", [64, 64, 32, 3]),  # 15
        (-1, "AConv", [48]),  # 16
        ([-1, 12], "Concat", []),  # 17
        (-1, "RepNCSPELAN4", [96, 96, 48, 3]),  # 18
        (-1, "AConv", [64]),  # 19
        ([-1, 9], "Concat", []),  # 20
        (-1, "RepNCSPELAN4", [128, 128, 64, 3]),  # 21
        ([15, 18, 21], "Detect", []),  # 22
    ],
    "s": [
        (-1, "Conv", [32, 3, 2]),  # 0
        (-1, "Conv", [64, 3, 2]),  # 1
        (-1, "ELAN1", [64, 64, 32]),  # 2
        (-1, "AConv", [128]),  # 3
        (-1, "RepNCSPELAN4", [128, 128, 64, 3]),  # 4
        (-1, "AConv", [192]),  # 5
        (-1, "RepNCSPELAN4", [192, 192, 96, 3]),  # 6
        (-1, "AConv", [256]),  # 7
        (-1, "RepNCSPELAN4", [256, 256, 128, 3]),  # 8
        (-1, "SPPELAN", [256, 128]),  # 9
        (-1, "Upsample", []),  # 10
        ([-1, 6], "Concat", []),  # 11
        (-1, "RepNCSPELAN4", [192, 192, 96, 3]),  # 12
        (-1, "Upsample", []),  # 13
        ([-1, 4], "Concat", []),  # 14
        (-1, "RepNCSPELAN4", [128, 128, 64, 3]),  # 15
        (-1, "AConv", [96]),  # 16
        ([-1, 12], "Concat", []),  # 17
        (-1, "RepNCSPELAN4", [192, 192, 96, 3]),  # 18
        (-1, "AConv", [128]),  # 19
        ([-1, 9], "Concat", []),  # 20
        (-1, "RepNCSPELAN4", [256, 256, 128, 3]),  # 21
        ([15, 18, 21], "Detect", []),  # 22
    ],
    "m": [
        (-1, "Conv", [32, 3, 2]),  # 0
        (-1, "Conv", [64, 3, 2]),  # 1
        (-1, "RepNCSPELAN4", [128, 128, 64, 1]),  # 2
        (-1, "AConv", [240]),  # 3
        (-1, "RepNCSPELAN4", [240, 240, 120, 1]),  # 4
        (-1, "AConv", [360]),  # 5
        (-1, "RepNCSPELAN4", [360, 360, 180, 1]),  # 6
        (-1, "AConv", [480]),  # 7
        (-1, "RepNCSPELAN4", [480, 480, 240, 1]),  # 8
        (-1, "SPPELAN", [480, 240]),  # 9
        (-1, "Upsample", []),  # 10
        ([-1, 6], "Concat", []),  # 11
        (-1, "RepNCSPELAN4", [360, 360, 180, 1]),  # 12
        (-1, "Upsample", []),  # 13
        ([-1, 4], "Concat", []),  # 14
        (-1, "RepNCSPELAN4", [240, 240, 120, 1]),  # 15
        (-1, "AConv", [180]),  # 16
        ([-1, 12], "Concat", []),  # 17
        (-1, "RepNCSPELAN4", [360, 360, 180, 1]),  # 18
        (-1, "AConv", [240]),  # 19
        ([-1, 9], "Concat", []),  # 20
        (-1, "RepNCSPELAN4", [480, 480, 240, 1]),  # 21
        ([15, 18, 21], "Detect", []),  # 22
    ],
    "c": [
        (-1, "Conv", [64, 3, 2]),  # 0
        (-1, "Conv", [128, 3, 2]),  # 1
        (-1, "RepNCSPELAN4", [256, 128, 64, 1]),  # 2
        (-1, "ADown", [256]),  # 3
        (-1, "RepNCSPELAN4", [512, 256, 128, 1]),  # 4
        (-1, "ADown", [512]),  # 5
        (-1, "RepNCSPELAN4", [512, 512, 256, 1]),  # 6
        (-1, "ADown", [512]),  # 7
        (-1, "RepNCSPELAN4", [512, 512, 256, 1]),  # 8
        (-1, "SPPELAN", [512, 256]),  # 9
        (-1, "Upsample", []),  # 10
        ([-1, 6], "Concat", []),  # 11
        (-1, "RepNCSPELAN4", [512, 512, 256, 1]),  # 12
        (-1, "Upsample", []),  # 13
        ([-1, 4], "Concat", []),  # 14
        (-1, "RepNCSPELAN4", [256, 256, 128, 1]),  # 15
        (-1, "ADown", [256]),  # 16
        ([-1, 12], "Concat", []),  # 17
        (-1, "RepNCSPELAN4", [512, 512, 256, 1]),  # 18
        (-1, "ADown", [512]),  # 19
        ([-1, 9], "Concat", []),  # 20
        (-1, "RepNCSPELAN4", [512, 512, 256, 1]),  # 21
        ([15, 18, 21], "Detect", []),  # 22
    ],
    "e": [
        (-1, "Identity", []),  # 0
        (-1, "Conv", [64, 3, 2]),  # 1
        (-1, "Conv", [128, 3, 2]),  # 2
        (-1, "RepNCSPELAN4", [256, 128, 64, 2]),  # 3
        (-1, "ADown", [256]),  # 4
        (-1, "RepNCSPELAN4", [512, 256, 128, 2]),  # 5
        (-1, "ADown", [512]),  # 6
        (-1, "RepNCSPELAN4", [1024, 512, 256, 2]),  # 7
        (-1, "ADown", [1024]),  # 8
        (-1, "RepNCSPELAN4", [1024, 512, 256, 2]),  # 9
        (1, "CBLinear", [[64]]),  # 10
        (3, "CBLinear", [[64, 128]]),  # 11
        (5, "CBLinear", [[64, 128, 256]]),  # 12
        (7, "CBLinear", [[64, 128, 256, 512]]),  # 13
        (9, "CBLinear", [[64, 128, 256, 512, 1024]]),  # 14
        (0, "Conv", [64, 3, 2]),  # 15
        ([10, 11, 12, 13, 14, -1], "CBFuse", [[0, 0, 0, 0, 0]]),  # 16
        (-1, "Conv", [128, 3, 2]),  # 17
        ([11, 12, 13, 14, -1], "CBFuse", [[1, 1, 1, 1]]),  # 18
        (-1, "RepNCSPELAN4", [256, 128, 64, 2]),  # 19
        (-1, "ADown", [256]),  # 20
        ([12, 13, 14, -1], "CBFuse", [[2, 2, 2]]),  # 21
        (-1, "RepNCSPELAN4", [512, 256, 128, 2]),  # 22
        (-1, "ADown", [512]),  # 23
        ([13, 14, -1], "CBFuse", [[3, 3]]),  # 24
        (-1, "RepNCSPELAN4", [1024, 512, 256, 2]),  # 25
        (-1, "ADown", [1024]),  # 26
        ([14, -1], "CBFuse", [[4]]),  # 27
        (-1, "RepNCSPELAN4", [1024, 512, 256, 2]),  # 28
        (-1, "SPPELAN", [512, 256]),  # 29
        (-1, "Upsample", []),  # 30
        ([-1, 25], "Concat", []),  # 31
        (-1, "RepNCSPELAN4", [512, 512, 256, 2]),  # 32
        (-1, "Upsample", []),  # 33
        ([-1, 22], "Concat", []),  # 34
        (-1, "RepNCSPELAN4", [256, 256, 128, 2]),  # 35
        (-1, "ADown", [256]),  # 36
        ([-1, 32], "Concat", []),  # 37
        (-1, "RepNCSPELAN4", [512, 512, 256, 2]),  # 38
        (-1, "ADown", [512]),  # 39
        ([-1, 29], "Concat", []),  # 40
        (-1, "RepNCSPELAN4", [512, 1024, 512, 2]),  # 41
        ([35, 38, 41], "Detect", []),  # 42
    ],
}


__all__ = ["YOLOV9_SPECS"]
