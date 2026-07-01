"""Anchor-point generation and distance<->box conversion (pure ``keras.ops``).

The anchor-free YOLO heads predict four distances (left, top, right, bottom)
from the centre of each grid cell. These helpers build the grid of anchor
points and convert between the distance and box parameterizations. All
coordinates here are in *feature-grid units*; multiply by the stride to get
pixel coordinates.
"""

from __future__ import annotations

from keras import ops

__all__ = ["make_anchors", "dist2bbox", "bbox2dist", "decode_raw_predictions"]


def make_anchors(feat_shapes, strides, grid_cell_offset=0.5):
    """Build anchor points and the matching per-anchor stride tensor.

    Args:
        feat_shapes: list of ``(H, W)`` (or objects indexable as ``[0]``/``[1]``)
            for each detection level, ordered to match ``strides``.
        strides: iterable of strides, e.g. ``(8, 16, 32)``.
        grid_cell_offset: centre offset within each cell (0.5 = cell centre).

    Returns:
        ``(anchor_points, stride_tensor)`` where ``anchor_points`` is ``(A, 2)``
        of ``(x, y)`` cell-centre coordinates and ``stride_tensor`` is ``(A, 1)``.
    """
    anchor_points = []
    stride_tensor = []
    for (h, w), stride in zip(feat_shapes, strides):
        sx = ops.arange(w, dtype="float32") + grid_cell_offset
        sy = ops.arange(h, dtype="float32") + grid_cell_offset
        gy, gx = ops.meshgrid(sy, sx, indexing="ij")
        pts = ops.reshape(ops.stack([gx, gy], axis=-1), (-1, 2))
        anchor_points.append(pts)
        stride_tensor.append(ops.full((h * w, 1), float(stride), dtype="float32"))
    return ops.concatenate(anchor_points, axis=0), ops.concatenate(stride_tensor, axis=0)


def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
    """Convert (l, t, r, b) distances to boxes.

    Args:
        distance: ``(..., 4)`` distances [left, top, right, bottom].
        anchor_points: broadcastable ``(..., 2)`` anchor centres (x, y).
        xywh: return (cx, cy, w, h) if True else (x1, y1, x2, y2).
        axis: split/concat axis (the size-4 axis).
    """
    lt, rb = ops.split(distance, 2, axis=axis)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        cxcy = (x1y1 + x2y2) / 2.0
        wh = x2y2 - x1y1
        return ops.concatenate([cxcy, wh], axis=axis)
    return ops.concatenate([x1y1, x2y2], axis=axis)


def bbox2dist(anchor_points, bbox, reg_max, axis=-1):
    """Inverse of :func:`dist2bbox` for xyxy boxes, clipped to ``[0, reg_max-1)``.

    Used to build DFL regression targets in the loss.
    """
    x1y1, x2y2 = ops.split(bbox, 2, axis=axis)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = ops.concatenate([lt, rb], axis=axis)
    return ops.clip(dist, 0.0, reg_max - 1 - 0.01)


def decode_raw_predictions(feats, strides, reg_max, nc, dfl_layer, data_format="channels_last"):
    """Decode a list of raw head feature maps into boxes + class scores.

    Args:
        feats: list of ``(B, H, W, 4*reg_max + nc)`` tensors (channels_last) or
            ``(B, C, H, W)`` (channels_first).
        strides: per-level strides.
        reg_max: DFL bin count.
        nc: number of classes.
        dfl_layer: a built :class:`kyolo.layers.dfl.DFL` (or ``None`` to skip the
            integral and treat the first 4 channels as raw distances - used by
            the DFL-free YOLO26 head).
        data_format: layout of ``feats``.

    Returns:
        ``(boxes_xywh, scores)``:
          * ``boxes_xywh``: ``(B, A, 4)`` in **pixel** coordinates.
          * ``scores``: ``(B, A, nc)`` sigmoid class probabilities.
    """
    no = 4 * reg_max + nc if dfl_layer is not None else 4 + nc
    shapes = []
    flat = []
    for f in feats:
        if data_format == "channels_first":
            f = ops.transpose(f, (0, 2, 3, 1))
        b = ops.shape(f)[0]
        h = ops.shape(f)[1]
        w = ops.shape(f)[2]
        shapes.append((h, w))
        flat.append(ops.reshape(f, (b, h * w, no)))
    x = ops.concatenate(flat, axis=1)  # (B, A, no)

    box = x[..., : no - nc]
    cls = x[..., no - nc :]

    anchors, stride_t = make_anchors(shapes, strides)  # (A,2), (A,1)
    anchors = ops.expand_dims(anchors, 0)  # (1,A,2)
    stride_t = ops.expand_dims(stride_t, 0)  # (1,A,1)

    if dfl_layer is not None:
        # DFL expects (B, 4*reg_max, A)
        box_t = ops.transpose(box, (0, 2, 1))
        dist = dfl_layer(box_t)  # (B, 4, A)
        dist = ops.transpose(dist, (0, 2, 1))  # (B, A, 4)
    else:
        dist = box  # already 4 distances

    boxes = dist2bbox(dist, anchors, xywh=True, axis=-1)  # grid units
    boxes = boxes * stride_t  # -> pixels
    scores = ops.sigmoid(cls)
    return boxes, scores
