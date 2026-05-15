"""
MegaDetector v5 ONNX output post-processing (NMS + animal-only + normalized xyxy).

Logic matches the reference Megadetector detector pipeline; NMS is implemented
iteratively (not recursively) so very low confidence thresholds do not hit the
Python recursion limit.
"""

from typing import Any, List, Sequence

import numpy as np

# Megadetector class index for animals in the MD v5 label set.
ANIMAL_CLASS_ID = 0


def _calc_ious(b0, bx):
    i_area = np.maximum(
        np.minimum(b0[2:4], bx[..., 2:4]) - np.maximum(b0[:2], bx[..., :2]), 0
    ).prod(axis=1)

    u_area = (
        (b0[2:4] - b0[:2]).prod(axis=0)
        + (bx[..., 2:4] - bx[..., :2]).prod(axis=-1)
        - i_area
    )

    return i_area / u_area


def _nms(pred, iou_thresh: float, npred: list) -> list:
    if len(pred) == 0:
        return npred

    cur = pred
    while len(cur) > 0:
        p0 = cur[0]
        npred.append(p0)

        px = cur[1:]
        if len(px) == 0:
            break

        ious = _calc_ious(p0, px)
        ious[px[..., 5] != p0[5]] = 0
        cur = px[ious < iou_thresh]

    return npred


def _xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xc, yc, half_w, half_h = xywh[:, 0], xywh[:, 1], xywh[:, 2] / 2, xywh[:, 3] / 2
    xyxy[:, 0] = xc - half_w
    xyxy[:, 1] = yc - half_h
    xyxy[:, 2] = xc + half_w
    xyxy[:, 3] = yc + half_h
    return xyxy


def non_max_suppression(pred, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
    pred = pred[pred[..., 4] > conf_thresh]
    pred = pred[np.flip(np.argsort(pred[..., 4], axis=-1), axis=0)]

    pred[..., 5] = np.argmax(pred[..., 5:], axis=-1)
    pred = pred[..., :6]
    pred[..., :4] = _xywh2xyxy(pred[..., :4])

    return _nms(pred, iou_thresh, [])


def megadetector_post_processing(
    outputs: Sequence[Any],
    confidence: float,
    input_image_width: int,
    input_image_height: int,
    *,
    animal_class_id: int = ANIMAL_CLASS_ID,
    iou_thresh: float = 0.45,
) -> List[np.ndarray]:
    preds: List[np.ndarray] = []
    for p in outputs[0]:
        p = non_max_suppression(p, confidence, iou_thresh)
        p = [pred for pred in p if pred[5] == animal_class_id]
        if len(p) > 0:
            p = np.array(p)
            p[..., :4] = p[..., :4] / [
                input_image_width,
                input_image_height,
                input_image_width,
                input_image_height,
            ]
        preds.append(p)
    return preds
