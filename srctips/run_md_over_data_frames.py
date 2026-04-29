#!/usr/bin/env python3
"""
Batch MegaDetector (ONNX) over still images: walk a tree, run inference, write JSON.

Each image gets one scalar: max confidence among detections with class 0 (animal).
Default output is ./confidences.json as { basename: confidence }; duplicate basenames
across folders would collide—use --with-paths for full paths.
"""
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime


logger = logging.getLogger("run_md_over_data_frames")

# MegaDetector v5 ONNX in this repo expects 640×640 RGB, NCHW float in [0, 1].
IMAGE_SIZE = 640


def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
    # Light blur before resize (same idea as video_picker/megadetector_video if aligned there).
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)

    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    nchw = np.expand_dims(chw, axis=0)
    return nchw / 255.0


# Post-processing matches video_picker/megadetector_video.py (NMS + animal-only), except
# _nms is iterative so very low -c does not blow the Python recursion limit.


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


def _nms(pred, iou_thresh, npred):
    # Iterative NMS (same logic as the recursive version, but avoids recursion limits
    # when conf_thresh is set very low).
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


def non_max_suppression(pred, conf_thresh=0.0, iou_thresh=0.45):
    pred = pred[pred[..., 4] > conf_thresh]
    pred = pred[np.flip(np.argsort(pred[..., 4], axis=-1), axis=0)]

    pred[..., 5] = np.argmax(pred[..., 5:], axis=-1)
    pred = pred[..., :6]
    pred[..., :4] = _xywh2xyxy(pred[..., :4])

    return _nms(pred, iou_thresh, [])


def megadetector_post_processing(
    outputs, confidence: float, input_image_width: int, input_image_height: int
):
    preds = []
    for p in outputs[0]:
        p = non_max_suppression(p, confidence, 0.45)
        p = [pred for pred in p if pred[5] == 0]  # class 0 = animal
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


def configure_ort_cpu_session_threads(sess_options: onnxruntime.SessionOptions) -> Dict[str, Any]:
    if os.environ.get("MEGADETECTOR_ORT_USE_DEFAULT_THREADS", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return {"note": "MEGADETECTOR_ORT_USE_DEFAULT_THREADS=1 (ORT defaults)"}

    cpu = os.cpu_count() or 4
    intra_default = max(1, min(cpu, 16))
    inter_default = 1

    intra_raw = os.environ.get("MEGADETECTOR_ORT_INTRA_OP_NUM_THREADS", "").strip()
    inter_raw = os.environ.get("MEGADETECTOR_ORT_INTER_OP_NUM_THREADS", "").strip()

    intra = int(intra_raw) if intra_raw else intra_default
    inter = int(inter_raw) if inter_raw else inter_default
    intra = max(1, intra)
    inter = max(1, inter)

    sess_options.intra_op_num_threads = intra
    sess_options.inter_op_num_threads = inter

    return {
        "intra_op_num_threads": intra,
        "inter_op_num_threads": inter,
        "source": "env" if (intra_raw or inter_raw) else f"heuristic (os.cpu_count()={cpu})",
    }


def run_onnx_with_stacked_batch(
    session: onnxruntime.InferenceSession, input_name: str, batch_tensor: np.ndarray
) -> Tuple[List[np.ndarray], str]:
    b = int(batch_tensor.shape[0])
    shape = session.get_inputs()[0].shape
    fixed_batch_1 = shape is not None and len(shape) > 0 and shape[0] == 1

    if b == 1:
        return session.run(None, {input_name: batch_tensor}), "single_forward"

    if not fixed_batch_1:
        return session.run(None, {input_name: batch_tensor}), "single_forward_batched"

    outs0 = []
    for i in range(b):
        one = session.run(None, {input_name: batch_tensor[i : i + 1]})
        outs0.append(one[0])
    return [np.concatenate(outs0, axis=0)], f"B={b}_sequential_fixed_batch1_model"


@dataclass(frozen=True)
class FrameResult:
    path: str
    animal_confidence: Optional[float]


def iter_image_paths(root_dir: str) -> List[str]:
    """All supported images under root_dir, sorted for stable output order."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def confidence_from_boxes(boxes: np.ndarray) -> float:
    # Column 4 is objectness/confidence after NMS; take the strongest animal box per image.
    if boxes is None or len(boxes) == 0:
        return 0.0
    boxes_arr = np.asarray(boxes)
    return float(np.max(boxes_arr[:, 4]))


def run_on_images(
    image_paths: Sequence[str],
    model_path: str,
    batch_size: int,
    confidence_threshold: float,
) -> List[FrameResult]:
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}")

    # CPU-only session; thread counts follow MEGADETECTOR_ORT_* env (see configure_ort_cpu_session_threads).
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_threads = configure_ort_cpu_session_threads(sess_options)
    logger.info("ORT threads: %s", ort_threads)

    session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"], sess_options=sess_options
    )
    input_name = session.get_inputs()[0].name

    results: List[FrameResult] = []
    frames_batch: List[np.ndarray] = []
    paths_batch: List[str] = []

    def flush_batch() -> None:
        """Run one ONNX forward for stacked (B,3,640,640) and append FrameResults."""
        if not frames_batch:
            return
        batch_tensor = np.concatenate(frames_batch, axis=0)
        outputs, _mode = run_onnx_with_stacked_batch(session, input_name, batch_tensor)
        preds = megadetector_post_processing(outputs, confidence_threshold, IMAGE_SIZE, IMAGE_SIZE)

        for path, boxes in zip(paths_batch, preds):
            conf = confidence_from_boxes(boxes)
            results.append(FrameResult(path=path, animal_confidence=conf))

        frames_batch.clear()
        paths_batch.clear()

    # Read → preprocess → batch; models with fixed batch=1 are handled inside run_onnx_with_stacked_batch.
    for idx, path in enumerate(image_paths):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("Failed to read image, storing null confidence: %s", path)
            results.append(FrameResult(path=path, animal_confidence=None))
            continue

        inp = preprocess_bgr_to_md_input(bgr)
        frames_batch.append(inp)
        paths_batch.append(path)

        if len(frames_batch) >= max(1, int(batch_size)):
            flush_batch()

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d/%d images…", idx + 1, len(image_paths))

    flush_batch()
    return results


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MEGADETECTOR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Run MegaDetector ONNX over all images under data/ and emit filename->animal_confidence."
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Root directory containing frames (default: data).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("MEGADETECTOR_MODEL_PATH", "models/md_v5a_1_3_640_640_static.onnx"),
        help="Path to MegaDetector ONNX model (default: models/md_v5a_1_3_640_640_static.onnx).",
    )
    p.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=int(os.environ.get("MEGADETECTOR_BATCH_SIZE", "16")),
        help="Images per ONNX call (default 16; or MEGADETECTOR_BATCH_SIZE).",
    )
    p.add_argument(
        "-c",
        "--confidence-threshold",
        type=float,
        default=float(os.environ.get("MEGADETECTOR_CONFIDENCE", "0.5")),
        help="Box filtering threshold before NMS (default 0.5; or MEGADETECTOR_CONFIDENCE).",
    )
    p.add_argument(
        "--with-paths",
        action="store_true",
        help="Output JSON objects with {path, animal_confidence} instead of filename->animal_confidence.",
    )
    p.add_argument(
        "-o",
        "--output",
        default="confidences.json",
        help="Output JSON file path (default: ./confidences.json). Use '-' to print to stdout.",
    )
    args = p.parse_args()

    image_paths = iter_image_paths(args.data_dir)
    if not image_paths:
        raise SystemExit(f"No images found under: {args.data_dir}")

    logger.info("Found %d images under %s", len(image_paths), os.path.abspath(args.data_dir))
    logger.info("Model: %s", os.path.abspath(args.model))

    frame_results = run_on_images(
        image_paths=image_paths,
        model_path=args.model,
        batch_size=int(args.batch_size),
        confidence_threshold=float(args.confidence_threshold),
    )

    if args.with_paths:
        payload: Any = [
            {"path": fr.path, "animal_confidence": fr.animal_confidence} for fr in frame_results
        ]
    else:
        # Keys are file names only; avoid duplicate names in different subfolders or use --with-paths.
        payload = {os.path.basename(fr.path): fr.animal_confidence for fr in frame_results}

    s = json.dumps(payload, indent=2)
    if args.output == "-":
        print(s)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(s)
            f.write("\n")
        logger.info("Wrote %s", os.path.abspath(args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

