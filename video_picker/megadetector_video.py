import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime


DEFAULT_MODEL_PATH = "./models/md_v5a_1_3_640_640_static.onnx"
IMAGE_SIZE = 640

logger = logging.getLogger("megadetector_video")


def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    nchw = np.expand_dims(chw, axis=0)
    return nchw / 255.0


# --- copied from src/megadetector_detector.py (Megadetector post-processing) ---


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
    if len(pred) == 0:
        return npred

    p0 = pred[0]
    px = pred[1:]

    npred.append(p0)
    if len(px) == 0:
        return npred

    ious = _calc_ious(p0, px)
    ious[px[..., 5] != p0[5]] = 0
    pp = px[ious < iou_thresh]

    return _nms(pp, iou_thresh, npred)


def _xywh2xyxy(xywh):
    xyxy = np.zeros_like(xywh)
    xc, yc, half_w, half_h = xywh[:, 0], xywh[:, 1], xywh[:, 2] / 2, xywh[:, 3] / 2
    xyxy[:, 0] = xc - half_w
    xyxy[:, 1] = yc - half_h
    xyxy[:, 2] = xc + half_w
    xyxy[:, 3] = yc + half_h
    return xyxy


def non_max_suppression(pred, conf_thresh=0.25, iou_thresh=0.45):
    pred = pred[pred[..., 4] > conf_thresh]
    pred = pred[np.flip(np.argsort(pred[..., 4], axis=-1), axis=0)]

    pred[..., 5] = np.argmax(pred[..., 5:], axis=-1)
    pred = pred[..., :6]
    pred[..., :4] = _xywh2xyxy(pred[..., :4])

    return _nms(pred, iou_thresh, [])


def megadetector_post_processing(outputs, confidence, input_image_width, input_image_height):
    preds = []
    for p in outputs[0]:
        p = non_max_suppression(p, confidence, 0.45)
        p = [pred for pred in p if pred[5] == 0]
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


def configure_ort_cpu_session_threads_from_cores(
    sess_options: onnxruntime.SessionOptions, cores: int
) -> Dict[str, Any]:
    """
    Explicitly set ORT CPU threading from a user-provided "cores" value.

    This intentionally ignores the env-based heuristics in configure_ort_cpu_session_threads()
    so the UI can directly control CPU utilization.
    """
    cores = max(1, int(cores))
    sess_options.intra_op_num_threads = cores
    sess_options.inter_op_num_threads = 1
    return {"intra_op_num_threads": cores, "inter_op_num_threads": 1, "source": "cli(--batch-size as cores)"}


def run_onnx_with_stacked_batch(
    session: onnxruntime.InferenceSession, input_name: str, batch_tensor: np.ndarray
) -> Tuple[List[np.ndarray], str]:
    """
    Run Megadetector ONNX on (B, 3, 640, 640).

    If the model input has fixed batch=1, run B forwards and concatenate output[0].
    """
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


def iter_frames_one_per_second(cap: cv2.VideoCapture, fps: float):
    """
    Yield (t_seconds, frame_index, frame_bgr) at ~1Hz using frame seeks.
    """
    if fps <= 0:
        fps = 30.0

    frame_step = max(1, int(round(fps)))
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield frame_idx / fps, frame_idx, frame
        frame_idx += frame_step


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MEGADETECTOR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Run Megadetector ONNX on 1 frame/second of a video and write detections to JSON."
    )
    p.add_argument("video_path", help="Path to a video file.")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <video_basename>_megadetector.json)",
    )
    p.add_argument(
        "-m",
        "--model",
        default=os.environ.get("MEGADETECTOR_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Path to Megadetector ONNX model (or MEGADETECTOR_MODEL_PATH).",
    )
    p.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=float(os.environ.get("MEGADETECTOR_CONFIDENCE", "0.5")),
        help="Confidence threshold (default 0.5; or MEGADETECTOR_CONFIDENCE).",
    )
    p.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=int(os.environ.get("MEGADETECTOR_CPU_CORES", str(max(1, (os.cpu_count() or 4) // 2)))),
        help="How many CPU cores/threads to give ONNX Runtime (default: half your CPUs; or MEGADETECTOR_CPU_CORES).",
    )
    args = p.parse_args()

    video_path = args.video_path
    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}")
    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    output_path = args.output
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_megadetector.json"

    logger.info("Video: %s", os.path.abspath(video_path))
    logger.info("Model: %s", os.path.abspath(args.model))
    logger.info("Output: %s", os.path.abspath(output_path))
    frames_per_batch = int(os.environ.get("MEGADETECTOR_FRAMES_PER_BATCH", "8"))
    frames_per_batch = max(1, frames_per_batch)
    logger.info(
        "Params: confidence=%.3f cpu_cores=%d frames_per_batch=%d sample_rate_hz=1.0",
        float(args.confidence),
        int(args.batch_size),
        int(frames_per_batch),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if fps > 0 else None

    logger.info("Video metadata: fps=%.3f total_frames=%d duration_seconds=%s", fps, total_frames, str(duration_s))

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_threads = configure_ort_cpu_session_threads_from_cores(sess_options, int(args.batch_size))
    logger.info("ORT threads: %s", ort_threads)
    t_load0 = time.perf_counter()
    session = onnxruntime.InferenceSession(
        args.model, providers=["CPUExecutionProvider"], sess_options=sess_options
    )
    logger.info("Loaded ONNX session in %.3fs", time.perf_counter() - t_load0)
    input_name = session.get_inputs()[0].name

    # Collect frames and associated timestamps, run in batches.
    frames_batch: List[np.ndarray] = []
    metas_batch: List[Tuple[float, int]] = []
    results: List[Dict[str, Any]] = []

    infer_total_s = 0.0
    post_total_s = 0.0
    onnx_mode_counts: Dict[str, int] = {}
    sampled = 0
    batches = 0

    def flush_batch():
        nonlocal infer_total_s, post_total_s, batches
        if not frames_batch:
            return

        batches += 1
        batch_tensor = np.concatenate(frames_batch, axis=0)

        t0 = time.perf_counter()
        outputs, onnx_mode = run_onnx_with_stacked_batch(session, input_name, batch_tensor)
        infer_s = time.perf_counter() - t0
        infer_total_s += infer_s
        onnx_mode_counts[onnx_mode] = onnx_mode_counts.get(onnx_mode, 0) + 1

        t1 = time.perf_counter()
        preds = megadetector_post_processing(outputs, args.confidence, IMAGE_SIZE, IMAGE_SIZE)
        post_s = time.perf_counter() - t1
        post_total_s += post_s

        logger.info(
            "Batch %d: B=%d onnx=%.3fs post=%.3fs mode=%s total_frames_written=%d",
            batches,
            int(batch_tensor.shape[0]),
            infer_s,
            post_s,
            onnx_mode,
            len(results) + len(metas_batch),
        )

        for (t_sec, frame_idx), boxes in zip(metas_batch, preds):
            boxes_arr = np.asarray(boxes)
            dets = []
            for b in boxes_arr:
                dets.append(
                    {
                        "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "confidence": float(b[4]),
                    }
                )
            results.append(
                {
                    "t_seconds": float(t_sec),
                    "frame_index": int(frame_idx),
                    "detections": dets,
                }
            )

        frames_batch.clear()
        metas_batch.clear()

    try:
        for t_sec, frame_idx, frame_bgr in iter_frames_one_per_second(cap, fps):
            inp = preprocess_bgr_to_md_input(frame_bgr)
            frames_batch.append(inp)
            metas_batch.append((t_sec, frame_idx))
            sampled += 1
            if sampled % 30 == 0:
                logger.info("Sampled %d frames @1Hz so far…", sampled)
            if len(frames_batch) >= frames_per_batch:
                flush_batch()
        flush_batch()
    finally:
        cap.release()

    out: Dict[str, Any] = {
        "video_path": os.path.abspath(video_path),
        "model_path": os.path.abspath(args.model),
        "confidence_threshold": float(args.confidence),
        "sample_rate_hz": 1.0,
        "video_fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_s,
        "ort_threads": ort_threads,
        "onnx_mode_counts": onnx_mode_counts,
        "timing_seconds": {
            "onnx_inference_total": infer_total_s,
            "postprocessing_total": post_total_s,
            "onnx_plus_post_total": infer_total_s + post_total_s,
        },
        "frames": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)

    logger.info("Wrote %d frames to %s", len(results), output_path)
    logger.info(
        "Timing: onnx=%.3fs post=%.3fs total=%.3fs",
        infer_total_s,
        post_total_s,
        infer_total_s + post_total_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

