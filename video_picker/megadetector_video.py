import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime

from .utils import (
    configure_ort_cpu_session_threads_from_cores,
    crop_norm_xyxy_from_bgr,
    load_txtset_labels_last_field,
    run_onnx_with_stacked_batch,
    softmax_2d,
)

DEFAULT_MEGADETECTOR_PATH = "./models/md_v5a_1_3_640_640_static.onnx"
DEFAULT_SPECIESNET_LABELS_PATH = "./static/spicesNet_labels_v401a.txtset"
DEFAULT_SPECIESNET_PATH = "./models/spicesNet_v401a.onnx"

IMAGE_SIZE = 640
SPECIESNET_IMAGE_SIZE = 480

logger = logging.getLogger("megadetector_video")

def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    nchw = np.expand_dims(chw, axis=0)
    return nchw / 255.0


class SpeciesNetRunner:
    def __init__(self, model_path: str, labels_path: str) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.labels = load_txtset_labels_last_field(labels_path)

        avail = set(onnxruntime.get_available_providers())
        providers: List[str] = []
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape

    def _preprocess_bgr(self, bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(bgr, (SPECIESNET_IMAGE_SIZE, SPECIESNET_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # If input is NCHW (common for PyTorch-exported ONNX), transpose.
        if isinstance(self.input_shape, list) and len(self.input_shape) >= 2 and self.input_shape[1] == 3:
            rgb = np.transpose(rgb, (2, 0, 1))

        return np.expand_dims(rgb, axis=0)

    def predict_crop_bgr(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        inp = self._preprocess_bgr(crop_bgr)
        logits = self.session.run(None, {self.input_name: inp})[0]
        probs = softmax_2d(np.asarray(logits))
        idx = int(np.argmax(probs, axis=1)[0])
        label = self.labels[idx] if 0 <= idx < len(self.labels) else str(idx)
        return label, float(probs[0, idx])


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


class MegaDetectorRunner:
    def __init__(self, model_path: str, cpu_cores: int) -> None:
        self.model_path = model_path
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_threads = configure_ort_cpu_session_threads_from_cores(sess_options, int(cpu_cores))
        t0 = time.perf_counter()
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=sess_options
        )
        self.load_seconds = time.perf_counter() - t0
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_frame_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        return preprocess_bgr_to_md_input(frame_bgr)

    def infer_batch(self, batch_tensor: np.ndarray) -> Tuple[List[np.ndarray], str]:
        return run_onnx_with_stacked_batch(self.session, self.input_name, batch_tensor)

    def postprocess(
        self, outputs: List[np.ndarray], confidence: float
    ) -> List[np.ndarray]:
        return megadetector_post_processing(outputs, confidence, IMAGE_SIZE, IMAGE_SIZE)


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
        default=os.environ.get("MEGADETECTOR_MODEL_PATH", DEFAULT_MEGADETECTOR_PATH),
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
    p.add_argument(
        "--species-model",
        default=os.environ.get("SPECIESNET_MODEL_PATH", DEFAULT_SPECIESNET_PATH).strip(),
        help="Path to SpeciesNet ONNX model (or SPECIESNET_MODEL_PATH). If empty, SpeciesNet is disabled.",
    )
    p.add_argument(
        "--species-labels",
        default=os.environ.get("SPECIESNET_LABELS_PATH", DEFAULT_SPECIESNET_LABELS_PATH),
        help="Path to SpeciesNet labels txtset (or SPECIESNET_LABELS_PATH).",
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
    if args.species_model:
        logger.info("SpeciesNet: enabled model=%s labels=%s", os.path.abspath(args.species_model), os.path.abspath(args.species_labels))
    else:
        logger.info("SpeciesNet: disabled (no model path provided)")
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

    md = MegaDetectorRunner(args.model, cpu_cores=int(args.batch_size))
    logger.info("ORT threads: %s", md.ort_threads)
    logger.info("Loaded MD ONNX session in %.3fs", md.load_seconds)

    species_runner: SpeciesNetRunner | None = None
    if args.species_model:
        if not os.path.exists(args.species_model):
            raise SystemExit(f"SpeciesNet model not found: {args.species_model}")
        if not os.path.exists(args.species_labels):
            raise SystemExit(f"SpeciesNet labels not found: {args.species_labels}")
        t_s0 = time.perf_counter()
        species_runner = SpeciesNetRunner(args.species_model, args.species_labels)
        logger.info("Loaded SpeciesNet ONNX session in %.3fs", time.perf_counter() - t_s0)

    # Collect frames and associated timestamps, run in batches.
    frames_batch: List[np.ndarray] = []
    metas_batch: List[Tuple[float, int]] = []
    orig_bgr_batch: List[np.ndarray] = []
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
        outputs, onnx_mode = md.infer_batch(batch_tensor)
        infer_s = time.perf_counter() - t0
        infer_total_s += infer_s
        onnx_mode_counts[onnx_mode] = onnx_mode_counts.get(onnx_mode, 0) + 1

        t1 = time.perf_counter()
        preds = md.postprocess(outputs, args.confidence)
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
                det = {
                    "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "confidence": float(b[4]),
                }
                dets.append(det)
            results.append(
                {
                    "t_seconds": float(t_sec),
                    "frame_index": int(frame_idx),
                    "detections": dets,
                }
            )

        # Optional SpeciesNet pass: classify crops for detections above the MD threshold.
        if species_runner is not None:
            for frame_out, frame_bgr in zip(results[-len(metas_batch) :], orig_bgr_batch):
                dets = frame_out.get("detections", [])
                if not isinstance(dets, list):
                    continue
                for det in dets:
                    if not isinstance(det, dict):
                        continue
                    conf = det.get("confidence", None)
                    if not isinstance(conf, (int, float)):
                        continue
                    if float(conf) < float(args.confidence):
                        continue
                    bbox = det.get("bbox_xyxy")
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    crop = crop_norm_xyxy_from_bgr(frame_bgr, bbox)
                    if crop is None:
                        continue
                    try:
                        class_name, prob = species_runner.predict_crop_bgr(crop)
                    except Exception as e:
                        logger.debug("SpeciesNet failed on crop: %s", e)
                        continue
                    det["speciesnet"] = {"class_name": class_name, "probability": float(prob)}

        frames_batch.clear()
        metas_batch.clear()
        orig_bgr_batch.clear()

    try:
        for t_sec, frame_idx, frame_bgr in iter_frames_one_per_second(cap, fps):
            inp = md.preprocess_frame_bgr(frame_bgr)
            frames_batch.append(inp)
            metas_batch.append((t_sec, frame_idx))
            orig_bgr_batch.append(frame_bgr)
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
        "ort_threads": md.ort_threads,
        "speciesnet": (
            {
                "enabled": True,
                "model_path": os.path.abspath(args.species_model),
                "labels_path": os.path.abspath(args.species_labels),
            }
            if species_runner is not None
            else {"enabled": False}
        ),
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

