import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

from .utils import crop_norm_xyxy_from_bgr


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


def process_video(
    *,
    video_path: str,
    output_path: str | None,
    md_runner: Any,
    species_runner: Any | None,
    confidence_threshold: float,
    frames_per_batch: int,
    sample_rate_hz: float = 1.0,
    on_progress: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """
    Run MegaDetector at ~1Hz over a video, optionally classify MD crops with SpeciesNet,
    and return a JSON-serializable dict. If output_path is provided, also writes JSON.
    """
    if sample_rate_hz != 1.0:
        raise ValueError("Only sample_rate_hz=1.0 is currently supported")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if fps > 0 else None

    frames_per_batch = max(1, int(frames_per_batch))

    frames_batch: List[np.ndarray] = []
    metas_batch: List[Tuple[float, int]] = []
    orig_bgr_batch: List[np.ndarray] = []
    results: List[Dict[str, Any]] = []

    infer_total_s = 0.0
    post_total_s = 0.0
    onnx_mode_counts: Dict[str, int] = {}
    sampled = 0
    batches = 0

    def emit(evt: Dict[str, Any]) -> None:
        if on_progress is not None:
            on_progress(evt)

    def flush_batch() -> None:
        nonlocal infer_total_s, post_total_s, batches
        if not frames_batch:
            return

        batches += 1
        batch_tensor = np.concatenate(frames_batch, axis=0)

        t0 = time.perf_counter()
        outputs, onnx_mode = md_runner.infer_batch(batch_tensor)
        infer_s = time.perf_counter() - t0
        infer_total_s += infer_s
        onnx_mode_counts[onnx_mode] = onnx_mode_counts.get(onnx_mode, 0) + 1

        t1 = time.perf_counter()
        preds = md_runner.postprocess(outputs, confidence_threshold)
        post_s = time.perf_counter() - t1
        post_total_s += post_s

        emit(
            {
                "type": "batch_done",
                "batches": batches,
                "batch_size": int(batch_tensor.shape[0]),
                "onnx_infer_s": float(infer_s),
                "post_s": float(post_s),
                "onnx_mode": str(onnx_mode),
                "frames_written_total": len(results) + len(metas_batch),
            }
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
                    if float(conf) < float(confidence_threshold):
                        continue
                    bbox = det.get("bbox_xyxy")
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    crop = crop_norm_xyxy_from_bgr(frame_bgr, bbox)
                    if crop is None:
                        continue
                    class_name, prob = species_runner.predict_crop_bgr(crop)
                    det["speciesnet"] = {"class_name": class_name, "probability": float(prob)}

        frames_batch.clear()
        metas_batch.clear()
        orig_bgr_batch.clear()

    try:
        for t_sec, frame_idx, frame_bgr in iter_frames_one_per_second(cap, fps):
            inp = md_runner.preprocess_frame_bgr(frame_bgr)
            frames_batch.append(inp)
            metas_batch.append((t_sec, frame_idx))
            orig_bgr_batch.append(frame_bgr)
            sampled += 1
            if sampled % 30 == 0:
                emit({"type": "progress", "sampled_frames": sampled})
            if len(frames_batch) >= frames_per_batch:
                flush_batch()
        flush_batch()
    finally:
        cap.release()

    out: Dict[str, Any] = {
        "video_path": os.path.abspath(video_path),
        "model_path": os.path.abspath(getattr(md_runner, "model_path", "")),
        "confidence_threshold": float(confidence_threshold),
        "sample_rate_hz": float(sample_rate_hz),
        "video_fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_s,
        "ort_threads": getattr(md_runner, "ort_threads", None),
        "speciesnet": (
            {
                "enabled": True,
                "model_path": os.path.abspath(getattr(species_runner, "model_path", "")),
                "labels_path": os.path.abspath(getattr(species_runner, "labels_path", "")),
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

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=False)

    return out

