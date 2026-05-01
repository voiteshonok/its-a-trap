import os
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime


def load_txtset_labels_last_field(label_path: str) -> List[str]:
    labels: List[str] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line.split(";")[-1])
    if not labels:
        raise ValueError(f"No labels found in: {label_path}")
    return labels


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape (B,C), got: {logits.shape}")
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def crop_norm_xyxy_from_bgr(frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray | None:
    if not (isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4):
        return None
    h, w = frame_bgr.shape[:2]
    try:
        x1n, y1n, x2n, y2n = (
            float(bbox_xyxy[0]),
            float(bbox_xyxy[1]),
            float(bbox_xyxy[2]),
            float(bbox_xyxy[3]),
        )
    except Exception:
        return None

    x1 = int(max(0.0, min(1.0, x1n)) * w)
    y1 = int(max(0.0, min(1.0, y1n)) * h)
    x2 = int(max(0.0, min(1.0, x2n)) * w)
    y2 = int(max(0.0, min(1.0, y2n)) * h)

    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


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
    cores = max(1, int(cores))
    sess_options.intra_op_num_threads = cores
    sess_options.inter_op_num_threads = 1
    return {
        "intra_op_num_threads": cores,
        "inter_op_num_threads": 1,
        "source": "cli(--batch-size as cores)",
    }


def run_onnx_with_stacked_batch(
    session: onnxruntime.InferenceSession, input_name: str, batch_tensor: np.ndarray
) -> Tuple[List[np.ndarray], str]:
    """
    Run an ONNX session on a stacked batch (B, ...).

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

