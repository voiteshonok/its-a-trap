#!/usr/bin/env python3
"""Measure MegaDetector influence of preprocess_bgr_to_md_input changes.

Runs the same ONNX inference path as video_picker on still images, once with the
original baseline preprocess and once with the current video_picker preprocess.
Writes JSON artifacts and a markdown comparison report.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = RESULTS_DIR / "data"
DEFAULT_MODEL = REPO_ROOT / "models" / "md_v5a_1_3_640_640_static.onnx"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_picker.md_postprocess import megadetector_post_processing
from video_picker.megadetector_video import IMAGE_SIZE, preprocess_bgr_to_md_input
from video_picker.utils import (
    configure_ort_cpu_session_threads,
    get_onnxruntime_providers,
    prepare_onnxruntime_cuda,
    run_onnx_with_stacked_batch,
)


PreprocessFn = Callable[[np.ndarray], np.ndarray]


def preprocess_bgr_to_md_input_baseline(bgr: np.ndarray) -> np.ndarray:
    """Original baseline before accepted experiment changes (0005/0010/0014)."""
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    nchw = np.expand_dims(chw, axis=0)
    return nchw / 255.0


@dataclass(frozen=True)
class FrameResult:
    path: str
    animal_confidence: Optional[float]


def iter_image_paths(root_dir: Path) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if Path(filename).suffix.lower() in exts:
                out.append(str(Path(dirpath) / filename))
    out.sort()
    return out


def infer_label_from_frame_path(frame: str) -> Optional[bool]:
    stem = Path(frame.replace("\\", "/")).stem.lower()
    if stem.endswith("_y"):
        return True
    if stem.endswith("_n"):
        return False
    return None


def build_labels_from_frame_keys(frame_keys: Sequence[str]) -> Dict[str, bool]:
    labels: Dict[str, bool] = {}
    unlabeled: List[str] = []

    for frame in frame_keys:
        label = infer_label_from_frame_path(frame)
        if label is None:
            unlabeled.append(frame)
            continue
        labels[frame] = label

    if unlabeled:
        preview = "\n".join(f"  - {frame}" for frame in unlabeled[:10])
        extra = "" if len(unlabeled) <= 10 else f"\n  ... and {len(unlabeled) - 10} more"
        raise SystemExit(
            "Every image must end with _y (expected positive) or _n (expected negative).\n"
            f"Found {len(unlabeled)} unlabeled frame(s):\n{preview}{extra}"
        )

    return labels


def load_labels(path: Optional[Path], frame_keys: Sequence[str]) -> Dict[str, bool]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit(f"Expected labels object JSON in {path}, got {type(payload).__name__}")
        labels = {str(key): bool(value) for key, value in payload.items()}
        missing = [frame for frame in frame_keys if frame not in labels]
        if missing:
            preview = "\n".join(f"  - {frame}" for frame in missing[:10])
            extra = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
            raise SystemExit(
                f"Labels JSON is missing {len(missing)} frame(s) from the dataset:\n{preview}{extra}"
            )
        return labels

    labels = build_labels_from_frame_keys(frame_keys)
    if not labels:
        raise SystemExit(
            "No ground-truth labels found. Use *_y/*_n filename suffixes or pass --labels."
        )
    return labels


def confidence_from_boxes(boxes: np.ndarray) -> float:
    if boxes is None or len(boxes) == 0:
        return 0.0
    boxes_arr = np.asarray(boxes)
    return float(np.max(boxes_arr[:, 4]))


def predicted_positive(confidence: Optional[float], threshold: float) -> bool:
    return confidence is not None and confidence >= threshold


def run_on_images(
    image_paths: Sequence[str],
    model_path: Path,
    batch_size: int,
    confidence_threshold: float,
    preprocess_fn: PreprocessFn,
) -> Tuple[List[FrameResult], float]:
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    configure_ort_cpu_session_threads(sess_options)
    prepare_onnxruntime_cuda()
    providers = get_onnxruntime_providers()
    session = onnxruntime.InferenceSession(
        str(model_path), providers=providers, sess_options=sess_options
    )
    input_name = session.get_inputs()[0].name

    results: List[FrameResult] = []
    frames_batch: List[np.ndarray] = []
    paths_batch: List[str] = []

    def flush_batch() -> None:
        if not frames_batch:
            return
        batch_tensor = np.concatenate(frames_batch, axis=0)
        outputs, _mode = run_onnx_with_stacked_batch(session, input_name, batch_tensor)
        preds = megadetector_post_processing(
            outputs, confidence_threshold, IMAGE_SIZE, IMAGE_SIZE
        )
        for path, boxes in zip(paths_batch, preds):
            results.append(FrameResult(path=path, animal_confidence=confidence_from_boxes(boxes)))
        frames_batch.clear()
        paths_batch.clear()

    started = time.perf_counter()
    for path in image_paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            results.append(FrameResult(path=path, animal_confidence=None))
            continue

        frames_batch.append(preprocess_fn(bgr))
        paths_batch.append(path)
        if len(frames_batch) >= max(1, batch_size):
            flush_batch()

    flush_batch()
    return results, time.perf_counter() - started


def frame_key(path: str, data_dir: Path) -> str:
    rel = Path(path).resolve().relative_to(data_dir.resolve())
    return rel.as_posix()


def confidences_from_results(results: Sequence[FrameResult], data_dir: Path) -> Dict[str, Optional[float]]:
    return {frame_key(result.path, data_dir): result.animal_confidence for result in results}


def pct(count: int, total: int) -> Optional[float]:
    if total == 0:
        return None
    return count / total * 100.0


def stddev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    return float(statistics.stdev(values))


def compute_labeled_metrics(
    confidences: Dict[str, Optional[float]],
    labels: Dict[str, bool],
    threshold: float,
) -> Dict[str, Any]:
    positive_total = 0
    negative_total = 0
    correct_positive_count = 0
    correct_negative_count = 0
    correct_positive_confidences: List[float] = []
    correct_negative_confidences: List[float] = []
    false_negative_frames: List[str] = []
    false_positive_frames: List[str] = []

    for frame, expected_positive in labels.items():
        confidence = confidences.get(frame)
        if frame not in confidences:
            raise ValueError(f"Missing confidence for labeled frame: {frame}")

        predicted = predicted_positive(confidence, threshold)

        if expected_positive:
            positive_total += 1
            if predicted:
                correct_positive_count += 1
                if confidence is not None:
                    correct_positive_confidences.append(confidence)
            else:
                false_negative_frames.append(frame)
        else:
            negative_total += 1
            if not predicted:
                correct_negative_count += 1
                if confidence is not None:
                    correct_negative_confidences.append(confidence)
            else:
                false_positive_frames.append(frame)

    return {
        "positive_frame_count": positive_total,
        "negative_frame_count": negative_total,
        "false_negative_count": len(false_negative_frames),
        "false_positive_count": len(false_positive_frames),
        "false_negative_frames": false_negative_frames,
        "false_positive_frames": false_positive_frames,
        "pct_correct_positive": pct(correct_positive_count, positive_total),
        "mean_confidence_correct_positive": (
            statistics.fmean(correct_positive_confidences)
            if correct_positive_confidences
            else None
        ),
        "std_confidence_correct_positive": stddev(correct_positive_confidences),
        "pct_correct_negative": pct(correct_negative_count, negative_total),
        "mean_confidence_correct_negative": (
            statistics.fmean(correct_negative_confidences)
            if correct_negative_confidences
            else None
        ),
        "std_confidence_correct_negative": stddev(correct_negative_confidences),
    }


def display_path(path: Path, *, resolve: bool = True) -> str:
    target = path.resolve() if resolve else path
    try:
        return str(target.relative_to(REPO_ROOT))
    except ValueError:
        return str(target)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def signed(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def signed_pp(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}pp"


def diff(before: Any, after: Any) -> Optional[float]:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def format_misclassified_frames(frames: Sequence[str], limit: int = 8) -> str:
    if not frames:
        return "None"
    shown = list(frames[:limit])
    lines = [f"- `{frame}`" for frame in shown]
    if len(frames) > limit:
        lines.append(f"- ... and {len(frames) - limit} more")
    return "\n".join(lines)


def render_report(
    *,
    data_dir: Path,
    model_path: Path,
    threshold: float,
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    before_runtime: float,
    after_runtime: float,
    frame_count: int,
) -> str:
    rows = [
        ("Percentage correct positive", "pct_correct_positive", fmt_pct, signed_pp),
        ("Mean confidence of correct positive", "mean_confidence_correct_positive", fmt, signed),
        ("STD confidence of correct positive", "std_confidence_correct_positive", fmt, signed),
        ("Percentage correct negative", "pct_correct_negative", fmt_pct, signed_pp),
        ("Mean confidence of correct negative", "mean_confidence_correct_negative", fmt, signed),
        ("STD confidence of correct negative", "std_confidence_correct_negative", fmt, signed),
    ]

    table_lines = [
        "| Metric | Before | After | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, formatter, diff_formatter in rows:
        before_value = before_metrics.get(key)
        after_value = after_metrics.get(key)
        table_lines.append(
            f"| {label} | {formatter(before_value)} | {formatter(after_value)} | "
            f"{diff_formatter(diff(before_value, after_value))} |"
        )

    return f"""# Preprocess Influence Report

Compares the original baseline `preprocess_bgr_to_md_input` pipeline against the current
`video_picker.megadetector_video.preprocess_bgr_to_md_input` implementation on the same
still-image dataset.

## Setup

| Setting | Value |
| --- | --- |
| Data directory | `{display_path(data_dir, resolve=False)}` |
| Model | `{display_path(model_path)}` |
| Detection threshold | {threshold:.2f} |
| Frames processed | {frame_count} |
| Positive frames (`*_y`) | {before_metrics["positive_frame_count"]} |
| Negative frames (`*_n`) | {before_metrics["negative_frame_count"]} |
| Before runtime (s) | {before_runtime:.3f} |
| After runtime (s) | {after_runtime:.3f} |

## Label Convention

- Filename ending in `_y` means the frame should be detected as **positive** (animal present, confidence >= {threshold:.2f}).
- Filename ending in `_n` means the frame should be detected as **negative** (no animal, confidence < {threshold:.2f}).

## Preprocess Versions

**Before (baseline):** pre-resize `GaussianBlur(3, 3)`, stretch resize to {IMAGE_SIZE}x{IMAGE_SIZE}.

**After (current app):** mild pre-resize Gaussian blur (`sigma=0.8`), stretch resize to {IMAGE_SIZE}x{IMAGE_SIZE}.

Both runs use `video_picker` ONNX inference and `megadetector_post_processing`.

## Results

{chr(10).join(table_lines)}

## Misclassified Frames

### Before

False negatives (`*_y` predicted below threshold):

{format_misclassified_frames(before_metrics.get("false_negative_frames", []))}

False positives (`*_n` predicted at or above threshold):

{format_misclassified_frames(before_metrics.get("false_positive_frames", []))}

### After

False negatives (`*_y` predicted below threshold):

{format_misclassified_frames(after_metrics.get("false_negative_frames", []))}

False positives (`*_n` predicted at or above threshold):

{format_misclassified_frames(after_metrics.get("false_positive_frames", []))}

## Notes

- Labels are inferred from `*_y` / `*_n` filename suffixes unless `--labels` is provided.
- Correct positive means an animal frame predicted above threshold.
- Correct negative means an empty frame predicted below threshold.
- Confidence metrics are computed only on correctly classified frames in each class.
"""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure before/after influence of preprocess_bgr_to_md_input."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Image root directory (default: {display_path(DEFAULT_DATA_DIR)})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="MegaDetector ONNX model path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("MEGADETECTOR_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.environ.get("MEGADETECTOR_CONFIDENCE", "0.5")),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional JSON mapping frame path to true/false ground truth.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=RESULTS_DIR / "report.md",
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir_resolved = data_dir.resolve()
    if not data_dir_resolved.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    image_paths = iter_image_paths(data_dir_resolved)
    if not image_paths:
        raise SystemExit(f"No images found under: {data_dir}")

    frame_keys = [frame_key(path, data_dir_resolved) for path in image_paths]
    labels = load_labels(args.labels.resolve() if args.labels else None, frame_keys)

    before_results, before_runtime = run_on_images(
        image_paths=image_paths,
        model_path=args.model.resolve(),
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        preprocess_fn=preprocess_bgr_to_md_input_baseline,
    )
    after_results, after_runtime = run_on_images(
        image_paths=image_paths,
        model_path=args.model.resolve(),
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        preprocess_fn=preprocess_bgr_to_md_input,
    )

    before_confidences = confidences_from_results(before_results, data_dir_resolved)
    after_confidences = confidences_from_results(after_results, data_dir_resolved)
    before_metrics = compute_labeled_metrics(
        before_confidences, labels, args.confidence_threshold
    )
    after_metrics = compute_labeled_metrics(
        after_confidences, labels, args.confidence_threshold
    )

    write_json(RESULTS_DIR / "before-confidences.json", before_confidences)
    write_json(RESULTS_DIR / "after-confidences.json", after_confidences)
    write_json(
        RESULTS_DIR / "before-metrics.json",
        {
            "runtime_seconds": before_runtime,
            "threshold": args.confidence_threshold,
            "metrics": {
                key: value
                for key, value in before_metrics.items()
                if key not in {"false_negative_frames", "false_positive_frames"}
            },
        },
    )
    write_json(
        RESULTS_DIR / "after-metrics.json",
        {
            "runtime_seconds": after_runtime,
            "threshold": args.confidence_threshold,
            "metrics": {
                key: value
                for key, value in after_metrics.items()
                if key not in {"false_negative_frames", "false_positive_frames"}
            },
        },
    )

    report = render_report(
        data_dir=data_dir,
        model_path=args.model.resolve(),
        threshold=args.confidence_threshold,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        before_runtime=before_runtime,
        after_runtime=after_runtime,
        frame_count=len(image_paths),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print(f"Processed {len(image_paths)} images from {display_path(data_dir, resolve=False)}")
    print(f"Wrote {display_path(args.report)}")
    print(f"Wrote {display_path(RESULTS_DIR / 'before-metrics.json')}")
    print(f"Wrote {display_path(RESULTS_DIR / 'after-metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
