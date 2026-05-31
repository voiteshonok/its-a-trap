"""Median-background motion gating on ~1 Hz video samples (in-memory, no image files)."""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Skip motion filter when video is shorter than this or has fewer 1 Hz samples.
MIN_DURATION_SECONDS = 10.0
MIN_SAMPLE_COUNT = 10


@dataclass(frozen=True)
class MotionFilterParams:
    diff_threshold: int = 30
    morph_kernel_size: int = 7
    min_contour_area: float = 1500.0
    # Max gap (in sample indices) between motion hits to stay in one span [min..max].
    max_cluster_gap_samples: int = 3


def compute_median_background(frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(frames_bgr, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def frame_has_motion(
    frame_bgr: np.ndarray,
    median_bg: np.ndarray,
    params: MotionFilterParams,
) -> bool:
    diff = cv2.absdiff(frame_bgr, median_bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, params.diff_threshold, 255, cv2.THRESH_BINARY)

    k = max(1, int(params.morph_kernel_size))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > params.min_contour_area:
            return True
    return False


def motion_sample_indices(
    frames_bgr: Sequence[np.ndarray],
    median_bg: np.ndarray,
    params: MotionFilterParams,
) -> List[int]:
    out: List[int] = []
    for i, frame in enumerate(frames_bgr):
        if frame_has_motion(frame, median_bg, params):
            out.append(i)
    return out


def motion_indices_to_ranges(
    motion_indices: Sequence[int],
    *,
    max_cluster_gap_samples: int,
) -> List[Tuple[int, int]]:
    """Merge motion hits into inclusive [min, max] spans; gaps inside a span are filled."""
    if not motion_indices:
        return []

    sorted_idx = sorted(set(int(i) for i in motion_indices))
    ranges: List[Tuple[int, int]] = []
    start = sorted_idx[0]
    end = sorted_idx[0]
    gap_limit = max(0, int(max_cluster_gap_samples))

    for i in sorted_idx[1:]:
        if i - end <= gap_limit + 1:
            end = i
        else:
            ranges.append((start, end))
            start = i
            end = i
    ranges.append((start, end))
    return ranges


def expand_ranges(ranges: Sequence[Tuple[int, int]], *, num_samples: int) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for start, end in ranges:
        for i in range(int(start), int(end) + 1):
            if 0 <= i < num_samples and i not in seen:
                seen.add(i)
                out.append(i)
    out.sort()
    return out


def _format_index_list(indices: Sequence[int], *, max_show: int = 40) -> str:
    items = [int(i) for i in indices]
    if len(items) <= max_show:
        return str(items)
    head = ", ".join(str(i) for i in items[:max_show])
    return f"[{head}, … +{len(items) - max_show} more]"


def log_motion_filter_summary(
    meta: Dict[str, Any],
    *,
    video_frame_indices: Sequence[int] | None = None,
) -> None:
    """Log how many 1 Hz samples were kept vs filtered out (and optional video frame indices)."""
    n = len(meta.get("processed_sample_indices", [])) + len(meta.get("filtered_out_sample_indices", []))
    if n == 0 and video_frame_indices is not None:
        n = len(video_frame_indices)

    processed: List[int] = list(meta.get("processed_sample_indices", []))
    filtered: List[int] = list(meta.get("filtered_out_sample_indices", []))
    n_total = len(processed) + len(filtered) if (processed or filtered) else n

    if meta.get("skipped"):
        logger.info(
            "Motion filter skipped (%s): processing all %d/%d samples %s",
            meta.get("skip_reason"),
            len(processed),
            n_total,
            _format_index_list(processed),
        )
        return

    logger.info(
        "Motion filter: %d motion hit(s) at sample indices %s",
        len(meta.get("motion_sample_indices", [])),
        _format_index_list(meta.get("motion_sample_indices", [])),
    )
    if meta.get("motion_ranges"):
        logger.info("Motion filter: spans (sample index min..max) %s", meta["motion_ranges"])
    if meta.get("skip_reason"):
        logger.info("Motion filter note: %s", meta["skip_reason"])

    logger.info(
        "Motion filter: keeping %d/%d samples for MegaDetector %s",
        len(processed),
        n_total,
        _format_index_list(processed),
    )
    logger.info(
        "Motion filter: filtered out %d/%d samples %s",
        len(filtered),
        n_total,
        _format_index_list(filtered),
    )

    if video_frame_indices is not None and n_total == len(video_frame_indices):
        proc_vf = [int(video_frame_indices[i]) for i in processed]
        filt_vf = [int(video_frame_indices[i]) for i in filtered]
        logger.info(
            "Motion filter: keeping video frame indices %s",
            _format_index_list(proc_vf),
        )
        logger.info(
            "Motion filter: filtered out video frame indices %s",
            _format_index_list(filt_vf),
        )


def should_skip_motion_filter(
    *,
    duration_seconds: float | None,
    num_samples: int,
) -> bool:
    if num_samples < MIN_SAMPLE_COUNT:
        return True
    if duration_seconds is not None and duration_seconds < MIN_DURATION_SECONDS:
        return True
    return False


def select_sample_indices_for_md(
    frames_bgr: Sequence[np.ndarray],
    *,
    duration_seconds: float | None,
    params: MotionFilterParams | None = None,
) -> Dict[str, Any]:
    """
    Choose which ~1 Hz sample indices to run through MegaDetector.

    Returns metadata plus sorted sample indices to process.
    """
    n = len(frames_bgr)
    all_indices = list(range(n))
    params = params or MotionFilterParams()

    if should_skip_motion_filter(duration_seconds=duration_seconds, num_samples=n):
        meta = {
            "enabled": True,
            "skipped": True,
            "skip_reason": (
                f"duration<{MIN_DURATION_SECONDS}s or samples<{MIN_SAMPLE_COUNT}"
            ),
            "params": asdict(params),
            "motion_sample_indices": all_indices,
            "motion_ranges": [[0, n - 1]] if n else [],
            "processed_sample_indices": all_indices,
            "filtered_out_sample_indices": [],
        }
        return meta

    median_bg = compute_median_background(frames_bgr)
    motion_hits = motion_sample_indices(frames_bgr, median_bg, params)
    ranges = motion_indices_to_ranges(
        motion_hits,
        max_cluster_gap_samples=params.max_cluster_gap_samples,
    )

    if not ranges:
        processed = all_indices
        skip_note = "no_motion_detected_using_all_samples"
    else:
        processed = expand_ranges(ranges, num_samples=n)
        skip_note = None

    processed_set = set(processed)
    filtered_out = [i for i in all_indices if i not in processed_set]

    meta = {
        "enabled": True,
        "skipped": False,
        "skip_reason": skip_note,
        "params": asdict(params),
        "motion_sample_indices": motion_hits,
        "motion_ranges": [[a, b] for a, b in ranges],
        "processed_sample_indices": processed,
        "filtered_out_sample_indices": filtered_out,
    }
    return meta
