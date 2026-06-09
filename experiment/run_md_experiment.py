#!/usr/bin/env python3
"""Run and compare MegaDetector confidence experiments."""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_md_over_data_frames.py"
EXPERIMENT_DIR = REPO_ROOT / "experiment"
ATTEMPTS_DIR = EXPERIMENT_DIR / "attempts"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_confidences(path: Path) -> Dict[str, Optional[float]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in {path}, got {type(payload).__name__}")
    return {str(k): (None if v is None else float(v)) for k, v in payload.items()}


def load_labels(path: Optional[Path]) -> Optional[Dict[str, bool]]:
    if path is None:
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected labels object JSON in {path}, got {type(payload).__name__}")
    return {str(k): parse_bool(v) for k, v in payload.items()}


def load_or_infer_labels(
    path: Optional[Path], confidences: Dict[str, Optional[float]]
) -> Optional[Dict[str, bool]]:
    if path is not None:
        return load_labels(path)

    labels = {
        frame: label
        for frame in confidences
        if (label := infer_label_from_frame_path(frame)) is not None
    }
    return labels or None


def infer_label_from_frame_path(frame: str) -> Optional[bool]:
    stem = Path(frame.replace("\\", "/")).stem.lower()
    if stem.endswith("_y"):
        return True
    if stem.endswith("_n"):
        return False
    return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "animal", "positive"}:
            return True
        if normalized in {"0", "false", "no", "n", "empty", "negative"}:
            return False
    return bool(value)


def summarize_confidences(
    confidences: Dict[str, Optional[float]],
    runtime_seconds: float,
    threshold: float,
    labels: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    values = [v for v in confidences.values() if v is not None]
    frame_count = len(confidences)
    summary: Dict[str, Any] = {
        "frame_count": frame_count,
        "null_count": frame_count - len(values),
        "runtime_seconds": runtime_seconds,
        "mean_frame_processing_seconds": runtime_seconds / frame_count if frame_count > 0 else None,
        "threshold": threshold,
        "mean_confidence": statistics.fmean(values) if values else None,
        "median_confidence": statistics.median(values) if values else None,
        "min_confidence": min(values) if values else None,
        "max_confidence": max(values) if values else None,
    }

    if labels is not None:
        classification = classify(confidences, labels, threshold)
        summary.update(classification)

    return summary


def classify(
    confidences: Dict[str, Optional[float]], labels: Dict[str, bool], threshold: float
) -> Dict[str, Any]:
    total = 0
    positives = 0
    negatives = 0
    positive_values = []
    negative_values = []
    false_positives = 0
    false_negatives = 0
    correct_hits = 0

    for frame, expected_positive in labels.items():
        if frame not in confidences:
            continue

        confidence = confidences[frame]
        predicted_positive = confidence is not None and confidence >= threshold
        total += 1
        positives += int(expected_positive)
        negatives += int(not expected_positive)
        if confidence is not None and expected_positive:
            positive_values.append(confidence)
        if confidence is not None and not expected_positive:
            negative_values.append(confidence)
        false_positives += int(predicted_positive and not expected_positive)
        false_negatives += int((not predicted_positive) and expected_positive)
        correct_hits += int(predicted_positive == expected_positive)

    return {
        "labeled_frame_count": total,
        "mean_confidence_positive": statistics.fmean(positive_values) if positive_values else None,
        "mean_confidence_negative": statistics.fmean(negative_values) if negative_values else None,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "correct_hit_count": correct_hits,
        "false_positive_pct": pct(false_positives, negatives),
        "false_negative_pct": pct(false_negatives, positives),
        "correct_hit_pct": pct(correct_hits, total),
    }


def pct(count: int, total: int) -> Optional[float]:
    if total == 0:
        return None
    return count / total * 100


def run_detector(args: argparse.Namespace) -> int:
    attempt_dir = resolve_run_attempt_dir(args)
    output_base = attempt_dir or EXPERIMENT_DIR
    output = resolve_repo_path(args.output) if args.output else output_base / f"{args.name}-confidences.json"
    metrics = resolve_repo_path(args.metrics) if args.metrics else output_base / f"{args.name}-metrics.json"

    command = [
        sys.executable,
        str(RUNNER),
        "--data-dir",
        args.data_dir,
        "--model",
        args.model,
        "--batch-size",
        str(args.batch_size),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--output",
        str(output),
    ]

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT)
    runtime_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        return completed.returncode

    confidences = load_confidences(output)
    labels = load_or_infer_labels(
        resolve_repo_path(args.labels) if args.labels else None, confidences
    )
    summary = summarize_confidences(
        confidences=confidences,
        runtime_seconds=runtime_seconds,
        threshold=args.confidence_threshold,
        labels=labels,
    )
    payload = {
        "name": args.name,
        "attempt": str(attempt_dir.relative_to(REPO_ROOT)) if attempt_dir else None,
        "command": " ".join(command),
        "output": str(output.relative_to(REPO_ROOT)),
        "summary": summary,
    }
    write_json(metrics, payload)

    if attempt_dir:
        print(f"Attempt: {attempt_dir.relative_to(REPO_ROOT)}")
    print(f"Wrote {output.relative_to(REPO_ROOT)}")
    print(f"Wrote {metrics.relative_to(REPO_ROOT)}")
    print(
        "Runtime: "
        f"{runtime_seconds:.3f}s, "
        f"mean frame processing: {fmt(summary['mean_frame_processing_seconds'])}s"
    )
    return 0


def build_report(args: argparse.Namespace) -> int:
    attempt_dir = resolve_report_attempt_dir(args)
    before_path = resolve_repo_path(args.before) if args.before else (attempt_dir or EXPERIMENT_DIR) / "before-metrics.json"
    after_path = resolve_repo_path(args.after) if args.after else (attempt_dir or EXPERIMENT_DIR) / "after-metrics.json"
    before = load_json(before_path)
    after = load_json(after_path)
    before_conf = load_confidences(REPO_ROOT / before["output"])
    after_conf = load_confidences(REPO_ROOT / after["output"])
    labels = load_or_infer_labels(
        resolve_repo_path(args.labels) if args.labels else None, before_conf
    )

    threshold = float(args.threshold)
    before_summary = summarize_confidences(
        before_conf, before["summary"]["runtime_seconds"], threshold, labels
    )
    after_summary = summarize_confidences(
        after_conf, after["summary"]["runtime_seconds"], threshold, labels
    )
    diff_summary = diff_summaries(before_summary, after_summary)
    frame_deltas = compare_frames(before_conf, after_conf, labels, threshold)
    git_diff = current_git_diff()

    report = render_report(
        hypothesis=args.hypothesis,
        before=before,
        after=after,
        before_summary=before_summary,
        after_summary=after_summary,
        diff_summary=diff_summary,
        frame_deltas=frame_deltas,
        git_diff=git_diff,
        decision=args.decision,
    )

    output_base = attempt_dir or EXPERIMENT_DIR
    comparison_path = resolve_repo_path(args.comparison) if args.comparison else output_base / "comparison.json"
    write_json(
        comparison_path,
        {
            "before": before["output"],
            "after": after["output"],
            "threshold": threshold,
            "summary": diff_summary,
            "frames": frame_deltas,
        },
    )

    report_path = resolve_repo_path(args.report) if args.report else output_base / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {comparison_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    return 0


def resolve_repo_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def resolve_run_attempt_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.attempt_dir:
        attempt_dir = resolve_repo_path(args.attempt_dir)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        return attempt_dir

    if not args.attempt:
        return None

    slug = slugify(args.attempt)
    if args.name == "before":
        return create_next_attempt_dir(slug)

    latest = latest_attempt_dir(slug)
    if latest is not None:
        return latest
    return create_next_attempt_dir(slug)


def resolve_report_attempt_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.attempt_dir:
        return resolve_repo_path(args.attempt_dir)

    if not args.attempt:
        return None

    latest = latest_attempt_dir(slugify(args.attempt))
    if latest is None:
        raise SystemExit(f"No attempt folder found for: {args.attempt}")
    return latest


def create_next_attempt_dir(slug: str) -> Path:
    ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    next_number = max(existing_attempt_numbers(), default=0) + 1
    attempt_dir = ATTEMPTS_DIR / f"{next_number:04d}-{slug}"
    attempt_dir.mkdir()
    return attempt_dir


def latest_attempt_dir(slug: str) -> Optional[Path]:
    if not ATTEMPTS_DIR.exists():
        return None

    matches = [
        path
        for path in ATTEMPTS_DIR.iterdir()
        if path.is_dir() and re.fullmatch(rf"\d{{4}}-{re.escape(slug)}", path.name)
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.name)


def existing_attempt_numbers() -> list[int]:
    if not ATTEMPTS_DIR.exists():
        return []

    numbers = []
    for path in ATTEMPTS_DIR.iterdir():
        match = re.match(r"^(\d{4})-", path.name)
        if path.is_dir() and match:
            numbers.append(int(match.group(1)))
    return numbers


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48].strip("-") or "experiment"


def compare_frames(
    before: Dict[str, Optional[float]],
    after: Dict[str, Optional[float]],
    labels: Optional[Dict[str, bool]],
    threshold: float,
) -> Dict[str, Any]:
    frames = sorted(set(before) | set(after))
    rows = []
    for frame in frames:
        before_conf = before.get(frame)
        after_conf = after.get(frame)
        expected = labels.get(frame) if labels else None
        rows.append(
            {
                "frame": frame,
                "expected_animal": expected,
                "before_confidence": before_conf,
                "after_confidence": after_conf,
                "score_difference": diff(before_conf, after_conf),
                "before_prediction": prediction(before_conf, threshold),
                "after_prediction": prediction(after_conf, threshold),
            }
        )

    numeric_rows = [row for row in rows if row["score_difference"] is not None]
    largest_positive = max(numeric_rows, key=lambda row: row["score_difference"], default=None)
    largest_negative = min(numeric_rows, key=lambda row: row["score_difference"], default=None)

    return {
        "largest_positive_movement": largest_positive,
        "largest_negative_movement": largest_negative,
        "rows": rows,
    }


def prediction(confidence: Optional[float], threshold: float) -> str:
    if confidence is None:
        return "negative"
    return "positive" if confidence >= threshold else "negative"


def diff_summaries(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Optional[float]]:
    keys = [
        "mean_confidence_positive",
        "mean_confidence_negative",
        "mean_frame_processing_seconds",
        "false_positive_count",
        "false_negative_count",
        "correct_hit_count",
        "false_positive_pct",
        "false_negative_pct",
        "correct_hit_pct",
    ]
    return {key: diff(before.get(key), after.get(key)) for key in keys}


def diff(before: Any, after: Any) -> Optional[float]:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def current_git_diff() -> str:
    completed = subprocess.run(
        ["git", "diff", "--", "scripts/run_md_over_data_frames.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "[No git diff captured]"


def render_report(
    hypothesis: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    before_summary: Dict[str, Any],
    after_summary: Dict[str, Any],
    diff_summary: Dict[str, Optional[float]],
    frame_deltas: Dict[str, Any],
    git_diff: str,
    decision: str,
) -> str:
    return f"""# MegaDetector Experiment

## Hypothesis

{hypothesis}

## Change

```diff
{git_diff}
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `{before["command"]}` | {fmt(before_summary["mean_frame_processing_seconds"])} | {before_summary["frame_count"]} | `{before["output"]}` |
| After | `{after["command"]}` | {fmt(after_summary["mean_frame_processing_seconds"])} | {after_summary["frame_count"]} | `{after["output"]}` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | {fmt(before_summary.get("mean_confidence_positive"))} | {fmt(after_summary.get("mean_confidence_positive"))} | {signed(diff_summary["mean_confidence_positive"])} |
| Mean confidence negative (_n) | {fmt(before_summary.get("mean_confidence_negative"))} | {fmt(after_summary.get("mean_confidence_negative"))} | {signed(diff_summary["mean_confidence_negative"])} |
| Mean frame processing seconds | {fmt(before_summary["mean_frame_processing_seconds"])} | {fmt(after_summary["mean_frame_processing_seconds"])} | {signed(diff_summary["mean_frame_processing_seconds"])} |
| False positive count | {count(before_summary.get("false_positive_count"))} | {count(after_summary.get("false_positive_count"))} | {signed_count(diff_summary["false_positive_count"])} |
| False negative count | {count(before_summary.get("false_negative_count"))} | {count(after_summary.get("false_negative_count"))} | {signed_count(diff_summary["false_negative_count"])} |
| Correct hit count | {count(before_summary.get("correct_hit_count"))} | {count(after_summary.get("correct_hit_count"))} | {signed_count(diff_summary["correct_hit_count"])} |
| False positives | {percent(before_summary.get("false_positive_pct"))} | {percent(after_summary.get("false_positive_pct"))} | {signed_pp(diff_summary["false_positive_pct"])} |
| False negatives | {percent(before_summary.get("false_negative_pct"))} | {percent(after_summary.get("false_negative_pct"))} | {signed_pp(diff_summary["false_negative_pct"])} |
| Correct hits | {percent(before_summary.get("correct_hit_pct"))} | {percent(after_summary.get("correct_hit_pct"))} | {signed_pp(diff_summary["correct_hit_pct"])} |

## Analysis

Largest positive frame movement: {frame_movement(frame_deltas["largest_positive_movement"])}

Largest negative frame movement: {frame_movement(frame_deltas["largest_negative_movement"])}

[Short interpretation of positive confidence, negative confidence, mean frame processing time, false positives, false negatives, and correct hits.]

## Decision

{decision}

Reason: [One or two sentences explaining the decision.]
"""


def frame_movement(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return "n/a"
    return (
        f'`{row["frame"]}` '
        f'{fmt(row["before_confidence"])} -> {fmt(row["after_confidence"])} '
        f'({signed(row["score_difference"])})'
    )


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def count(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def signed(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def signed_count(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):+d}"


def percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def signed_pp(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}pp"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and compare MegaDetector experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run srctips/run_md_over_data_frames.py and collect metrics.")
    run_parser.add_argument("--name", required=True, help="Run name, for example before or after.")
    run_parser.add_argument("--data-dir", default="data")
    run_parser.add_argument("--model", default="models/md_v5a_1_3_640_640_static.onnx")
    run_parser.add_argument("--batch-size", type=int, default=16)
    run_parser.add_argument("--confidence-threshold", type=float, default=0.5)
    run_parser.add_argument(
        "--labels",
        help="Optional JSON mapping frame path to true/false. If omitted, *_y and *_n filename suffixes are used.",
    )
    run_parser.add_argument("--attempt", help="Short hypothesis name for experiment/attempts/0001-name.")
    run_parser.add_argument("--attempt-dir", help="Explicit attempt directory to use.")
    run_parser.add_argument("--output", help="Optional confidences output path.")
    run_parser.add_argument("--metrics", help="Optional metrics output path.")
    run_parser.set_defaults(func=run_detector)

    report_parser = subparsers.add_parser("report", help="Compare two collected runs and write a report.")
    report_parser.add_argument("--before")
    report_parser.add_argument("--after")
    report_parser.add_argument(
        "--labels",
        help="Optional JSON mapping frame path to true/false. If omitted, *_y and *_n filename suffixes are used.",
    )
    report_parser.add_argument("--threshold", type=float, default=0.5)
    report_parser.add_argument("--attempt", help="Short hypothesis name for experiment/attempts/0001-name.")
    report_parser.add_argument("--attempt-dir", help="Explicit attempt directory to use.")
    report_parser.add_argument("--hypothesis", default="[What change are we testing, and what result should improve?]")
    report_parser.add_argument("--decision", default="[Continue / do not continue / rerun with more labeled data]")
    report_parser.add_argument("--report")
    report_parser.add_argument("--comparison")
    report_parser.set_defaults(func=build_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
