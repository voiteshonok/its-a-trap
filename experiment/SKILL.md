---
name: megadetector-confidence-experiment
description: Runs MegaDetector confidence experiments using scripts/run_md_over_data_frames.py and compares before/after confidences.json outputs. Use when evaluating changes to frame preprocessing, model inference, post-processing, thresholds, or confidence scoring.
disable-model-invocation: true
---

# MegaDetector Confidence Experiment

Use this skill when the user wants to test whether a code change improves the confidence output from `scripts/run_md_over_data_frames.py`.

The core goal is to increase `Mean confidence positive (_y)` and decrease `Mean confidence negative (_n)`.

## Workflow

1. Define the hypothesis in plain language before changing code.
2. Choose a short hypothesis slug, for example `blur-before-resize`. Outputs go under `experiment/attempts/0001-blur-before-resize`.
3. Capture the baseline output and metrics. This creates the next numbered attempt folder:

```powershell
python experiment/run_md_experiment.py run --name before --attempt blur-before-resize
```

4. Make the experimental code change.
5. Record the exact changed code with:

```powershell
git diff -- srctips/run_md_over_data_frames.py
```

6. Capture the after output and metrics. This reuses the latest attempt folder with the same slug:

```powershell
python experiment/run_md_experiment.py run --name after --attempt blur-before-resize
```

7. Generate the report:

```powershell
python experiment/run_md_experiment.py report --attempt blur-before-resize --hypothesis "[what changed and what should improve]"
```

8. Review `report.md` and `comparison.json` inside the attempt folder.
9. Apply the report decision to the working tree:
   - If the report says `Do not continue`, revert the experimental code change.
   - If the report says `Continue`, leave the experimental code change in place so the next experiment builds on it.
10. If and only if the report says `Continue`, create or update `experiment/RESULTS.md` with the latest accepted metrics.
11. Save visual previews for the current preprocessing output in the attempt folder.

## Metrics

Use a fixed detection threshold for before and after. If no threshold is specified, use `0.5` because it is the script default.

For each frame:

- Predicted positive: confidence >= threshold.
- Predicted negative: confidence < threshold or confidence is null.
- Score difference: after confidence - before confidence.
- Mean frame processing time: total wall-clock runtime / processed frame count.
- Correct hit: prediction matches the expected label.
- False positive: predicted positive but expected label is negative.
- False negative: predicted negative but expected label is positive.

False-positive, false-negative, and correct-hit percentages require ground-truth labels. If labels are unavailable, report only score movement and mark label-based metrics as not measured.

## Report Requirements

Every experiment report must include:

- Hypothesis: phrase exactly what is being tested.
- Overview: summarize the experiment and include the exact code changed.
- Results table: show before and after results in one table.
- Analysis: explain positive mean confidence (`_y`), negative mean confidence (`_n`), mean frame processing time, false positives, false negatives, correct hits, and notable frame-level changes.
- Summary: include positive mean confidence difference, negative mean confidence difference, mean frame processing time difference, percentage of false positives, percentage of false negatives, percentage of correct hits, and the final decision.

## Visual Previews

For each experiment, save side-by-side preview images under:

`experiment/attempts/<attempt>/preprocess-previews/`

Use the first image (frame_0001) from each immediate folder under `data/` (v1-v5).

Each preview must show:

- Left: original image resized to height `IMAGE_SIZE` while preserving the original aspect ratio.
- Right: current `preprocess_bgr_to_md_input` output converted back to an image. This side is always `IMAGE_SIZE x IMAGE_SIZE`.

Do not stretch the original preview to a square. The original preview height should be `IMAGE_SIZE`; its width should follow the source image aspect ratio.

## RESULTS.md

Maintain `experiment/RESULTS.md` as the summary of accepted progress only.

- Update `RESULTS.md` only when an experiment report decision is `Continue`.
- Do not update `RESULTS.md` for rejected experiments.
- Include a table comparing the original baseline from before all experiments with the latest accepted metrics after all continued experiments.
- Use the "before" metrics from the first experiment report as the original baseline.
- Use the current continued experiment's "after" metrics as the latest accepted metrics.
- Include at minimum: accepted attempt, accepted change, `Mean confidence positive (_y)`, `Mean confidence negative (_n)`, mean frame processing seconds, false positives, false negatives, correct hits, and final decision.

## Decision Rule

Continue only if the after result moves the core metrics in the right direction without a meaningful regression in mean frame processing time, false positives, false negatives, or correct hits:

- `Mean confidence positive (_y)` should be higher.
- `Mean confidence negative (_n)` should be lower.

If only one core metric improves, continue only when the trade-off is clearly useful. If the output changes are ambiguous, request a larger labeled frame sample before continuing.

After writing the decision, make the code state match it: rejected experiments must be reverted, accepted experiments must remain applied.
