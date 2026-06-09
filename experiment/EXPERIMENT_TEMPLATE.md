# MegaDetector Experiment

## Hypothesis

`[What change are we testing, and what result should improve?]`

## Change

```diff
[Exact relevant diff from scripts/run_md_over_data_frames.py]
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `python experiment/run_md_experiment.py run --name before --attempt [short-hypothesis-name]` | `[0.00]` | `[count]` | `experiment/attempts/0001-short-hypothesis-name/before-confidences.json` |
| After | `python experiment/run_md_experiment.py run --name after --attempt [short-hypothesis-name]` | `[0.00]` | `[count]` | `experiment/attempts/0001-short-hypothesis-name/after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | `[0.0000]` | `[0.0000]` | `[+0.0000]` |
| Mean confidence negative (_n) | `[0.0000]` | `[0.0000]` | `[+0.0000]` |
| Mean frame processing seconds | `[0.0000]` | `[0.0000]` | `[+0.0000]` |
| False positives | `[0.00%]` | `[0.00%]` | `[+0.00pp]` |
| False negatives | `[0.00%]` | `[0.00%]` | `[+0.00pp]` |
| Correct hits | `[0.00%]` | `[0.00%]` | `[+0.00pp]` |

## Analysis

`[Short interpretation of positive confidence, negative confidence, mean frame processing time, false positives, false negatives, and correct hits. Mention any important frame-level regressions.]`

## Decision

`[Continue / do not continue / rerun with more labeled data]`

Reason: `[One or two sentences explaining the decision.]`
