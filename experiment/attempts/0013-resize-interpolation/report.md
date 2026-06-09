# MegaDetector Experiment

## Hypothesis

Changing resize interpolation may preserve small-animal detail differently: `INTER_AREA` may help downscaling, while `INTER_CUBIC` or `INTER_LANCZOS4` may preserve sharper detail.

## Change

```diff
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
-    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
+    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=[INTER_AREA, INTER_CUBIC, INTER_LANCZOS4])
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Mean Frame Processing Seconds | Frames | Output |
| --- | ---: | ---: | --- |
| INTER_LINEAR | 0.0507 | 75 | `experiment/attempts/0013-resize-interpolation/before-confidences.json` |
| INTER_AREA | 0.0473 | 75 | `experiment/attempts/0013-resize-interpolation/inter-area-confidences.json` |
| INTER_CUBIC | 0.0466 | 75 | `experiment/attempts/0013-resize-interpolation/inter-cubic-confidences.json` |
| INTER_LANCZOS4 | 0.0474 | 75 | `experiment/attempts/0013-resize-interpolation/inter-lanczos4-confidences.json` |

## Results

| Metric | INTER_LINEAR | INTER_AREA | INTER_CUBIC | INTER_LANCZOS4 |
| --- | ---: | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9000 | 0.8996 | 0.8992 | 0.8992 |
| Mean confidence positive delta | 0.0000 | -0.0004 | -0.0008 | -0.0009 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mean confidence negative delta | 0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0507 | 0.0473 | 0.0466 | 0.0474 |
| Mean frame processing delta | 0.0000 | -0.0034 | -0.0042 | -0.0033 |
| False positives | 0.00% | 0.00% | 0.00% | 0.00% |
| False negatives | 0.00% | 0.00% | 0.00% | 0.00% |
| Correct hits | 100.00% | 100.00% | 100.00% | 100.00% |

## Analysis

All alternative interpolation modes were faster than `INTER_LINEAR`, with `INTER_CUBIC` having the lowest mean frame processing time. However, all three alternatives reduced `Mean confidence positive (_y)`, and none improved `Mean confidence negative (_n)`, which was already 0.0000. Classification quality stayed unchanged across all variants: 0 false positives, 0 false negatives, and 100% correct hits.

## Decision

Do not continue

Reason: The alternatives improve processing time, but the core goal is higher `_y` confidence and lower `_n` confidence. Since `_y` regressed for every tested interpolation and `_n` did not improve, keep `INTER_LINEAR`.
