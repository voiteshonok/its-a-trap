# MegaDetector Experiment

## Hypothesis

Tuning the accepted after-resize unsharp mask may improve `Mean confidence positive (_y)` more than the current `sigma=1.0`, `1.5/-0.5` setting while avoiding false negatives and keeping mean frame processing time close to current.

## Change

```diff
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     # ... letterbox resize ...
-    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
-    resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
+    blurred = cv2.GaussianBlur(resized, (0, 0), [0.75, 1.0, 1.25])
+    resized = cv2.addWeighted(resized, [1.25, 1.3, 1.4], blurred, [-0.25, -0.3, -0.4], 0)
```

## Runs

| Run | Mean Frame Processing Seconds | Frames | Output |
| --- | ---: | ---: | --- |
| Current unsharp `1.0, 1.5/-0.5` | 0.0470 | 75 | `experiment/attempts/0017-tune-unsharp-strength/before-confidences.json` |
| `sigma=0.75, 1.25/-0.25` | 0.0498 | 75 | `experiment/attempts/0017-tune-unsharp-strength/sigma-0-75-a-1-25-confidences.json` |
| `sigma=1.0, 1.3/-0.3` | 0.0501 | 75 | `experiment/attempts/0017-tune-unsharp-strength/sigma-1-0-a-1-3-confidences.json` |
| `sigma=1.25, 1.4/-0.4` | 0.0511 | 75 | `experiment/attempts/0017-tune-unsharp-strength/sigma-1-25-a-1-4-confidences.json` |

## Results

| Metric | Current | `0.75, 1.25/-0.25` | `1.0, 1.3/-0.3` | `1.25, 1.4/-0.4` |
| --- | ---: | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.9036 | 0.9028 | 0.8825 |
| Mean confidence positive delta | 0.0000 | +0.0001 | -0.0007 | -0.0209 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mean confidence negative delta | 0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0470 | 0.0498 | 0.0501 | 0.0511 |
| Mean frame processing delta | 0.0000 | +0.0028 | +0.0030 | +0.0041 |
| False positives | 0.00% | 0.00% | 0.00% | 0.00% |
| False negatives | 0.00% | 0.00% | 0.00% | 3.77% |
| Correct hits | 100.00% | 100.00% | 100.00% | 97.33% |

## Analysis

The mildest variant, `sigma=0.75` with `1.25/-0.25`, was the only tuned setting that improved `Mean confidence positive (_y)`, but the gain was only +0.0001 while mean frame processing time regressed by +0.0028s per frame. `Mean confidence negative (_n)` stayed flat at 0.0000. The largest positive frame movement for that variant was `v1_frames/frame_0015_y.jpg` 0.5233 -> 0.5482 (+0.0249), while the largest negative movement was `v1_frames/frame_0006_y.jpg` 0.8941 -> 0.8830 (-0.0111).

The `sigma=1.0, 1.3/-0.3` variant reduced `_y`, and the `sigma=1.25, 1.4/-0.4` variant clearly regressed `_y` and introduced 2 false negatives.

## Decision

Do not continue

Reason: The best tuned setting gives only a negligible `_y` improvement, does not improve `_n`, and makes processing slower. Keep the current accepted unsharp mask: `sigma=1.0`, `1.5/-0.5`.
