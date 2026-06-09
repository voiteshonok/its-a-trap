# MegaDetector Experiment

## Hypothesis

Changing the letterbox padding color may improve model behavior because the constant `114` padding may not be optimal for this dataset. Better padding should increase `Mean confidence positive (_y)` or lower `Mean confidence negative (_n)` without hurting classification quality.

## Change

```diff
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     # ... aspect-ratio resize ...
-    resized = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=bgr.dtype)
+    resized = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), [0, 128, per-image mean], dtype=bgr.dtype)
```

## Runs

| Run | Mean Frame Processing Seconds | Frames | Output |
| --- | ---: | ---: | --- |
| Padding 114 current | 0.0470 | 75 | `experiment/attempts/0018-tune-letterbox-padding-color/before-confidences.json` |
| Padding 0 | 0.0455 | 75 | `experiment/attempts/0018-tune-letterbox-padding-color/padding-0-confidences.json` |
| Padding 128 | 0.0464 | 75 | `experiment/attempts/0018-tune-letterbox-padding-color/padding-128-confidences.json` |
| Padding mean color | 0.0472 | 75 | `experiment/attempts/0018-tune-letterbox-padding-color/padding-mean-confidences.json` |

## Results

| Metric | Padding 114 | Padding 0 | Padding 128 | Padding mean |
| --- | ---: | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8722 | 0.9029 | 0.9035 |
| Mean confidence positive delta | 0.0000 | -0.0312 | -0.0006 | -0.0000 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mean confidence negative delta | 0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0470 | 0.0455 | 0.0464 | 0.0472 |
| Mean frame processing delta | 0.0000 | -0.0015 | -0.0006 | +0.0002 |
| False positives | 0.00% | 0.00% | 0.00% | 0.00% |
| False negatives | 0.00% | 3.77% | 0.00% | 0.00% |
| Correct hits | 100.00% | 97.33% | 100.00% | 100.00% |

## Analysis

Padding `0` is faster, but it strongly regresses `Mean confidence positive (_y)` and introduces 2 false negatives. Its largest negative movement was `v1_frames/frame_0014_y.jpg` 0.5317 -> 0.0000 (-0.5317).

Padding `128` is slightly faster than `114`, but it also lowers `_y` and does not improve `_n`. Mean-color padding is effectively tied with `114` on `_y` and `_n`, but it is slightly slower due to the extra per-frame mean calculation.

## Decision

Do not continue

Reason: No tested padding color improves the core metrics over `114`. Keep the current accepted constant padding value `114`.
