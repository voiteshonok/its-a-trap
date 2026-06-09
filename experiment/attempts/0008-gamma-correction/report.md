# MegaDetector Experiment

## Hypothesis

Gamma correction before resize may help MegaDetector recover animals in underexposed or overexposed frames by brightening or darkening the input image.

## Change

```diff
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    gamma = [0.8, 1.2, 1.5]
+    lookup = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
+    bgr = cv2.LUT(bgr, lookup)
+
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
```

## Runs

| Run | Runtime Seconds | Frames | FPS | Output |
| --- | ---: | ---: | ---: | --- |
| Before | 3.5127 | 75 | 21.3508 | `experiment/attempts/0008-gamma-correction/before-confidences.json` |
| Gamma 0.8 | 3.6361 | 75 | 20.6267 | `experiment/attempts/0008-gamma-correction/gamma-0-8-confidences.json` |
| Gamma 1.2 | 3.5702 | 75 | 21.0074 | `experiment/attempts/0008-gamma-correction/gamma-1-2-confidences.json` |
| Gamma 1.5 | 3.5793 | 75 | 20.9540 | `experiment/attempts/0008-gamma-correction/gamma-1-5-confidences.json` |

## Results

| Metric | Before | Gamma 0.8 | Gamma 1.2 | Gamma 1.5 |
| --- | ---: | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8964 | 0.8974 | 0.8836 | 0.8729 |
| Mean confidence positive delta | 0.0000 | +0.0010 | -0.0128 | -0.0236 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mean confidence negative delta | 0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Runtime seconds delta | 0.0000 | +0.1233 | +0.0574 | +0.0665 |
| FPS delta | 0.0000 | -0.7242 | -0.3435 | -0.3969 |
| False positives | 0.00% | 0.00% | 0.00% | 0.00% |
| False negatives | 0.00% | 0.00% | 1.89% | 3.77% |
| Correct hits | 100.00% | 100.00% | 98.67% | 97.33% |

## Analysis

Gamma 0.8 produced a very small mean-confidence gain (+0.0007) without changing false positives, false negatives, or correct hits. Its largest positive movement was `v1_frames/frame_0015_y.jpg` 0.5445 -> 0.5864 (+0.0419), while its largest negative movement was `v2_frames/frame_0002_y.jpg` 0.7208 -> 0.6986 (-0.0223). It was slower than baseline by 0.1233 seconds.

Gamma 1.2 and gamma 1.5 both regressed the important metrics. Gamma 1.2 reduced mean confidence by 0.0091 and introduced 1 false negative. Gamma 1.5 reduced mean confidence by 0.0166 and introduced 2 false negatives. The strongest regressions were `v1_frames/frame_0015_y.jpg` dropping to 0.0 for gamma 1.2 and `v1_frames/frame_0014_y.jpg` dropping to 0.0 for gamma 1.5.

## Decision

Do not continue with fixed gamma correction.

Reason: Gamma 0.8 is directionally best, but the confidence gain is too small and comes with slower processing. Darkening with gamma 1.2 or 1.5 clearly hurts recall by creating false negatives.
