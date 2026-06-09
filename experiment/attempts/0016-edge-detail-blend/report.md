# MegaDetector Experiment

## Hypothesis

Blending a small Canny edge/detail map back into the image will emphasize contours, increasing Mean confidence positive (_y), while checking whether background texture raises Mean confidence negative (_n) or false positives.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..4200fe5 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -39,6 +39,10 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     left = (IMAGE_SIZE - resized_width) // 2
     resized[top : top + resized_height, left : left + resized_width] = resized_content
 
+    edges = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 50, 150)
+    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
+    resized = cv2.addWeighted(resized, 1.0, edge_bgr, 0.08, 0)
+
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0016-edge-detail-blend\before-confidences.json` | 0.0483 | 75 | `experiment\attempts\0016-edge-detail-blend\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0016-edge-detail-blend\after-confidences.json` | 0.0498 | 75 | `experiment\attempts\0016-edge-detail-blend\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8731 | -0.0303 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0483 | 0.0498 | +0.0015 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 2 | +2 |
| Correct hit count | 75 | 73 | -2 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 3.77% | +3.77pp |
| Correct hits | 100.00% | 97.33% | -2.67pp |

## Analysis

Largest positive frame movement: `v5_frames/frame_0012_y.jpg` 0.8713 -> 0.8806 (+0.0093)

Largest negative frame movement: `v1_frames/frame_0014_y.jpg` 0.5317 -> 0.0000 (-0.5317)

The Canny edge blend strongly regressed the core positive-confidence metric: `Mean confidence positive (_y)` dropped from 0.9035 to 0.8731 (-0.0303). `Mean confidence negative (_n)` stayed at 0.0000, so there was no negative-sample improvement. Classification quality also regressed, adding 2 false negatives and dropping correct hits from 100.00% to 97.33%. Mean frame processing time increased from 0.0483s to 0.0498s (+0.0015s per frame).

## Decision

Do not continue

Reason: Edge blending reduced `_y` confidence, introduced false negatives, and made processing slower. Keep the accepted letterbox and unsharp steps, but do not add Canny edge/detail blending.
