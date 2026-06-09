# MegaDetector Experiment

## Hypothesis

A very mild bilateral filter after letterbox and before unsharp will reduce fixed-size noise while preserving edges, increasing positive confidence without increasing negative confidence and with minimal frame-time cost.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..9e79faf 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -39,6 +39,8 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     left = (IMAGE_SIZE - resized_width) // 2
     resized[top : top + resized_height, left : left + resized_width] = resized_content
 
+    resized = cv2.bilateralFilter(resized, 3, 20, 20)
+
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0025-very-mild-bilateral-after-resize\before-confidences.json` | 0.0496 | 75 | `experiment\attempts\0025-very-mild-bilateral-after-resize\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0025-very-mild-bilateral-after-resize\after-confidences.json` | 0.0503 | 75 | `experiment\attempts\0025-very-mild-bilateral-after-resize\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.9018 | -0.0016 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0496 | 0.0503 | +0.0006 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0014_y.jpg` 0.5317 -> 0.5785 (+0.0468)

Largest negative frame movement: `v1_frames/frame_0006_y.jpg` 0.8941 -> 0.8740 (-0.0201)

The mild bilateral filter preserved negative confidence and classification quality, with no false positives or false negatives. It did not improve the core score, though: positive confidence decreased by 0.0016 and mean frame processing time increased slightly by 0.0006 seconds.

## Decision

Do not continue

Reason: The change lowers positive confidence and adds a small processing cost without improving negative confidence or classification. Revert this preprocessing change and keep the current accepted pipeline.
