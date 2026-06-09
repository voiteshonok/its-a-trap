# MegaDetector Experiment

## Hypothesis

Lifting only dark HSV value pixels with a capped curve will help low-light animal regions without changing bright backgrounds, increasing positive confidence while keeping negative confidence unchanged and frame-time cost small.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..b3cd876 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -39,6 +39,13 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     left = (IMAGE_SIZE - resized_width) // 2
     resized[top : top + resized_height, left : left + resized_width] = resized_content
 
+    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
+    value = hsv[..., 2]
+    shadow_mask = value < 96.0
+    value[shadow_mask] = np.minimum(value[shadow_mask] + (96.0 - value[shadow_mask]) * 0.25, 112.0)
+    hsv[..., 2] = value
+    resized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
+
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0024-shadow-lift-dark-pixels\before-confidences.json` | 0.0456 | 75 | `experiment\attempts\0024-shadow-lift-dark-pixels\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0024-shadow-lift-dark-pixels\after-confidences.json` | 0.0496 | 75 | `experiment\attempts\0024-shadow-lift-dark-pixels\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.9022 | -0.0012 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0456 | 0.0496 | +0.0040 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0002_y.jpg` 0.9092 -> 0.9316 (+0.0224)

Largest negative frame movement: `v1_frames/frame_0013_y.jpg` 0.6926 -> 0.6710 (-0.0216)

Targeted shadow lift left negative confidence and classification unchanged, with no false positives or false negatives. However, positive confidence decreased by 0.0012 and mean frame processing time increased by 0.0040 seconds, so the change does not improve the accepted pipeline.

## Decision

Do not continue

Reason: The change lowers positive confidence and adds processing cost without improving negative confidence or classification. Revert this preprocessing change and keep the current accepted pipeline.
