# MegaDetector Experiment

## Hypothesis

Applying very mild CLAHE to the resized LAB luminance channel after letterbox and before unsharp will improve local animal contrast without the larger cost and regressions seen from pre-resize CLAHE.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..bb558e1 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -39,6 +39,15 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     left = (IMAGE_SIZE - resized_width) // 2
     resized[top : top + resized_height, left : left + resized_width] = resized_content
 
+    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
+    l_channel, a_channel, b_channel = cv2.split(lab)
+    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
+    enhanced_l = clahe.apply(l_channel)
+    resized = cv2.cvtColor(
+        cv2.merge((enhanced_l, a_channel, b_channel)),
+        cv2.COLOR_LAB2BGR,
+    )
+
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0023-mild-local-contrast-resized-luminance\before-confidences.json` | 0.0456 | 75 | `experiment\attempts\0023-mild-local-contrast-resized-luminance\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0023-mild-local-contrast-resized-luminance\after-confidences.json` | 0.0480 | 75 | `experiment\attempts\0023-mild-local-contrast-resized-luminance\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8889 | -0.0145 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0456 | 0.0480 | +0.0023 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 1 | +1 |
| Correct hit count | 75 | 74 | -1 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 1.89% | +1.89pp |
| Correct hits | 100.00% | 98.67% | -1.33pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0002_y.jpg` 0.9092 -> 0.9231 (+0.0139)

Largest negative frame movement: `v1_frames/frame_0014_y.jpg` 0.5317 -> 0.0000 (-0.5317)

Mild local contrast reduced positive confidence by 0.0145 while leaving negative confidence unchanged. It also added one false negative, reduced correct hits by 1.33 percentage points, and slightly increased mean frame processing time, so it fails the confidence and classification goals.

## Decision

Do not continue

Reason: The change lowered positive confidence and introduced a false negative. Revert this preprocessing change and keep the current accepted pipeline.
