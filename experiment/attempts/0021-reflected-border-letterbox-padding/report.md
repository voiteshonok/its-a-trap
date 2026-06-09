# MegaDetector Experiment

## Hypothesis

Using reflected image content for letterbox padding will avoid the single-row smearing artifact from BORDER_REPLICATE while still reducing artificial constant-padding borders, improving positive confidence without increasing negative confidence.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..04c0f81 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -34,10 +34,18 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
         bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
     )
 
-    resized = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=bgr.dtype)
     top = (IMAGE_SIZE - resized_height) // 2
     left = (IMAGE_SIZE - resized_width) // 2
-    resized[top : top + resized_height, left : left + resized_width] = resized_content
+    bottom = IMAGE_SIZE - resized_height - top
+    right = IMAGE_SIZE - resized_width - left
+    resized = cv2.copyMakeBorder(
+        resized_content,
+        top,
+        bottom,
+        left,
+        right,
+        cv2.BORDER_REFLECT_101,
+    )
 
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0021-reflected-border-letterbox-padding\before-confidences.json` | 0.0472 | 75 | `experiment\attempts\0021-reflected-border-letterbox-padding\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0021-reflected-border-letterbox-padding\after-confidences.json` | 0.0471 | 75 | `experiment\attempts\0021-reflected-border-letterbox-padding\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8964 | -0.0070 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0472 | 0.0471 | -0.0001 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0015_y.jpg` 0.5233 -> 0.5741 (+0.0509)

Largest negative frame movement: `v4_frames/frame_0005_y.jpg` 0.8938 -> 0.6478 (-0.2461)

Reflected border padding avoids the severe one-row smear seen in the earlier `BORDER_REPLICATE` preview, and it keeps negative confidence, false positives, false negatives, and correct hits unchanged. However, positive confidence still drops by 0.0070, so this border-fill direction does not improve the detector signal compared with constant padding.

## Decision

Do not continue

Reason: The corrected border padding keeps speed and classification stable but reduces positive confidence. Revert this preprocessing change and keep the existing constant letterbox padding.
