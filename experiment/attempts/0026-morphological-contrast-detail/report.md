# MegaDetector Experiment

## Hypothesis

A low-weight LAB luminance top-hat and black-hat blend after letterbox will enhance animal texture without Canny-like artifacts, increasing positive confidence while keeping negative confidence and false negatives unchanged.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..ed01999 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -39,6 +39,18 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     left = (IMAGE_SIZE - resized_width) // 2
     resized[top : top + resized_height, left : left + resized_width] = resized_content
 
+    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
+    l_channel, a_channel, b_channel = cv2.split(lab)
+    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
+    top_hat = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, kernel)
+    black_hat = cv2.morphologyEx(l_channel, cv2.MORPH_BLACKHAT, kernel)
+    enhanced_l = np.clip(
+        l_channel.astype(np.float32) + top_hat.astype(np.float32) * 0.08 - black_hat.astype(np.float32) * 0.05,
+        0,
+        255,
+    ).astype(np.uint8)
+    resized = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
+
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0026-morphological-contrast-detail\before-confidences.json` | 0.0477 | 75 | `experiment\attempts\0026-morphological-contrast-detail\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0026-morphological-contrast-detail\after-confidences.json` | 0.0520 | 75 | `experiment\attempts\0026-morphological-contrast-detail\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8932 | -0.0103 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0477 | 0.0520 | +0.0043 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 1 | +1 |
| Correct hit count | 75 | 74 | -1 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 1.89% | +1.89pp |
| Correct hits | 100.00% | 98.67% | -1.33pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0010_y.jpg` 0.8234 -> 0.8366 (+0.0132)

Largest negative frame movement: `v1_frames/frame_0015_y.jpg` 0.5233 -> 0.0000 (-0.5233)

The morphology detail blend left negative confidence unchanged, but it reduced positive confidence by 0.0103 and introduced one false negative. Mean frame processing time also increased by 0.0043 seconds, so the added luminance texture harms the current accepted pipeline.

## Decision

Do not continue

Reason: The change lowers positive confidence, adds a false negative, and increases processing time. Revert this preprocessing change and keep the current accepted pipeline.
