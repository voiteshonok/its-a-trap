# MegaDetector Experiment

## Hypothesis

Applying the unsharp mask only inside the pasted content rectangle will avoid sharpening artificial letterbox padding edges, improving positive confidence or reducing negative confidence without classification regression and with small frame-time cost.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..c276f7c 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -39,8 +39,15 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     left = (IMAGE_SIZE - resized_width) // 2
     resized[top : top + resized_height, left : left + resized_width] = resized_content
 
-    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
-    resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
+    content = resized[top : top + resized_height, left : left + resized_width]
+    blurred = cv2.GaussianBlur(content, (0, 0), 1.0)
+    resized[top : top + resized_height, left : left + resized_width] = cv2.addWeighted(
+        content,
+        1.5,
+        blurred,
+        -0.5,
+        0,
+    )
 
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
     chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0027-padding-aware-unsharp-mask\before-confidences.json` | 0.0460 | 75 | `experiment\attempts\0027-padding-aware-unsharp-mask\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0027-padding-aware-unsharp-mask\after-confidences.json` | 0.0450 | 75 | `experiment\attempts\0027-padding-aware-unsharp-mask\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.9025 | -0.0010 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0460 | 0.0450 | -0.0010 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0011_y.jpg` 0.8533 -> 0.8546 (+0.0012)

Largest negative frame movement: `v1_frames/frame_0015_y.jpg` 0.5233 -> 0.5084 (-0.0149)

Masking unsharp to the content rectangle slightly improved mean frame processing time and preserved classification quality, with no false positives or false negatives. However, positive confidence decreased by 0.0010 while negative confidence stayed unchanged, so the faster path does not improve the detector signal.

## Decision

Do not continue

Reason: The change lowers positive confidence and does not improve negative confidence, despite a small speed gain. Revert this preprocessing change and keep the current accepted pipeline.
