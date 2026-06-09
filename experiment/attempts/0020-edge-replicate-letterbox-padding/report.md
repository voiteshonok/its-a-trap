# MegaDetector Experiment

## Hypothesis

Replicating the resized content edges into the letterbox padding will look less artificial than constant gray padding, improving positive confidence without increasing negative confidence or hurting classification.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..784e5a4 100644
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
+        cv2.BORDER_REPLICATE,
+    )
 
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0020-edge-replicate-letterbox-padding\before-confidences.json` | 0.0464 | 75 | `experiment\attempts\0020-edge-replicate-letterbox-padding\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0020-edge-replicate-letterbox-padding\after-confidences.json` | 0.0459 | 75 | `experiment\attempts\0020-edge-replicate-letterbox-padding\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8995 | -0.0039 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0464 | 0.0459 | -0.0005 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0015_y.jpg` 0.5233 -> 0.5665 (+0.0432)

Largest negative frame movement: `v4_frames/frame_0005_y.jpg` 0.8938 -> 0.8573 (-0.0365)

Edge-replicate padding kept negative confidence at 0.0000 and preserved classification quality with no false positives or false negatives. Mean frame processing time improved slightly by 0.0005 seconds, but positive confidence dropped by 0.0039, so the experiment does not meet the main goal.

## Decision

Do not continue

Reason: The change did not improve positive confidence, even though speed and classification stayed stable. Revert this preprocessing change and keep the existing constant letterbox padding.
