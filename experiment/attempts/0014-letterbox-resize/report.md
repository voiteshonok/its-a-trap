# MegaDetector Experiment

## Hypothesis

Preserving aspect ratio with letterbox padding to 640x640 will avoid geometric distortion, increasing Mean confidence positive (_y) and lowering Mean confidence negative (_n) on elongated animals or non-square frames.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b5858d7..b9d20e8 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,7 +26,19 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
-    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
+    height, width = bgr.shape[:2]
+    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
+    resized_width = max(1, int(round(width * scale)))
+    resized_height = max(1, int(round(height * scale)))
+    resized_content = cv2.resize(
+        bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
+    )
+
+    resized = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=bgr.dtype)
+    top = (IMAGE_SIZE - resized_height) // 2
+    left = (IMAGE_SIZE - resized_width) // 2
+    resized[top : top + resized_height, left : left + resized_width] = resized_content
+
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0014-letterbox-resize\before-confidences.json` | 0.0450 | 75 | `experiment\attempts\0014-letterbox-resize\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0014-letterbox-resize\after-confidences.json` | 0.0458 | 75 | `experiment\attempts\0014-letterbox-resize\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9000 | 0.9035 | +0.0034 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0450 | 0.0458 | +0.0008 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v4_frames/frame_0005_y.jpg` 0.6486 -> 0.8938 (+0.2453)

Largest negative frame movement: `v1_frames/frame_0013_y.jpg` 0.7934 -> 0.6926 (-0.1007)

Letterbox resizing improved `Mean confidence positive (_y)` from 0.9000 to 0.9035 (+0.0034), while `Mean confidence negative (_n)` stayed at 0.0000. Classification quality did not regress: 0 false positives, 0 false negatives, and 100% correct hits. Mean frame processing time increased slightly from 0.0450s to 0.0458s (+0.0008s per frame), which is small relative to the positive-confidence gain. The largest positive movement (+0.2453) was larger than the largest negative movement (-0.1007).

## Decision

Continue

Reason: The core `_y` metric improved, `_n` did not worsen, classification quality stayed perfect, and the mean frame processing slowdown is small. Keep letterbox resizing and continue building on it.
