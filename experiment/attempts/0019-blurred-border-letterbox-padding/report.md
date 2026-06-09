# MegaDetector Experiment

## Hypothesis

Using a stretched and heavily blurred copy of each frame as letterbox padding will reduce artificial constant-border artifacts while preserving aspect ratio, increasing positive confidence without increasing negative confidence or frame processing time much.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..d2f111f 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -34,7 +34,8 @@ def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
         bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
     )
 
-    resized = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=bgr.dtype)
+    resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
+    resized = cv2.GaussianBlur(resized, (0, 0), 24.0)
     top = (IMAGE_SIZE - resized_height) // 2
     left = (IMAGE_SIZE - resized_width) // 2
     resized[top : top + resized_height, left : left + resized_width] = resized_content
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0019-blurred-border-letterbox-padding\before-confidences.json` | 0.0488 | 75 | `experiment\attempts\0019-blurred-border-letterbox-padding\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0019-blurred-border-letterbox-padding\after-confidences.json` | 0.0912 | 75 | `experiment\attempts\0019-blurred-border-letterbox-padding\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8720 | -0.0314 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0488 | 0.0912 | +0.0424 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 2 | +2 |
| Correct hit count | 75 | 73 | -2 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 3.77% | +3.77pp |
| Correct hits | 100.00% | 97.33% | -2.67pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0002_y.jpg` 0.9092 -> 0.9116 (+0.0024)

Largest negative frame movement: `v1_frames/frame_0014_y.jpg` 0.5317 -> 0.0000 (-0.5317)

The blurred padding background reduced positive confidence by 0.0314 while leaving negative confidence unchanged. It also introduced 2 false negatives and increased mean frame processing time by 0.0424 seconds, so it fails both the confidence and speed goals.

## Decision

Do not continue

Reason: Positive confidence regressed, false negatives increased, and processing time nearly doubled. Revert this preprocessing change and keep the existing constant letterbox padding.
