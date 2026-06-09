# MegaDetector Experiment

## Hypothesis

A mild gray-world white balance with conservative per-channel gain limits will correct color casts enough to make animals stand out, increasing positive confidence without increasing negative confidence and with lower cost than HSV enhancement.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..f8614fe 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,6 +26,17 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    channel_means = bgr.reshape(-1, 3).mean(axis=0)
+    gray_mean = channel_means.mean()
+    gains = np.divide(
+        gray_mean,
+        channel_means,
+        out=np.ones_like(channel_means, dtype=np.float64),
+        where=channel_means > 1.0,
+    )
+    gains = np.clip(gains, 0.9, 1.1)
+    bgr = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
+
     height, width = bgr.shape[:2]
     scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
     resized_width = max(1, int(round(width * scale)))
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0022-mild-gray-world-white-balance\before-confidences.json` | 0.0502 | 75 | `experiment\attempts\0022-mild-gray-world-white-balance\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0022-mild-gray-world-white-balance\after-confidences.json` | 0.1427 | 75 | `experiment\attempts\0022-mild-gray-world-white-balance\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.9036 | +0.0002 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0502 | 0.1427 | +0.0925 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v3_frames/frame_0013_y.jpg` 0.9427 -> 0.9449 (+0.0022)

Largest negative frame movement: `v4_frames/frame_0005_y.jpg` 0.8938 -> 0.8854 (-0.0084)

Mild gray-world white balance produced only a negligible positive-confidence gain of 0.0002 and left negative confidence unchanged. Classification stayed perfect, but mean frame processing time increased by 0.0925 seconds per frame, which is too expensive for such a small signal change.

## Decision

Do not continue

Reason: The tiny positive-confidence gain does not justify the large processing-time regression. Revert this preprocessing change and keep the current accepted pipeline.
