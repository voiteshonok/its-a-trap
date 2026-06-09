# MegaDetector Experiment

## Hypothesis

A mild HSV saturation and value boost will make animals stand out from vegetation or background, increasing Mean confidence positive (_y) and lowering Mean confidence negative (_n).

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..89d632b 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,6 +26,11 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
+    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
+    hsv[..., 2] = np.clip(hsv[..., 2] * 1.10, 0, 255)
+    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
+
     height, width = bgr.shape[:2]
     scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
     resized_width = max(1, int(round(width * scale)))
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0015-hsv-saturation-value-boost\before-confidences.json` | 0.0470 | 75 | `experiment\attempts\0015-hsv-saturation-value-boost\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0015-hsv-saturation-value-boost\after-confidences.json` | 0.0873 | 75 | `experiment\attempts\0015-hsv-saturation-value-boost\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.9045 | +0.0011 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0470 | 0.0873 | +0.0403 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v2_frames/frame_0002_y.jpg` 0.7030 -> 0.7363 (+0.0333)

Largest negative frame movement: `v5_frames/frame_0012_y.jpg` 0.8713 -> 0.8638 (-0.0075)

The HSV saturation/value boost improved `Mean confidence positive (_y)` slightly, from 0.9035 to 0.9045 (+0.0011), and `Mean confidence negative (_n)` stayed at 0.0000. Classification quality was unchanged: 0 false positives, 0 false negatives, and 100% correct hits. However, mean frame processing time regressed heavily from 0.0470s to 0.0873s (+0.0403s per frame), which is much too expensive for the small confidence gain.

## Decision

Do not continue

Reason: The `_y` improvement is small, `_n` does not improve, and mean frame processing time nearly doubles. Keep the accepted letterbox and unsharp steps, but do not add HSV saturation/value boost.
