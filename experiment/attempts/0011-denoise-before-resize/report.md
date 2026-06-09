# MegaDetector Experiment

## Hypothesis

A small bilateral denoise before resize will remove sensor or compression noise and improve MegaDetector confidence on true animal frames, especially low-light frames.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b5858d7..6936960 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,6 +26,8 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    bgr = cv2.bilateralFilter(bgr, 5, 50, 50)
+
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
     blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
     resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0011-denoise-before-resize\before-confidences.json` | 3.4288 | 75 | 21.8738 | `experiment\attempts\0011-denoise-before-resize\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0011-denoise-before-resize\after-confidences.json` | 3.7667 | 75 | 19.9116 | `experiment\attempts\0011-denoise-before-resize\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9000 | 0.9019 | +0.0018 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 3.4288 | 3.7667 | +0.3379 |
| FPS | 21.8738 | 19.9116 | -1.9622 |
| Labeled frames | 75 | 75 | +0 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v3_frames/frame_0013_y.jpg` 0.8457 -> 0.8947 (+0.0489)

Largest negative frame movement: `v2_frames/frame_0002_y.jpg` 0.7132 -> 0.6933 (-0.0199)

The bilateral denoise produced a small mean-confidence gain, from 0.6360 to 0.6373 (+0.0013), and did not change classification quality: 0 false positives, 0 false negatives, and 100% correct hits. The largest positive movement (+0.0489) was larger than the largest negative movement (-0.0199). However, the processing cost was meaningful: runtime increased by 0.3379 seconds and throughput dropped from 21.8738 FPS to 19.9116 FPS.

## Decision

Do not continue

Reason: The confidence gain is too small for the processing slowdown. Keep the previous after-resize unsharp mask, but do not add the bilateral denoise step.
