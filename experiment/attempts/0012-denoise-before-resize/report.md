# MegaDetector Experiment

## Hypothesis

A small bilateral denoise before resize will remove sensor or compression noise, increasing Mean confidence positive (_y) and lowering Mean confidence negative (_n), especially on low-light frames.

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

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0012-denoise-before-resize\before-confidences.json` | 0.0462 | 75 | `experiment\attempts\0012-denoise-before-resize\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0012-denoise-before-resize\after-confidences.json` | 0.0500 | 75 | `experiment\attempts\0012-denoise-before-resize\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9000 | 0.9019 | +0.0018 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0462 | 0.0500 | +0.0038 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v3_frames/frame_0013_y.jpg` 0.8457 -> 0.8947 (+0.0489)

Largest negative frame movement: `v2_frames/frame_0002_y.jpg` 0.7132 -> 0.6933 (-0.0199)

The bilateral denoise improved `Mean confidence positive (_y)` slightly, from 0.9000 to 0.9019 (+0.0018), but it did not lower `Mean confidence negative (_n)`, which stayed at 0.0000. Classification quality was unchanged: 0 false positives, 0 false negatives, and 100% correct hits. Mean frame processing time regressed from 0.0462s to 0.0500s (+0.0038s per frame), which is too much overhead for the small confidence gain.

## Decision

Do not continue

Reason: The core `_y` metric improved only slightly, `_n` did not improve, and mean frame processing time got meaningfully slower. Keep the previous after-resize unsharp mask, but do not add the bilateral denoise step.
