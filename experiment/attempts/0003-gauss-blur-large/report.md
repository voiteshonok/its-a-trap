# MegaDetector Experiment

## Hypothesis

Increasing the pre-resize Gaussian blur from a light 3x3 kernel to a heavy 31x31 kernel should smooth noise and improve MegaDetector confidence stability.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index 592c94f..ad24208 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,8 +26,8 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
-    # Light blur before resize (same idea as video_picker/megadetector_video if aligned there).
-    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
+    # Heavy blur before resize to test whether smoothing suppresses distracting texture.
+    bgr = cv2.GaussianBlur(bgr, (31, 31), 0)
 
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0003-gauss-blur-large\before-confidences.json` | 9.4095 | 75 | 7.9707 | `experiment\attempts\0003-gauss-blur-large\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0003-gauss-blur-large\after-confidences.json` | 4.0629 | 75 | 18.4599 | `experiment\attempts\0003-gauss-blur-large\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8948 | 0.7464 | -0.1484 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 9.4095 | 4.0629 | -5.3466 |
| FPS | 7.9707 | 18.4599 | +10.4892 |
| False positives | n/a | n/a | n/a |
| False negatives | n/a | n/a | n/a |
| Correct hits | n/a | n/a | n/a |

## Analysis

Largest positive frame movement: `v1_frames/frame_0006.jpg` 0.8934 -> 0.9101 (+0.0166)

Largest negative frame movement: `v3_frames/frame_0011.jpg` 0.8813 -> 0.0000 (-0.8813)

The heavy blur reduced mean confidence by 0.1048 across 75 frames. Frame-level movement was mostly negative: 52 frames decreased, 22 were unchanged, and only 1 increased. Six frames crossed from positive to negative at the 0.5 threshold, and no frames crossed from negative to positive. Seventeen frames dropped by at least 0.10 confidence.

Runtime improved from 9.4095s to 4.0629s, increasing throughput from 7.9707 FPS to 18.4599 FPS. That speed result should be treated cautiously because the ONNX session warm-up and provider initialization are included in the first run; the confidence regression is the more important signal.

False positives, false negatives, and correct hits were not measured because no ground-truth labels JSON was provided.

## Decision

Do not continue

Reason: The heavy 31x31 blur causes a clear confidence regression and flips six previously positive frames below threshold. Without labels, there is no evidence that those drops are desirable false-positive suppression.
