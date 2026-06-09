# MegaDetector Experiment

## Hypothesis

Removing the pre-resize Gaussian blur should preserve sharper image details and improve MegaDetector confidence compared with the original 3x3 blur.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index 592c94f..7a3f68f 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,9 +26,6 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
-    # Light blur before resize (same idea as video_picker/megadetector_video if aligned there).
-    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
-
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
     chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0005-no-gaussian-blur\before-confidences.json` | 3.6252 | 75 | 20.6885 | `experiment\attempts\0005-no-gaussian-blur\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0005-no-gaussian-blur\after-confidences.json` | 3.3598 | 75 | 22.3226 | `experiment\attempts\0005-no-gaussian-blur\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8948 | 0.8964 | +0.0017 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 3.6252 | 3.3598 | -0.2654 |
| FPS | 20.6885 | 22.3226 | +1.6340 |
| False positives | n/a | n/a | n/a |
| False negatives | n/a | n/a | n/a |
| Correct hits | n/a | n/a | n/a |

## Analysis

Largest positive frame movement: `v4_frames/frame_0004.jpg` 0.7296 -> 0.7670 (+0.0374)

Largest negative frame movement: `v3_frames/frame_0013.jpg` 0.8951 -> 0.8791 (-0.0160)

Removing the Gaussian blur produced a small positive confidence movement: mean confidence increased from 0.6323 to 0.6335, a +0.0012 change across 75 frames. Frame-level movement was mildly favorable: 35 frames increased, 18 decreased, and 22 were unchanged.

No frames crossed the 0.5 threshold in either direction: 53 stayed positive and 22 stayed negative. The largest gain was +0.0374, while the largest drop was -0.0160. Five frames gained at least 0.01 confidence, and only one frame dropped by at least 0.01 confidence.

Runtime improved from 3.6252s to 3.3598s, with throughput increasing from 20.6885 FPS to 22.3226 FPS. False positives, false negatives, and correct hits were not measured because no ground-truth labels JSON was provided.

## Decision

Continue

Reason: Removing blur is the first blur-related change that improves mean confidence and throughput without causing any threshold-level regressions on this sample. The confidence gain is small, so the next step should be validating it on labeled frames before treating it as a final quality improvement.
