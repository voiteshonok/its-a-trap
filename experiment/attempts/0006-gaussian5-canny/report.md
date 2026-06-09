# MegaDetector Experiment

## Hypothesis

Applying a 5x5 Gaussian blur followed by Canny edge detection before resizing should emphasize animal outlines and improve MegaDetector confidence compared with the no-blur RGB input.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index 7a3f68f..a11e1ea 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,6 +26,10 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
+    edges = cv2.Canny(blurred, 100, 200)
+    bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
+
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
     chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0006-gaussian5-canny\before-confidences.json` | 3.6177 | 75 | 20.7315 | `experiment\attempts\0006-gaussian5-canny\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0006-gaussian5-canny\after-confidences.json` | 4.1550 | 75 | 18.0504 | `experiment\attempts\0006-gaussian5-canny\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8964 | 0.1269 | -0.7695 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 3.6177 | 4.1550 | +0.5374 |
| FPS | 20.7315 | 18.0504 | -2.6811 |
| False positives | n/a | n/a | n/a |
| False negatives | n/a | n/a | n/a |
| Correct hits | n/a | n/a | n/a |

## Analysis

Largest positive frame movement: `v4_frames/frame_0004.jpg` 0.7670 -> 0.9045 (+0.1375)

Largest negative frame movement: `v2_frames/frame_0001.jpg` 0.9675 -> 0.0000 (-0.9675)

Feeding Canny edges into the model caused a severe confidence regression. Mean confidence dropped from 0.6335 to 0.0897, a -0.5438 change across 75 frames. Frame-level movement was overwhelmingly negative: 52 frames decreased, 22 were unchanged, and only 1 increased.

The threshold impact was also severe. Forty-five frames crossed from positive to negative at the 0.5 threshold, no frames crossed from negative to positive, and only 8 frames remained positive after Canny preprocessing. Sixty-seven frames had an after confidence of exactly 0.0.

Runtime also regressed: 3.6177s before and 4.1550s after, with throughput dropping from 20.7315 FPS to 18.0504 FPS. False positives, false negatives, and correct hits were not measured because no ground-truth labels JSON was provided.

## Decision

Do not continue

Reason: Canny edge input removes too much of the natural image signal MegaDetector expects, causing most detections to disappear and making inference slower. This change should be rejected.
