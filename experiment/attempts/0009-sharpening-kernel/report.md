# MegaDetector Experiment

## Hypothesis

A mild unsharp mask before resize will strengthen edge detail around animal bodies and increase MegaDetector confidence.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index 7a3f68f..45d3af0 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,6 +26,9 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
+    bgr = cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)
+
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
     chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0009-sharpening-kernel\before-confidences.json` | 3.3039 | 75 | 22.7005 | `experiment\attempts\0009-sharpening-kernel\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0009-sharpening-kernel\after-confidences.json` | 3.7759 | 75 | 19.8630 | `experiment\attempts\0009-sharpening-kernel\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8964 | 0.8963 | -0.0002 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 3.3039 | 3.7759 | +0.4720 |
| FPS | 22.7005 | 19.8630 | -2.8375 |
| Labeled frames | 75 | 75 | +0 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v4_frames/frame_0004_y.jpg` 0.7670 -> 0.7965 (+0.0295)

Largest negative frame movement: `v1_frames/frame_0015_y.jpg` 0.5445 -> 0.5151 (-0.0294)

The mild unsharp mask did not improve the aggregate confidence metric. Mean confidence moved from 0.6335 to 0.6334, effectively flat but slightly negative. Classification quality stayed unchanged at threshold 0.5: 0 false positives, 0 false negatives, and 100% correct hits. The frame-level gains and losses were balanced, with the largest positive movement (+0.0295) almost exactly offset by the largest negative movement (-0.0294). Runtime regressed by 0.4720 seconds and throughput dropped from 22.7005 FPS to 19.8630 FPS.

## Decision

Do not continue

Reason: The sharpening step added processing cost without improving mean confidence or classification quality. It is not worth keeping in this form.
