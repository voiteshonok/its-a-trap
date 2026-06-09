# MegaDetector Experiment

## Hypothesis

Sharpening at the model input resolution after resize will improve detector-visible texture and increase MegaDetector confidence.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index 7a3f68f..b5858d7 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -27,6 +27,9 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
+    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
+    resized = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
+
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
     chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
     nchw = np.expand_dims(chw, axis=0)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0010-unsharp-after-resize\before-confidences.json` | 3.4733 | 75 | 21.5936 | `experiment\attempts\0010-unsharp-after-resize\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0010-unsharp-after-resize\after-confidences.json` | 3.5234 | 75 | 21.2865 | `experiment\attempts\0010-unsharp-after-resize\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8964 | 0.9000 | +0.0036 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 3.4733 | 3.5234 | +0.0501 |
| FPS | 21.5936 | 21.2865 | -0.3070 |
| Labeled frames | 75 | 75 | +0 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v4_frames/frame_0004_y.jpg` 0.7670 -> 0.8597 (+0.0928)

Largest negative frame movement: `v3_frames/frame_0013_y.jpg` 0.8791 -> 0.8457 (-0.0334)

Sharpening after resize improved the main confidence metric without hurting classification quality. Mean confidence increased by 0.0025, with the largest positive frame movement (+0.0928) notably larger than the largest negative movement (-0.0334). False positives stayed at 0, false negatives stayed at 0, and correct hits stayed at 100%. Runtime regressed only slightly by 0.0501 seconds, reducing throughput from 21.5936 FPS to 21.2865 FPS.

## Decision

Continue

Reason: This is a small but clean confidence improvement with no classification regression and only minor processing overhead. It is worth testing on a larger or harder sample.
