# MegaDetector Experiment

## Hypothesis

CLAHE on the LAB luminance channel will increase animal detection confidence for low-contrast or shadowed frames without over-amplifying background noise.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index 7a3f68f..d9b1639 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,6 +26,12 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
+    l_channel, a_channel, b_channel = cv2.split(lab)
+    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
+    enhanced_l = clahe.apply(l_channel)
+    bgr = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
+
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
     chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0007-clahe-luminance\before-confidences.json` | 3.4259 | 75 | 21.8923 | `experiment\attempts\0007-clahe-luminance\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0007-clahe-luminance\after-confidences.json` | 4.3886 | 75 | 17.0896 | `experiment\attempts\0007-clahe-luminance\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8964 | 0.8880 | -0.0084 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 3.4259 | 4.3886 | +0.9628 |
| FPS | 21.8923 | 17.0896 | -4.8027 |
| Labeled frames | 75 | 75 | +0 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 0 | +0 |
| Correct hit count | 75 | 75 | +0 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 0.00% | +0.00pp |
| Correct hits | 100.00% | 100.00% | +0.00pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0010_y.jpg` 0.8508 -> 0.8672 (+0.0164)

Largest negative frame movement: `v2_frames/frame_0002_y.jpg` 0.7208 -> 0.5917 (-0.1291)

CLAHE did not improve the main confidence metric on this sample. Mean confidence dropped by 0.0059, while the largest negative frame movement was much larger than the largest positive movement. Runtime also regressed by 0.9628 seconds, reducing throughput from 21.8923 FPS to 17.0896 FPS. Classification quality at threshold 0.5 stayed unchanged: 0 false positives, 0 false negatives, and 100% correct hits.

## Decision

Do not continue

Reason: CLAHE made processing slower and slightly reduced mean confidence without improving false positives, false negatives, or correct hits. It is not worth keeping in this form.
