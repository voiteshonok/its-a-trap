# MegaDetector Experiment

## Hypothesis

Cropping a conservative bottom timestamp/metadata band before letterbox will remove distracting camera overlay text, giving the detector more useful image signal and increasing positive confidence without changing negative confidence or adding false negatives.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index b9d20e8..9dafb73 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,6 +26,9 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
+    crop_bottom = max(1, int(round(bgr.shape[0] * 0.08)))
+    bgr = bgr[: bgr.shape[0] - crop_bottom]
+
     height, width = bgr.shape[:2]
     scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
     resized_width = max(1, int(round(width * scale)))
```

## Runs

| Run | Command | Mean Frame Processing Seconds | Frames | Output |
| --- | --- | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0028-crop-bottom-timestamp-band\before-confidences.json` | 0.0497 | 75 | `experiment\attempts\0028-crop-bottom-timestamp-band\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0028-crop-bottom-timestamp-band\after-confidences.json` | 0.0456 | 75 | `experiment\attempts\0028-crop-bottom-timestamp-band\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.9035 | 0.8843 | -0.0192 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.0497 | 0.0456 | -0.0040 |
| False positive count | 0 | 0 | +0 |
| False negative count | 0 | 2 | +2 |
| Correct hit count | 75 | 73 | -2 |
| False positives | 0.00% | 0.00% | +0.00pp |
| False negatives | 0.00% | 3.77% | +3.77pp |
| Correct hits | 100.00% | 97.33% | -2.67pp |

## Analysis

Largest positive frame movement: `v1_frames/frame_0013_y.jpg` 0.6926 -> 0.7417 (+0.0490)

Largest negative frame movement: `v1_frames/frame_0014_y.jpg` 0.5317 -> 0.0000 (-0.5317)

Cropping the bottom timestamp band improved mean frame processing time by 0.0040 seconds, but it reduced positive confidence by 0.0192 and introduced two false negatives. Negative confidence stayed unchanged, so the crop removes useful scene signal or changes geometry enough to hurt positives.

## Decision

Do not continue

Reason: The speed gain does not offset the positive-confidence regression and false-negative increase. Revert this preprocessing change and keep the current accepted pipeline.
