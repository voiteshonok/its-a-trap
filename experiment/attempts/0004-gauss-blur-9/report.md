# MegaDetector Experiment

## Hypothesis

Increasing the pre-resize Gaussian blur from a light 3x3 kernel to a moderate 9x9 kernel should smooth noise without destroying useful animal features.

## Change

```diff
diff --git a/scripts/run_md_over_data_frames.py b/scripts/run_md_over_data_frames.py
index 592c94f..f780d96 100644
--- a/scripts/run_md_over_data_frames.py
+++ b/scripts/run_md_over_data_frames.py
@@ -26,8 +26,8 @@ _DLL_DIRECTORY_HANDLES: List[Any] = []
 
 
 def preprocess_bgr_to_md_input(bgr: np.ndarray) -> np.ndarray:
-    # Light blur before resize (same idea as video_picker/megadetector_video if aligned there).
-    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
+    # Moderate blur before resize to test whether smoothing suppresses distracting texture.
+    bgr = cv2.GaussianBlur(bgr, (9, 9), 0)
 
     resized = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
```

## Runs

| Run | Command | Runtime Seconds | Frames | FPS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0004-gauss-blur-9\before-confidences.json` | 3.5414 | 75 | 21.1779 | `experiment\attempts\0004-gauss-blur-9\before-confidences.json` |
| After | `C:\Users\Vlad\cv\its-a-trap\.venv\Scripts\python.exe C:\Users\Vlad\cv\its-a-trap\scripts\run_md_over_data_frames.py --data-dir data --model models/md_v5a_1_3_640_640_static.onnx --batch-size 16 --confidence-threshold 0.5 --output C:\Users\Vlad\cv\its-a-trap\experiment\attempts\0004-gauss-blur-9\after-confidences.json` | 3.5106 | 75 | 21.3638 | `experiment\attempts\0004-gauss-blur-9\after-confidences.json` |

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8948 | 0.8871 | -0.0077 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Runtime seconds | 3.5414 | 3.5106 | -0.0308 |
| FPS | 21.1779 | 21.3638 | +0.1859 |
| False positives | n/a | n/a | n/a |
| False negatives | n/a | n/a | n/a |
| Correct hits | n/a | n/a | n/a |

## Analysis

Largest positive frame movement: `v2_frames/frame_0002.jpg` 0.7071 -> 0.7176 (+0.0105)

Largest negative frame movement: `v1_frames/frame_0014.jpg` 0.5752 -> 0.5241 (-0.0511)

The 9x9 blur caused a small overall confidence regression: mean confidence moved from 0.6323 to 0.6269, a -0.0054 change. Across 75 frames, 43 decreased, 10 increased, and 22 were unchanged. No frames crossed the 0.5 threshold in either direction: 53 stayed positive and 22 stayed negative.

The largest gain was small (+0.0105), while the largest drop was -0.0511. Sixteen frames dropped by at least 0.01 confidence, and only one frame gained at least 0.01 confidence.

Runtime was effectively flat: 3.5414s before and 3.5106s after, with throughput moving from 21.1779 FPS to 21.3638 FPS. False positives, false negatives, and correct hits were not measured because no ground-truth labels JSON was provided.

## Decision

Do not continue

Reason: The 9x9 blur does not improve the target confidence metric and provides no classification benefit at the 0.5 threshold. It is far less harmful than 31x31, but the result is still slightly negative without labeled evidence that the lowered scores are desirable.
