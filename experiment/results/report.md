# Preprocess Influence Report

Compares the original baseline `preprocess_bgr_to_md_input` pipeline against the current
`video_picker.megadetector_video.preprocess_bgr_to_md_input` implementation on the same
still-image dataset.

## Setup

| Setting | Value |
| --- | --- |
| Data directory | `experiment\results\data` |
| Model | `models\md_v5a_1_3_640_640_static.onnx` |
| Detection threshold | 0.50 |
| Frames processed | 232 |
| Positive frames (`*_y`) | 210 |
| Negative frames (`*_n`) | 22 |
| Before runtime (s) | 6.405 |
| After runtime (s) | 6.200 |

## Label Convention

- Filename ending in `_y` means the frame should be detected as **positive** (animal present, confidence >= 0.50).
- Filename ending in `_n` means the frame should be detected as **negative** (no animal, confidence < 0.50).

## Preprocess Versions

**Before (baseline):** pre-resize `GaussianBlur(3, 3)`, stretch resize to 640x640.

**After (current app):** mild pre-resize Gaussian blur (`sigma=0.8`), stretch resize to 640x640.

Both runs use `video_picker` ONNX inference and `megadetector_post_processing`.

## Results

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Percentage correct positive | 96.67% | 97.14% | +0.48pp |
| Mean confidence of correct positive | 0.9035 | 0.9007 | -0.0027 |
| STD confidence of correct positive | 0.0808 | 0.0854 | +0.0046 |
| Percentage correct negative | 100.00% | 100.00% | +0.00pp |
| Mean confidence of correct negative | 0.0000 | 0.0000 | +0.0000 |
| STD confidence of correct negative | 0.0000 | 0.0000 | +0.0000 |

## Misclassified Frames

### Before

False negatives (`*_y` predicted below threshold):

- `v11_frames/frame_0009_y.jpg`
- `v13_frames/frame_0005_y.jpg`
- `v13_frames/frame_0006_y.jpg`
- `v13_frames/frame_0007_y.jpg`
- `v16_frames/frame_0006_y.jpg`
- `v18_frames/frame_0005_y.jpg`
- `v18_frames/frame_0006_y.jpg`

False positives (`*_n` predicted at or above threshold):

None

### After

False negatives (`*_y` predicted below threshold):

- `v11_frames/frame_0009_y.jpg`
- `v13_frames/frame_0005_y.jpg`
- `v13_frames/frame_0006_y.jpg`
- `v13_frames/frame_0007_y.jpg`
- `v18_frames/frame_0005_y.jpg`
- `v18_frames/frame_0006_y.jpg`

False positives (`*_n` predicted at or above threshold):

None

## Notes

- Labels are inferred from `*_y` / `*_n` filename suffixes unless `--labels` is provided.
- Correct positive means an animal frame predicted above threshold.
- Correct negative means an empty frame predicted below threshold.
- Confidence metrics are computed only on correctly classified frames in each class.
