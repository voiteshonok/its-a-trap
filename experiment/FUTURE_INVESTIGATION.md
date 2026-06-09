# Future Investigation

Scope: only classic computer vision changes inside `preprocess_bgr_to_md_input`.

Current accepted baseline:

- Remove pre-resize Gaussian blur.
- Use letterbox resize to preserve aspect ratio.
- Apply unsharp mask after resize.

Current goal:

- Increase `Mean confidence positive (_y)`.
- Decrease `Mean confidence negative (_n)`.
- Avoid regressions in mean frame processing time, false positives, false negatives, and correct hits.

## Lessons From Attempts

| Pattern | Result | Future Direction |
| --- | --- | --- |
| Heavy or moderate pre-resize blur | Rejected: `_y` confidence dropped. | Avoid smoothing that removes animal texture. |
| No pre-resize blur | Accepted: `_y` improved and runtime improved. | Keep source detail before resize. |
| Unsharp before resize | Rejected: no useful gain and slower. | Prefer sharpening at model input resolution. |
| Unsharp after resize | Accepted: `_y` improved with small overhead. | Tune this carefully. |
| Letterbox resize | Accepted: `_y` improved and `_n` stayed flat. | Keep aspect-ratio preservation; tune padding. |
| CLAHE, HSV boost, gamma, denoise | Mostly tiny `_y` gains or regressions with speed cost. | Only retry in milder/local forms if they target a specific failure. |
| Canny/edge-only or edge blend | Rejected: `_y` collapsed or false negatives increased. | Avoid explicit edge injection into model input. |
| Resize interpolation alternatives | Faster, but `_y` regressed. | Keep `INTER_LINEAR` unless testing a hybrid. |
| Blurred, replicated, or reflected letterbox padding | Rejected: padding artifacts reduced `_y`; blurred padding was much slower and caused false negatives. | Keep constant `114` padding unless testing a very cheap neutral variant. |
| Post-letterbox color, contrast, denoise, and morphology filters | Rejected: most reduced `_y`; gray-world had a tiny gain but was too slow. | Avoid global transforms after letterbox; focus on removing known nuisance regions. |
| Padding-aware unsharp | Rejected: slightly faster, but `_y` dropped. | Keep current full-frame unsharp unless a new mask targets a real artifact. |

## Next Experiments

Remove each experiment from this list after it is finished, regardless of the decision.

| Priority | Experiment | Hypothesis | Implementation Sketch | Success Criteria |
| ---: | --- | --- | --- | --- |
| 1 | Neutralize bottom timestamp band before letterbox | Replacing only the overlay strip may reduce text artifacts without changing image geometry as much as cropping. | Detect or assume the bottom metadata band and fill it with a neutral value or local median before letterbox resize. | Higher `_y`, `_n` unchanged, no classification regression, small frame-time cost. |
| 2 | Crop black camera borders before letterbox | Removing black border/overlay margins before letterbox may give the animal more useful model-input pixels. | Detect near-black horizontal/vertical border bands from row/column luminance and crop them before aspect-ratio resize. | Higher `_y`, `_n` unchanged, no false negatives, low runtime overhead. |
| 3 | Content-area-only letterbox padding without overlay strip | If bottom overlays are cropped first, constant padding may work better with less non-animal text competing for detail. | Combine conservative bottom-strip crop with current constant `114` letterbox padding and existing unsharp settings. | Higher `_y` than current accepted pipeline, `_n` unchanged, frame time close to current. |
| 4 | Smaller input unsharp radius | Current sigma `1.0` may over-emphasize overlay text and padding transitions; a smaller radius may preserve animal texture with fewer artifacts. | Keep strength `1.5/-0.5` but test `cv2.GaussianBlur(..., 0.5)` or `0.6` after letterbox. | Higher or equal `_y`, `_n` unchanged, no false negatives, frame time not worse. |
| 5 | Mild content contrast normalization before unsharp | Normalizing only the pasted content area may reduce frame-to-frame exposure differences without changing constant padding. | On `resized[top:top+h, left:left+w]`, apply a cheap percentile clamp/stretch on luminance, then current unsharp. | Higher `_y`, `_n` unchanged, no classification regression, lower cost than CLAHE. |
