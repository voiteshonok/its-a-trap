# MegaDetector Experiment Results

## Current Accepted Pipeline

Accepted changes:

| Attempt | Change | Decision |
| --- | --- | --- |
| `0005-no-gaussian-blur` | Removed pre-resize Gaussian blur | Continue |
| `0010-unsharp-after-resize` | Added unsharp mask after resize | Continue |
| `0014-letterbox-resize` | Replaced forced resize with letterbox resize | Continue |

## Latest Metrics

| Metric | Original Baseline | Latest Accepted | Difference |
| --- | ---: | ---: | ---: |
| Mean confidence positive (_y) | 0.8948 | 0.9035 | +0.0087 |
| Mean confidence negative (_n) | 0.0000 | 0.0000 | +0.0000 |
| Mean frame processing seconds | 0.1255 | 0.0458 | -0.0797 |
| False positives | n/a | 0.00% | n/a |
| False negatives | n/a | 0.00% | n/a |
| Correct hits | n/a | 100.00% | n/a |

Original baseline source: `experiment/attempts/0003-gauss-blur-large/report.md` before metrics.

Latest accepted source: `experiment/attempts/0014-letterbox-resize/report.md` after metrics.
