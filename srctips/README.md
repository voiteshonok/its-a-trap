# `run_md_over_data_frames.py`

Runs the MegaDetector ONNX model over every image under a directory tree and writes per-image **animal** confidence (max score among class-0 detections).

## Prerequisites

From the **repository root** (`its-a-trap/`):

```bash
uv venv
uv pip install -e .
```

Dependencies are listed in `pyproject.toml` (`opencv-python`, `onnxruntime`, `numpy`, …).

## Run

Default: read `data/`, model `models/md_v5a_1_3_640_640_static.onnx`, write **`./confidences.json`** in the current working directory as `{ "frame_0001.jpg": 0.91, ... }`.

```bash
uv run python srctips/run_md_over_data_frames.py
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `--data-dir DIR` | Root folder to scan (default: `data`) |
| `--model PATH` | ONNX path (default: `models/md_v5a_1_3_640_640_static.onnx`, or `MEGADETECTOR_MODEL_PATH`) |
| `-b N` | Batch size for inference (default: `16`, or `MEGADETECTOR_BATCH_SIZE`) |
| `-c FLOAT` | Pre-NMS score filter (default: `0.5`, or `MEGADETECTOR_CONFIDENCE`) |
| `-o PATH` | Output JSON (default: `confidences.json`; use `-` for stdout) |
| `--with-paths` | JSON array of `{ "path", "animal_confidence" }` instead of basename map |

### Environment

- `MEGADETECTOR_MODEL_PATH`, `MEGADETECTOR_CONFIDENCE`, `MEGADETECTOR_BATCH_SIZE`
- `MEGADETECTOR_LOG_LEVEL` (e.g. `DEBUG`)
- `MEGADETECTOR_ORT_INTRA_OP_NUM_THREADS`, `MEGADETECTOR_ORT_INTER_OP_NUM_THREADS`, or `MEGADETECTOR_ORT_USE_DEFAULT_THREADS=1`

### Example

```bash
cd /path/to/its-a-trap
uv run python srctips/run_md_over_data_frames.py --data-dir data -b 8 -c 0.5 -o confidences.json
```

If the same file name appears in different subfolders, the default map uses **basename only**—use `--with-paths` to keep full paths distinct.
