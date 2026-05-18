## Video picker (PyQt + uv)

Simple PyQt app that lets you pick a video file, choose **batch size** and **confidence**, then run MegaDetector and write results to `./output.json`.

### Setup (uv)

Create a virtual environment and install deps:

```bash
uv venv
uv pip install -e .
```

Place ONNX models in `models/` (not in git):

- `models/md_v5a_1_3_640_640_static.onnx`
- `models/spicesNet_v401a.onnx`

Labels file: `static/spicesNet_labels_v401a.txtset`

### Run

```bash
uv run video-picker
```

(Alternative)

```bash
uv run python -m video_picker
```

To run the batch script:

```bash
uv run python srctips/run_md_over_data_frames.py -b 8 -c 0.0
```

### Build (one-file installer)

Builds a single executable with models and static assets embedded (~800 MB). Run the build **on the target OS** (Linux/macOS/Windows wheels are platform-specific).

Prerequisites: [uv](https://docs.astral.sh/uv/), project `.venv`, and `models/` populated as above.

| OS | Command | Output |
|----|---------|--------|
| Linux | `./scripts/build_onefile.sh` | `dist/video-picker-linux` |
| macOS | `./scripts/build_onefile.sh` | `dist/video-picker-macos` |
| Windows | `.\scripts\build_onefile.ps1` | `dist\video-picker-windows.exe` |

Example (Linux):

```bash
uv venv
uv pip install -e .
./scripts/build_onefile.sh
./dist/video-picker-linux
```

The first launch may take a moment while PyInstaller extracts bundled files to a temp directory.
