# Video Picker Dashboard (PyQt6 + uv)

An animal detection and classification dashboard. It scans folders for videos, identifies animals using **MegaDetector**, and classifies species with **SpeciesNet**. Results are stored in a local SQLite database for fast browsing and statistics.

## 🚀 Key Features
- **Folder-based Workflow:** Open a directory and see all videos at once.
- **Batch Processing:** Detect animals across multiple videos in the background.
- **SQLite Storage:** Results are saved in a hidden `.detections.db` inside each processed folder.
- **Interactive Statistics:** Folder-wide species distribution charts using `PyQt6.QtCharts`.
- **Enhanced Sidebar:** Sortable table showing detected animals and unique species counts.

## 🛠️ Setup

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Download Models:**
   Place the following `.onnx` files into a `models/` directory in the project root:
   - `md_v5a_1_3_640_640_static.onnx` (MegaDetector)
   - `spicesNet_v401a.onnx` (SpeciesNet)

## 🏃 How to Run

Launch the dashboard with:
```bash
uv run python -m video_picker.app
```

## 📂 Data Management
- **Database:** The app creates a `.detections.db` file in the folder you open. This contains all bounding boxes, species names, and timestamps.
- **Exporting:** You can still run the legacy batch script for images if needed:
  ```bash
  uv run python srctips/run_md_over_data_frames.py --data-dir ./my_images -b 8
  ```

## 🧩 Requirements
- Python >= 3.9
- PyQt6 & PyQt6-Charts
- OpenCV
- ONNX Runtime
- NumPy
