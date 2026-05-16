import os
import sys
import json
from pathlib import Path
from uuid import uuid4
from typing import Any

import cv2
from PyQt6.QtCore import QProcess, Qt, QModelIndex, QDir, QSortFilterProxyModel
from PyQt6.QtGui import QBrush, QColor, QCloseEvent, QFont, QImage, QPainter, QPen, QPixmap, QFileSystemModel
from PyQt6.QtCharts import QChart, QChartView, QPieSeries, QPieSlice
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
    QTreeView,
    QSplitter,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QSizePolicy,
)
from .database import DetectionDatabase


DEFAULT_VIDEO_PATH = Path("/home/slava/Videos/v1.AVI")


class SettingsDialog(QDialog):
    def __init__(self, parent=None, cpu_cores=4, confidence=0.5):
        super().__init__(parent)
        self.setWindowTitle("Detection Settings")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(cpu_cores)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setValue(confidence)
        
        form.addRow("CPU cores:", self.batch_spin)
        form.addRow("Confidence:", self.conf_spin)
        layout.addLayout(form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_values(self) -> tuple[int, float]:
        return self.batch_spin.value(), self.conf_spin.value()


class FolderTreeModel(QFileSystemModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stats_cache: dict[str, dict] = {}  # path -> {species_str, unique_count}

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 4

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if section == 0: return "Name"
            if section == 1: return "Size"
            if section == 2: return "Animals Found"
            if section == 3: return "Unique Species"
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid(): return None
        if role == Qt.ItemDataRole.DisplayRole:
            path = self.filePath(index)
            if index.column() == 0: return super().data(index, role)
            if index.column() == 1: return super().data(self.index(index.row(), 1, index.parent()), role)
            if index.column() == 2:
                if not os.path.isfile(path): return ""
                return self.stats_cache.get(path, {}).get("species_str", "")
            if index.column() == 3:
                if not os.path.isfile(path): return ""
                count = self.stats_cache.get(path, {}).get("unique_count")
                return str(count) if count is not None else ""
        return None

    def update_stats_info(self, video_path: str, species_str: str, unique_count: int) -> None:
        self.stats_cache[video_path] = {
            "species_str": species_str,
            "unique_count": unique_count
        }
        idx = self.index(video_path)
        if idx.isValid():
            # Trigger data changed for the entire row's added columns
            p = idx.parent()
            r = idx.row()
            start_idx = self.index(r, self.columnCount() - 2, p)
            end_idx = self.index(r, self.columnCount() - 1, p)
            self.dataChanged.emit(start_idx, end_idx)


class StatisticsWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.series = QPieSeries()
        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.setTitle("Species Distribution (Folder-wide)")
        self.chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        self.chart.legend().setAlignment(Qt.AlignmentFlag.AlignRight)

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.layout.addWidget(self.chart_view)
        
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_stats(self, species_counts: dict[str, int]) -> None:
        self.series.clear()
        for species, count in species_counts.items():
            self.series.append(f"{species} ({count})", count)
        
        # Make slices more visible
        for slice in self.series.slices():
            slice.setLabelVisible(True)


class VideoPicker(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Picker Dashboard")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(800)

        # 2x bigger UI font
        f = self.font()
        if f.pointSizeF() > 0:
            f.setPointSizeF(f.pointSizeF() * 1.5)
        else:
            f.setPixelSize(max(16, int(f.pixelSize() * 1.5)))
        self.setFont(f)

        self._worker: QProcess | None = None
        self._worker_stdout_buf = ""
        self._queue: list[tuple[str, str]] = []
        self._active_job_id: str | None = None
        self._frames: list[dict] = []
        self._frame_i: int = 0
        self._cap: cv2.VideoCapture | None = None
        self._cap_path: str | None = None
        self._current_folder: str | None = None
        self._all_species_counts: dict[str, int] = {}
        self._db: DetectionDatabase | None = None
        
        # Default settings
        self._cpu_cores = max(1, (os.cpu_count() or 4) - 2)
        self._confidence = 0.5

        self.stack = QStackedWidget()
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.stack)

        self._setup_landing_page()
        self._setup_dashboard()

    def _setup_landing_page(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()

        title = QLabel("Welcome to Video Picker")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel("Select a folder containing videos to start detection.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        self.open_folder_btn = QPushButton("Open Folder…")
        self.open_folder_btn.setFixedSize(200, 60)
        self.open_folder_btn.clicked.connect(self._open_folder)  # type: ignore[arg-type]
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.stack.addWidget(page)

    def _setup_dashboard(self) -> None:
        self.dashboard = QWidget()
        outer_layout = QVBoxLayout(self.dashboard)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Header (Web Pattern)
        header = QFrame()
        header.setStyleSheet("background-color: #222; color: #eee; border-bottom: 1px solid #444;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)

        self.back_btn = QPushButton("← Back to Landing")
        self.back_btn.setStyleSheet("background-color: transparent; border: 1px solid #666; padding: 5px 15px;")
        self.back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        header_layout.addWidget(self.back_btn)
        
        header_layout.addStretch()
        
        self.header_title = QLabel("Dashboard")
        self.header_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        header_layout.addWidget(self.header_title)
        
        header_layout.addStretch()

        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setStyleSheet("background-color: transparent; border: 1px solid #666; padding: 5px 15px;")
        self.settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(self.settings_btn)

        self.process_all_btn = QPushButton("Process All Videos")
        self.process_all_btn.setStyleSheet("background-color: #2d5a27; color: white; padding: 5px 15px; font-weight: bold;")
        self.process_all_btn.clicked.connect(self._process_all)  # type: ignore[arg-type]
        header_layout.addWidget(self.process_all_btn)

        outer_layout.addWidget(header, 0)

        # Content Area with Horizontal Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(main_splitter, 1)

        # Left Sidebar: Tree View
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)

        self.tree_model = FolderTreeModel()
        self.tree_model.setFilter(QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.tree_model.setNameFilters(["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mkv", "*.MKV", "*.mov", "*.MOV"])
        self.tree_model.setNameFilterDisables(False)

        # Proxy model for sorting
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.tree_model)
        self.proxy_model.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.proxy_model)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.header().setStretchLastSection(True)
        self.tree_view.clicked.connect(self._on_tree_item_clicked)  # type: ignore[arg-type]

        sidebar_layout.addWidget(QLabel("Folder Contents:"))
        sidebar_layout.addWidget(self.tree_view)

        # Right Area: Preview & Statistics with Vertical Splitter
        right_area_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(5, 5, 5, 5)

        # Compact video info bar
        info_bar = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Select a video...")
        self.path_edit.setStyleSheet("background-color: #333; color: #ccc; border: none; padding: 2px;")
        
        info_bar.addWidget(QLabel("Video:"))
        info_bar.addWidget(self.path_edit)
        preview_layout.addLayout(info_bar)
        
        self.image_label = QLabel()
        self.image_label.setMinimumHeight(300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #000; border: 1px solid #333;")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self._prev_frame)  # type: ignore[arg-type]
        self.next_btn.clicked.connect(self._next_frame)  # type: ignore[arg-type]
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)

        self._show_zero_state()

        preview_layout.addWidget(self.image_label, 1) # Set stretch factor to 1
        preview_layout.addLayout(nav)

        # Bottom: Stats & Controls
        stats_widget_container = QWidget()
        stats_layout = QVBoxLayout(stats_widget_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        self.stats_view = StatisticsWidget()
        stats_layout.addWidget(self.stats_view)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        stats_layout.addWidget(self.status)

        right_area_splitter.addWidget(preview_widget)
        right_area_splitter.addWidget(stats_widget_container)
        right_area_splitter.setStretchFactor(0, 2)
        right_area_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(sidebar)
        main_splitter.addWidget(right_area_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        self.stack.addWidget(self.dashboard)


    def _show_zero_state(self, message: str = "Select a video from the list to view detections.") -> None:
        self.image_label.setText(f"<div style='color: #888; font-size: 18px;'>{message}</div>")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.path_edit.setText("")

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        
        self._current_folder = folder
        self._db = DetectionDatabase(str(Path(folder) / ".detections.db"))
        self.header_title.setText(f"Dashboard: {os.path.basename(folder)}")

        self.tree_model.setRootPath(folder)
        self.tree_view.setRootIndex(self.proxy_model.mapFromSource(self.tree_model.index(folder)))
        self.stack.setCurrentIndex(1)
        self._show_zero_state()
        self._scan_and_update_all()

    def _scan_and_update_all(self) -> None:
        """Scan SQLite database for existing detections and update tree/stats."""
        if not self._db or not self._current_folder:
            return

        # Folder-wide stats from SQLite
        self._all_species_counts = self._db.get_all_species_counts()
        self.stats_view.update_stats(self._all_species_counts)

        # Update TreeView column for all videos in folder
        folder_path = Path(self._current_folder)
        for v_path in folder_path.glob("*"):
            if v_path.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
                stats = self._db.get_video_stats(str(v_path.absolute()))
                if stats:
                    species_str = ", ".join(stats["top_species"])
                    self.tree_model.update_stats_info(
                        str(v_path.absolute()), 
                        species_str, 
                        stats["unique_species_count"]
                    )

    def _on_tree_item_clicked(self, index: QModelIndex) -> None:
        source_index = self.proxy_model.mapToSource(index)
        path = self.tree_model.filePath(source_index)
        if not os.path.isfile(path):
            return
        
        self.path_edit.setText(path)
        
        if self._db:
            frames = self._db.get_video_frames(str(Path(path).absolute()))
            if frames:
                self._frames = frames
                self._frame_i = 0
                self._update_nav_enabled()
                self._render_current_frame()
                self._set_status(f"Loaded {len(frames)} detection frames from database.")
            else:
                self._close_cap()
                self._frames = []
                self._frame_i = 0
                self._update_nav_enabled()
                self.image_label.setText("<div style='color: #888; font-size: 18px;'>Video not processed yet.</div>")
                self._set_status("Selected unprocessed video.")

    def _process_all(self) -> None:
        if not self._current_folder or not self._db:
            return
        
        folder_path = Path(self._current_folder)
        videos_to_process = []
        
        # Simple glob for supported videos
        for ext in ["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mkv", "*.MKV", "*.mov", "*.MOV"]:
            for v_path in folder_path.glob(ext):
                abs_v_path = str(v_path.absolute())
                # Check if video is in DB
                stats = self._db.get_video_stats(abs_v_path)
                if not stats:
                    videos_to_process.append(abs_v_path)

        if not videos_to_process:
            self._set_status("All videos in this folder are already processed.")
            return

        self.process_all_btn.setEnabled(False)
        self._ensure_worker_started()
        
        for video_path in videos_to_process:
            job_id = str(uuid4())
            msg = {
                "type": "enqueue",
                "job_id": job_id,
                "video_path": video_path,
                "output_path": None, # Disable JSON output
            }
            assert self._worker is not None
            self._worker.write((json.dumps(msg) + "\n").encode("utf-8"))

        self._set_status(f"Enqueued {len(videos_to_process)} jobs to worker…")

    def _set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def _on_worker_stdout(self) -> None:
        if self._worker is None:
            return
        text = bytes(self._worker.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self._worker_stdout_buf += text
        while "\n" in self._worker_stdout_buf:
            line, self._worker_stdout_buf = self._worker_stdout_buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            self._handle_worker_msg(msg)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self, self._cpu_cores, self._confidence)
        if dlg.exec():
            self._cpu_cores, self._confidence = dlg.get_values()
            self._set_status(f"Settings updated: CPU={self._cpu_cores}, Confidence={self._confidence:.3f}")
            # If worker is running, it won't see these until next init/restart. 
            # For now, we simple store them for the next _ensure_worker_started.

    def _ensure_worker_started(self) -> None:
        if self._worker is not None and self._worker.state() != QProcess.ProcessState.NotRunning:
            return

        self._worker = QProcess(self)
        self._worker.setProgram(sys.executable)
        self._worker.setArguments(["-m", "video_picker.worker"])
        self._worker.setWorkingDirectory(str(Path.cwd()))
        self._worker.readyReadStandardOutput.connect(self._on_worker_stdout)  # type: ignore[arg-type]
        self._worker.readyReadStandardError.connect(self._on_worker_stderr)  # type: ignore[arg-type]
        self._worker.finished.connect(self._on_worker_finished)  # type: ignore[arg-type]
        self._worker.start()

        md_model_path = os.environ.get("MEGADETECTOR_MODEL_PATH", "./models/md_v5a_1_3_640_640_static.onnx")
        species_model_path = os.environ.get("SPECIESNET_MODEL_PATH", "./models/spicesNet_v401a.onnx")
        species_labels_path = os.environ.get("SPECIESNET_LABELS_PATH", "./static/spicesNet_labels_v401a.txtset")

        if not os.path.exists(md_model_path):
             print(f"Warning: MegaDetector model not found at {md_model_path}")
        
        if not os.path.exists(species_model_path):
             print(f"Note: SpeciesNet model not found at {species_model_path}, disabling species classification.")
             species_model_path = ""

        init_msg = {
            "type": "init",
            "md_model_path": md_model_path,
            "species_model_path": species_model_path,
            "species_labels_path": species_labels_path,
            "db_path": self._db.db_path if self._db else "",
            "cpu_cores": self._cpu_cores,
            "confidence": self._confidence,
            "frames_per_batch": int(os.environ.get("MEGADETECTOR_FRAMES_PER_BATCH", "8")),
        }
        self._worker.write((json.dumps(init_msg) + "\n").encode("utf-8"))

    def _handle_worker_msg(self, msg: dict) -> None:
        # Log every event to stdout (so you can see full timeline).
        try:
            sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

        t = msg.get("type")
        if t == "ready":
            self._set_status("Worker ready (models preloaded).")
            return
        if t == "model_load_started":
            model = str(msg.get("model", "model"))
            mp = str(msg.get("model_path", ""))
            self._set_status(f"Loading {model}… {mp}")
            return
        if t == "model_load_finished":
            model = str(msg.get("model", "model"))
            ls = msg.get("load_seconds", None)
            self._set_status(f"Loaded {model} ({ls}s)")
            return
        if t == "job_started":
            self._active_job_id = str(msg.get("job_id", "")) or None
            self._set_status(f"Running job {self._active_job_id}…")
            return
        if t == "job_progress":
            jid = str(msg.get("job_id", ""))
            sf = int(msg.get("sampled_frames", 0) or 0)
            self._set_status(f"Job {jid}: sampled_frames={sf}")
            return
        if t == "job_finished":
            jid = str(msg.get("job_id", ""))
            self._active_job_id = None
            elapsed = msg.get("elapsed_seconds", None)
            self._set_status(f"Done job {jid} ({elapsed}s).")
            
            # Update tree and stats
            self._scan_and_update_all()
            
            # If the finished job is the one currently selected, reload from DB
            if self.path_edit.text():
                 frames = self._db.get_video_frames(str(Path(self.path_edit.text()).absolute())) if self._db else []
                 if frames:
                     self._frames = frames
                     self._frame_i = 0
                     self._update_nav_enabled()
                     self._render_current_frame()
            
            return
        if t == "job_failed":
            jid = str(msg.get("job_id", ""))
            err = str(msg.get("error", "unknown error"))
            self._active_job_id = None
            self._set_status(f"Job {jid} failed: {err}")
            return

    def _on_worker_stderr(self) -> None:
        if self._worker is None:
            return
        text = bytes(self._worker.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except Exception:
                pass

    def _on_worker_finished(self, exit_code: int, _status) -> None:
        self._set_status(f"Worker exited with code {exit_code}.")
        self.process_all_btn.setEnabled(True)
        self._worker = None
        self._worker_stdout_buf = ""
        self._active_job_id = None

    def _stop_worker(self) -> None:
        if self._worker is None:
            return
        try:
            self._worker.write((json.dumps({"type": "shutdown"}) + "\n").encode("utf-8"))
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API name)
        # Best-effort graceful shutdown of the always-on worker.
        try:
            self._close_cap()
        except Exception:
            pass

        w = self._worker
        if w is not None and w.state() != QProcess.ProcessState.NotRunning:
            try:
                w.write((json.dumps({"type": "shutdown"}) + "\n").encode("utf-8"))
                w.waitForBytesWritten(250)
            except Exception:
                pass

            if not w.waitForFinished(1500):
                try:
                    w.terminate()
                except Exception:
                    pass
                if not w.waitForFinished(1500):
                    try:
                        w.kill()
                    except Exception:
                        pass
                    w.waitForFinished(1500)

        self._worker = None
        self._worker_stdout_buf = ""
        self._active_job_id = None
        event.accept()

    def _update_nav_enabled(self) -> None:
        n = len(self._frames)
        self.prev_btn.setEnabled(n > 0 and self._frame_i > 0)
        self.next_btn.setEnabled(n > 0 and self._frame_i < n - 1)

    def _prev_frame(self) -> None:
        if self._frame_i <= 0:
            return
        self._frame_i -= 1
        self._update_nav_enabled()
        self._render_current_frame()

    def _next_frame(self) -> None:
        if self._frame_i >= len(self._frames) - 1:
            return
        self._frame_i += 1
        self._update_nav_enabled()
        self._render_current_frame()

    def _ensure_cap(self, video_path: str) -> cv2.VideoCapture | None:
        if self._cap is not None and self._cap_path == video_path and self._cap.isOpened():
            return self._cap
        self._close_cap()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        self._cap = cap
        self._cap_path = video_path
        return cap

    def _close_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._cap_path = None

    def _render_current_frame(self) -> None:
        if not self._frames:
            return
        video_path = self.path_edit.text().strip()
        if not video_path:
            self._set_status("No video path set.")
            return

        frame_entry = self._frames[self._frame_i]
        frame_index = int(frame_entry.get("frame_index", 0))
        dets = frame_entry.get("detections", [])
        if not isinstance(dets, list):
            dets = []

        cap = self._ensure_cap(video_path)
        if cap is None:
            self.image_label.setText("Failed to open video for preview")
            self._set_status(f"Failed to open video: {video_path}")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            self.image_label.setText("Failed to read frame")
            self._set_status(f"Failed to read frame_index={frame_index}")
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())  # detach from numpy buffer

        painter = QPainter(pix)
        pen = QPen(Qt.GlobalColor.green)
        pen.setWidth(6)
        painter.setPen(pen)

        # 2x bigger overlay font for bbox labels
        font = QFont(painter.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 2.0)
        else:
            font.setPixelSize(max(18, int(font.pixelSize() * 2)))
        painter.setFont(font)

        for d in dets:
            if not isinstance(d, dict):
                continue
            bbox = d.get("bbox_xyxy")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            conf = d.get("confidence", None)
            label_parts: list[str] = []
            if isinstance(conf, (int, float)):
                label_parts.append(f"{float(conf):.3f}")

            sn = d.get("speciesnet")
            if isinstance(sn, dict):
                cn = sn.get("class_name")
                pr = sn.get("probability")
                if isinstance(cn, str) and isinstance(pr, (int, float)):
                    label_parts.append(f"{cn} {float(pr):.3f}")

            conf_text = " | ".join(label_parts) if label_parts else None
            try:
                x1n, y1n, x2n, y2n = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except Exception:
                continue
            x1 = max(0.0, min(1.0, x1n)) * w
            y1 = max(0.0, min(1.0, y1n)) * h
            x2 = max(0.0, min(1.0, x2n)) * w
            y2 = max(0.0, min(1.0, y2n)) * h
            x = int(x1)
            y = int(y1)
            rw = int(max(1.0, x2 - x1))
            rh = int(max(1.0, y2 - y1))
            painter.drawRect(x, y, rw, rh)

            if conf_text is not None:
                fm = painter.fontMetrics()
                pad = 3
                tw = fm.horizontalAdvance(conf_text) + 2 * pad
                th = fm.height() + 2 * pad
                tx = x
                ty = max(0, y - th)

                painter.fillRect(tx, ty, tw, th, QBrush(QColor(0, 0, 0, 170)))
                painter.drawText(tx + pad, ty + th - pad - fm.descent(), conf_text)
        painter.end()

        # Fit to label size while keeping aspect ratio
        target = self.image_label.size()
        scaled = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)

        self._set_status(
            f"Frame {self._frame_i+1}/{len(self._frames)} (frame_index={frame_index}) | detections={len(dets)}"
        )


def main() -> None:
    print("Starting VideoPicker...", flush=True)
    app = QApplication(sys.argv)
    print("QApplication created", flush=True)
    w = VideoPicker()
    print("VideoPicker initialized", flush=True)
    w.show()
    print("VideoPicker window shown, entering event loop.", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
 
