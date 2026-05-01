import os
import sys
import json
from pathlib import Path

import cv2
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
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
)


DEFAULT_VIDEO_PATH = Path("/home/slava/Videos/v1.AVI")
DEFAULT_OUTPUT_PATH = Path("output.json")


class VideoPicker(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video picker")
        self.setMinimumWidth(1000)

        # 2x bigger UI font
        f = self.font()
        if f.pointSizeF() > 0:
            f.setPointSizeF(f.pointSizeF() * 2.0)
        else:
            f.setPixelSize(max(18, int(f.pixelSize() * 2)))
        self.setFont(f)

        self._proc: QProcess | None = None
        self._frames: list[dict] = []
        self._frame_i: int = 0
        self._cap: cv2.VideoCapture | None = None
        self._cap_path: str | None = None

        title = QLabel("Select a video, then run MegaDetector")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("No file selected")
        if DEFAULT_VIDEO_PATH.exists():
            self.path_edit.setText(str(DEFAULT_VIDEO_PATH))

        self.select_btn = QPushButton("Select video…")
        self.select_btn.clicked.connect(self._select_video)  # type: ignore[arg-type]

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        # "Batch size" now controls ORT CPU cores/threads.
        cpu = os.cpu_count() or 4
        self.batch_spin.setValue(max(1, cpu - 1))

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)

        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_PATH))
        self.output_edit.setReadOnly(True)

        self.start_btn = QPushButton("Start processing")
        self.start_btn.clicked.connect(self._start_processing)  # type: ignore[arg-type]

        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self._prev_frame)  # type: ignore[arg-type]
        self.next_btn.clicked.connect(self._next_frame)  # type: ignore[arg-type]
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        self.image_label = QLabel()
        # 2x bigger preview area
        self.image_label.setMinimumHeight(400)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #333;")
        self.image_label.setText("No results loaded.")

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.addRow("CPU cores:", self.batch_spin)
        form.addRow("Confidence:", self.conf_spin)
        form.addRow("Output file:", self.output_edit)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addWidget(QLabel("Selected video path:"))
        layout.addWidget(self.path_edit)
        layout.addWidget(self.select_btn)
        layout.addSpacing(8)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(self.start_btn)
        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status)
        self.setLayout(layout)

        self._set_status("Pick a video (or use the default), then click Start processing.")

    def _set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def _select_video(self) -> None:
        start_dir = (
            os.path.dirname(self.path_edit.text())
            if self.path_edit.text()
            else os.path.expanduser("~")
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video file",
            start_dir,
            "Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mpeg *.mpg);;All files (*)",
        )
        if not path:
            self._set_status("Selection cancelled.")
            return

        self.path_edit.setText(path)
        self._close_cap()
        self._frames = []
        self._frame_i = 0
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.image_label.setText("No results loaded.")
        self._set_status("Video selected (in-memory).")

    def _start_processing(self) -> None:
        if self._proc is not None and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._set_status("Processing already running.")
            return

        video_path = self.path_edit.text().strip()
        if not video_path:
            self._set_status("Select a video first.")
            return

        output_path = str(DEFAULT_OUTPUT_PATH)
        batch = int(self.batch_spin.value())
        conf = float(self.conf_spin.value())

        args = [
            "-m",
            "video_picker.megadetector_video",
            video_path,
            "--output",
            output_path,
            "--batch-size",
            str(batch),
            "--confidence",
            str(conf),
        ]

        self._proc = QProcess(self)
        self._proc.setProgram(sys.executable)
        self._proc.setArguments(args)
        self._proc.setWorkingDirectory(str(Path.cwd()))

        self._proc.readyReadStandardOutput.connect(self._on_proc_stdout)  # type: ignore[arg-type]
        self._proc.readyReadStandardError.connect(self._on_proc_stderr)  # type: ignore[arg-type]
        self._proc.finished.connect(self._on_proc_finished)  # type: ignore[arg-type]

        self.start_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self._set_status(f"Running… (logs in terminal) writing ./{output_path}")
        self._proc.start()

    def _on_proc_stdout(self) -> None:
        if self._proc is None:
            return
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            # Forward logs to the terminal where the GUI was launched.
            try:
                sys.stdout.write(text)
                sys.stdout.flush()
            except Exception:
                pass

    def _on_proc_stderr(self) -> None:
        if self._proc is None:
            return
        text = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            # Forward logs to the terminal where the GUI was launched.
            try:
                sys.stderr.write(text)
                sys.stderr.flush()
            except Exception:
                pass

    def _on_proc_finished(self, exit_code: int, _status) -> None:
        self.start_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        if exit_code == 0:
            self._set_status(f"Done. Wrote ./{DEFAULT_OUTPUT_PATH}. Loading results…")
            self._load_results_and_show_first()
        else:
            self._set_status(f"Failed with exit code {exit_code}. See last message above.")

    def _load_results_and_show_first(self) -> None:
        try:
            with open(DEFAULT_OUTPUT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            frames = data.get("frames", [])
            if not isinstance(frames, list):
                raise ValueError("output.json: 'frames' is not a list")
            self._frames = [x for x in frames if isinstance(x, dict)]
            self._frame_i = 0
        except Exception as e:
            self._frames = []
            self._frame_i = 0
            self.image_label.setText("Failed to load output.json")
            self._set_status(f"Failed to read {DEFAULT_OUTPUT_PATH}: {e}")
            return

        if not self._frames:
            self.image_label.setText("No frames in output.json")
            self._set_status("Loaded output.json, but it contains 0 frames.")
            return

        self._update_nav_enabled()
        self._render_current_frame()

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
    app = QApplication(sys.argv)
    w = VideoPicker()
    w.show()
    raise SystemExit(app.exec())
 
