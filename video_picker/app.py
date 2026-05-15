import os
import sys
import json
from pathlib import Path
from uuid import uuid4

import cv2
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QBrush, QColor, QCloseEvent, QFont, QImage, QPainter, QPen, QPixmap
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

        self._worker: QProcess | None = None
        self._worker_stdout_buf = ""
        self._job_id_to_output: dict[str, str] = {}
        self._queue: list[tuple[str, str]] = []  # (video_path, output_path)
        self._active_job_id: str | None = None
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

        self.add_queue_btn = QPushButton("Add videos to queue…")
        self.add_queue_btn.clicked.connect(self._add_videos_to_queue)  # type: ignore[arg-type]

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        # "Batch size" now controls ORT CPU cores/threads.
        cpu = os.cpu_count() or 4
        self.batch_spin.setValue(max(1, cpu - 2))

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)

        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_PATH))
        self.output_edit.setReadOnly(True)

        self.start_btn = QPushButton("Start queue")
        self.start_btn.clicked.connect(self._start_queue)  # type: ignore[arg-type]

        self.stop_btn = QPushButton("Stop worker")
        self.stop_btn.clicked.connect(self._stop_worker)  # type: ignore[arg-type]
        self.stop_btn.setEnabled(False)

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
        layout.addWidget(self.add_queue_btn)
        layout.addSpacing(8)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status)
        self.setLayout(layout)

        self._set_status("Add videos to the queue, then click Start queue.")

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
            "Videos (*.mp4 *.MP4 *.mkv *.MKV *.mov *.MOV *.avi *.AVI *.webm *.WEBM *.m4v *.M4V *.mpeg *.MPEG *.mpg *.MPG);;All files (*)",
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

    def _add_videos_to_queue(self) -> None:
        start_dir = (
            os.path.dirname(self.path_edit.text())
            if self.path_edit.text()
            else os.path.expanduser("~")
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            start_dir,
            "Videos (*.mp4 *.MP4 *.mkv *.MKV *.mov *.MOV *.avi *.AVI *.webm *.WEBM *.m4v *.M4V *.mpeg *.MPEG *.mpg *.MPG);;All files (*)",
        )
        if not paths:
            self._set_status("No videos added.")
            return

        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            out = str(Path.cwd() / f"{base}_output.json")
            self._queue.append((p, out))

        self._set_status(f"Queued {len(paths)} videos (total queued={len(self._queue)}).")

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

        self.stop_btn.setEnabled(True)

        init_msg = {
            "type": "init",
            "md_model_path": os.environ.get("MEGADETECTOR_MODEL_PATH", "./models/md_v5a_1_3_640_640_static.onnx"),
            "species_model_path": os.environ.get("SPECIESNET_MODEL_PATH", "./models/spicesNet_v401a.onnx"),
            "species_labels_path": os.environ.get("SPECIESNET_LABELS_PATH", "./static/spicesNet_labels_v401a.txtset"),
            "cpu_cores": int(self.batch_spin.value()),
            "confidence": float(self.conf_spin.value()),
            "frames_per_batch": int(os.environ.get("MEGADETECTOR_FRAMES_PER_BATCH", "8")),
        }
        self._worker.write((json.dumps(init_msg) + "\n").encode("utf-8"))

    def _start_queue(self) -> None:
        # If user hasn't used the queue UI, treat the single selected video as a 1-item queue.
        if not self._queue:
            video_path = self.path_edit.text().strip()
            if video_path:
                base = os.path.splitext(os.path.basename(video_path))[0]
                out = str(Path.cwd() / f"{base}_output.json")
                self._queue.append((video_path, out))

        if not self._queue:
            self._set_status("Queue is empty. Add videos first.")
            return

        self._ensure_worker_started()

        # Enqueue everything currently in the queue to the worker.
        # Worker processes sequentially.
        for video_path, output_path in list(self._queue):
            job_id = str(uuid4())
            self._job_id_to_output[job_id] = output_path
            msg = {
                "type": "enqueue",
                "job_id": job_id,
                "video_path": video_path,
                "output_path": output_path,
            }
            assert self._worker is not None
            self._worker.write((json.dumps(msg) + "\n").encode("utf-8"))

        self.start_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.add_queue_btn.setEnabled(False)
        self._set_status(f"Enqueued {len(self._queue)} jobs to worker…")
        self._queue.clear()

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
            outp = str(msg.get("output_path", self._job_id_to_output.get(jid, DEFAULT_OUTPUT_PATH)))
            self._active_job_id = None
            elapsed = msg.get("elapsed_seconds", None)
            self._set_status(f"Done job {jid} ({elapsed}s). Loading {outp}…")
            self._load_results_and_show_first(output_path=Path(outp))
            self.start_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.add_queue_btn.setEnabled(True)
            return
        if t == "job_failed":
            jid = str(msg.get("job_id", ""))
            err = str(msg.get("error", "unknown error"))
            self._active_job_id = None
            self._set_status(f"Job {jid} failed: {err}")
            self.start_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.add_queue_btn.setEnabled(True)
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
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.add_queue_btn.setEnabled(True)
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

    def _load_results_and_show_first(self, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
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
            self._set_status(f"Failed to read {output_path}: {e}")
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
 
