"""Manual detector ROIs mapped as spatial features across scan bins."""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QProcess, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QSpinBox,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import catalogs, io, reflection_sum, roi_detection, roi_map
from .lifecycle import start_process, stop_process, stop_process_async
from .palette import _get_cmap
from .viewer import DetectorView, HeatmapView, _RectItem, _scalar_to_rgba

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
_PROGRESS_RE = re.compile(r"PROGRESS\s+(\d+)/(\d+)")


class ROIShapeWindow(QMainWindow):
    """Select detector ROIs and map their total counts across spatial bins."""

    bin_size_changed = pyqtSignal(int)

    def __init__(self, project_root=".", scan=None, bin_size=3, embedded=False):
        super().__init__()
        self.dm = DataManager(project_root, scan=scan)
        self.project_root = str(Path(project_root).resolve())
        self.scan = self.dm._scan(scan)
        self.bin_size = int(bin_size)
        self.embedded = embedded
        self.source = None
        self.image = None
        self.roi = None
        self.result_path = None
        self.preview_feature = None
        self.pending = []
        self.spatial_keys = []
        self.spatial_index = 0
        self.sum_process = None
        self._detect_process = None
        self._batch_process = None
        self._save_process = None
        self._output_buffers = {}
        self._sum_prompted = False

        self.setWindowTitle("ROI > Shape")
        self.resize(1500, 900)
        self._build_ui()
        self._populate_controls()
        self._load_saved_features()
        self._load_detector_image()

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        self._header = QWidget()
        h = QHBoxLayout(self._header)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("<b>Bin:</b>"))
        self.bin_combo = QComboBox()
        bins = sorted({1, 2, 3, 4, 5, self.bin_size})
        self.bin_combo.addItems([f"{n}x{n}" for n in bins])
        self.bin_combo.setCurrentText(f"{self.bin_size}x{self.bin_size}")
        self.bin_combo.currentTextChanged.connect(self._bin_changed)
        h.addWidget(self.bin_combo)
        h.addWidget(QLabel("<b>Detector image:</b>"))
        self.image_mode = QComboBox()
        self.image_mode.addItems(["Fully summed scan", "Selected spatial bin"])
        self.image_mode.currentIndexChanged.connect(self._image_mode_changed)
        h.addWidget(self.image_mode)
        self.compute_sum_btn = QPushButton("Compute grand sum")
        self.compute_sum_btn.setToolTip(
            "Sum every frame in the scan into the fully-summed detector image.\n"
            "Slow on a fresh project; build bins first for quick loading.")
        self.compute_sum_btn.clicked.connect(self._on_compute_btn)
        h.addWidget(self.compute_sum_btn)
        self.prev_bin_btn = QPushButton("< Bin")
        self.prev_bin_btn.clicked.connect(lambda: self._step_spatial_bin(-1))
        h.addWidget(self.prev_bin_btn)
        self.spatial_label = QLabel("bin -/-")
        h.addWidget(self.spatial_label)
        self.next_bin_btn = QPushButton("Bin >")
        self.next_bin_btn.clicked.connect(lambda: self._step_spatial_bin(1))
        h.addWidget(self.next_bin_btn)
        h.addStretch()
        if not self.embedded:
            root.addWidget(self._header)

        split = QSplitter(Qt.Horizontal)
        self.heatmap = HeatmapView()
        self.heatmap.set_click_callback(self._heatmap_clicked)
        split.addWidget(self.heatmap)

        detector_host = QWidget()
        dl = QVBoxLayout(detector_host)
        dl.setContentsMargins(0, 0, 0, 0)
        self.detector = DetectorView()
        self.detector.drag_enabled = True
        self.detector.click_while_drag_enabled = True
        self.detector.set_drag_callback(self._roi_selected)
        self.detector.set_click_callback(self._detector_clicked)
        dl.addWidget(self.detector, 1)
        self.detector_status = QLabel("Click a feature or drag a detector ROI.")
        dl.addWidget(self.detector_status)
        split.addWidget(detector_host)

        controls = QWidget()
        controls.setMinimumWidth(310)
        form = QFormLayout(controls)
        metric_help = QLabel(
            "Heatmap value: total detector counts inside the selected ROI for each spatial bin.")
        metric_help.setWordWrap(True)
        metric_help.setStyleSheet("color:#888; font-size:0.9em;")
        form.addRow(metric_help)
        self.roi_algo = QComboBox()
        for algorithm in roi_detection.discover_algorithms(self.dm):
            self.roi_algo.addItem(algorithm["name"], algorithm)
        form.addRow("ROI detector", self.roi_algo)
        detect_row = QWidget(); dr = QHBoxLayout(detect_row); dr.setContentsMargins(0, 0, 0, 0)
        self.detect_btn = QPushButton("Detect ROIs")
        self.detect_btn.clicked.connect(self._detect_rois)
        self.roi_sensitivity = QDoubleSpinBox()
        self.roi_sensitivity.setRange(0.05, 0.95)
        self.roi_sensitivity.setSingleStep(0.05)
        self.roi_sensitivity.setValue(0.50)
        self.roi_sensitivity.setToolTip(
            "Classifier threshold. Lower finds more weak peaks; higher reduces extras.")
        self.roi_algo.currentIndexChanged.connect(self._roi_algorithm_changed)
        self._roi_algorithm_changed()
        self.run_shapes_btn = QPushButton("Run shape finding")
        self.run_shapes_btn.clicked.connect(self._run_detected_shapes)
        dr.addWidget(self.detect_btn); dr.addWidget(QLabel("threshold"))
        dr.addWidget(self.roi_sensitivity); dr.addWidget(self.run_shapes_btn)
        form.addRow(detect_row)
        fast_row = QWidget(); fl = QHBoxLayout(fast_row); fl.setContentsMargins(0, 0, 0, 0)
        self.fast_preview_cb = QCheckBox("Fast preview")
        self.fast_preview_cb.setToolTip(
            "Approximate coarse-to-fine spatial sampling. Save always recomputes exactly.")
        self.fast_stride = QSpinBox()
        self.fast_stride.setRange(2, 10); self.fast_stride.setValue(3)
        self.fast_stride.setPrefix("stride ")
        fl.addWidget(self.fast_preview_cb); fl.addWidget(self.fast_stride); fl.addStretch()
        form.addRow(fast_row)
        self.name = QLineEdit("manual_roi")
        form.addRow("Catalog tag", self.name)
        self.roi_label = QLabel("not selected")
        form.addRow("Detector ROI", self.roi_label)

        self.pending_list = QListWidget()
        self.pending_list.currentRowChanged.connect(self._pending_selected)
        form.addRow("Features", self.pending_list)
        pending_row = QWidget(); pl = QHBoxLayout(pending_row); pl.setContentsMargins(0, 0, 0, 0)
        self.run_btn = QPushButton("Save all ready")
        self.run_btn.clicked.connect(self._run)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_pending)
        self.cancel_btn = QPushButton("Cancel job")
        self.cancel_btn.setEnabled(False); self.cancel_btn.clicked.connect(self._cancel)
        pl.addWidget(self.run_btn); pl.addWidget(self.remove_btn); pl.addWidget(self.cancel_btn)
        form.addRow(pending_row)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        form.addRow(self.progress)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        form.addRow(self.status)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(190)
        form.addRow(self.log)
        split.addWidget(controls)
        split.setSizes([500, 650, 350])
        root.addWidget(split, 1)
        self.setCentralWidget(central)

    def header_bar(self):
        return self._header if self.embedded else None

    def _populate_controls(self):
        pass

    def current_bin_size(self):
        return self.bin_size

    def _expensive_job_active(self):
        return any(process is not None for process in
                   (self.sum_process, self._batch_process, self._save_process))

    def _update_job_controls(self):
        shape_job = self._batch_process is not None or self._save_process is not None
        active = self._expensive_job_active() or self._detect_process is not None
        self.run_shapes_btn.setEnabled(not active)
        self.run_btn.setEnabled(not active)
        self.detect_btn.setEnabled(not active)
        self.remove_btn.setEnabled(not shape_job)
        self.cancel_btn.setEnabled(shape_job)
        self.bin_combo.setEnabled(not active)
        self._update_compute_btn()

    def _roi_algorithm_changed(self, *_):
        algorithm = self.roi_algo.currentData() or {}
        default = float(algorithm.get("default_sensitivity", 0.5))
        if default > self.roi_sensitivity.maximum():
            self.roi_sensitivity.setMaximum(max(20.0, default))
        elif self.roi_sensitivity.maximum() > 1.0 and default <= 1.0:
            self.roi_sensitivity.setMaximum(0.95)
        self.roi_sensitivity.setValue(default)

    def _bin_changed(self, text):
        try:
            self.bin_size = int(text.split("x", 1)[0])
        except ValueError:
            return
        if self.source is not None:
            self.source.close()
            self.source = None
        self.spatial_keys = []
        self.spatial_index = 0
        self._load_saved_features()
        self._load_detector_image()
        self.bin_size_changed.emit(self.bin_size)

    def _load_saved_features(self):
        """Restore dedicated ROI catalogs for the active scan and bin size."""
        from ..core import roi_catalog

        for entry in self.pending:
            process = entry.get("process")
            if process is not None and process.state() != QProcess.NotRunning:
                stop_process(process)
            rect = entry.get("rect")
            if rect is not None:
                rect.remove()
        self.pending = []
        self.preview_feature = None
        self.result_path = None

        paths = roi_catalog.discover(self.dm.labels_dir(self.scan), self.bin_size)
        for path in paths:
            data = roi_catalog.load(path)
            for feature in data.get("features", []):
                roi_data = feature.get("manual_roi") or {}
                try:
                    roi = roi_map.normalize_roi((roi_data["x0"], roi_data["y0"],
                                                 roi_data["x1"], roi_data["y1"]))
                except (KeyError, TypeError, ValueError):
                    continue
                self.pending.append({
                    "roi": roi,
                    "status": "saved",
                    "rect": None,
                    "feature": feature,
                    "catalog_path": path,
                    "preview_path": None,
                    "process": None,
                })
        if len(paths) == 1:
            self.result_path = paths[0]
            self.name.setText(roi_catalog.load(paths[0]).get("name") or "manual_roi")
        self._refresh_pending_list(select=0 if self.pending else None)
        if self.pending:
            self.status.setText(
                f"Loaded {len(self.pending)} saved ROI features from {len(paths)} catalog(s).")

    def _ensure_source(self):
        if self.source is None:
            self.source = io.open_bin_source(self.dm, self.bin_size, self.scan)
        if not self.spatial_keys:
            self.spatial_keys = list(self.source.keys())
            self.spatial_index = min(self.spatial_index, max(0, len(self.spatial_keys) - 1))
            self._update_spatial_label()
        return self.source

    def _current_spatial_key(self):
        if not self.spatial_keys:
            return ""
        return self.spatial_keys[self.spatial_index]

    def _update_spatial_label(self):
        key = self._current_spatial_key()
        total = len(self.spatial_keys)
        self.spatial_label.setText(
            f"{self.spatial_index + 1}/{total}: {key}" if total else "bin -/-")

    def _step_spatial_bin(self, amount):
        source = self._ensure_source()
        if not self.spatial_keys:
            return
        self.spatial_index = (self.spatial_index + amount) % len(self.spatial_keys)
        self._update_spatial_label()
        self.image_mode.setCurrentIndex(1)
        self._load_detector_image()

    def _grand_sum_cached(self) -> bool:
        """True when this scan's fully-summed image is already on disk."""
        try:
            return reflection_sum.sum_path(self.dm, self.scan).exists()
        except Exception:
            return False

    def _update_compute_btn(self):
        if self.sum_process is not None:
            self.compute_sum_btn.setText("Computing grand sum...")
            self.compute_sum_btn.setEnabled(False)
            return
        self.compute_sum_btn.setEnabled(
            self._batch_process is None and self._save_process is None)
        self.compute_sum_btn.setText(
            "Recompute grand sum" if self._grand_sum_cached() else "Compute grand sum")

    def _image_mode_changed(self, index):
        # Selecting the grand-sum view while it is missing offers to build it,
        # but never blocks: the bin browser stays usable either way.
        if index == 0 and not self._grand_sum_cached() and self.sum_process is None:
            self._prompt_grand_sum()
        self._load_detector_image()

    def _load_detector_image(self, *_):
        try:
            if self.image_mode.currentIndex() == 0:
                path = reflection_sum.sum_path(self.dm, self.scan)
                if not path.exists():
                    # Lazy: don't sum the whole scan on load. Show a placeholder
                    # and let the user compute it (or browse single bins instead).
                    self._show_grand_sum_placeholder()
                    return
                with np.load(path) as saved:
                    image = saved["image"].astype(np.float64)
                title = "Fully summed detector image"
            else:
                source = self._ensure_source()
                key = self._current_spatial_key()
                image = source.image(key)
                title = f"Spatial bin {key} ({self.bin_size}x{self.bin_size})"
            if image is None:
                raise ValueError("selected image is empty")
            self.image = np.asarray(image, dtype=float)
            finite = self.image[np.isfinite(self.image)]
            vmin, vmax = (np.percentile(finite, [2, 99.7]) if finite.size else (0, 1))
            self.detector.show_image(self.image, float(vmin), float(vmax), "inferno")
            self.detector.set_title(title)
            self._redraw_pending_rects()
        except Exception as exc:
            self.detector.clear_image()
            self.status.setText(f"Could not load detector image: {exc}")
        finally:
            self._update_compute_btn()

    def _show_grand_sum_placeholder(self):
        """Render nothing but explain how to get the grand sum, without blocking."""
        self.image = None
        self.detector.clear_image()
        self.detector.set_title("Grand sum not loaded")
        self.detector_status.setText(
            "Grand sum not loaded. Click 'Compute grand sum', or switch "
            "'Detector image' to 'Selected spatial bin' to browse individual bins.")
        self._update_compute_btn()
        # Prompt once per session so opening the tab explains the situation, but
        # re-opening or re-rendering doesn't nag. Deferred so it shows after the
        # window is up (safe during embedded tab construction).
        if not self._sum_prompted:
            self._sum_prompted = True
            QTimer.singleShot(0, self._prompt_grand_sum)

    def _prompt_grand_sum(self):
        if self._grand_sum_cached() or self.sum_process is not None:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Grand sum not loaded")
        box.setText("The fully summed detector image has not been computed for this scan.")
        box.setInformativeText(
            "Computing it reads every frame in the scan, which is slow on a fresh "
            "project.\n\n"
            "Tip: build bins first for quick loading, e.g.\n"
            f"    xrd-app bin --scan {self.scan or '<N>'} "
            f"--bin-size {self.bin_size}\n\n"
            "If no bins are built, the sum is computed from raw frames using the "
            "scan's real positions where available.\n\n"
            "You can also switch 'Detector image' to 'Selected spatial bin' to "
            "browse individual bins without computing the sum.")
        compute = box.addButton("Compute now", QMessageBox.AcceptRole)
        box.addButton("Not now", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is compute:
            self._compute_grand_sum(prompt=False)

    def _on_compute_btn(self):
        # When the sum already exists the button recomputes it (e.g. after bins
        # were built); otherwise it computes it for the first time.
        self._compute_grand_sum(prompt=False, force=self._grand_sum_cached())

    def _compute_grand_sum(self, prompt=True, force=False):
        """Run 'xrd-app reflection-sum' as a non-blocking job, then display it."""
        if self._expensive_job_active():
            self.status.setText("Another batch, save, or grand-sum job is already active.")
            return
        if self._grand_sum_cached() and not force:
            self.image_mode.setCurrentIndex(0)
            self._load_detector_image()
            return
        if prompt and not force:
            self._prompt_grand_sum()
            return
        args = ["reflection-sum", "--root", self.project_root]
        if force:
            args += ["--overwrite"]
        if self.scan:
            args += ["--scan", str(self.scan)]
        cmd = [sys.executable, "-m", "xrd_app.cli", *args]
        self.log.appendPlainText("$ " + " ".join(cmd))
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda process=process: self._on_sum_output(process))
        process.finished.connect(
            lambda code, status, process=process: self._on_sum_finished(process, code))
        process.errorOccurred.connect(
            lambda error, process=process: self._on_process_error(
                process, error, lambda: self._on_sum_finished(process, -1)))
        self.sum_process = process
        self.progress.setVisible(True); self.progress.setValue(0)
        self.status.setText("Computing grand sum across all frames...")
        self.detector_status.setText(
            "Computing grand sum... (slow on raw data; build bins for quick loading)")
        self._update_job_controls()
        start_process(process, cmd[0], cmd[1:])

    def _on_process_error(self, process, error, on_failed):
        if error != QProcess.FailedToStart:
            return
        self.log.appendPlainText(f"[failed to start: {process.errorString()}]")
        on_failed()

    def _on_sum_output(self, process):
        text = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            match = _PROGRESS_RE.search(line)
            if match:
                i, n = int(match.group(1)), int(match.group(2))
                self.progress.setVisible(True)
                self.progress.setValue(int(100 * i / n) if n else 0)
                self.status.setText(f"Computing grand sum: {i}/{n} frames")
            else:
                self.log.appendPlainText(line)

    def _on_sum_finished(self, process, code):
        if self.sum_process is not process:
            return
        self.sum_process = None
        self._update_job_controls()
        if code != 0 or not self._grand_sum_cached():
            self.status.setText(f"Grand sum failed (exit {code}); see log.")
            return
        self.progress.setValue(100)
        self.status.setText("Grand sum computed.")
        self.image_mode.setCurrentIndex(0)
        self._load_detector_image()

    def _roi_selected(self, x0, y0, x1, y1, rect):
        self.roi = roi_map.normalize_roi((x0, y0, x1, y1))
        row = self._add_feature_roi(self.roi, rect, status="detected")
        self._refresh_pending_list(select=row)
        self.detector_status.setText(
            f"Added ROI {self.roi}. Press Run shape finding to build its spatial map.")

    def _add_feature_roi(self, roi, rect, status="detected", score=None):
        if any(entry.get("roi") == roi for entry in self.pending):
            if rect is not None:
                rect.remove()
            return next(i for i, entry in enumerate(self.pending) if entry.get("roi") == roi)
        job_id = len(self.pending) + 1
        preview_dir = self.dm.metadata_scan_dir(self.scan) / "roi_previews"
        entry = {"roi": roi, "status": status, "rect": rect,
                 "preview_path": preview_dir / f"preview_{job_id}.json",
                 "process": None, "score": score}
        self.pending.append(entry)
        return len(self.pending) - 1

    def _detector_clicked(self, x, y):
        if self.image is None:
            return
        roi = self._auto_roi_from_click(int(x), int(y))
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        rect = _RectItem(self.detector.vb, x0, y0, x1, y1,
                         edge="#f0a030", face="#f0a030", alpha=0.2)
        self._roi_selected(x0, y0, x1, y1, rect)

    def _auto_roi_from_click(self, x, y):
        return roi_map.auto_roi_from_click(self.image, x, y)

    def _detect_rois(self):
        if self._expensive_job_active() or self._detect_process is not None:
            self.status.setText("Another ROI job is already active.")
            return
        output = self.dm.metadata_scan_dir(self.scan) / "roi_previews" / "detected_rois.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        algorithm = self.roi_algo.currentData() or {}
        args = ["roi-detect", "--root", self.project_root,
                "--algorithm", str(algorithm.get("file")),
                "--sensitivity", str(self.roi_sensitivity.value()),
                "--output", str(output)]
        if self.scan:
            args += ["--scan", str(self.scan)]
        cmd = [sys.executable, "-m", "xrd_app.cli", *args]
        self.detect_btn.setEnabled(False)
        self.log.appendPlainText("$ " + " ".join(cmd))
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda process=process: self.log.appendPlainText(
                bytes(process.readAllStandardOutput()).decode("utf-8", "replace")))
        process.finished.connect(
            lambda code, status, process=process, output=output:
            self._on_detection_finished(process, output, code, status))
        process.errorOccurred.connect(
            lambda error, process=process, output=output: self._on_process_error(
                process, error,
                lambda: self._on_detection_finished(
                    process, output, -1, QProcess.CrashExit)))
        self._detect_process = process
        self._update_job_controls()
        start_process(process, cmd[0], cmd[1:])

    def _on_detection_finished(self, process, output, code, _status):
        if getattr(self, "_detect_process", None) is not process:
            return
        self._detect_process = None
        self._update_job_controls()
        if code != 0:
            self.status.setText(f"ROI detection failed (exit {code}); see log.")
            return
        try:
            with open(output) as handle:
                data = json.load(handle)
        except Exception as exc:
            self.status.setText(f"Could not load ROI candidates: {exc}")
            return
        added = 0
        for candidate in data.get("candidates", []):
            try:
                roi = roi_map.normalize_roi(candidate["roi"])
            except (KeyError, TypeError, ValueError):
                continue
            x0, y0, x1, y1 = roi
            rect = _RectItem(self.detector.vb, x0, y0, x1, y1,
                             edge="#f0a030", face="#f0a030", alpha=0.2)
            before = len(self.pending)
            self._add_feature_roi(roi, rect, status="detected", score=candidate.get("score"))
            added += len(self.pending) - before
        self._refresh_pending_list(select=0 if self.pending else None)
        self.status.setText(
            f"Added {added} detector candidates. Review/remove them, then press Run shape finding.")

    def _run_detected_shapes(self):
        if self._expensive_job_active() or self._detect_process is not None:
            self.status.setText("Another batch, save, or grand-sum job is already active.")
            return
        entries = [entry for entry in self.pending if entry.get("status") == "detected"]
        if not entries:
            QMessageBox.information(self, "No detected features",
                                    "Detect or select one or more ROIs first.")
            return
        preview = (self.dm.metadata_scan_dir(self.scan) / "roi_previews" /
                   f"batch_preview_{uuid.uuid4().hex}.json")
        preview.parent.mkdir(parents=True, exist_ok=True)
        args = ["roi-shapes", "--root", self.project_root,
                "--bin-size", str(self.bin_size),
                "--name", self.name.text().strip() or "manual_roi",
                "--preview-output", str(preview),
                "--normalize-frames"]
        for entry in entries:
            args += ["--roi", ",".join(str(v) for v in entry["roi"])]
        if self.fast_preview_cb.isChecked():
            args += ["--fast", "--stride", str(self.fast_stride.value())]
        if self.scan:
            args += ["--scan", str(self.scan)]
        cmd = [sys.executable, "-m", "xrd_app.cli", *args]
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda process=process: self._on_batch_output(process))
        process.finished.connect(
            lambda code, status, process=process, entries=entries, preview=preview:
            self._on_batch_finished(process, entries, preview, code, status))
        process.errorOccurred.connect(
            lambda error, process=process, entries=entries, preview=preview:
            self._on_process_error(
                process, error,
                lambda: self._on_batch_finished(
                    process, entries, preview, -1, QProcess.CrashExit)))
        for entry in entries:
            entry["process"] = process
            entry["status"] = "running"
            entry["preview_path"] = preview
        self._batch_process = process
        self._output_buffers[process] = ""
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._refresh_pending_list(select=self.pending_list.currentRow())
        self._update_job_controls()
        self.log.appendPlainText("$ " + " ".join(cmd))
        self.status.setText(f"Running one batch pass for {len(entries)} ROIs...")
        start_process(process, cmd[0], cmd[1:])

    def _consume_batch_output(self, process, final=False):
        chunk = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        text = self._output_buffers.get(process, "") + chunk
        lines = text.splitlines(keepends=True)
        remainder = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            remainder = lines.pop()
        if final and remainder:
            lines.append(remainder)
            remainder = ""
        self._output_buffers[process] = remainder
        for line in lines:
            line = line.rstrip("\r\n")
            match = _PROGRESS_RE.search(line)
            if match:
                i, n = int(match.group(1)), int(match.group(2))
                self.progress.setValue(int(100 * i / n) if n else 0)
            elif line:
                self.log.appendPlainText(line)

    def _on_batch_output(self, process):
        self._consume_batch_output(process)

    def _on_batch_finished(self, process, entries, preview, code, _status):
        if self._batch_process is not process:
            return
        self._consume_batch_output(process, final=True)
        self._output_buffers.pop(process, None)
        self._batch_process = None
        active = [entry for entry in entries if entry in self.pending]
        for entry in active:
            entry["process"] = None
            entry["status"] = "failed"
        error = None
        data = None
        if code != 0 or not preview.exists():
            error = f"Batch shape finding failed (exit {code}); see log."
        else:
            try:
                with open(preview) as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    raise ValueError("preview root is not an object")
                if data.get("scan") != self.dm.scan_name:
                    raise ValueError(f"preview scan {data.get('scan')!r} does not match {self.dm.scan_name!r}")
                if int(data.get("bin_size", -1)) != self.bin_size:
                    raise ValueError("preview bin size does not match the current context")
            except Exception as exc:
                error = f"Could not load batch preview: {exc}"
        ready_count = 0
        if data is not None and error is None:
            by_roi = {}
            for feature in data.get("features") or []:
                try:
                    roi_data = feature["manual_roi"]
                    roi = tuple(int(roi_data[key]) for key in ("x0", "y0", "x1", "y1"))
                    if not isinstance(feature.get("intensity_profile"), dict):
                        raise ValueError
                    by_roi[roi] = feature
                except (KeyError, TypeError, ValueError):
                    continue
            for entry in active:
                feature = by_roi.get(tuple(entry["roi"]))
                if feature is None:
                    continue
                feature["n_bin_rows"] = int(data.get("n_bin_rows", 0))
                feature["n_bin_cols"] = int(data.get("n_bin_cols", 0))
                entry["feature"] = feature
                entry["status"] = "ready"
                ready_count += 1
                rect = entry.get("rect")
                if rect is not None:
                    rect.set_color("yellow", "yellow", 0.2)
        current = self.pending_list.currentRow()
        self._refresh_pending_list(select=current)
        self._update_job_controls()
        if 0 <= current < len(self.pending) and self.pending[current].get("feature"):
            self._pending_selected(current)
        if error:
            self.status.setText(error)
        else:
            approximate = bool(data.get("approximate"))
            self.status.setText(
                f"Batch preview complete for {ready_count}/{len(active)} ROIs" +
                (" (approximate; Save recomputes exactly)." if approximate else "."))

    def _refresh_pending_list(self, select=None):
        self.pending_list.blockSignals(True)
        self.pending_list.clear()
        for i, entry in enumerate(self.pending, 1):
            roi = entry["roi"]
            item = QListWidgetItem(
                f"{i}. x={roi[0]}:{roi[2]} y={roi[1]}:{roi[3]} "
                f"[{entry['status']}]")
            if entry.get("status") == "saved":
                item.setForeground(QColor("lime"))
            elif entry.get("status") == "detected":
                item.setForeground(QColor("#f0a030"))
            elif entry.get("status") == "ready":
                item.setForeground(QColor("#d4b000"))
            elif entry.get("status") == "failed":
                item.setForeground(QColor("#ff6666"))
            self.pending_list.addItem(item)
        self.pending_list.blockSignals(False)
        if select is not None and self.pending:
            self.pending_list.setCurrentRow(min(select, len(self.pending) - 1))

    def _pending_selected(self, row):
        if not (0 <= row < len(self.pending)):
            return
        entry = self.pending[row]
        self.roi = entry["roi"]
        for index, candidate in enumerate(self.pending):
            rect = candidate.get("rect")
            if rect is None:
                continue
            if index == row:
                rect.set_color("red", "red", 0.28)
            elif candidate.get("status") == "saved":
                rect.set_color("lime", "lime", 0.18)
            elif candidate.get("status") == "ready":
                rect.set_color("yellow", "yellow", 0.2)
            else:
                rect.set_color("#f0a030", "#f0a030", 0.2)
        self.roi_label.setText(", ".join(str(v) for v in self.roi))
        if entry.get("feature"):
            self.preview_feature = entry["feature"]
            self._render_feature()
        self._load_detector_image()

    def _remove_pending(self):
        row = self.pending_list.currentRow()
        if not (0 <= row < len(self.pending)):
            return
        entry = self.pending[row]
        process = entry.get("process")
        if process is not None and process.state() != QProcess.NotRunning:
            # Shape previews are batched: removing one running ROI stops the shared
            # pass and returns its remaining members to detected for a clean rerun.
            if getattr(self, "_batch_process", None) is process:
                self._batch_process = None
                for other in self.pending:
                    if other.get("process") is process:
                        other["process"] = None
                        if other is not entry:
                            other["status"] = "detected"
            else:
                entry["process"] = None
            stop_process(process)
            process.deleteLater()
        feature = entry.get("feature") if entry.get("status") == "saved" else None
        catalog_path = entry.get("catalog_path") or self.result_path
        if feature is not None and catalog_path and Path(catalog_path).exists():
            try:
                from ..core import roi_catalog
                roi_catalog.remove_feature(catalog_path, feature.get("manual_roi"))
            except Exception as exc:
                QMessageBox.warning(self, "Could not remove saved feature", str(exc))
                return
        rect = entry.get("rect")
        if rect is not None:
            rect.remove()
        preview_path = entry.get("preview_path")
        if preview_path is not None:
            try:
                preview_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.pending.pop(row)
        self.roi = None
        self.roi_label.setText("not selected")
        self._refresh_pending_list(select=max(0, row - 1))

    def _redraw_pending_rects(self):
        selected = self.pending_list.currentRow()
        for index, entry in enumerate(self.pending):
            old = entry.get("rect")
            if old is not None:
                old.remove()
            x0, y0, x1, y1 = entry["roi"]
            if index == selected:
                edge, face, alpha = "red", "red", 0.28
            elif entry.get("status") == "saved":
                edge, face, alpha = "lime", "lime", 0.18
            elif entry.get("status") == "ready":
                edge, face, alpha = "yellow", "yellow", 0.2
            else:
                edge, face, alpha = "#f0a030", "#f0a030", 0.2
            entry["rect"] = _RectItem(
                self.detector.vb, x0, y0, x1, y1, edge=edge, face=face, alpha=alpha)

    def _draw_roi(self):
        """Queue a preselected ROI (used by the raw-scan project handoff)."""
        if self.roi is None:
            return
        x0, y0, x1, y1 = self.roi
        rect = _RectItem(self.detector.vb, x0, y0, x1, y1,
                         edge="#f0a030", face="#f0a030", alpha=0.2)
        self._roi_selected(x0, y0, x1, y1, rect)

    def _run(self):
        if self._expensive_job_active() or self._detect_process is not None:
            self.status.setText("Another batch, save, or grand-sum job is already active.")
            return
        ready = [entry for entry in self.pending if entry.get("status") == "ready"]
        if not ready:
            QMessageBox.information(
                self, "No ready features",
                "Wait for at least one search to finish (yellow) before saving.")
            return
        tag = self.name.text().strip()
        if not tag:
            QMessageBox.information(self, "Catalog name", "Enter a catalog name before saving.")
            return
        args = ["roi-save", "--root", self.project_root,
                "--bin-size", str(self.bin_size), "--name", tag]
        for entry in ready:
            args += ["--roi", ",".join(str(v) for v in entry["roi"])]
        if self.scan:
            args += ["--scan", str(self.scan)]
        cmd = [sys.executable, "-m", "xrd_app.cli", *args]
        self.log.appendPlainText("$ " + " ".join(cmd))
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda process=process: self.log.appendPlainText(
                bytes(process.readAllStandardOutput()).decode("utf-8", "replace")))
        process.finished.connect(
            lambda code, status, ready=ready, process=process:
            self._on_save_finished(ready, process, code, status))
        process.errorOccurred.connect(
            lambda error, ready=ready, process=process: self._on_process_error(
                process, error,
                lambda: self._on_save_finished(
                    ready, process, -1, QProcess.CrashExit)))
        for entry in ready:
            entry["process"] = process
            entry["status"] = "saving"
        self._save_process = process
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._refresh_pending_list(select=self.pending_list.currentRow())
        self._update_job_controls()
        start_process(process, cmd[0], cmd[1:])

    def _on_save_finished(self, entries, process, code, _status):
        if self._save_process is not process:
            return
        self._save_process = None
        active = [entry for entry in entries
                  if entry in self.pending and entry.get("process") is process]
        if not active:
            self._update_job_controls()
            return
        for entry in active:
            entry["process"] = None
        if code != 0:
            for entry in active:
                entry["status"] = "ready"
            self.status.setText(f"Save failed (exit {code}); see log.")
            self._refresh_pending_list(select=self.pending_list.currentRow())
            self._update_job_controls()
            return
        tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.name.text().strip()).strip('_.-')
        self.result_path = self.dm.roi_map_json(tag, self.bin_size, self.scan)
        for entry in active:
            entry["status"] = "saved"
            entry["catalog_path"] = self.result_path
            rect = entry.get("rect")
            if rect is not None:
                rect.set_color("lime", "lime", 0.18)
        self._refresh_pending_list(select=self.pending_list.currentRow())
        self._update_job_controls()
        self.status.setText(
            f"Saved {len(active)} features to ROI > Shape catalog {self.result_path.name}.")

    def _render_feature(self):
        feature = self.preview_feature
        if feature is None:
            self.heatmap.img.clear(); self.heatmap._grid_data = None
            return
        profile = feature.get("intensity_profile") or {}
        metric = "integrated"
        result = {"profile": profile,
                  "n_bin_rows": int(feature.get("n_bin_rows", 0)),
                  "n_bin_cols": int(feature.get("n_bin_cols", 0)),
                  "metric": metric}
        if not result["n_bin_rows"] or not result["n_bin_cols"]:
            rows = []; cols = []
            for key in profile:
                try:
                    row, col = (int(v) for v in key.split("_", 1))
                    rows.append(row); cols.append(col)
                except ValueError:
                    pass
            result["n_bin_rows"] = max(rows, default=-1) + 1
            result["n_bin_cols"] = max(cols, default=-1) + 1
        grid = roi_map.grid_array(result, metric)
        if grid.size and np.isfinite(grid).any():
            finite = grid[np.isfinite(grid)]
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
            rgba = _scalar_to_rgba(grid, vmin, vmax, _get_cmap("inferno"))
            self.heatmap.img.setImage(rgba, autoLevels=False)
            self.heatmap.img.setRect(QRectF(-0.5, -0.5, grid.shape[1], grid.shape[0]))
            self.heatmap._grid_data = grid
            self.heatmap._grid_r_lo = self.heatmap._grid_c_lo = 0
            self.heatmap.fit_to_rect(-0.5, -0.5, grid.shape[1], grid.shape[0])
        center = feature.get("center_bin")
        try:
            center_rc = tuple(int(v) for v in center.split("_", 1))
        except (AttributeError, ValueError):
            center_rc = None
        self.heatmap.set_markers(center_rc, None)
        self.heatmap.plot.setTitle(
            f"ROI feature {feature.get('feature_id', '?')} - total ROI counts",
            color="w", size="10pt")

    def _heatmap_clicked(self, row, col):
        key = f"{row}_{col}"
        source = self._ensure_source()
        if key not in self.spatial_keys:
            return
        self.spatial_index = self.spatial_keys.index(key)
        self._update_spatial_label()
        self.image_mode.setCurrentIndex(1)
        self._load_detector_image()

    def _cancel(self):
        process = self._batch_process or self._save_process
        if process is None:
            return
        row = self.pending_list.currentRow()
        if process is self._batch_process:
            self._batch_process = None
            self._output_buffers.pop(process, None)
            for entry in self.pending:
                if entry.get("process") is process:
                    entry["process"] = None
                    entry["status"] = "detected"
            message = "Batch preview cancelled; detected ROIs are ready to rerun"
        else:
            self._save_process = None
            for entry in self.pending:
                if entry.get("process") is process:
                    entry["process"] = None
                    entry["status"] = "ready"
            message = "Save cancelled; ready ROIs were not changed"
        stop_process_async(process)
        self._refresh_pending_list(select=row)
        self._update_job_controls()
        self.status.setText(message)

    def closeEvent(self, event):  # noqa: N802
        processes = {entry.get("process") for entry in self.pending}
        processes.update((self._detect_process, self._batch_process,
                          self._save_process, self.sum_process))
        for process in processes:
            stop_process(process)
        self._detect_process = None
        self._batch_process = None
        self._save_process = None
        self.sum_process = None
        if self.source is not None:
            self.source.close()
            self.source = None
        super().closeEvent(event)


def build_window(project_root=".", scan=None, bin_size=3, embedded=False):
    return ROIShapeWindow(project_root, scan=scan, bin_size=bin_size, embedded=embedded)
