"""Manual detector ROIs mapped as spatial features across scan bins."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QProcess, QRectF, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import catalogs, io, reflection_sum, roi_map
from .palette import _get_cmap
from .viewer import DetectorView, HeatmapView, _RectItem, _scalar_to_rgba

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
_PROGRESS_RE = re.compile(r"PROGRESS\s+(\d+)/(\d+)")


class ROIShapeWindow(QMainWindow):
    """Select detector ROIs and map their total counts across spatial bins."""

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
        self.features = []
        self.feature_index = 0
        self.pending = []
        self.spatial_keys = []
        self.spatial_index = 0

        self.setWindowTitle("ROI > Shape")
        self.resize(1500, 900)
        self._build_ui()
        self._populate_controls()
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
        self.bin_combo.addItems([f"{n}x{n}" for n in (1, 2, 3, 4, 5)])
        self.bin_combo.setCurrentText(f"{self.bin_size}x{self.bin_size}")
        self.bin_combo.currentTextChanged.connect(self._bin_changed)
        h.addWidget(self.bin_combo)
        h.addWidget(QLabel("<b>Detector image:</b>"))
        self.image_mode = QComboBox()
        self.image_mode.addItems(["Fully summed scan", "Selected spatial bin"])
        self.image_mode.currentIndexChanged.connect(self._load_detector_image)
        h.addWidget(self.image_mode)
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

        nav = QWidget(); nl = QHBoxLayout(nav); nl.setContentsMargins(0, 0, 0, 0)
        self.prev_btn = QPushButton("Previous"); self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn = QPushButton("Next"); self.next_btn.clicked.connect(lambda: self._step(1))
        self.feature_label = QLabel("No feature preview")
        nl.addWidget(self.prev_btn); nl.addWidget(self.next_btn); nl.addWidget(self.feature_label)
        form.addRow("Feature preview", nav)
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
        self._load_detector_image()

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

    def _load_detector_image(self, *_):
        try:
            if self.image_mode.currentIndex() == 0:
                path = reflection_sum.sum_path(self.dm, self.scan)
                if path.exists():
                    with np.load(path) as saved:
                        image = saved["image"].astype(np.float64)
                else:
                    source = self._ensure_source()
                    image = source.sum_all()
                    reflection_sum.save(self.dm, self.scan, image, source.is_raw)
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

    def _roi_selected(self, x0, y0, x1, y1, rect):
        self.roi = roi_map.normalize_roi((x0, y0, x1, y1))
        job_id = len(self.pending) + 1
        preview_dir = self.dm.metadata_scan_dir(self.scan) / "roi_previews"
        entry = {
            "roi": self.roi,
            "status": "running",
            "rect": rect,
            "preview_path": preview_dir / f"preview_{job_id}.json",
            "process": None,
        }
        self.pending.append(entry)
        row = len(self.pending) - 1
        self._refresh_pending_list(select=row)
        self.detector_status.setText(
            f"Searching ROI {self.roi} across all spatial bins...")
        self._start_search(row)

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

    def _search_args(self, entry, preview=False):
        tag = self.name.text().strip() or "manual_roi"
        args = ["roi-shapes", "--root", self.project_root,
                "--bin-size", str(self.bin_size),
                "--roi", ",".join(str(v) for v in entry["roi"]),
                "--name", tag]
        if preview:
            args += ["--preview-output", str(entry["preview_path"])]
        if self.scan:
            args += ["--scan", str(self.scan)]
        return args

    def _start_search(self, row):
        if not (0 <= row < len(self.pending)):
            return
        entry = self.pending[row]
        entry["preview_path"].parent.mkdir(parents=True, exist_ok=True)
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda entry=entry, process=process: self._on_search_output(entry, process))
        process.finished.connect(
            lambda code, status, entry=entry, process=process:
            self._on_search_finished(entry, process, code, status))
        entry["process"] = process
        entry["status"] = "running"
        rect = entry.get("rect")
        if rect is not None:
            rect.set_color("#f0a030", "#f0a030", 0.2)
        cmd = [sys.executable, "-m", "xrd_app.cli", *self._search_args(entry, preview=True)]
        self.log.appendPlainText("$ " + " ".join(cmd))
        process.start(cmd[0], cmd[1:])

    def _on_search_output(self, entry, process):
        text = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        row = self.pending.index(entry) if entry in self.pending else -1
        for line in text.splitlines():
            match = _PROGRESS_RE.search(line)
            if match and row == self.pending_list.currentRow():
                i, n = int(match.group(1)), int(match.group(2))
                self.progress.setVisible(True)
                self.progress.setValue(int(100 * i / n) if n else 0)
                self.status.setText(f"Searching selected ROI: {i}/{n} spatial bins")
            elif not match:
                self.log.appendPlainText(line)

    def _on_search_finished(self, entry, process, code, _status):
        if entry not in self.pending:
            return
        row = self.pending.index(entry)
        if entry.get("process") is not process:
            return
        entry["process"] = None
        if code != 0 or not entry["preview_path"].exists():
            entry["status"] = "failed"
            self.status.setText(f"ROI search failed (exit {code}); see log.")
            self._refresh_pending_list(select=row)
            return
        try:
            kept, _ = catalogs.load_features_any(entry["preview_path"])
            entry["feature"] = kept[0]
        except Exception as exc:
            entry["status"] = "failed"
            self.status.setText(f"Could not load ROI preview: {exc}")
            self._refresh_pending_list(select=row)
            return
        entry["status"] = "ready"
        rect = entry.get("rect")
        if rect is not None:
            rect.set_color("yellow", "yellow", 0.2)
        current = self.pending_list.currentRow()
        self._refresh_pending_list(select=current)
        if row == current:
            self.progress.setValue(100)
            self._pending_selected(row)
            self.status.setText(
                "ROI search complete. Preview is shown at left; Save all ready commits it.")

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
            self.features = [("Saved", entry["feature"])]
            self.feature_index = 0
            self._render_feature()
        self._load_detector_image()

    def _remove_pending(self):
        row = self.pending_list.currentRow()
        if not (0 <= row < len(self.pending)):
            return
        entry = self.pending[row]
        process = entry.get("process")
        if process is not None and process.state() != QProcess.NotRunning:
            entry["process"] = None
            process.kill()
            process.deleteLater()
        feature = entry.get("feature") if entry.get("status") == "saved" else None
        if feature is not None and self.result_path and self.result_path.exists():
            try:
                from ..core import roi_catalog
                roi_catalog.remove_feature(self.result_path, feature.get("manual_roi"))
            except Exception as exc:
                QMessageBox.warning(self, "Could not remove saved feature", str(exc))
                return
        rect = entry.get("rect")
        if rect is not None:
            rect.remove()
        try:
            entry.get("preview_path").unlink(missing_ok=True)
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
            args += ["--preview", str(entry["preview_path"])]
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
        for entry in ready:
            entry["process"] = process
            entry["status"] = "saving"
        self._refresh_pending_list(select=self.pending_list.currentRow())
        process.start(cmd[0], cmd[1:])

    def _on_save_finished(self, entries, process, code, _status):
        active = [entry for entry in entries
                  if entry in self.pending and entry.get("process") is process]
        if not active:
            return
        for entry in active:
            entry["process"] = None
        if code != 0:
            for entry in active:
                entry["status"] = "ready"
            self.status.setText(f"Save failed (exit {code}); see log.")
            self._refresh_pending_list(select=self.pending_list.currentRow())
            return
        tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.name.text().strip()).strip('_.-')
        self.result_path = self.dm.roi_map_json(tag, self.bin_size, self.scan)
        for entry in active:
            entry["status"] = "saved"
            rect = entry.get("rect")
            if rect is not None:
                rect.set_color("lime", "lime", 0.18)
        self._refresh_pending_list(select=self.pending_list.currentRow())
        self.status.setText(
            f"Saved {len(active)} features to ROI > Shape catalog {self.result_path.name}.")

    def _step(self, amount):
        if not self.features:
            return
        self.feature_index = (self.feature_index + amount) % len(self.features)
        self._render_feature()

    def _render_feature(self):
        if not self.features:
            self.feature_label.setText("No feature preview")
            self.heatmap.img.clear(); self.heatmap._grid_data = None
            return
        category, feature = self.features[self.feature_index]
        profile = feature.get("intensity_profile") or {}
        metric = "integrated"
        result = {"profile": profile, "n_bin_rows": 0, "n_bin_cols": 0,
                  "metric": metric}
        rows = []; cols = []
        for key in profile:
            try:
                row, col = (int(v) for v in key.split("_", 1)); rows.append(row); cols.append(col)
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
            f"{category}: ROI feature "
            f"{feature.get('feature_id', self.feature_index + 1)} - total ROI counts",
            color="w", size="10pt")
        self.feature_label.setText(f"{self.feature_index + 1}/{len(self.features)} {category}")

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
        row = self.pending_list.currentRow()
        if not (0 <= row < len(self.pending)):
            return
        process = self.pending[row].get("process")
        if process is not None and process.state() != QProcess.NotRunning:
            self.pending[row]["process"] = None
            process.kill()
            self.pending[row]["status"] = "cancelled"
            self._refresh_pending_list(select=row)
            self.status.setText("Selected ROI job cancelled")

    def closeEvent(self, event):  # noqa: N802
        for entry in self.pending:
            process = entry.get("process")
            if process is not None and process.state() != QProcess.NotRunning:
                process.kill()
        if self.source is not None:
            self.source.close()
        super().closeEvent(event)


def build_window(project_root=".", scan=None, bin_size=3, embedded=False):
    return ROIShapeWindow(project_root, scan=scan, bin_size=bin_size, embedded=embedded)
