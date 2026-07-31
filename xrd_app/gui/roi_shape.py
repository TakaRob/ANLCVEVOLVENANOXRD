"""Manual detector-ROI search followed by the standard peak/shape pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QProcess, QRectF, Qt
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGraphicsRectItem, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import catalogs, io, reflection_sum, roi_map
from .palette import _get_cmap
from .viewer import DetectorView, HeatmapView, _scalar_to_rgba

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
_PROGRESS_RE = re.compile(r"PROGRESS\s+(\d+)/(\d+)")


class ROIShapeWindow(QMainWindow):
    """Select a feature on a summed detector image and build standard shapes."""

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
        self.roi_item = None
        self.result_path = None
        self.features = []
        self.feature_index = 0
        self._proc = None

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
        self.spatial_bin = QComboBox()
        self.spatial_bin.currentTextChanged.connect(self._load_detector_image)
        h.addWidget(self.spatial_bin)
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
        self.detector.set_drag_callback(self._roi_selected)
        dl.addWidget(self.detector, 1)
        self.detector_status = QLabel("Drag a rectangle around one reflection feature.")
        dl.addWidget(self.detector_status)
        split.addWidget(detector_host)

        controls = QWidget()
        controls.setMinimumWidth(310)
        form = QFormLayout(controls)
        self.reflection_combo = QComboBox()
        form.addRow("Reflection", self.reflection_combo)
        self.peak_combo = QComboBox()
        form.addRow("Peak detector", self.peak_combo)
        self.shape_combo = QComboBox()
        form.addRow("Shape algorithm", self.shape_combo)
        self.snr = QDoubleSpinBox()
        self.snr.setRange(0.1, 100.0); self.snr.setValue(4.0); self.snr.setDecimals(2)
        form.addRow("SNR", self.snr)
        self.link_tolerance = QSpinBox()
        self.link_tolerance.setRange(1, 50); self.link_tolerance.setValue(5)
        form.addRow("Link tolerance (px)", self.link_tolerance)
        self.name = QLineEdit("manual_roi")
        form.addRow("Catalog tag", self.name)
        self.roi_label = QLabel("not selected")
        form.addRow("Detector ROI", self.roi_label)

        run_row = QWidget(); rl = QHBoxLayout(run_row); rl.setContentsMargins(0, 0, 0, 0)
        self.run_btn = QPushButton("Find peaks + shapes")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False); self.cancel_btn.clicked.connect(self._cancel)
        rl.addWidget(self.run_btn); rl.addWidget(self.cancel_btn)
        form.addRow(run_row)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        form.addRow(self.progress)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        form.addRow(self.status)

        nav = QWidget(); nl = QHBoxLayout(nav); nl.setContentsMargins(0, 0, 0, 0)
        self.prev_btn = QPushButton("Previous"); self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn = QPushButton("Next"); self.next_btn.clicked.connect(lambda: self._step(1))
        self.feature_label = QLabel("No saved shapes")
        nl.addWidget(self.prev_btn); nl.addWidget(self.next_btn); nl.addWidget(self.feature_label)
        form.addRow("Saved result", nav)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(190)
        form.addRow(self.log)
        split.addWidget(controls)
        split.setSizes([500, 650, 350])
        root.addWidget(split, 1)
        self.setCentralWidget(central)

    def header_bar(self):
        return self._header if self.embedded else None

    def _populate_controls(self):
        try:
            _degs, labels = io.load_reflections(self.dm.reflections(scan=self.scan))
            self.reflection_combo.addItems(labels)
        except Exception as exc:
            self.status.setText(f"Could not load reflections: {exc}")
        compatible = {"5x5_tophat_band_adaptive_snr"}
        for entry in self.dm.list_detectors(self.bin_size):
            if entry.get("name") in compatible:
                self.peak_combo.addItem(entry["name"], entry["name"])
        if not self.peak_combo.count():
            path = self.dm.detector_script(bin_size=self.bin_size)
            self.peak_combo.addItem(Path(path).stem, str(path))
        for entry in self.dm.list_shapes():
            if self.bin_size != 1 and entry.get("name") == "territory":
                continue
            self.shape_combo.addItem(entry["name"], entry["name"])
        if self.shape_combo.findData("gaussian") >= 0:
            self.shape_combo.setCurrentIndex(self.shape_combo.findData("gaussian"))

    def _bin_changed(self, text):
        try:
            self.bin_size = int(text.split("x", 1)[0])
        except ValueError:
            return
        if self.source is not None:
            self.source.close()
            self.source = None
        self.spatial_bin.clear()
        self._load_detector_image()

    def _ensure_source(self):
        if self.source is None:
            self.source = io.open_bin_source(self.dm, self.bin_size, self.scan)
        if not self.spatial_bin.count():
            self.spatial_bin.blockSignals(True)
            self.spatial_bin.addItems(self.source.keys())
            self.spatial_bin.blockSignals(False)
        return self.source

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
                key = self.spatial_bin.currentText() or (source.keys()[0] if source.keys() else "")
                image = source.image(key)
                title = f"Spatial bin {key} ({self.bin_size}x{self.bin_size})"
            if image is None:
                raise ValueError("selected image is empty")
            self.image = np.asarray(image, dtype=float)
            finite = self.image[np.isfinite(self.image)]
            vmin, vmax = (np.percentile(finite, [2, 99.7]) if finite.size else (0, 1))
            self.detector.show_image(self.image, float(vmin), float(vmax), "inferno")
            self.detector.set_title(title)
            if self.roi is not None:
                self._draw_roi()
        except Exception as exc:
            self.detector.clear_image()
            self.status.setText(f"Could not load detector image: {exc}")

    def _roi_selected(self, x0, y0, x1, y1, rect):
        if self.roi_item is not None:
            self.roi_item.remove()
        self.roi_item = rect
        self.roi = roi_map.normalize_roi((x0, y0, x1, y1))
        self.roi_label.setText(", ".join(str(v) for v in self.roi))
        self.detector_status.setText(
            f"Selected x={self.roi[0]}:{self.roi[2]}, y={self.roi[1]}:{self.roi[3]}. "
            "Choose the reflection, then run peak + shape finding.")

    def _draw_roi(self):
        x0, y0, x1, y1 = self.roi
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setPen(pg.mkPen("#f0a030", width=2)); rect.setBrush(pg.mkBrush(240, 160, 48, 35))
        self.detector.add_overlay(rect)
        self.roi_item = None

    def _run(self):
        if self.roi is None:
            QMessageBox.information(self, "Select detector ROI", "Drag a rectangle on the detector image first.")
            return
        tag = self.name.text().strip()
        if not tag:
            QMessageBox.information(self, "Catalog tag", "Enter a catalog tag before running.")
            return
        args = ["roi-shapes", "--root", self.project_root, "--bin-size", str(self.bin_size),
                "--reflection", self.reflection_combo.currentText(),
                "--roi", ",".join(str(v) for v in self.roi), "--name", tag,
                "--snr", str(self.snr.value()), "--link-tolerance", str(self.link_tolerance.value()),
                "--peak-algorithm", str(self.peak_combo.currentData()),
                "--shape-algorithm", str(self.shape_combo.currentData())]
        if self.scan:
            args += ["--scan", str(self.scan)]
        cmd = [sys.executable, "-m", "xrd_app.cli", *args]
        self.log.clear(); self.log.appendPlainText("$ " + " ".join(cmd))
        self.progress.setVisible(True); self.progress.setValue(0)
        self.run_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.status.setText("Running detector across all spatial bins...")
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(cmd[0], cmd[1:])

    def _on_output(self):
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            match = _PROGRESS_RE.search(line)
            if match:
                i, n = int(match.group(1)), int(match.group(2))
                if n:
                    self.progress.setValue(int(100 * i / n))
                self.status.setText(line.split("  ", 1)[-1])
            else:
                self.log.appendPlainText(line)

    def _on_finished(self, code, _status):
        self.run_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        if code != 0:
            self.status.setText(f"Processing failed (exit {code}); see log.")
            return
        self.progress.setValue(100)
        tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.name.text().strip()).strip('_.-')
        shape_algo = Path(str(self.shape_combo.currentData())).stem
        self.result_path = self.dm.shapes_json(shape_algo, self.bin_size, self.scan, variant=tag)
        self._load_result()

    def _load_result(self):
        try:
            kept, filtered = catalogs.load_features_any(self.result_path)
            self.features = [("Kept", f) for f in kept] + [("Filtered", f) for f in filtered]
            self.feature_index = 0
            self.status.setText(
                f"Saved standard Shape/Verify catalog: {self.result_path.name} "
                f"({len(kept)} kept, {len(filtered)} filtered)")
            self._render_feature()
        except Exception as exc:
            self.status.setText(f"Saved, but could not load result: {exc}")

    def _step(self, amount):
        if not self.features:
            return
        self.feature_index = (self.feature_index + amount) % len(self.features)
        self._render_feature()

    def _render_feature(self):
        if not self.features:
            self.feature_label.setText("No shapes found")
            self.heatmap.img.clear(); self.heatmap._grid_data = None
            return
        category, feature = self.features[self.feature_index]
        profile = feature.get("intensity_profile") or {}
        result = {"profile": profile, "n_bin_rows": 0, "n_bin_cols": 0,
                  "metric": "integrated"}
        rows = []; cols = []
        for key in profile:
            try:
                row, col = (int(v) for v in key.split("_", 1)); rows.append(row); cols.append(col)
            except ValueError:
                pass
        result["n_bin_rows"] = max(rows, default=-1) + 1
        result["n_bin_cols"] = max(cols, default=-1) + 1
        grid = roi_map.grid_array(result, "integrated")
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
            f"{category}: {feature.get('reflection', '?')} feature "
            f"{feature.get('feature_id', self.feature_index + 1)}",
            color="w", size="10pt")
        self.feature_label.setText(f"{self.feature_index + 1}/{len(self.features)} {category}")

    def _heatmap_clicked(self, row, col):
        self.image_mode.setCurrentIndex(1)
        self.spatial_bin.setCurrentText(f"{row}_{col}")

    def _cancel(self):
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
            self.status.setText("Cancelled")

    def closeEvent(self, event):  # noqa: N802
        self._cancel()
        if self.source is not None:
            self.source.close()
        super().closeEvent(event)


def build_window(project_root=".", scan=None, bin_size=3, embedded=False):
    return ROIShapeWindow(project_root, scan=scan, bin_size=bin_size, embedded=embedded)
