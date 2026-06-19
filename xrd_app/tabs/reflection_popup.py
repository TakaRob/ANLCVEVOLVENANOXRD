"""Manual reflection popup (pyqtgraph).

Computes a radial (2θ) histogram on the fully-summed #x# binned image, shows the
summed image with interactive 2θ ring overlays and the histogram with reflection
bands, lets the user enter [Name, Bragg angle, Width], and on Create writes the
reflection set (per-scan reflections.json + generated reflections.py) and a PNG.

First component of the pyqtgraph GUI rewrite: pyqtgraph gives fast image
display, draggable band regions, and click-on-histogram-to-set-2θ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout,
)

from ..config import DataManager
from ..core import io
from ..core import reflections as refl_io

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

_RING_PEN = pg.mkPen((255, 255, 255, 180), width=1)
_BAND_BRUSH = pg.mkBrush(255, 165, 0, 60)
_LINE_PEN = pg.mkPen((255, 80, 80), width=1)


class ReflectionDialog(QDialog):
    def __init__(self, project_root, scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.dm = DataManager(project_root, scan=scan)
        self.scan = scan
        self.bin_size = bin_size
        self.reflections = list(refl_io.read_json(self.dm.reflections_json(scan)))
        self._image = None          # raw summed image
        self._cleaned = None        # summed image after optional noise reduction
        self._tth = None            # currently displayed 2θ map (may be edited)
        self._tth_raw = None        # 2θ map as loaded from disk
        self._centers = None
        self._profile = None
        self._rings = []
        self._center = None         # estimated (y0, x0) beam center
        self._det = None            # detector module (for noise reduction)
        self._tth_data = None

        self.setWindowTitle(f"Reflections — {self.dm.scan_name or project_root}")
        self.resize(1200, 720)
        lay = QVBoxLayout(self)

        # ---- figures (pyqtgraph) ---------------------------------------
        figs = QHBoxLayout()
        self.img_view = pg.ImageView()
        self.img_view.ui.roiBtn.hide()
        self.img_view.ui.menuBtn.hide()
        try:
            self.img_view.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            pass
        figs.addWidget(self.img_view, 1)

        self.hist = pg.PlotWidget()
        self.hist.setLabel("bottom", "2θ", units="deg")
        self.hist.setLabel("left", "mean intensity")
        self.hist.showGrid(x=True, y=True, alpha=0.3)
        self.hist.scene().sigMouseClicked.connect(self._on_hist_click)
        figs.addWidget(self.hist, 1)
        lay.addLayout(figs, 1)

        # ---- compute controls ------------------------------------------
        crow = QHBoxLayout()
        crow.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["2θ map (tth.tiff)", "Summed image"])
        self.view_combo.currentIndexChanged.connect(self._redraw_image)
        crow.addWidget(self.view_combo)
        crow.addWidget(QLabel("Bin:"))
        self.bin_combo = QComboBox()
        self.bin_combo.addItems([f"{b}x{b}" for b in (1, 3, 4, 5)])
        self.bin_combo.setCurrentText(f"{self.bin_size}x{self.bin_size}")
        crow.addWidget(self.bin_combo)
        self.noise_cb = QCheckBox("Noise reduction")
        self.noise_cb.setToolTip(
            "Radial background subtraction before the 2θ histogram is computed")
        crow.addWidget(self.noise_cb)
        crow.addWidget(QLabel("Max bins to sum (0 = all):"))
        self.max_bins = QSpinBox()
        self.max_bins.setRange(0, 1_000_000)
        crow.addWidget(self.max_bins)
        compute_btn = QPushButton("Compute histogram")
        compute_btn.clicked.connect(self._compute)
        crow.addWidget(compute_btn)
        self.status = QLabel("click 'Compute' (then click the histogram to set 2θ)")
        crow.addWidget(self.status, 1)
        lay.addLayout(crow)

        # ---- geometry / 2θ calibration (edit the tth map) --------------
        grow = QHBoxLayout()
        self.center_label = QLabel("beam center: —")
        self.center_label.setStyleSheet("color:#555;")
        grow.addWidget(self.center_label)
        grow.addSpacing(12)
        grow.addWidget(QLabel("Detected 2θ:"))
        self.detect_combo = QComboBox()
        self.detect_combo.setMinimumWidth(110)
        self.detect_combo.activated.connect(self._on_detected)
        grow.addWidget(self.detect_combo)
        grow.addSpacing(12)
        grow.addWidget(QLabel("2θ offset:"))
        self.tth_offset = QDoubleSpinBox()
        self.tth_offset.setRange(-20, 20)
        self.tth_offset.setDecimals(4)
        self.tth_offset.setSingleStep(0.01)
        grow.addWidget(self.tth_offset)
        grow.addWidget(QLabel("2θ scale:"))
        self.tth_scale = QDoubleSpinBox()
        self.tth_scale.setRange(0.5, 2.0)
        self.tth_scale.setDecimals(4)
        self.tth_scale.setSingleStep(0.001)
        self.tth_scale.setValue(1.0)
        grow.addWidget(self.tth_scale)
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Preview tth' = scale·tth + offset")
        apply_btn.clicked.connect(self._apply_geometry)
        grow.addWidget(apply_btn)
        self.save_tth_btn = QPushButton("Save tth…")
        self.save_tth_btn.clicked.connect(self._save_tth)
        grow.addWidget(self.save_tth_btn)
        grow.addStretch()
        lay.addLayout(grow)

        # ---- reflection entry ------------------------------------------
        erow = QHBoxLayout()
        erow.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setMaximumWidth(120)
        erow.addWidget(self.name_edit)
        erow.addWidget(QLabel("Bragg 2θ:"))
        self.tth_spin = QDoubleSpinBox()
        self.tth_spin.setRange(0, 180)
        self.tth_spin.setDecimals(4)
        self.tth_spin.setSingleStep(0.1)
        erow.addWidget(self.tth_spin)
        erow.addWidget(QLabel("Width ±:"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.01, 10)
        self.width_spin.setDecimals(3)
        self.width_spin.setValue(refl_io.DEFAULT_WIDTH)
        erow.addWidget(self.width_spin)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add)
        erow.addWidget(add_btn)
        erow.addStretch()
        lay.addLayout(erow)

        # ---- list + actions --------------------------------------------
        self.list = QListWidget()
        self.list.setMaximumHeight(120)
        lay.addWidget(self.list)

        brow = QHBoxLayout()
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._remove)
        brow.addWidget(rm_btn)
        brow.addStretch()
        create_btn = QPushButton("Create (save reflections + figure)")
        create_btn.clicked.connect(self._create)
        brow.addWidget(create_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        brow.addWidget(close_btn)
        lay.addLayout(brow)

        self._refresh_list()
        self._redraw_hist()
        self._load_tth_for_display()

    # ----- bin / noise reduction --------------------------------------
    def _selected_bin(self):
        return int(self.bin_combo.currentText().split("x")[0])

    def _apply_noise_reduction(self, image):
        """Radial background subtraction via the project detector module."""
        if self._det is None:
            det_path = self.dm.detector_script()
            self._det = io.load_module(det_path)
            self._tth_data = self._det.precompute_tth(self._tth_raw)
        return self._det.radial_median_subtract(image, self._tth_data)

    # ----- compute ----------------------------------------------------
    def _compute(self):
        bin_size = self._selected_bin()
        h5 = self.dm.binned_h5(bin_size)
        tth_path = self.dm.tth_map(scan=self.scan)
        if self._tth_raw is None and Path(tth_path).exists():
            self._tth_raw = io.load_tth_map(tth_path)
            self._tth = self._tth_raw.copy()
        if self._tth is None:
            self.status.setText("missing tth map (load tth.tiff or convert a .poni)")
            return
        if not Path(h5).exists():
            self.status.setText(f"missing bins for {bin_size}×{bin_size}: "
                                f"build them in Programs → Create bins")
            return
        self.status.setText("summing bins…")
        self.status.repaint()
        mb = self.max_bins.value() or None
        self._image = io.sum_binned_image(h5, max_bins=mb)

        self._cleaned = self._image
        if self.noise_cb.isChecked():
            try:
                self._cleaned = self._apply_noise_reduction(self._image)
            except Exception as e:
                self.status.setText(f"noise reduction failed: {e}")
                self._cleaned = self._image

        self._centers, self._profile = io.radial_profile(self._cleaned, self._tth)
        self._populate_detected()
        nr = " (noise-reduced)" if self.noise_cb.isChecked() else ""
        self.status.setText(
            f"summed {self._image.shape} @ {bin_size}×{bin_size}{nr}; "
            f"tth {self._tth.shape}")
        if not self.view_combo.currentText().startswith("2θ"):
            pass  # keep summed view
        self._redraw_image()
        self._redraw_hist()

    def _load_tth_for_display(self):
        """Load the linked tth.tiff so it can be shown before summing bins."""
        tth_path = self.dm.tth_map(scan=self.scan)
        if tth_path and Path(tth_path).exists():
            try:
                self._tth_raw = io.load_tth_map(tth_path)
                self._tth = self._tth_raw.copy()
                self._update_center()
                self.status.setText(
                    f"tth.tiff {self._tth.shape}  "
                    f"2θ range {float(self._tth.min()):.3f}–{float(self._tth.max()):.3f}°  "
                    f"({tth_path})")
            except Exception as e:
                self.status.setText(f"could not read tth.tiff: {e}")
        else:
            self.status.setText("no tth.tiff loaded — use Setup → Load tth.tiff")
        self._redraw_image()

    # ----- detected 2θ peaks + beam center ----------------------------
    def _populate_detected(self):
        """Auto-list 2θ peaks found in the radial profile (pick → fills 2θ)."""
        self.detect_combo.clear()
        self.detect_combo.addItem("—")
        if self._profile is None:
            return
        try:
            from scipy.signal import find_peaks
            prof = np.asarray(self._profile, dtype=float)
            finite = prof[np.isfinite(prof)]
            if finite.size == 0:
                return
            prom = max((finite.max() - finite.min()) * 0.03, 1e-9)
            idx, _ = find_peaks(prof, prominence=prom, distance=3)
            order = sorted(idx, key=lambda i: prof[i], reverse=True)[:20]
            for i in sorted(order):
                self.detect_combo.addItem(f"{self._centers[i]:.3f}°", float(self._centers[i]))
        except Exception:
            pass

    def _on_detected(self, _idx):
        val = self.detect_combo.currentData()
        if val is not None:
            self.tth_spin.setValue(float(val))
            self.status.setText(f"2θ set to {val:.4f} (enter a name and Add)")

    def _update_center(self):
        if self._tth_raw is None:
            return
        try:
            from ..core.processing import estimate_beam_center
            y0, x0 = estimate_beam_center(self._tth_raw)
            self._center = (y0, x0)
            self.center_label.setText(f"beam center (x, y): ({x0:.1f}, {y0:.1f})")
        except Exception:
            self.center_label.setText("beam center: —")

    # ----- geometry edit (2θ calibration) -----------------------------
    def _apply_geometry(self):
        if self._tth_raw is None:
            self.status.setText("no tth.tiff loaded to adjust")
            return
        scale = self.tth_scale.value()
        offset = self.tth_offset.value()
        self._tth = self._tth_raw * scale + offset
        if self._cleaned is not None:
            self._centers, self._profile = io.radial_profile(self._cleaned, self._tth)
            self._populate_detected()
        self.status.setText(
            f"preview: 2θ' = {scale:.4f}·2θ + {offset:.4f}  "
            f"(range {float(self._tth.min()):.3f}–{float(self._tth.max()):.3f}°)  "
            f"— Save tth… to keep")
        self._redraw_image()
        self._redraw_hist()

    def _save_tth(self):
        if self._tth is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Save 2θ map")
        box.setText("Save the adjusted 2θ map (tth.tiff)?")
        save_btn = box.addButton("Save", QMessageBox.AcceptRole)
        new_btn = box.addButton("Save as new", QMessageBox.ActionRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is None or clicked not in (save_btn, new_btn):
            return

        import tifffile
        if clicked is save_btn:
            path = Path(self.dm.tth_map(scan=self.scan))
        else:
            start = str(self.dm.metadata_dir / "tth_edited.tiff")
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save new 2θ map", start, "TIFF (*.tif *.tiff)")
            if not chosen:
                return
            path = Path(chosen)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(path), self._tth.astype(np.float32))
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Could not write {path}:\n{e}")
            return
        # The saved map becomes the new baseline; reset the offset/scale preview.
        self._tth_raw = self._tth.copy()
        self.tth_offset.blockSignals(True); self.tth_scale.blockSignals(True)
        self.tth_offset.setValue(0.0); self.tth_scale.setValue(1.0)
        self.tth_offset.blockSignals(False); self.tth_scale.blockSignals(False)
        self._update_center()
        self.status.setText(f"saved tth → {path}")

    def _redraw_image(self):
        vb = self.img_view.getView()
        for r in self._rings:
            vb.removeItem(r)
        self._rings = []

        summed = self._cleaned if self._cleaned is not None else self._image
        showing_tth = self.view_combo.currentText().startswith("2θ")
        shown = False
        if showing_tth and self._tth is not None:
            # Raw 2θ-per-pixel map (linear, not log) — this is the loaded tiff.
            self.img_view.setImage(self._tth, autoLevels=True)
            shown = True
        elif not showing_tth and summed is not None:
            self.img_view.setImage(np.log1p(np.clip(summed, 0, None)),
                                   autoLevels=True)
            shown = True
        elif summed is not None:
            self.img_view.setImage(np.log1p(np.clip(summed, 0, None)),
                                   autoLevels=True)
            shown = True
        elif self._tth is not None:
            self.img_view.setImage(self._tth, autoLevels=True)
            shown = True

        if shown and self._tth is not None:
            for r in self.reflections:
                iso = pg.IsocurveItem(data=self._tth, level=float(r["two_theta"]),
                                      pen=_RING_PEN)
                vb.addItem(iso)
                self._rings.append(iso)

    def _redraw_hist(self):
        self.hist.clear()
        if self._profile is not None:
            self.hist.plot(self._centers, self._profile, pen=pg.mkPen((70, 130, 220), width=1))
        for r in self.reflections:
            t, w = float(r["two_theta"]), float(r.get("width", refl_io.DEFAULT_WIDTH))
            region = pg.LinearRegionItem(values=[t - w, t + w], movable=False, brush=_BAND_BRUSH)
            region.setZValue(-10)
            self.hist.addItem(region)
            line = pg.InfiniteLine(pos=t, angle=90, pen=_LINE_PEN,
                                   label=r["name"], labelOpts={"position": 0.9})
            self.hist.addItem(line)

    # ----- interactivity ----------------------------------------------
    def _on_hist_click(self, ev):
        vb = self.hist.getViewBox()
        if vb is None:
            return
        pt = vb.mapSceneToView(ev.scenePos())
        self.tth_spin.setValue(float(pt.x()))
        self.status.setText(f"2θ set to {pt.x():.4f} (enter a name and Add)")

    # ----- list mgmt --------------------------------------------------
    def _add(self):
        name = self.name_edit.text().strip()
        if not name:
            self.status.setText("enter a reflection name")
            return
        self.reflections.append({
            "name": name,
            "two_theta": self.tth_spin.value(),
            "width": self.width_spin.value(),
        })
        self.name_edit.clear()
        self._refresh_list()
        self._redraw_image()
        self._redraw_hist()

    def _remove(self):
        i = self.list.currentRow()
        if 0 <= i < len(self.reflections):
            self.reflections.pop(i)
            self._refresh_list()
            self._redraw_image()
            self._redraw_hist()

    def _refresh_list(self):
        self.list.clear()
        for r in self.reflections:
            self.list.addItem(
                f"{r['name']:<10}  2θ={float(r['two_theta']):.4f}  "
                f"±{float(r.get('width', refl_io.DEFAULT_WIDTH)):.3f}")

    # ----- save -------------------------------------------------------
    def _create(self):
        if not self.reflections:
            self.status.setText("no reflections to save")
            return
        mdir = self.dm.metadata_scan_dir(self.scan) if self.scan else self.dm.metadata_dir
        json_path = mdir / "reflections.json"
        refl_io.save(self.reflections, json_path, mdir / "reflections.py")

        fig_path = None
        if self._profile is not None:
            fdir = self.dm.figures_dir
            fdir.mkdir(parents=True, exist_ok=True)
            fig_path = fdir / f"{self.dm.scan_name or 'project'}_reflections.png"
            try:
                exporter = pg.exporters.ImageExporter(self.hist.plotItem)
                exporter.export(str(fig_path))
            except Exception:
                fig_path = None
        QMessageBox.information(
            self, "Saved",
            f"Wrote {len(self.reflections)} reflections:\n{json_path}\n"
            + (f"Figure: {fig_path}" if fig_path else "(compute the histogram to also save a figure)"))
        self.accept()


def open_reflection_dialog(project_root, scan=None, bin_size=3, parent=None):
    return ReflectionDialog(project_root, scan=scan, bin_size=bin_size, parent=parent)
