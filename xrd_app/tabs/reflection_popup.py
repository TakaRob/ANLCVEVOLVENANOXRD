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
    QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
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
        self._image = None
        self._tth = None
        self._centers = None
        self._profile = None
        self._rings = []

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

    # ----- compute ----------------------------------------------------
    def _compute(self):
        h5 = self.dm.binned_h5(self.bin_size)
        tth_path = self.dm.tth_map(scan=self.scan)
        if not Path(h5).exists():
            self.status.setText(f"missing bins: {h5}")
            return
        if not Path(tth_path).exists():
            self.status.setText("missing tth map (load tth.tiff or convert a .poni)")
            return
        self.status.setText("summing bins…")
        self.status.repaint()
        mb = self.max_bins.value() or None
        self._image = io.sum_binned_image(h5, max_bins=mb)
        self._tth = io.load_tth_map(tth_path)
        self._centers, self._profile = io.radial_profile(self._image, self._tth)
        self.status.setText(f"summed {self._image.shape}; tth {self._tth.shape}")
        self._redraw_image()
        self._redraw_hist()

    def _redraw_image(self):
        if self._image is None:
            return
        disp = np.log1p(np.clip(self._image, 0, None))
        self.img_view.setImage(disp, autoLevels=True)
        vb = self.img_view.getView()
        for r in self._rings:
            vb.removeItem(r)
        self._rings = []
        if self._tth is not None:
            for r in self.reflections:
                iso = pg.IsocurveItem(data=self._tth, level=float(r["two_theta"]), pen=_RING_PEN)
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
