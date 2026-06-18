"""Manual reflection popup.

Computes a radial (2θ) histogram on the fully-summed #x# binned image, shows the
summed image with 2θ ring overlays and the histogram with reflection bands, lets
the user enter [Name, Bragg angle, Width], and on Create writes the reflection
set (per-scan reflections.json + generated reflections.py) and saves PNGs.

Uses matplotlib for now (consistent with the other tabs); a pyqtgraph rewrite is
deferred to the final phase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import io
from ..core import reflections as refl_io


class ReflectionDialog(QDialog):
    def __init__(self, project_root, scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.dm = DataManager(project_root, scan=scan)
        self.scan = scan
        self.bin_size = bin_size
        self.reflections = list(refl_io.read_json(self.dm.reflections_json(scan)))
        self._image = None
        self._centers = None
        self._profile = None

        self.setWindowTitle(f"Reflections — {self.dm.scan_name or project_root}")
        self.resize(1200, 720)

        lay = QVBoxLayout(self)

        # ---- figures ----------------------------------------------------
        figs = QHBoxLayout()
        self.img_fig = Figure(figsize=(5, 5))
        self.img_canvas = FigureCanvasQTAgg(self.img_fig)
        self.img_ax = self.img_fig.add_subplot(111)
        self.img_ax.set_title("Fully-summed image — click 'Compute'")
        figs.addWidget(self.img_canvas, 1)

        self.hist_fig = Figure(figsize=(5, 5))
        self.hist_canvas = FigureCanvasQTAgg(self.hist_fig)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_ax.set_title("Radial 2θ histogram")
        self.hist_ax.set_xlabel("2θ (degrees)")
        self.hist_ax.set_ylabel("mean intensity")
        figs.addWidget(self.hist_canvas, 1)
        lay.addLayout(figs, 1)

        # ---- compute controls ------------------------------------------
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Max bins to sum (0 = all):"))
        self.max_bins = QSpinBox()
        self.max_bins.setRange(0, 1_000_000)
        self.max_bins.setValue(0)
        crow.addWidget(self.max_bins)
        compute_btn = QPushButton("Compute histogram")
        compute_btn.clicked.connect(self._compute)
        crow.addWidget(compute_btn)
        self.status = QLabel("")
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
        tth = io.load_tth_map(tth_path)
        self._tth = tth
        self._centers, self._profile = io.radial_profile(self._image, tth)
        self.status.setText(f"summed {self._image.shape} image; tth {tth.shape}")
        self._redraw()

    def _redraw(self):
        # image
        self.img_ax.clear()
        if self._image is not None:
            vmax = np.percentile(self._image[self._image > 0], 99) if np.any(self._image > 0) else 1
            self.img_ax.imshow(self._image, origin="lower", cmap="viridis",
                               norm=mcolors.LogNorm(vmin=1, vmax=max(vmax, 2)))
            # 2θ ring overlays at each reflection (contour of the tth map)
            if getattr(self, "_tth", None) is not None and self.reflections:
                levels = sorted(float(r["two_theta"]) for r in self.reflections)
                try:
                    self.img_ax.contour(self._tth, levels=levels, colors="white",
                                        linewidths=0.6, alpha=0.7)
                except Exception:
                    pass
        self.img_ax.set_title("Fully-summed image (log) with 2θ rings")
        self.img_canvas.draw_idle()

        # histogram
        self.hist_ax.clear()
        if self._profile is not None:
            self.hist_ax.plot(self._centers, self._profile, lw=1.0, color="#1f77b4")
        for r in self.reflections:
            t, w = float(r["two_theta"]), float(r.get("width", refl_io.DEFAULT_WIDTH))
            self.hist_ax.axvspan(t - w, t + w, color="orange", alpha=0.2)
            self.hist_ax.axvline(t, color="red", lw=0.8)
            top = self.hist_ax.get_ylim()[1]
            self.hist_ax.text(t, top, r["name"], rotation=90, va="top", ha="right", fontsize=7)
        self.hist_ax.set_xlabel("2θ (degrees)")
        self.hist_ax.set_ylabel("mean intensity")
        self.hist_ax.set_title("Radial 2θ histogram")
        self.hist_canvas.draw_idle()

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
        self._redraw()

    def _remove(self):
        i = self.list.currentRow()
        if 0 <= i < len(self.reflections):
            self.reflections.pop(i)
            self._refresh_list()
            self._redraw()

    def _refresh_list(self):
        self.list.clear()
        for r in self.reflections:
            self.list.addItem(
                f"{r['name']:<10}  2θ={float(r['two_theta']):.4f}  ±{float(r.get('width', refl_io.DEFAULT_WIDTH)):.3f}")

    # ----- save -------------------------------------------------------
    def _create(self):
        if not self.reflections:
            self.status.setText("no reflections to save")
            return
        mdir = self.dm.metadata_scan_dir(self.scan) if self.scan else self.dm.metadata_dir
        json_path = mdir / "reflections.json"
        refl_io.save(self.reflections, json_path, mdir / "reflections.py")

        fig_path = None
        if self._image is not None:
            fdir = self.dm.figures_dir
            fdir.mkdir(parents=True, exist_ok=True)
            fig_path = fdir / f"{self.dm.scan_name or 'project'}_reflections.png"
            self.hist_fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        QMessageBox.information(
            self, "Saved",
            f"Wrote {len(self.reflections)} reflections:\n{json_path}\n"
            + (f"Figure: {fig_path}" if fig_path else "(compute the histogram to also save a figure)"))
        self.accept()


def open_reflection_dialog(project_root, scan=None, bin_size=3, parent=None):
    dlg = ReflectionDialog(project_root, scan=scan, bin_size=bin_size, parent=parent)
    return dlg
