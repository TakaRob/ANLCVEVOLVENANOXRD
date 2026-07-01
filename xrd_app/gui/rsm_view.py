"""Reciprocal-space (RSM) view — projections + per-grain feature cloud (pyqtgraph).

Render-only face over ``core/rsm.py``:

* the **RSM heatmap** is a max-intensity projection of the fused 3D volume
  (``Study/rsm.npz`` from ``xrd-app rsm``), shown on true q-axes (1/Å);
* the **feature cloud** scatters every detected grain from the per-scan
  ``Study/qspace/<scan>_features_q.csv`` (from ``xrd-app qspace``) at its
  reciprocal-space position, colored by reflection, θ, or intensity.

All data prep lives in ``core.rsm`` (``load_rsm`` / ``load_feature_cloud``); this
module only draws. Missing artifacts degrade to an inline hint, never a crash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..core import rsm as rsm_core
from .palette import ARC_COLORS, _get_cmap

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

# The three orthogonal planes: label -> (proj key, q-index for x, y, and the
# projected-out axis used for the scatter's in-plane coords).
_PLANES = {
    "qx–qy (top)":  ("qx_qy", 0, 1),
    "qx–qz (side)": ("qx_qz", 0, 2),
    "qy–qz (front)": ("qy_qz", 1, 2),
}
_AXIS_LABELS = {0: "qx", 1: "qy", 2: "qz"}
_COLOR_MODES = ["reflection", "θ", "intensity"]


class RSMView(QWidget):
    """Self-contained tab widget; reads Study/ artifacts under ``project_root``."""

    def __init__(self, project_root=".", scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.scan = scan
        self._study = self.project_root / "Study"
        self._rsm_path = self._study / "rsm.npz"
        self._qspace_dir = self._study / "qspace"
        self._rsm = None
        self._cloud = None
        self._colorbar = None

        self._build_ui()
        self._reload()

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()

        bar.addWidget(QLabel("Plane:"))
        self.plane_cb = QComboBox()
        self.plane_cb.addItems(list(_PLANES))
        self.plane_cb.currentIndexChanged.connect(self._refresh)
        bar.addWidget(self.plane_cb)

        self.heat_chk = QCheckBox("RSM heatmap")
        self.heat_chk.setChecked(True)
        self.heat_chk.stateChanged.connect(self._refresh)
        bar.addWidget(self.heat_chk)

        self.log_chk = QCheckBox("log I")
        self.log_chk.setChecked(True)
        self.log_chk.stateChanged.connect(self._refresh)
        bar.addWidget(self.log_chk)

        self.cloud_chk = QCheckBox("Feature cloud")
        self.cloud_chk.setChecked(True)
        self.cloud_chk.stateChanged.connect(self._refresh)
        bar.addWidget(self.cloud_chk)

        bar.addWidget(QLabel("Color by:"))
        self.color_cb = QComboBox()
        self.color_cb.addItems(_COLOR_MODES)
        self.color_cb.currentIndexChanged.connect(self._refresh)
        bar.addWidget(self.color_cb)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload)
        bar.addWidget(reload_btn)

        bar.addStretch(1)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #888;")
        bar.addWidget(self.status)
        root.addLayout(bar)

        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.plot.setAspectLocked(False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.legend = None
        root.addWidget(self.glw, 1)

    # ---- data --------------------------------------------------------------
    def _reload(self):
        try:
            self._rsm = rsm_core.load_rsm(self._rsm_path) if self._rsm_path.exists() else None
        except Exception as e:
            self._rsm = None
            self.status.setText(f"rsm.npz error: {e}")
        try:
            self._cloud = (rsm_core.load_feature_cloud(self._qspace_dir)
                           if self._qspace_dir.is_dir() else None)
        except Exception as e:
            self._cloud = None
            self.status.setText(f"features error: {e}")
        self._refresh()

    # ---- rendering ---------------------------------------------------------
    def _clear(self):
        self.plot.clear()
        if self.legend is not None:
            try:
                self.legend.scene().removeItem(self.legend)
            except Exception:
                pass
            self.legend = None
        if self._colorbar is not None:
            try:
                self.glw.removeItem(self._colorbar)
            except Exception:
                pass
            self._colorbar = None

    def _refresh(self):
        self._clear()
        have_rsm = self._rsm is not None
        have_cloud = self._cloud is not None and self._cloud["n"] > 0
        if not have_rsm and not have_cloud:
            self.status.setText(
                "No RSM data. Run `xrd-app qspace` then `xrd-app rsm` "
                "(writes Study/rsm.npz + Study/qspace/*_features_q.csv).")
            return

        plane_label = self.plane_cb.currentText()
        proj_key, ax_x, ax_y = _PLANES[plane_label]
        self.plot.setLabel("bottom", f"{_AXIS_LABELS[ax_x]} (1/Å)")
        self.plot.setLabel("left", f"{_AXIS_LABELS[ax_y]} (1/Å)")

        if have_rsm and self.heat_chk.isChecked():
            self._draw_heatmap(proj_key, ax_x, ax_y)
        if have_cloud and self.cloud_chk.isChecked():
            self._draw_cloud(ax_x, ax_y)

        bits = []
        if have_rsm:
            bits.append(f"{len(self._rsm['scans'])} scans fused")
        if have_cloud:
            bits.append(f"{self._cloud['n']} features")
        self.status.setText("  •  ".join(bits))

    def _draw_heatmap(self, proj_key, ax_x, ax_y):
        proj = np.asarray(self._rsm["proj"][proj_key], dtype=np.float64)
        edges = self._rsm["edges"]
        ex, ey = edges[ax_x], edges[ax_y]
        img = proj.copy()
        if self.log_chk.isChecked():
            img = np.log1p(np.clip(img, 0, None))
        item = pg.ImageItem()
        # proj axis0 = ax_x (horizontal), axis1 = ax_y (vertical); row-major
        # wants [row=y, col=x], so transpose.
        item.setImage(img.T, autoLevels=False)
        finite = img[np.isfinite(img)]
        if finite.size:
            lo = float(np.percentile(finite, 1.0))
            hi = float(np.percentile(finite, 99.5))
            if hi <= lo:
                hi = lo + 1.0
            item.setLevels((lo, hi))
        try:
            item.setLookupTable(_get_cmap("inferno").getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        x0, y0 = float(ex[0]), float(ey[0])
        item.setRect(pg.QtCore.QRectF(x0, y0, float(ex[-1] - x0), float(ey[-1] - y0)))
        item.setZValue(-10)
        self.plot.addItem(item)

    def _draw_cloud(self, ax_x, ax_y):
        c = self._cloud
        coords = {0: c["qx"], 1: c["qy"], 2: c["qz"]}
        xs, ys = coords[ax_x], coords[ax_y]
        good = np.isfinite(xs) & np.isfinite(ys)
        mode = self.color_cb.currentText()

        if mode == "reflection":
            self.legend = self.plot.addLegend(offset=(-10, 10))
            refls = [c["reflection"][i] for i in range(c["n"]) if good[i]]
            uniq = sorted(set(refls))
            for k, ref in enumerate(uniq):
                sel = good & np.array([c["reflection"][i] == ref for i in range(c["n"])])
                if not np.any(sel):
                    continue
                col = QColor(ARC_COLORS[k % len(ARC_COLORS)])
                sp = pg.ScatterPlotItem(
                    x=xs[sel], y=ys[sel], size=7, pen=pg.mkPen("k", width=0.3),
                    brush=pg.mkBrush(col), name=(ref or "?"))
                self.plot.addItem(sp)
            return

        # continuous color by θ or intensity
        vals = c["theta"].copy() if mode == "θ" else c["intensity"].copy()
        sel = good & np.isfinite(vals)
        if not np.any(sel):
            return
        v = vals[sel]
        if mode == "intensity":
            v = np.log1p(np.clip(v, 0, None))
        vmin, vmax = float(np.min(v)), float(np.max(v))
        norm = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
        cmap = _get_cmap("viridis")
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        idx = np.clip((norm * 255).astype(int), 0, 255)
        brushes = [pg.mkBrush(int(lut[i][0]), int(lut[i][1]), int(lut[i][2])) for i in idx]
        sp = pg.ScatterPlotItem(
            x=xs[sel], y=ys[sel], size=7, pen=pg.mkPen("k", width=0.3), brush=brushes)
        self.plot.addItem(sp)
        try:
            bar = pg.ColorBarItem(values=(vmin, vmax), colorMap=cmap,
                                  label=("log I" if mode == "intensity" else "θ (°)"))
            self.glw.addItem(bar)
            self._colorbar = bar
        except Exception:
            self._colorbar = None
