"""Territorial (cell-model) device map + shape inspector (pyqtgraph).

Renders the **variable-footprint** territories of the skew-free reference
binning (``core/territory.py``) to-scale: each territory is drawn as its true
(X, Y) polygon, coloured by a switchable metric (frame count, cell area, or a
linked shape's per-territory peak intensity). Selecting a shape in the list
highlights the territories it spans.

This is a focused, self-contained viewer (not a clone of the 4k-line grid
viewer): the grid device-map assumes one pixel per ``"r_c"`` bin, which cannot
represent irregular territories, so this draws real polygons instead. It reads
the territorial grid mapping (``..._territory.json``, which carries each cell's
polygon / centroid / area / count) and the territorial shapes catalog.

``build_window(project_root, scan, bin_size, catalog=None)`` mirrors
``gui.device_map`` so the tab wrappers and standalone runner work unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QColor, QPolygonF, QBrush
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QGroupBox, QSplitter, QGraphicsPolygonItem,
)

from ..config import DataManager
from . import palette

pg.setConfigOptions(antialias=True)

_BASE_PEN = pg.mkPen(QColor(40, 40, 40, 120), width=0)
_HILITE_PEN = pg.mkPen(QColor("#00e5ff"), width=2)
_DIM = QColor(70, 70, 70, 90)


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────
def _load_territory_mapping(dm: DataManager, scan, bin_size) -> dict | None:
    """The territorial grid mapping (carries the ``territories`` polygon block)."""
    gm_path = dm.grid_mapping(bin_size=1, variant="territory", scan=scan)
    if not Path(gm_path).exists():
        return None
    with open(gm_path) as f:
        gm = json.load(f)
    return gm if gm.get("territories") else None


def _load_shapes(dm: DataManager, scan, catalog=None) -> list:
    """Kept features from the territorial shapes catalog (empty if none yet)."""
    if catalog and Path(catalog).exists():
        path = Path(catalog)
    else:
        path = dm.shapes_json("territory", 1, scan, variant="territory")
        if not Path(path).exists():
            # Fall back to any territorial shapes file in the labels dir.
            ldir = dm.labels_dir(scan)
            hits = sorted(ldir.glob("*_shapes*territory*.json")) if ldir.is_dir() else []
            if not hits:
                return []
            path = hits[-1]
    with open(path) as f:
        data = json.load(f)
    return data.get("kept", [])


# ─────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────
def _lut(cmap_name: str):
    return palette._get_cmap(cmap_name).getLookupTable(0.0, 1.0, 256)


def _scalar_color(lut, t: float) -> QColor:
    i = int(np.clip(t, 0.0, 1.0) * 255)
    r, g, b = int(lut[i][0]), int(lut[i][1]), int(lut[i][2])
    return QColor(r, g, b)


# ─────────────────────────────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────────────────────────────
class TerritoryCanvas(pg.PlotWidget):
    """Draws one filled polygon per territory; recolours by metric in place."""

    def __init__(self, territories: dict):
        super().__init__()
        self.territories = territories
        self.items: dict = {}            # territory key -> QGraphicsPolygonItem
        vb = self.getViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)                 # match the device-map (X right, Y down)
        self.setBackground("#101014")
        self.setMenuEnabled(False)
        self._build()

    def _build(self):
        vb = self.getViewBox()
        for key, info in self.territories.items():
            poly = info.get("polygon") or []
            if len(poly) < 3:
                continue
            qpoly = QPolygonF([QPointF(float(x), float(y)) for x, y in poly])
            item = QGraphicsPolygonItem(qpoly)
            item.setPen(_BASE_PEN)
            item.setBrush(QBrush(_DIM))
            vb.addItem(item)
            self.items[key] = item
        vb.autoRange()

    def color_by_values(self, values: dict, cmap_name: str, log_scale=False):
        """Colour each territory by ``values[key]`` (missing keys → dim)."""
        lut = _lut(cmap_name)
        vals = np.array([v for v in values.values() if v is not None], dtype=float)
        if len(vals) == 0:
            for it in self.items.values():
                it.setBrush(QBrush(_DIM))
            return
        if log_scale:
            vals = np.log1p(vals)
        lo, hi = float(vals.min()), float(vals.max())
        span = (hi - lo) or 1.0
        for key, item in self.items.items():
            v = values.get(key)
            if v is None:
                item.setBrush(QBrush(_DIM))
                continue
            t = ((np.log1p(v) if log_scale else v) - lo) / span
            item.setBrush(QBrush(_scalar_color(lut, t)))

    def highlight(self, keys):
        """Outline ``keys`` in the highlight pen; reset everything else."""
        keyset = set(keys or [])
        for key, item in self.items.items():
            item.setPen(_HILITE_PEN if key in keyset else _BASE_PEN)
            item.setZValue(1 if key in keyset else 0)


# ─────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────
class TerritoryMap(QWidget):
    def __init__(self, gm: dict, shapes: list):
        super().__init__()
        self.territories = gm["territories"]
        self.shapes = shapes
        self._build_ui(gm)
        self._refresh_metric()

    def _build_ui(self, gm: dict):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        # ── left: controls ───────────────────────────────────────────
        left = QWidget()
        lyt = QVBoxLayout(left)
        n_terr = len(self.territories)
        counts = [t.get("count", 0) for t in self.territories.values()]
        lyt.addWidget(QLabel(
            f"<b>{n_terr}</b> territories · target {gm.get('target_size','?')} "
            f"frames/cell<br>{gm.get('n_total_frames','?')} frames · "
            f"step {gm.get('step', 0):.3g}"))

        box = QGroupBox("Colour by")
        bl = QVBoxLayout(box)
        self.metric = QComboBox()
        self.metric.addItems(["Frame count", "Cell area", "Shape peak intensity"])
        bl.addWidget(self.metric)
        self.reflection = QComboBox()
        self.reflection.addItem("(all reflections)")
        for r in sorted({s.get("reflection", "?") for s in self.shapes}):
            self.reflection.addItem(r)
        bl.addWidget(QLabel("Reflection (shapes):"))
        bl.addWidget(self.reflection)
        self.cmap = QComboBox()
        self.cmap.addItems(palette.COLORMAPS)
        bl.addWidget(QLabel("Colormap:"))
        bl.addWidget(self.cmap)
        lyt.addWidget(box)

        sbox = QGroupBox(f"Shapes ({len(self.shapes)} kept)")
        sl = QVBoxLayout(sbox)
        self.flist = QListWidget()
        for s in self.shapes:
            label = (f"#{s.get('feature_id','?')} {s.get('reflection','?')} · "
                     f"{s.get('n_bins','?')} cells · χ={s.get('chi_deg','?')}°")
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, s)
            self.flist.addItem(it)
        sl.addWidget(self.flist)
        lyt.addWidget(sbox, 1)

        self.info = QLabel("Select a shape to inspect its territories.")
        self.info.setWordWrap(True)
        lyt.addWidget(self.info)

        # ── right: canvas ────────────────────────────────────────────
        self.canvas = TerritoryCanvas(self.territories)

        split.addWidget(left)
        split.addWidget(self.canvas)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 900])

        self.metric.currentIndexChanged.connect(self._refresh_metric)
        self.reflection.currentIndexChanged.connect(self._refresh_metric)
        self.cmap.currentIndexChanged.connect(self._refresh_metric)
        self.flist.currentItemChanged.connect(self._on_select)

    # ── metric colouring ─────────────────────────────────────────────
    def _refresh_metric(self):
        mode = self.metric.currentText()
        cmap = self.cmap.currentText()
        if mode == "Frame count":
            vals = {k: t.get("count") for k, t in self.territories.items()}
            self.canvas.color_by_values(vals, cmap)
        elif mode == "Cell area":
            vals = {k: t.get("area") for k, t in self.territories.items()}
            self.canvas.color_by_values(vals, cmap, log_scale=True)
        else:
            self.canvas.color_by_values(self._shape_intensity_by_territory(), cmap,
                                        log_scale=True)

    def _shape_intensity_by_territory(self) -> dict:
        """Max per-territory peak intensity over kept shapes (reflection-filtered)."""
        want = self.reflection.currentText()
        out: dict = {}
        for s in self.shapes:
            if want != "(all reflections)" and s.get("reflection") != want:
                continue
            for key, entry in (s.get("intensity_profile") or {}).items():
                if isinstance(entry, dict):
                    v = entry.get("intensity", 0)
                    out[key] = max(out.get(key, 0), v)
        return out

    # ── selection ────────────────────────────────────────────────────
    def _on_select(self, cur, _prev):
        if cur is None:
            self.canvas.highlight([])
            return
        s = cur.data(Qt.UserRole)
        self.canvas.highlight(s.get("spatial_extent", []))
        self.info.setText(
            f"<b>#{s.get('feature_id','?')} {s.get('reflection','?')}</b><br>"
            f"cells: {s.get('n_bins','?')} · peak I: {s.get('peak_intensity','?')} · "
            f"SNR: {s.get('mean_snr','?')}<br>"
            f"χ = {s.get('chi_deg','?')}° · χ FWHM = {s.get('chi_fwhm','?')} · "
            f"Δ2θ FWHM = {s.get('tth_fwhm','?')}<br>"
            f"detector: ({s.get('detector_x','?')}, {s.get('detector_y','?')})<br>"
            f"<i>{s.get('reason','')}</i>")


# ─────────────────────────────────────────────────────────────────────
# Window / tab entry points
# ─────────────────────────────────────────────────────────────────────
def _placeholder(msg: str) -> QWidget:
    w = QWidget()
    lyt = QVBoxLayout(w)
    lbl = QLabel(msg)
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignCenter)
    lyt.addWidget(lbl)
    return w


def build_window(project_root=".", scan=None, bin_size=1, catalog=None) -> QWidget:
    """Build the territorial device-map widget for the current scan.

    Returns a placeholder with instructions when the territorial grid mapping is
    missing, so the tab never crashes the window.
    """
    dm = DataManager(project_root, scan=scan)
    gm = _load_territory_mapping(dm, scan, bin_size)
    if gm is None:
        return _placeholder(
            "No territorial grid mapping for this scan.\n\n"
            "Build one with:\n"
            "    xrd-app territory-grid --target-size 9\n"
            "    xrd-app bin    --bin-size 1 --variant territory\n"
            "    xrd-app peaks  --bin-size 1 --variant territory\n"
            "    xrd-app shapes --bin-size 1 --variant territory --algorithm territory")
    shapes = _load_shapes(dm, scan, catalog)
    return TerritoryMap(gm, shapes)


def launch_gui(project_root=".", scan=None, bin_size=1):
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Territorial Device Map")
    win.setCentralWidget(build_window(project_root, scan=scan, bin_size=bin_size))
    win.resize(1280, 820)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_gui()
