"""HD Device View — 1×1 intensity beneath the N×N feature map (pyqtgraph).

The binned Device View (:mod:`gui.device_map`) shows one value per N×N bin. This
view keeps the same N×N feature *segmentation* (outlines colored by reflection)
but paints the heatmap **beneath** them at true 1×1 resolution — the raw
intensity at each feature's detector peak sampled per unbinned pixel by
``xrd-app hd-device-map`` (see :mod:`core.hd_map`). At 1×1 you see the real scan
with all its holes and 9× finer spatial detail, so nearly-overlapping features
separate.

Two display spaces, switchable:
  * **Grid (1×1)** — the fine regular-grid heatmap under the N×N outlines.
  * **Real position (x, y)** — each sampled pixel plotted at its true stage
    position colored by raw intensity (the actual scan geometry, holes included).

The heavy sampling lives in the CLI/``core``; this module only renders the cached
``*_hdmap_NxN.json``. Rendering helpers (colormap, RGBA, chi range slider) are
reused from :mod:`gui.device_map` so the two views stay visually consistent.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QPushButton, QGroupBox, QComboBox, QSplitter,
    QSpinBox, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ..config import DataManager
from ..core import hd_map as hd_core
from . import device_map as dv

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

_ISO_GHOST_ALPHA = 0   # opacity of everything but the isolated pick (0 = hide)

METRICS = [
    ("intensity",  "Intensity (peak max)"),
    ("integrated", "Integrated (window sum)"),
    ("none",       "None (outlines only)"),
]
METRIC_ZLABELS = {"intensity": "Peak intensity", "integrated": "Integrated intensity"}

DISPLAY_GRID = "grid"
DISPLAY_XY = "xy"


# ── catalog discovery / loading ────────────────────────────────────
def list_hd_catalogs(results_dir, bin_size):
    """``*_hdmap_<NxN>*.json`` in a scan's Labels dir, sorted by name."""
    rd = Path(results_dir)
    if not rd.is_dir():
        return []
    return sorted(rd.glob(f"*_hdmap_{bin_size}x{bin_size}*.json"))


def _resolve_catalog(results_dir, bin_size, catalog):
    if catalog:
        p = Path(catalog)
        return p if p.exists() else (Path(results_dir) / catalog)
    cats = list_hd_catalogs(results_dir, bin_size)
    return cats[-1] if cats else None


def _parse_cell(key):
    r, c = key.split("_")
    return int(r), int(c)


def load_hd_features(catalog_path):
    """Parse an hd_map JSON → (features, n_rows, n_cols, positions_real).

    Each returned feature gets a 1×1 boolean ``_mask``, cached ``center_row/col``
    and ``center_x/y`` for hover, and keeps its ``hd_profile`` cells.
    """
    with open(catalog_path) as f:
        data = json.load(f)
    feats = data.get("features", [])
    n_rows = int(data.get("n_bin_rows_1x1") or 0)
    n_cols = int(data.get("n_bin_cols_1x1") or 0)
    if not (n_rows and n_cols):
        mr = mc = -1
        for feat in feats:
            for k in feat.get("hd_profile", {}):
                r, c = _parse_cell(k)
                mr, mc = max(mr, r), max(mc, c)
        n_rows, n_cols = mr + 1, mc + 1

    for feat in feats:
        prof = feat.get("hd_profile", {})
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        rs, cs, xs, ys = [], [], [], []
        for k, e in prof.items():
            r, c = _parse_cell(k)
            if 0 <= r < n_rows and 0 <= c < n_cols:
                mask[r, c] = True
                rs.append(r); cs.append(c)
                if "x" in e and "y" in e:
                    xs.append(e["x"]); ys.append(e["y"])
        feat["_mask"] = mask
        feat["center_row"] = float(np.mean(rs)) if rs else 0.0
        feat["center_col"] = float(np.mean(cs)) if cs else 0.0
        feat["center_x"] = float(np.mean(xs)) if xs else None
        feat["center_y"] = float(np.mean(ys)) if ys else None
    return feats, n_rows, n_cols, bool(data.get("positions_real"))


# ── main window ────────────────────────────────────────────────────
class HDDeviceMapWindow(QMainWindow):
    def __init__(self, results_dir, bin_size, catalog_path, catalogs_list,
                 trajectory=None):
        super().__init__()
        self.setWindowTitle("HD Device View — 1×1 beneath the feature map")
        self.setGeometry(50, 30, 1450, 900)

        self._results_dir = results_dir
        self._bin_size = bin_size
        self._catalog_path = catalog_path
        self._catalogs = list(catalogs_list)
        self._trajectory = trajectory or {"grid": [], "xy": None}

        self.metric = "intensity"
        self.display = DISPLAY_GRID
        self.show_points = False
        self.show_trajectory = False
        self._isolate = True
        self._locked_idx = None
        self._highlighted_idx = None
        self._items = []          # transient overlay items (outlines/highlight/scatter)

        self._load(catalog_path)
        self._build_ui()
        self._redraw()

    # ----- data ------------------------------------------------------
    def _load(self, catalog_path):
        self.features, self.n_rows, self.n_cols, self.positions_real = \
            load_hd_features(catalog_path)
        self.reflections = sorted(set(f.get("reflection", "unknown")
                                      for f in self.features))
        self.ref_colors = dv._assign_ref_colors(self.reflections)
        chis = [f.get("chi_deg") for f in self.features if f.get("chi_deg") is not None]
        if chis:
            self._chi_data_min = int(np.floor(min(chis)))
            self._chi_data_max = int(np.ceil(max(chis)))
        else:
            self._chi_data_min, self._chi_data_max = -180, 180
        self._chi_lo = float(self._chi_data_min)
        self._chi_hi = float(self._chi_data_max)

    def _chi_range(self):
        if self._chi_lo <= self._chi_data_min and self._chi_hi >= self._chi_data_max:
            return None
        return (self._chi_lo, self._chi_hi)

    def _visible_refs(self):
        return [r for r in self.reflections if self.layer_cbs[r].isChecked()]

    def _feature_visible(self, feat, visible, chi_range):
        return (feat.get("reflection") in visible
                and dv._feat_in_chi_range(feat, chi_range))

    # ----- UI --------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: plot + hover line
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        self.plot.getViewBox().setBackgroundColor("w")
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.colorbar = None
        self.legend = self.plot.addLegend(offset=(-10, 10))
        ll.addWidget(self.glw, 1)
        self.hover_label = QLabel("Hover over a feature to see details")
        self.hover_label.setStyleSheet(
            "font-family: monospace; font-size: 0.9em; color: #555; "
            "padding: 4px; background: #f0f0f0;")
        self.hover_label.setFixedHeight(22)
        ll.addWidget(self.hover_label)
        splitter.addWidget(left)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        # Right: controls
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)

        # Catalog selector (multiple hd maps for a bin)
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("HD map:"))
        self.cat_combo = QComboBox()
        for p in self._catalogs:
            self.cat_combo.addItem(p.name, str(p))
        i = self.cat_combo.findData(str(self._catalog_path))
        self.cat_combo.setCurrentIndex(i if i >= 0 else 0)
        self.cat_combo.currentIndexChanged.connect(self._on_catalog_changed)
        cat_row.addWidget(self.cat_combo, 1)
        rl.addLayout(cat_row)

        # Display space
        disp = QGroupBox("Display space")
        dl = QVBoxLayout(disp)
        self.display_combo = QComboBox()
        self.display_combo.addItem("Grid (1×1)", DISPLAY_GRID)
        self.display_combo.addItem("Real position (x, y)", DISPLAY_XY)
        if not self.positions_real:
            # Disable the real-position row with an explanatory tooltip.
            self.display_combo.model().item(1).setEnabled(False)
            self.display_combo.setToolTip(
                "Real-position scatter needs a real stage position CSV; this "
                "hd map has none (positions_real = false).")
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        dl.addWidget(self.display_combo)
        rl.addWidget(disp)

        # Layers
        lg = QGroupBox("Layers")
        lgl = QVBoxLayout(lg)
        brow = QHBoxLayout()
        ab = QPushButton("All"); ab.setFixedWidth(45); ab.clicked.connect(self._check_all)
        nb = QPushButton("None"); nb.setFixedWidth(45); nb.clicked.connect(self._uncheck_all)
        brow.addWidget(ab); brow.addWidget(nb); brow.addStretch()
        lgl.addLayout(brow)
        self.layer_cbs = {}
        row = QHBoxLayout()
        for i, ref in enumerate(self.reflections):
            cb = QCheckBox(ref)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {self.ref_colors[ref]}; }}")
            cb.toggled.connect(self._redraw)
            row.addWidget(cb)
            self.layer_cbs[ref] = cb
            if (i + 1) % 4 == 0:
                lgl.addLayout(row); row = QHBoxLayout()
        if row.count():
            lgl.addLayout(row)
        trow = QHBoxLayout()
        self.points_cb = QCheckBox("Points")
        self.points_cb.toggled.connect(self._on_points_toggle)
        trow.addWidget(self.points_cb)
        self.isolate_cb = QCheckBox("Isolate selection")
        self.isolate_cb.setChecked(self._isolate)
        self.isolate_cb.toggled.connect(self._on_isolate_toggle)
        trow.addWidget(self.isolate_cb); trow.addStretch()
        lgl.addLayout(trow)
        trow2 = QHBoxLayout()
        self.traj_cb = QCheckBox("Scan trajectory")
        self.traj_cb.setToolTip(
            "Dotted line following the beam's acquisition path (serpentine "
            "raster) through the 1×1 pixels — it threads through the pixels "
            "inside each feature.")
        has_traj = bool(self._trajectory.get("grid"))
        self.traj_cb.setEnabled(has_traj)
        if not has_traj:
            self.traj_cb.setToolTip("No scan trajectory (1×1 grid mapping unavailable).")
        self.traj_cb.setChecked(self.show_trajectory and has_traj)
        self.traj_cb.toggled.connect(self._on_trajectory_toggle)
        trow2.addWidget(self.traj_cb); trow2.addStretch()
        lgl.addLayout(trow2)
        rl.addWidget(lg)

        # Metric + contrast
        mg = QGroupBox("Metric")
        mgl = QVBoxLayout(mg)
        self.metric_combo = QComboBox()
        for key, label in METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        mgl.addWidget(self.metric_combo)
        ch = QHBoxLayout()
        ch.addWidget(QLabel("Contrast %:"))
        self.lo_spin = QSpinBox(); self.lo_spin.setRange(0, 100); self.lo_spin.setValue(2)
        self.lo_spin.valueChanged.connect(self._redraw)
        ch.addWidget(self.lo_spin)
        self.hi_spin = QSpinBox(); self.hi_spin.setRange(0, 100); self.hi_spin.setValue(99)
        self.hi_spin.valueChanged.connect(self._redraw)
        ch.addWidget(self.hi_spin); ch.addStretch()
        mgl.addLayout(ch)
        rl.addWidget(mg)

        # Chi range
        cg = QGroupBox("χ angle range")
        cgl = QVBoxLayout(cg)
        self.chi_slider = dv.QRangeSlider(self._chi_data_min, self._chi_data_max)
        self.chi_slider.rangeChanged.connect(self._on_chi_range)
        cgl.addWidget(self.chi_slider)
        self.chi_label = QLabel(f"χ: {self._chi_data_min}° to {self._chi_data_max}°")
        self.chi_label.setStyleSheet(
            "font-family: monospace; font-size: 0.9em; color:#555; padding:2px;")
        cgl.addWidget(self.chi_label)
        rl.addWidget(cg)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop)
        self.info_label.setStyleSheet("QLabel { font-size: 10pt; padding: 6px; }")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(self.info_label); scroll.setFrameShape(QScrollArea.NoFrame)
        rl.addWidget(scroll, 1)

        splitter.addWidget(right)
        splitter.setSizes([980, 420])

    # ----- grid / value helpers --------------------------------------
    def _combined_grid(self, visible, chi_range):
        grid = np.full((self.n_rows, self.n_cols), np.nan)
        for f in self.features:
            if not self._feature_visible(f, visible, chi_range):
                continue
            for k, e in f.get("hd_profile", {}).items():
                v = e.get(self.metric)
                if v is None:
                    continue
                r, c = _parse_cell(k)
                if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                    grid[r, c] = v if np.isnan(grid[r, c]) else max(grid[r, c], v)
        return grid

    def _levels(self, grid):
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(self.lo_spin.value()), float(self.hi_spin.value())
        if hi <= lo:
            hi = min(100.0, lo + 1.0)
        vmin, vmax = float(np.percentile(finite, lo)), float(np.percentile(finite, hi))
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1
        return vmin, vmax

    def _clear_items(self):
        for it in self._items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._items.clear()

    # ----- redraw ----------------------------------------------------
    def _redraw(self):
        self._clear_items()
        visible = self._visible_refs()
        chi_range = self._chi_range()
        isolate = (self._isolate and self._locked_idx is not None
                   and 0 <= self._locked_idx < len(self.features)
                   and self._feature_visible(
                       self.features[self._locked_idx], visible, chi_range))
        if self.display == DISPLAY_XY and self.positions_real:
            self.img_item.setVisible(False)
            self._redraw_xy(visible, chi_range, isolate)
        else:
            self.img_item.setVisible(True)
            self._redraw_grid(visible, chi_range, isolate)
        if self.show_trajectory:
            self._draw_trajectory(
                DISPLAY_XY if (self.display == DISPLAY_XY and self.positions_real)
                else DISPLAY_GRID)
        self._rebuild_legend(visible)
        if self._locked_idx is not None and isolate:
            self._show_feature_info(self._locked_idx)

    def _redraw_grid(self, visible, chi_range, isolate):
        self.plot.setLabel("bottom", "Col (1×1)")
        self.plot.setLabel("left", "Row (1×1)")
        self.plot.invertY(True)
        cmap = dv._get_cmap("viridis")
        if self.metric == "none":
            self.img_item.setVisible(False)
        else:
            iso_feat = self.features[self._locked_idx] if isolate else None
            if isolate:
                grid = np.full((self.n_rows, self.n_cols), np.nan)
                m = iso_feat.get("_mask")
                full = self._combined_grid(visible, chi_range)
                if m is not None:
                    grid[m] = full[m]
            else:
                grid = self._combined_grid(visible, chi_range)
            vmin, vmax = self._levels(grid)
            self.img_item.setVisible(True)
            self.img_item.setImage(dv._scalar_to_rgba(grid, vmin, vmax, cmap),
                                   autoLevels=False)
            self._update_colorbar(cmap, vmin, vmax)
        if self.metric == "none":
            self._update_colorbar(None, None, None)
        # Outlines per reflection
        for ref in visible:
            merged = np.zeros((self.n_rows, self.n_cols), dtype=bool)
            for f in self.features:
                if f.get("reflection") == ref and dv._feat_in_chi_range(f, chi_range):
                    if isolate and f is not self.features[self._locked_idx]:
                        continue
                    m = f.get("_mask")
                    if m is not None:
                        merged |= m
            if merged.any():
                col = QColor(self.ref_colors[ref])
                iso = pg.IsocurveItem(data=merged.astype(float), level=0.5,
                                      pen=pg.mkPen(col, width=1.5))
                iso.setZValue(5)
                self.plot.addItem(iso)
                self._items.append(iso)
        if self.show_points:
            self._draw_points(visible, chi_range, space=DISPLAY_GRID,
                              only_idx=self._locked_idx if isolate else None)

    def _redraw_xy(self, visible, chi_range, isolate):
        self.plot.setLabel("bottom", "Stage X (µm)")
        self.plot.setLabel("left", "Stage Y (µm)")
        self.plot.invertY(False)
        self._update_colorbar(None, None, None)
        cmap = dv._get_cmap("viridis")
        # Gather sampled points (x, y, value, ref).
        xs, ys, vals, refs, fidx = [], [], [], [], []
        for i, f in enumerate(self.features):
            if not self._feature_visible(f, visible, chi_range):
                continue
            for e in f.get("hd_profile", {}).values():
                if "x" not in e or "y" not in e:
                    continue
                v = e.get(self.metric if self.metric != "none" else "intensity")
                if v is None:
                    continue
                xs.append(e["x"]); ys.append(e["y"]); vals.append(v)
                refs.append(f.get("reflection")); fidx.append(i)
        if not xs:
            return
        vals = np.asarray(vals, float)
        if self.metric == "none":
            # Color by reflection only.
            spots = [{"pos": (x, y), "size": 6, "pen": None,
                      "brush": pg.mkBrush(self.ref_colors.get(r, "#888"))}
                     for x, y, r in zip(xs, ys, refs)]
        else:
            vmin, vmax = self._levels(vals.reshape(1, -1))
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            norm = np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0, 1)
            idx = (norm * 255).astype(int)
            spots = [{"pos": (x, y), "size": 6, "pen": None,
                      "brush": pg.mkBrush(int(lut[j, 0]), int(lut[j, 1]), int(lut[j, 2]))}
                     for x, y, j in zip(xs, ys, idx)]
            self._update_colorbar(cmap, vmin, vmax)
        if isolate:
            for s, i in zip(spots, fidx):
                if i != self._locked_idx:
                    b = s["brush"].color(); b.setAlpha(40); s["brush"] = pg.mkBrush(b)
        sc = pg.ScatterPlotItem(spots=spots)
        sc.setZValue(4)
        self.plot.addItem(sc)
        self._items.append(sc)

    def _draw_points(self, visible, chi_range, space, only_idx=None):
        spots = []
        for i, f in enumerate(self.features):
            if only_idx is not None and i != only_idx:
                continue
            if not self._feature_visible(f, visible, chi_range):
                continue
            if space == DISPLAY_GRID:
                pos = (f["center_col"], f["center_row"])
            else:
                if f.get("center_x") is None:
                    continue
                pos = (f["center_x"], f["center_y"])
            spots.append({"pos": pos, "size": 7, "pen": pg.mkPen("k", width=0.5),
                          "brush": pg.mkBrush(self.ref_colors.get(f.get("reflection"), "k"))})
        if spots:
            sc = pg.ScatterPlotItem(spots=spots)
            sc.setZValue(14)
            self.plot.addItem(sc)
            self._items.append(sc)

    def _rebuild_legend(self, visible):
        self.legend.clear()
        for ref in visible:
            s = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(self.ref_colors[ref]), size=10)
            self.legend.addItem(s, ref)

    def _update_colorbar(self, cmap, vmin, vmax):
        if self.colorbar is not None:
            try:
                self.glw.removeItem(self.colorbar)
            except Exception:
                pass
            self.colorbar = None
        if cmap is None:
            return
        self.colorbar = pg.ColorBarItem(
            interactive=False, values=(vmin, vmax), colorMap=cmap,
            label=METRIC_ZLABELS.get(self.metric, ""))
        self.glw.addItem(self.colorbar, row=0, col=1)

    # ----- interactions ----------------------------------------------
    def _check_all(self):
        for cb in self.layer_cbs.values():
            cb.setChecked(True)

    def _uncheck_all(self):
        for cb in self.layer_cbs.values():
            cb.setChecked(False)

    def _on_points_toggle(self, checked):
        self.show_points = checked
        self._redraw()

    def _on_isolate_toggle(self, checked):
        self._isolate = bool(checked)
        self._redraw()

    def _on_trajectory_toggle(self, checked):
        self.show_trajectory = bool(checked)
        self._redraw()

    def _draw_trajectory(self, space):
        """Dotted polyline along the acquisition path in the current space."""
        pts = self._trajectory.get("grid") if space == DISPLAY_GRID \
            else self._trajectory.get("xy")
        if not pts:
            return
        arr = np.asarray(pts, dtype=float)
        pen = pg.mkPen(QColor(60, 60, 60, 150), width=0.8, style=Qt.DotLine)
        line = pg.PlotDataItem(arr[:, 0], arr[:, 1], pen=pen,
                               antialias=True, connect="all")
        line.setZValue(8)
        self.plot.addItem(line)
        self._items.append(line)

    def _on_metric_changed(self):
        self.metric = self.metric_combo.currentData()
        self._redraw()

    def _on_display_changed(self):
        self.display = self.display_combo.currentData()
        self._redraw()
        self.plot.autoRange()  # grid↔xy have very different coordinate ranges

    def _on_chi_range(self, lo, hi):
        self._chi_lo, self._chi_hi = float(lo), float(hi)
        self.chi_label.setText(f"χ: {lo}° to {hi}°")
        self._redraw()

    def _on_catalog_changed(self):
        path = self.cat_combo.currentData()
        if not path:
            return
        self._catalog_path = path
        self._locked_idx = None
        self._load(path)
        # Reflections/layers may differ between catalogs; rebuild the panel.
        # setCentralWidget replaces (and deletes) the previous central widget.
        self._build_ui()
        self._redraw()

    def _nearest_feature(self, x, y, space):
        best_idx, best_d = None, float("inf")
        visible = self._visible_refs()
        chi_range = self._chi_range()
        for i, f in enumerate(self.features):
            if not self._feature_visible(f, visible, chi_range):
                continue
            if space == DISPLAY_GRID:
                cx, cy = f["center_col"], f["center_row"]
            else:
                if f.get("center_x") is None:
                    continue
                cx, cy = f["center_x"], f["center_y"]
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d < best_d:
                best_d, best_idx = d, i
        return best_idx, best_d

    def _scene_to_view(self, scene_pos):
        vb = self.plot.getViewBox()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        pt = vb.mapSceneToView(scene_pos)
        return pt.x(), pt.y()

    def _hover_tol(self):
        # Grid space: a few 1×1 cells. XY space: a few % of the data span.
        if self.display == DISPLAY_XY:
            span = max(self.n_rows, self.n_cols, 1)
            return max(1.0, 0.03 * span)
        return 4.0

    def _on_mouse_move(self, scene_pos):
        if self._locked_idx is not None:
            return
        pos = self._scene_to_view(scene_pos)
        if pos is None:
            return
        idx, d = self._nearest_feature(*pos, space=self.display)
        if idx is None or d > self._hover_tol() * 4:
            self.hover_label.setText("Hover over a feature to see details")
            return
        self._show_feature_info(idx, hover=True)

    def _on_click(self, ev):
        pos = self._scene_to_view(ev.scenePos())
        if pos is None:
            return
        idx, d = self._nearest_feature(*pos, space=self.display)
        if idx is None or d > self._hover_tol() * 4:
            if self._locked_idx is not None:
                self._locked_idx = None
                self._redraw()
            return
        self._locked_idx = None if idx == self._locked_idx else idx
        self._redraw()

    def _show_feature_info(self, idx, hover=False):
        f = self.features[idx]
        fid = f.get("feature_id", "?")
        ref = f.get("reflection", "?")
        chi = f.get("chi_deg")
        ref_tth = f.get("ref_tth")
        n_cells = int(f["_mask"].sum())
        lines = [f"#{fid}  {ref}"]
        if ref_tth is not None:
            lines.append(f"2θ={ref_tth:.3f}°" + (f"  χ={chi:.1f}°" if chi is not None else ""))
        elif chi is not None:
            lines.append(f"χ={chi:.1f}°")
        lines.append(f"{n_cells} × 1×1 cells  (det {f.get('detector_x')},{f.get('detector_y')})")
        pinned = "  [pinned]" if self._locked_idx is not None else ""
        self.info_label.setText("\n".join(lines))
        self.hover_label.setText(" · ".join(lines) + pinned)

    # ----- view-state carry-over (BinnedTab rebuilds on bin change) ---
    def get_view_state(self):
        return {"metric": self.metric, "display": self.display,
                "isolate": self._isolate, "trajectory": self.show_trajectory}

    def apply_view_state(self, state):
        if not state:
            return
        if state.get("metric") and self.metric_combo.findData(state["metric"]) >= 0:
            self.metric_combo.setCurrentIndex(self.metric_combo.findData(state["metric"]))
        if state.get("display") and self.display_combo.findData(state["display"]) >= 0:
            item = self.display_combo.model().item(
                self.display_combo.findData(state["display"]))
            if item.isEnabled():
                self.display_combo.setCurrentIndex(
                    self.display_combo.findData(state["display"]))
        self._isolate = bool(state.get("isolate", self._isolate))
        self.isolate_cb.setChecked(self._isolate)
        if "trajectory" in state and self.traj_cb.isEnabled():
            self.show_trajectory = bool(state["trajectory"])
            self.traj_cb.setChecked(self.show_trajectory)
        self._redraw()


def _message_window(message, detail=""):
    win = QMainWindow()
    win.setWindowTitle("HD Device View")
    w = QWidget(); lay = QVBoxLayout(w)
    lay.addStretch()
    lbl = QLabel(message); lbl.setAlignment(Qt.AlignCenter); lbl.setWordWrap(True)
    lbl.setStyleSheet("font-size: 1.15em;")
    lay.addWidget(lbl)
    if detail:
        d = QLabel(detail); d.setAlignment(Qt.AlignCenter); d.setWordWrap(True)
        d.setStyleSheet("color:#999; font-size:0.9em;")
        lay.addWidget(d)
    lay.addStretch()
    win.setCentralWidget(w)
    return win


def _load_trajectory(dm, scan):
    """Acquisition-order scan path from the default 1×1 grid mapping (best-effort)."""
    try:
        gm_path = dm.grid_mapping(bin_size=1, scan=scan)
        if not gm_path or not Path(gm_path).exists():
            return {"grid": [], "xy": None}
        with open(gm_path) as f:
            gm = json.load(f)
        pos_csv = gm.get("positions_csv") or dm.position_csv(scan=scan)
        return hd_core.scan_trajectory(gm, pos_csv)
    except Exception:
        return {"grid": [], "xy": None}


def build_window(project_root=".", scan=None, bin_size=3, catalog=None):
    """Construct the HD device view (no event loop; embeddable as a tab)."""
    dm = DataManager(project_root, scan=scan)
    results_dir = dm.results_dir(scan)
    cats = list_hd_catalogs(results_dir, bin_size)
    path = _resolve_catalog(results_dir, bin_size, catalog)
    if not path or not Path(path).exists():
        return _message_window(
            f"No HD map for {bin_size}×{bin_size} yet.",
            f"Run:  xrd-app hd-device-map --scan {scan or ''} "
            f"--bin-size {bin_size}\nto sample 1×1 intensity beneath the feature map.")
    if not cats:
        cats = [Path(path)]
    trajectory = _load_trajectory(dm, scan)
    return HDDeviceMapWindow(results_dir, bin_size, path, cats, trajectory=trajectory)


def launch_gui(project_root=".", bin_size=3, scan=None):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    win = build_window(project_root=project_root, scan=scan, bin_size=bin_size)
    win.show()
    app.exec_()


def main():
    import argparse
    p = argparse.ArgumentParser(description="HD Device View (1×1 under N×N)")
    p.add_argument("--project-root", default=".")
    p.add_argument("--bin-size", type=int, default=3)
    p.add_argument("--scan", default=None)
    args = p.parse_args()
    launch_gui(args.project_root, args.bin_size, args.scan)


if __name__ == "__main__":
    main()
