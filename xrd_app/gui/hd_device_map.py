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
import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QPushButton, QGroupBox, QComboBox, QSplitter,
    QSpinBox, QScrollArea, QProgressBar, QPlainTextEdit,
)
from PyQt5.QtCore import Qt, QTimer, QProcess
from PyQt5.QtGui import QColor, QFont

from ..config import DataManager
from ..core import hd_map as hd_core
from . import device_map as dv
from .lifecycle import start_process, stop_process

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
                 trajectory=None, xrf=None):
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
        self.dot_size = 6           # scatter point diameter (screen px)
        self._isolate = True
        self._locked_idx = None
        self._highlighted_idx = None
        self._items = []          # transient overlay items (outlines/highlight/scatter)

        # Debounce: rapid spinbox/slider changes coalesce into one redraw so a
        # burst of clicks doesn't re-render 10k+ points on every step.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(220)
        self._redraw_timer.timeout.connect(self._redraw)

        self._load(catalog_path)
        # XRF underlay: only usable if its 1×1 grid matches this HD grid.
        if xrf and any(np.asarray(m).shape != (self.n_rows, self.n_cols)
                       for m in xrf["maps"].values()):
            xrf = None
        self.xrf = xrf
        self.xrf_elements = list(xrf["elements"]) if xrf else []
        self.xrf_colors = dv._element_colors(self.xrf_elements)
        self.visible_xrf = list(self.xrf_elements)
        self.xrf_on = False
        self.xrf_mode = "dominant"
        self.xrf_normalize = True
        self.xrf_opacity = 0.75
        self._build_ui()
        self._redraw()

    def _schedule_redraw(self):
        """Coalesce a burst of control changes into a single delayed redraw."""
        self._redraw_timer.start()   # restarts the countdown on each call

    # ----- data ------------------------------------------------------
    def _load(self, catalog_path):
        self.features, self.n_rows, self.n_cols, self.positions_real = \
            load_hd_features(catalog_path)
        self.xy_orientation = hd_core.infer_xy_orientation(self.features)
        self.reflections = sorted(set(f.get("reflection", "unknown")
                                      for f in self.features))
        self.ref_colors = dv._assign_ref_colors(self.reflections)
        # χ is wrapped to ±180 in the catalog; a cluster straddling ±180 would
        # otherwise span nearly the whole circle. Unwrap (like Device View) so the
        # slider/histogram cover only the data's real angular span (e.g. 163–235).
        all_chi = [f.get("chi_deg") for f in self.features if f.get("chi_deg") is not None]
        self._chi_wraps = bool(all_chi) and (max(all_chi) - min(all_chi)) > 180
        unwrapped = [self._unwrap_chi(c) for c in all_chi]
        if unwrapped:
            self._chi_data_min = int(np.floor(min(unwrapped)))
            self._chi_data_max = int(np.ceil(max(unwrapped)))
        else:
            self._chi_data_min, self._chi_data_max = -180, 180
        self._chi_lo = float(self._chi_data_min)
        self._chi_hi = float(self._chi_data_max)

    def _unwrap_chi(self, c):
        """Lift a wrapped (±180) angle into the continuous range (matches Device View)."""
        return c + 360 if (self._chi_wraps and c < 0) else c

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
        # XRF underlay beneath the 1×1 heatmap (shows through the scan's holes and
        # the "none" metric); same lower-Z pattern as the binned Device View.
        self.xrf_img = pg.ImageItem()
        self.xrf_img.setZValue(-10)
        self.xrf_img.setVisible(False)
        self.plot.addItem(self.xrf_img)
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
        dot_row = QHBoxLayout()
        dot_row.addWidget(QLabel("Dot size:"))
        self.dot_spin = QSpinBox()
        self.dot_spin.setRange(1, 60)
        self.dot_spin.setValue(self.dot_size)
        self.dot_spin.setToolTip(
            "Diameter (px) of each 1×1 pixel dot in the real-position scatter. "
            "Increase it when zooming in so the sampled pixels don't look sparse.")
        self.dot_spin.valueChanged.connect(self._on_dot_size_changed)
        dot_row.addWidget(self.dot_spin)
        dot_row.addStretch()
        dl.addLayout(dot_row)
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
        self.lo_spin.valueChanged.connect(self._schedule_redraw)
        ch.addWidget(self.lo_spin)
        self.hi_spin = QSpinBox(); self.hi_spin.setRange(0, 100); self.hi_spin.setValue(99)
        self.hi_spin.valueChanged.connect(self._schedule_redraw)
        ch.addWidget(self.hi_spin); ch.addStretch()
        mgl.addLayout(ch)
        rl.addWidget(mg)

        self._build_xrf_group(rl)

        # Chi range: histogram above the slider (like Device View)
        cg = QGroupBox("χ angle range")
        cgl = QVBoxLayout(cg)
        self.chi_hist = pg.PlotWidget()
        self.chi_hist.setBackground("w")
        self.chi_hist.setFixedHeight(150)
        self.chi_hist.setLabel("bottom", "χ (°)")
        self.chi_hist.setLabel("left", "Features")
        self.chi_hist.setMouseEnabled(False, False)
        cgl.addWidget(self.chi_hist)
        self.chi_slider = dv.QRangeSlider(self._chi_data_min, self._chi_data_max)
        self.chi_slider.rangeChanged.connect(self._on_chi_range)
        cgl.addWidget(self.chi_slider)
        self.chi_label = QLabel(f"χ: {self._chi_data_min}° to {self._chi_data_max}°")
        self.chi_label.setStyleSheet(
            "font-family: monospace; font-size: 0.9em; color:#555; padding:2px;")
        cgl.addWidget(self.chi_label)
        rl.addWidget(cg)
        self._draw_chi_histogram()

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
    def _feature_peak(self, feat):
        """A feature's representative brightness = its max sample in the current metric."""
        return max((e.get(self.metric) for e in feat.get("hd_profile", {}).values()
                    if e.get(self.metric) is not None), default=None)

    def _combined_grid(self, visible, chi_range):
        # Where features overlap in a 1×1 cell, the *brighter feature* should own
        # the cell (show its own pixels), not whichever feature happens to have the
        # larger local sample there. Rank visible features by peak intensity and
        # paint brightest last, so it overwrites dimmer features on any shared cell.
        grid = np.full((self.n_rows, self.n_cols), np.nan)
        ranked = []
        for f in self.features:
            if not self._feature_visible(f, visible, chi_range):
                continue
            peak = self._feature_peak(f)
            if peak is not None:
                ranked.append((peak, f))
        ranked.sort(key=lambda t: t[0])           # ascending → brightest painted last
        for _peak, f in ranked:
            for k, e in f.get("hd_profile", {}).items():
                v = e.get(self.metric)
                if v is None:
                    continue
                r, c = _parse_cell(k)
                if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                    grid[r, c] = v
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
        if hasattr(self, "chi_hist"):
            self._draw_chi_histogram()   # reflect layer changes in the histogram
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
        self._update_xrf_underlay()
        if self.show_trajectory:
            self._draw_trajectory(
                DISPLAY_XY if (self.display == DISPLAY_XY and self.positions_real)
                else DISPLAY_GRID)
        self._rebuild_legend(visible)
        self.plot.setTitle(f"HD 1×1  —  {self.n_cols}×{self.n_rows} (col×row)")
        if self._locked_idx is not None and isolate:
            self._show_feature_info(self._locked_idx)

    def _redraw_grid(self, visible, chi_range, isolate):
        self.plot.setLabel("bottom", "Col (1×1)")
        self.plot.setLabel("left", "Row (1×1)")
        self.plot.getViewBox().invertX(False)
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
            rgba = dv._scalar_to_rgba(grid, vmin, vmax, cmap)
            self.img_item.setVisible(True)
            self.img_item.setImage(rgba, autoLevels=False)
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
                data = merged
                col = QColor(self.ref_colors[ref])
                iso = pg.IsocurveItem(data=data.astype(float), level=0.5,
                                      pen=pg.mkPen(col, width=1.5))
                iso.setZValue(5)
                self.plot.addItem(iso)
                self._items.append(iso)
        if self.show_points:
            self._draw_points(visible, chi_range, space=DISPLAY_GRID,
                              only_idx=self._locked_idx if isolate else None)

    def _redraw_xy(self, visible, chi_range, isolate):
        orient = self.xy_orientation
        self.plot.setLabel("bottom", f"Stage {orient['horizontal'].upper()} (µm)")
        self.plot.setLabel("left", f"Stage {orient['vertical'].upper()} (µm)")
        self.plot.getViewBox().invertX(orient["invert_x"])
        self.plot.invertY(orient["invert_y"])
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
        size = self.dot_size
        if self.metric == "none":
            # Color by reflection only.
            spots = [{"pos": self._pxy(x, y), "size": size, "pen": None,
                      "brush": pg.mkBrush(self.ref_colors.get(r, "#888"))}
                     for x, y, r in zip(xs, ys, refs)]
        else:
            vmin, vmax = self._levels(vals.reshape(1, -1))
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            norm = np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0, 1)
            idx = (norm * 255).astype(int)
            spots = [{"pos": self._pxy(x, y), "size": size, "pen": None,
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
        if self.show_points:
            self._draw_points(visible, chi_range, space=DISPLAY_XY,
                              only_idx=self._locked_idx if isolate else None)

    def _draw_points(self, visible, chi_range, space, only_idx=None):
        entries = []
        for i, f in enumerate(self.features):
            if only_idx is not None and i != only_idx:
                continue
            if not self._feature_visible(f, visible, chi_range):
                continue
            if space == DISPLAY_GRID:
                pos = self._pxy(f["center_col"] + 0.5, f["center_row"] + 0.5)
            else:
                if f.get("center_x") is None:
                    continue
                pos = self._pxy(f["center_x"], f["center_y"])
            entries.append((f, pos))
        font_pt = max(8, min(24, int(round((self.glw.height() or 600) / 45))))
        offset = (0.5, -0.5) if space == DISPLAY_GRID else (0, 0)
        self._items.extend(dv._draw_feature_point_labels(
            self.plot, entries, self.ref_colors, font_pt, label_offset=offset))

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

    # ----- XRF underlay ----------------------------------------------
    def _build_xrf_group(self, parent_layout):
        """1×1 element-fluorescence underlay controls (mirrors Device View)."""
        grp = QGroupBox("XRF underlay (1×1)")
        gl = QVBoxLayout(grp)
        have = bool(self.xrf)

        self.xrf_on_cb = QCheckBox("Underlay XRF map")
        self.xrf_on_cb.setEnabled(have)
        self.xrf_on_cb.setToolTip(
            "Show the 1×1 ME7 fluorescence element map beneath the feature map."
            if have else
            "No 1×1 XRF product for this scan.\n"
            "Run:  xrd-app xrf --scans <scan> --bin-size 1 --refine-roi")
        self.xrf_on_cb.toggled.connect(self._on_xrf_toggle)
        gl.addWidget(self.xrf_on_cb)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Show:"))
        self.xrf_mode_combo = QComboBox()
        self.xrf_mode_combo.addItem("Dominant element (ranked)", "dominant")
        self.xrf_mode_combo.addItem("Total intensity", "total")
        self.xrf_mode_combo.setEnabled(have)
        self.xrf_mode_combo.currentIndexChanged.connect(self._on_xrf_mode_changed)
        mode_row.addWidget(self.xrf_mode_combo)
        mode_row.addStretch()
        gl.addLayout(mode_row)

        opt_row = QHBoxLayout()
        self.xrf_norm_cb = QCheckBox("Normalize per element")
        self.xrf_norm_cb.setChecked(self.xrf_normalize)
        self.xrf_norm_cb.setEnabled(have)
        self.xrf_norm_cb.toggled.connect(self._on_xrf_norm_toggle)
        opt_row.addWidget(self.xrf_norm_cb)
        opt_row.addStretch()
        opt_row.addWidget(QLabel("Opacity %:"))
        self.xrf_opacity_spin = QSpinBox()
        self.xrf_opacity_spin.setRange(0, 100)
        self.xrf_opacity_spin.setValue(int(self.xrf_opacity * 100))
        self.xrf_opacity_spin.setEnabled(have)
        self.xrf_opacity_spin.valueChanged.connect(self._on_xrf_opacity_changed)
        opt_row.addWidget(self.xrf_opacity_spin)
        gl.addLayout(opt_row)

        self.xrf_cbs = {}
        if have:
            btn_row = QHBoxLayout()
            ab = QPushButton("All"); ab.setFixedWidth(45)
            ab.clicked.connect(lambda: self._set_all_xrf(True))
            nb = QPushButton("None"); nb.setFixedWidth(45)
            nb.clicked.connect(lambda: self._set_all_xrf(False))
            btn_row.addWidget(ab); btn_row.addWidget(nb); btn_row.addStretch()
            gl.addLayout(btn_row)
            row = QHBoxLayout()
            for i, el in enumerate(self.xrf_elements):
                cb = QCheckBox(el)
                cb.setChecked(True)
                cb.setStyleSheet(f"QCheckBox {{ color: {self.xrf_colors[el]}; }}")
                cb.toggled.connect(self._on_xrf_layer_toggle)
                row.addWidget(cb)
                self.xrf_cbs[el] = cb
                if (i + 1) % 3 == 0:
                    gl.addLayout(row)
                    row = QHBoxLayout()
            if row.count():
                gl.addLayout(row)
        parent_layout.addWidget(grp)

    def _update_xrf_underlay(self):
        """Refresh the underlay image (grid space only) from the current state."""
        if not hasattr(self, "xrf_img"):
            return
        grid_space = not (self.display == DISPLAY_XY and self.positions_real)
        if not (self.xrf_on and self.xrf and grid_space):
            self.xrf_img.setVisible(False)
            return
        rgba = dv.compute_xrf_rgba(self.xrf, self.visible_xrf, self.xrf_mode,
                                   self.xrf_normalize, self.xrf_opacity, self.xrf_colors)
        if rgba is None:
            self.xrf_img.setVisible(False)
            return
        self.xrf_img.setImage(rgba, autoLevels=False)
        self.xrf_img.setVisible(True)

    def _on_xrf_toggle(self, checked):
        self.xrf_on = bool(checked)
        self._update_xrf_underlay()

    def _on_xrf_mode_changed(self):
        self.xrf_mode = self.xrf_mode_combo.currentData()
        self._update_xrf_underlay()

    def _on_xrf_norm_toggle(self, checked):
        self.xrf_normalize = bool(checked)
        self._update_xrf_underlay()

    def _on_xrf_opacity_changed(self, val):
        self.xrf_opacity = val / 100.0
        self._update_xrf_underlay()

    def _on_xrf_layer_toggle(self):
        self.visible_xrf = [e for e in self.xrf_elements if self.xrf_cbs[e].isChecked()]
        self._update_xrf_underlay()

    def _set_all_xrf(self, state):
        for cb in self.xrf_cbs.values():
            cb.blockSignals(True)
            cb.setChecked(state)
            cb.blockSignals(False)
        self.visible_xrf = [e for e in self.xrf_elements if self.xrf_cbs[e].isChecked()]
        self._update_xrf_underlay()

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

    def _on_dot_size_changed(self, val):
        self.dot_size = int(val)
        self._schedule_redraw()

    def _draw_trajectory(self, space):
        """Dotted polyline along the acquisition path in the current space."""
        pts = self._trajectory.get("grid") if space == DISPLAY_GRID \
            else self._trajectory.get("xy")
        if not pts:
            return
        arr = np.asarray(pts, dtype=float)
        if space == DISPLAY_XY:
            orient = self.xy_orientation
            axes = {"x": arr[:, 0], "y": arr[:, 1]}
            px, py = axes[orient["horizontal"]], axes[orient["vertical"]]
        else:
            px, py = arr[:, 0], arr[:, 1]
        pen = pg.mkPen(QColor(60, 60, 60, 150), width=0.8, style=Qt.DotLine)
        line = pg.PlotDataItem(px, py, pen=pen, antialias=True, connect="all")
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
        self.chi_label.setText(f"χ: {lo}° to {hi}°")   # label updates immediately
        self._draw_chi_histogram()                     # selection band moves live
        self._schedule_redraw()                        # heavy redraw coalesced

    def _draw_chi_histogram(self):
        """Feature-count distribution over χ, with the selected band highlighted
        (unwrapped/continuous — the slider, label and axis all agree)."""
        self.chi_hist.clear()
        visible = self._visible_refs()
        chis = [self._unwrap_chi(f.get("chi_deg")) for f in self.features
                if f.get("chi_deg") is not None and f.get("reflection") in visible]
        if not chis:
            return
        edges = np.arange(self._chi_data_min, self._chi_data_max + 5, 5.0)
        if len(edges) < 2:
            edges = np.array([self._chi_data_min, self._chi_data_max + 1.0])
        centers = (edges[:-1] + edges[1:]) / 2
        h_all, _ = np.histogram(chis, bins=edges)
        inside = [c for c in chis if self._chi_lo <= c <= self._chi_hi]
        h_in, _ = np.histogram(inside, bins=edges) if inside \
            else (np.zeros(len(centers)), None)
        w = float(edges[1] - edges[0]) * 0.9
        self.chi_hist.addItem(pg.BarGraphItem(
            x=centers, height=h_all, width=w, brush=(204, 204, 204, 130), pen=None))
        self.chi_hist.addItem(pg.BarGraphItem(
            x=centers, height=h_in, width=w, brush=(67, 99, 216, 220), pen=None))
        for v in (self._chi_lo, self._chi_hi):
            self.chi_hist.addItem(pg.InfiniteLine(
                pos=v, angle=90, pen=pg.mkPen("r", width=1.2, style=Qt.DashLine)))

    def _pxy(self, x, y):
        """Map grid or stage coordinates into the active plot-axis order."""
        if self.display == DISPLAY_XY and self.positions_real:
            orient = self.xy_orientation
            values = {"x": x, "y": y}
            return values[orient["horizontal"]], values[orient["vertical"]]
        return x, y

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
                cx, cy = self._pxy(f["center_col"], f["center_row"])
            else:
                if f.get("center_x") is None:
                    continue
                cx, cy = self._pxy(f["center_x"], f["center_y"])
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
                "isolate": self._isolate, "trajectory": self.show_trajectory,
                "dot_size": self.dot_size,
                "xrf_on": self.xrf_on, "xrf_mode": self.xrf_mode,
                "xrf_normalize": self.xrf_normalize, "xrf_opacity": self.xrf_opacity,
                "xrf_hidden": [e for e, cb in self.xrf_cbs.items()
                               if not cb.isChecked()]}

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
        if "dot_size" in state:
            self.dot_size = int(state["dot_size"])
            self.dot_spin.setValue(self.dot_size)
        self._apply_xrf_view_state(state)
        self._redraw()

    def _apply_xrf_view_state(self, state):
        if "xrf_mode" in state and self.xrf_mode_combo.findData(state["xrf_mode"]) >= 0:
            self.xrf_mode = state["xrf_mode"]
            self.xrf_mode_combo.blockSignals(True)
            self.xrf_mode_combo.setCurrentIndex(self.xrf_mode_combo.findData(self.xrf_mode))
            self.xrf_mode_combo.blockSignals(False)
        if "xrf_normalize" in state:
            self.xrf_normalize = bool(state["xrf_normalize"])
            self.xrf_norm_cb.blockSignals(True)
            self.xrf_norm_cb.setChecked(self.xrf_normalize)
            self.xrf_norm_cb.blockSignals(False)
        if "xrf_opacity" in state:
            self.xrf_opacity = float(state["xrf_opacity"])
            self.xrf_opacity_spin.blockSignals(True)
            self.xrf_opacity_spin.setValue(int(self.xrf_opacity * 100))
            self.xrf_opacity_spin.blockSignals(False)
        hidden = set(state.get("xrf_hidden", []))
        for e, cb in self.xrf_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(e not in hidden)
            cb.blockSignals(False)
        self.visible_xrf = [e for e in self.xrf_elements if self.xrf_cbs[e].isChecked()]
        if "xrf_on" in state and self.xrf:
            self.xrf_on = bool(state["xrf_on"])
            self.xrf_on_cb.blockSignals(True)
            self.xrf_on_cb.setChecked(self.xrf_on)
            self.xrf_on_cb.blockSignals(False)


_PROGRESS_RE = re.compile(r"PROGRESS\s+(\d+)\s*/\s*(\d+)")


class HDMapBuilder(QMainWindow):
    """Shown in place of the view when no HD map exists yet: a Run button that
    builds it in-app.

    Runs ``xrd-app hd-device-map`` in a subprocess (same CLI-is-the-engine path
    the Programs tab uses), shows an ``(i/n)`` status parsed from the CLI's
    ``PROGRESS`` markers, and calls ``on_built`` on success so the embedding tab
    swaps in the real HD view — no manual CLI step or tab reload needed.
    """

    def __init__(self, project_root, scan, bin_size, on_built=None):
        super().__init__()
        self.setWindowTitle("HD Device View")
        self._project_root = project_root
        self._scan = scan
        self._bin_size = bin_size
        self._on_built = on_built
        self._proc = None
        self._cancelled = False

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch()

        msg = QLabel(f"No HD map for {bin_size}×{bin_size} yet.")
        msg.setAlignment(Qt.AlignCenter); msg.setWordWrap(True)
        msg.setStyleSheet("font-size: 1.15em;")
        lay.addWidget(msg)
        detail = QLabel(
            f"Sample 1×1 intensity beneath the {bin_size}×{bin_size} feature map "
            "at each feature's detector peak. Reads raw frames (heavy) — runs "
            "once, then the JSON is cached. Also builds the 1×1 grid mapping if "
            "missing, so the real (x, y) stage-position scatter is available.")
        detail.setAlignment(Qt.AlignCenter); detail.setWordWrap(True)
        detail.setStyleSheet("color:#999; font-size:0.9em;")
        lay.addWidget(detail)

        # Run button + (i/n) status.
        row = QHBoxLayout()
        row.addStretch()
        self._run_btn = QPushButton("Build HD device map")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setToolTip(
            f"Run  xrd-app hd-device-map --scan {scan or ''} --bin-size {bin_size}\n"
            "using this bin's newest shapes/combined catalog.")
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self._cancel_btn)
        self._status = QLabel("")
        self._status.setStyleSheet(
            "font-family: monospace; color:#555; padding-left:10px;")
        row.addWidget(self._status)
        row.addStretch()
        lay.addLayout(row)

        prow = QHBoxLayout()
        prow.addStretch()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100); self._progress.setValue(0)
        self._progress.setMaximumWidth(420)
        self._progress.setVisible(False)
        prow.addWidget(self._progress)
        prow.addStretch()
        lay.addLayout(prow)

        # Compact output log so failures (e.g. no catalog) are visible.
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setMaximumHeight(150)
        self._log.setFont(QFont("monospace", 8))
        self._log.setVisible(False)
        lay.addWidget(self._log)

        lay.addStretch()
        self.setCentralWidget(w)

    # ----- run the CLI -----------------------------------------------
    def _run(self):
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            return
        self._cancelled = False
        args = ["hd-device-map", "--root", str(self._project_root),
                "--bin-size", str(self._bin_size)]
        if self._scan:
            args += ["--scan", str(self._scan)]
        cmd = [sys.executable, "-m", "xrd_app.cli", *args]
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status.setText("starting…")
        self._progress.setVisible(True); self._progress.setValue(0)
        self._log.setVisible(True); self._log.clear()
        self._log.appendPlainText("$ " + " ".join(cmd))
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)
        start_process(self._proc, cmd[0], cmd[1:])

    def _cancel(self):
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            self._cancelled = True
            stop_process(self._proc)  # _on_finished re-enables the controls

    def _on_error(self, error):
        if error == QProcess.FailedToStart and self._proc is not None:
            self._log.appendPlainText(f"\n[failed to start: {self._proc.errorString()}]")
            self._on_finished(-1, QProcess.CrashExit)

    def _on_output(self):
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            m = _PROGRESS_RE.search(line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if n:
                    self._progress.setValue(int(100 * i / n))
                self._status.setText(f"({i}/{n})")
                continue  # don't echo the raw PROGRESS marker
            self._log.appendPlainText(line)

    def _on_finished(self, code, _status):
        self._cancel_btn.setEnabled(False)
        self._run_btn.setEnabled(True)
        if self._cancelled:
            self._status.setText("cancelled")
            self._log.appendPlainText("\n[cancelled]")
            return
        if code == 0:
            self._status.setText("done ✓")
            self._progress.setValue(100)
            # Defer the swap-in so we don't rebuild the parent (which deletes
            # this widget) while still inside the QProcess.finished handler.
            if self._on_built is not None:
                QTimer.singleShot(0, self._on_built)
            return
        self._status.setText(f"failed (exit {code})")

    def closeEvent(self, event):  # noqa: N802 (Qt signature)
        stop_process(self._proc)
        super().closeEvent(event)


def _load_trajectory(dm, scan):
    """Acquisition-order scan path from the default 1×1 grid mapping (best-effort)."""
    try:
        gm_path = dm.grid_mapping(bin_size=1, scan=scan)
        if not gm_path or not Path(gm_path).exists():
            return {"grid": [], "xy": None}
        with open(gm_path) as f:
            gm = json.load(f)
        # The stored positions_csv may be an absolute path from another machine
        # (e.g. /net/micdata on the LAN host vs /mnt/z on a laptop). Fall back to
        # the project-relative CSV when it doesn't resolve here, so the real-(x,y)
        # trajectory still loads regardless of where the GUI runs.
        pos_csv = gm.get("positions_csv")
        if not pos_csv or not Path(pos_csv).exists():
            pos_csv = dm.position_csv(scan=scan)
        return hd_core.scan_trajectory(
            gm, pos_csv, archive=dm.unbinned_archive_h5(scan=scan))
    except Exception:
        return {"grid": [], "xy": None}


def build_window(project_root=".", scan=None, bin_size=3, catalog=None,
                 on_built=None):
    """Construct the HD device view (no event loop; embeddable as a tab).

    When no HD map exists for the bin yet, returns an :class:`HDMapBuilder` with
    a Run button that builds it in-app; ``on_built`` (called on a successful
    build) lets the embedding tab swap in the real view.
    """
    dm = DataManager(project_root, scan=scan)
    results_dir = dm.results_dir(scan)
    cats = list_hd_catalogs(results_dir, bin_size)
    path = _resolve_catalog(results_dir, bin_size, catalog)
    if not path or not Path(path).exists():
        return HDMapBuilder(project_root, scan, bin_size, on_built=on_built)
    if not cats:
        cats = [Path(path)]
    trajectory = _load_trajectory(dm, scan)
    # HD is 1×1, so load the 1×1 XRF product (matching the fine grid) for the
    # underlay. Shape is validated against the HD grid inside the window.
    xrf = _load_hd_xrf(dm, scan)
    return HDDeviceMapWindow(results_dir, bin_size, path, cats,
                             trajectory=trajectory, xrf=xrf)


def _load_hd_xrf(dm, scan):
    """1×1 XRF element maps for the HD underlay, or ``None`` (best-effort)."""
    try:
        from ..core import xrf as xrf_core
        path = dm.xrf_product(scan=scan)
        if not path.exists():
            return None
        prod = xrf_core.load_product(path)
    except Exception:
        return None
    maps = prod.get("maps") or {}
    elements = [e for e in prod.get("elements", []) if e in maps]
    if not elements:
        return None
    norm = {}
    for e in elements:
        m = np.asarray(maps[e], dtype=float)
        pos = m[m > 0]
        hi = np.percentile(pos, 99.0) if pos.size else 0.0
        norm[e] = np.clip(m / hi, 0.0, 1.0) if hi > 0 else np.zeros_like(m)
    return {"elements": elements,
            "maps": {e: np.asarray(maps[e], dtype=float) for e in elements},
            "norm": norm}


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
