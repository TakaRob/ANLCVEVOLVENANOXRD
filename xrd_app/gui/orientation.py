"""
Orientation map — adaptive sector chart on detector geometry (pyqtgraph).

Finds natural clusters of features along each Bragg ring using Gaussian KDE with
valley-finding, paints each ring with a smooth density gradient, and labels
clusters with percentage and angular span. Hovering over a sector shows detailed
azimuthal and Δ2θ histograms.

pyqtgraph rewrite of the matplotlib version (full feature parity). Clustering /
density logic is framework-agnostic and unchanged; rendering is pyqtgraph.
"""

import sys
from collections import defaultdict

import numpy as np
import tifffile
import matplotlib
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QPushButton,
    QHBoxLayout, QLabel, QMainWindow, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import orientation as orientation_core

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

# Resolved at runtime by configure(); see launch_gui().
_DM = None
_BIN_SIZE = 3
RESULTS_DIR = None
HOLDOUT_DIR = None
CATALOG_PATH = None   # selected feature-map JSON; None → canonical per-bin file


def configure(project_root=".", bin_size=3, scan=None, catalog=None):
    global _DM, _BIN_SIZE, RESULTS_DIR, HOLDOUT_DIR, CATALOG_PATH, WEIGHT_MODE
    global DEGS, DEG_LABELS, LABELED_DEGS
    _DM = DataManager(project_root, scan=scan)
    _BIN_SIZE = bin_size
    RESULTS_DIR = _DM.results_dir()
    HOLDOUT_DIR = _DM.holdout_dir
    CATALOG_PATH = catalog
    WEIGHT_MODE = "count"   # fresh build starts on a plain head-count
    # Honor the per-scan/selected reflections file (fall back to defaults).
    try:
        from ..core import io as _io
        degs, labels = _io.load_reflections(_DM.reflections(scan=_DM._scan()))
        if degs:
            DEGS = list(degs)
            DEG_LABELS = list(labels)
            LABELED_DEGS = {lab: deg for lab, deg in zip(DEG_LABELS, DEGS)}
    except Exception:
        pass


DEGS = [6.81319, 7.51422, 10.61748, 13.00831, 15.01266,
        16.07224, 16.79944, 18.42549, 21.29655, 22.59817, 26.16205]
DEG_LABELS = ["PbI2", "(001)", "(011)", "(111)", "(002)",
              "ITO", "(012)", "(112)"]
LABELED_DEGS = {lab: deg for lab, deg in zip(DEG_LABELS, DEGS)}

ARC_COLORS = [
    "#b71c1c", "#2e7d32", "#b8860b", "#1a5276", "#d35400",
    "#6a1b9a", "#00838f", "#ad1457",
]
COLORMAPS = [
    "inferno", "viridis", "plasma", "magma", "cividis",
    "hot", "coolwarm", "gray", "jet", "turbo",
]


# ── Data loading ──────────────────────────────────────────────────
def load_features():
    """Kept-feature list from the selected feature map (or the newest shapes/
    combined catalog for the bin when none is selected). Reads any catalog kind
    via ``load_features_any``.
    """
    from ..core import catalogs
    path = CATALOG_PATH or catalogs.default_feature_source(RESULTS_DIR, _BIN_SIZE)
    if not path:
        return []
    kept, _ = catalogs.load_features_any(path)
    return kept


def load_tth_map():
    return tifffile.imread(str(_DM.tth_map())).astype(np.float64)


estimate_beam_center = orientation_core.estimate_beam_center
compute_chi_map = orientation_core.compute_chi_map


# ── Feature weighting ─────────────────────────────────────────────
# Clumps (KDE clusters), the ring-density gradient, and the hover histograms
# can each weight a feature by more than a plain head-count. The active mode is
# a module global so the existing global-state functions (clustering / density)
# need no extra threading — the window sets it before re-clustering.
WEIGHT_MODES = [
    ("count",     "Count — each feature = 1"),
    ("area",      "Area — large features win (n_bins)"),
    ("intensity", "Intensity — bright features win (peak)"),
    ("bright_big", "Peak × spread × area — bright, large, spread win"),
]
WEIGHT_LABELS = {k: v.split(" — ")[0] for k, v in WEIGHT_MODES}
WEIGHT_MODE = "count"


def feature_weight(feature, mode=None):
    """Weight a feature using the active orientation-map mode."""
    return orientation_core.feature_weight(feature, mode or WEIGHT_MODE)


# ── Clustering ────────────────────────────────────────────────────
def cluster_features_by_chi(features, bandwidth=5.0):
    """Delegate weighted circular KDE clustering to the core engine."""
    return orientation_core.cluster_features_by_chi(
        features, bandwidth=bandwidth, weight_mode=WEIGHT_MODE)


def _chi_mask(chi_map, chi_lo, chi_hi, wraps):
    if wraps:
        return (chi_map >= chi_lo) | (chi_map <= chi_hi)
    return (chi_map >= chi_lo) & (chi_map <= chi_hi)


def _circular_frame(chis):
    """Frame azimuthal (χ) data on its minimal enclosing arc.

    χ is circular (−180..180 wraps). Unwrapping with a *fixed* hinge at χ=0 splits
    any distribution that straddles the 0 (or ±180) boundary into two clumps at
    opposite ends of a linear axis, leaving a false empty gap — the "dead space"
    seen on the wide, often-multimodal inner rings. Instead put the cut at the
    largest empty gap on the circle so all data stays contiguous. Returns a mapping
    function ``χ → plot coordinate`` and the ``(lo, hi)`` extent of the data.
    """
    v = np.sort(np.asarray(chis, dtype=float))
    if len(v) < 2:
        c = float(v[0]) if len(v) else 0.0
        return (lambda x: x), c, c
    gaps = np.diff(v)
    wrap_gap = 360.0 - (v[-1] - v[0])
    if gaps.size == 0 or wrap_gap >= gaps.max():
        # data already contiguous; the empty side is the natural wrap-around
        return (lambda x: x), float(v[0]), float(v[-1])
    k = int(np.argmax(gaps))          # widest interior gap sits between v[k], v[k+1]
    thr = (v[k] + v[k + 1]) / 2.0     # everything at/below the gap lifts by +360
    return (lambda x: x + 360.0 if x < thr else x), float(v[k + 1]), float(v[k] + 360.0)


# ── Density overlay ───────────────────────────────────────────────
def build_density_overlay(tth_map, chi_map, features_by_ref, active_refs,
                          band_tol, cmap_name, sigma=3.0,
                          low_pct=0.0, high_pct=100.0):
    """Paint each Bragg ring with a smooth chi-density gradient.

    The colormap spans the [low_pct, high_pct] percentile window of the density
    values; alpha still follows the raw (global-max-normalized) density so faint
    rings stay translucent.

    Returns (overlay_rgba_float, global_max, vmin, vmax). vmin/vmax label the
    colorbar.
    """
    h, w = tth_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    chi_idx = np.clip(((chi_map + 180)).astype(int), 0, 359)
    cmap = _mpl_cmap(cmap_name)

    refs = [ref for ref in active_refs if ref in LABELED_DEGS]
    densities, global_max, vmin, vmax = orientation_core.orientation_densities(
        features_by_ref, refs, sigma=sigma, low_pct=low_pct,
        high_pct=high_pct, weight_mode=WEIGHT_MODE)
    if global_max == 0:
        return overlay, 0, 0.0, 1.0
    span = vmax - vmin

    for ref, density in densities.items():
        ref_tth = LABELED_DEGS[ref]
        band = np.abs(tth_map - ref_tth) < band_tol
        cscale = np.clip((density - vmin) / span, 0, 1)  # contrast-scaled colour
        normed = density / global_max                    # alpha as before
        pixel_c = cscale[chi_idx]
        pixel_a = normed[chi_idx]
        paint = band & (pixel_a > 0.01)
        if not np.any(paint):
            continue
        rgba = cmap(pixel_c[paint])
        overlay[paint, :3] = rgba[:, :3]
        overlay[paint, 3] = np.clip(pixel_a[paint] * 1.2, 0.05, 0.85)
    return overlay, global_max, vmin, vmax


def _mpl_cmap(name):
    """Matplotlib Colormap by name, version-safe.

    `matplotlib.cm.get_cmap` was deprecated in 3.7 and removed in 3.9; the
    supported access is `matplotlib.colormaps[name]`. Fall back to the old API
    for matplotlib < 3.5, and to viridis on an unknown name.
    """
    try:
        return matplotlib.colormaps[name]
    except KeyError:
        return matplotlib.colormaps["viridis"]
    except AttributeError:  # matplotlib < 3.5
        import matplotlib.cm as _cm
        try:
            return _cm.get_cmap(name)
        except Exception:
            return _cm.get_cmap("viridis")


def _get_cmap(name):
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        try:
            return pg.colormap.get(name)
        except Exception:
            return pg.colormap.get("viridis")


# ── Main window ───────────────────────────────────────────────────
class OrientationMapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Orientation Map — Detector Geometry")
        self.setGeometry(50, 30, 1450, 900)

        self._band_tol = 0.4
        self._bandwidth = 5.0
        self._cmap_name = "inferno"
        self._show_arcs = True
        self._show_boundaries = True
        self._show_markers = True
        self._show_labels = True
        self._cbar = None
        self._dyn_items = []
        self._current_hover = None
        self._locked_sector = None

        self._load_data()
        self._cluster_all()
        self._build_ui()
        self._connect_signals()
        self._render_main()
        self._clear_histograms()

    # ── data ──
    def _load_data(self):
        self._features = load_features()
        self._tth_map = load_tth_map()
        self._beam_center = estimate_beam_center(self._tth_map)
        self._chi_map = compute_chi_map(self._tth_map.shape, self._beam_center)
        self._all_reflections = sorted(
            set(f["reflection"] for f in self._features),
            key=lambda r: LABELED_DEGS.get(r, 99))
        self._active_refs = set(self._all_reflections)
        self._by_ref = defaultdict(list)
        for f in self._features:
            self._by_ref[f["reflection"]].append(f)

    def _cluster_all(self):
        self._clusters_by_ref = {}
        self._valleys_by_ref = {}
        for ref in self._all_reflections:
            clusters, valleys = cluster_features_by_chi(
                self._by_ref.get(ref, []), self._bandwidth)
            self._clusters_by_ref[ref] = clusters
            self._valleys_by_ref[ref] = valleys
        self._build_sector_id_map()

    def _build_sector_id_map(self):
        h, w = self._tth_map.shape
        self._sector_map = np.full((h, w, 2), -1, dtype=np.int16)
        ref_list = [r for r in self._all_reflections if r in self._active_refs]
        for ri, ref in enumerate(ref_list):
            if ref not in LABELED_DEGS:
                continue
            band = np.abs(self._tth_map - LABELED_DEGS[ref]) < self._band_tol
            for ci, cl in enumerate(self._clusters_by_ref.get(ref, [])):
                cell = band & _chi_mask(self._chi_map, cl["chi_lo"], cl["chi_hi"], cl["wraps"])
                self._sector_map[cell, 0] = ri
                self._sector_map[cell, 1] = ci
        self._ref_list = ref_list

    # ── ui ──
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setTitle("Orientation Map — Adaptive Sectors")
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        left_lay.addWidget(self.glw, 1)
        self.plot.scene().sigMouseMoved.connect(self._on_scene_move)
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)

        self.hover_label = QLabel("Hover over a sector to see details")
        self.hover_label.setStyleSheet(
            "font-family: monospace; font-size: 0.9em; color: #555; "
            "padding: 4px; background: #f0f0f0;")
        self.hover_label.setFixedHeight(22)
        left_lay.addWidget(self.hover_label)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)

        rg = QGroupBox("Reflections")
        rgl = QVBoxLayout(rg)
        btn_row = QHBoxLayout()
        ab = QPushButton("All"); ab.setFixedWidth(45); ab.clicked.connect(self._check_all)
        nb = QPushButton("None"); nb.setFixedWidth(45); nb.clicked.connect(self._uncheck_all)
        btn_row.addWidget(ab); btn_row.addWidget(nb); btn_row.addStretch()
        rgl.addLayout(btn_row)
        self._ref_checks = {}
        row = QHBoxLayout()
        for i, ref in enumerate(self._all_reflections):
            cb = QCheckBox(ref); cb.setChecked(True)
            idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
            cb.setStyleSheet(f"QCheckBox {{ color: {ARC_COLORS[idx % len(ARC_COLORS)]}; }}")
            row.addWidget(cb)
            self._ref_checks[ref] = cb
            if (i + 1) % 4 == 0:
                rgl.addLayout(row); row = QHBoxLayout()
        if row.count():
            rgl.addLayout(row)
        rl.addWidget(rg)

        sg = QGroupBox("Settings")
        sl = QVBoxLayout(sg)
        gr = QHBoxLayout()
        gr.addWidget(QLabel("KDE bandwidth"))
        self.bw_spin = QSpinBox()
        self.bw_spin.setRange(1, 30)
        self.bw_spin.setValue(int(self._bandwidth))
        self.bw_spin.setSuffix("°")
        self.bw_spin.setToolTip("Gaussian KDE bandwidth used to cluster features along χ")
        gr.addWidget(self.bw_spin)
        gr.addStretch()
        sl.addLayout(gr)
        tr = QHBoxLayout()
        tr.addWidget(QLabel("Band tol"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0.10, 1.00)
        self.tol_spin.setSingleStep(0.05)
        self.tol_spin.setDecimals(2)
        self.tol_spin.setValue(self._band_tol)
        self.tol_spin.setSuffix("°")
        self.tol_spin.setToolTip("2θ half-width of each Bragg band painted around a reflection")
        tr.addWidget(self.tol_spin)
        tr.addStretch()
        sl.addLayout(tr)
        cr = QHBoxLayout()
        cr.addWidget(QLabel("Colormap"))
        self.cmap_combo = QComboBox(); self.cmap_combo.addItems(COLORMAPS)
        cr.addWidget(self.cmap_combo)
        sl.addLayout(cr)
        # Contrast: colormap spans [min%, max%] percentiles of the density.
        xr = QHBoxLayout()
        xr.addWidget(QLabel("Contrast %"))
        self.contrast_lo = QSpinBox()
        self.contrast_lo.setRange(0, 100); self.contrast_lo.setValue(0)
        self.contrast_lo.setToolTip("Lower percentile — colormap bottom maps here")
        xr.addWidget(self.contrast_lo)
        self.contrast_hi = QSpinBox()
        self.contrast_hi.setRange(0, 100); self.contrast_hi.setValue(100)
        self.contrast_hi.setToolTip("Upper percentile — colormap top maps here")
        xr.addWidget(self.contrast_hi)
        sl.addLayout(xr)
        wr = QHBoxLayout()
        wr.addWidget(QLabel("Weight by"))
        self.weight_combo = QComboBox()
        for key, label in WEIGHT_MODES:
            self.weight_combo.addItem(label, key)
        i = self.weight_combo.findData(WEIGHT_MODE)
        self.weight_combo.setCurrentIndex(i if i >= 0 else 0)
        self.weight_combo.setToolTip(
            "How much each feature contributes to clumps and histograms: a "
            "head-count, or weighted by area / intensity / peak×spread×area.")
        wr.addWidget(self.weight_combo)
        sl.addLayout(wr)
        tg = QHBoxLayout()
        self.arcs_cb = QCheckBox("Arcs"); self.arcs_cb.setChecked(True)
        self.bounds_cb = QCheckBox("Bounds"); self.bounds_cb.setChecked(True)
        self.markers_cb = QCheckBox("Markers"); self.markers_cb.setChecked(True)
        self.labels_cb = QCheckBox("Labels"); self.labels_cb.setChecked(True)
        for c in (self.arcs_cb, self.bounds_cb, self.markers_cb, self.labels_cb):
            tg.addWidget(c)
        sl.addLayout(tg)
        rl.addWidget(sg)

        rl.addWidget(QLabel("  Azimuthal distribution (χ)"))
        self.az_hist = pg.PlotWidget(); self.az_hist.setBackground("w")
        self.az_hist.setFixedHeight(220)
        rl.addWidget(self.az_hist)
        rl.addWidget(QLabel("  Along-arc distribution (Δ2θ)"))
        self.arc_hist = pg.PlotWidget(); self.arc_hist.setBackground("w")
        self.arc_hist.setFixedHeight(220)
        rl.addWidget(self.arc_hist)
        self.az_hist.scene().sigMouseClicked.connect(lambda ev: self._unlock_histograms())
        self.arc_hist.scene().sigMouseClicked.connect(lambda ev: self._unlock_histograms())

        rl.addStretch()
        # Keep the control panel a fixed, compact width (it would otherwise claim
        # ~half the splitter); the map (left) takes all the extra space so the
        # window resizes by growing the map rather than stretching the side panel.
        right.setMinimumWidth(360)
        right.setMaximumWidth(470)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(1, False)
        splitter.setSizes([1000, 450])

    # ── signals ──
    def _connect_signals(self):
        for cb in self._ref_checks.values():
            cb.toggled.connect(self._on_filter_changed)
        self.bw_spin.valueChanged.connect(self._on_bw_changed)
        self.tol_spin.valueChanged.connect(self._on_tol_changed)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.contrast_lo.valueChanged.connect(lambda _v: self._render_main())
        self.contrast_hi.valueChanged.connect(lambda _v: self._render_main())
        self.weight_combo.currentIndexChanged.connect(self._on_weight_changed)
        for c in (self.arcs_cb, self.bounds_cb, self.markers_cb, self.labels_cb):
            c.toggled.connect(self._on_toggle)

    def _check_all(self):
        for cb in self._ref_checks.values():
            cb.setChecked(True)

    def _uncheck_all(self):
        for cb in self._ref_checks.values():
            cb.setChecked(False)

    def _on_filter_changed(self):
        self._active_refs = {r for r, cb in self._ref_checks.items() if cb.isChecked()}
        self._build_sector_id_map()
        self._render_main()

    def _on_bw_changed(self, value):
        self._bandwidth = float(value)
        self._cluster_all()
        self._render_main()

    def _on_tol_changed(self, value):
        self._band_tol = float(value)
        self._build_sector_id_map()
        self._render_main()

    def _on_cmap_changed(self, name):
        self._cmap_name = name
        self._render_main()

    def _on_weight_changed(self, _idx):
        global WEIGHT_MODE
        WEIGHT_MODE = self.weight_combo.currentData() or "count"
        # Weights change the KDE → re-cluster, repaint, and refresh histograms.
        self._cluster_all()          # also rebuilds the sector-id map
        self._render_main()
        if self._locked_sector is not None:
            self._restore_locked()   # redraw pinned histograms under new weights
        for hist in (self.az_hist, self.arc_hist):
            hist.autoRange()

    def _on_toggle(self):
        self._show_arcs = self.arcs_cb.isChecked()
        self._show_boundaries = self.bounds_cb.isChecked()
        self._show_markers = self.markers_cb.isChecked()
        self._show_labels = self.labels_cb.isChecked()
        self._render_main()

    # ── view-state carry-over (across a feature-catalog switch) ──
    def get_view_state(self):
        """Selected layers + weighting + colormap, so a catalog switch keeps them."""
        return {
            "hidden_layers": [r for r, cb in self._ref_checks.items()
                              if not cb.isChecked()],
            "weight": self.weight_combo.currentData(),
            "cmap": self._cmap_name,
        }

    def apply_view_state(self, state):
        """Re-apply a saved view state; reflections absent from this catalog and
        unknown weight/colormap values are ignored (new reflections stay on)."""
        if not state:
            return
        global WEIGHT_MODE
        hidden = set(state.get("hidden_layers", []))
        for ref, cb in self._ref_checks.items():
            cb.blockSignals(True)
            cb.setChecked(ref not in hidden)
            cb.blockSignals(False)
        self._active_refs = {r for r, cb in self._ref_checks.items() if cb.isChecked()}
        cmap = state.get("cmap")
        if cmap and self.cmap_combo.findText(cmap) >= 0:
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentText(cmap)
            self.cmap_combo.blockSignals(False)
            self._cmap_name = cmap
        w = state.get("weight")
        if w and self.weight_combo.findData(w) >= 0:
            self.weight_combo.blockSignals(True)
            self.weight_combo.setCurrentIndex(self.weight_combo.findData(w))
            self.weight_combo.blockSignals(False)
            WEIGHT_MODE = w
        self._cluster_all()      # weighting may have changed; also rebuilds sectors
        self._render_main()

    # ── rendering ──
    def _clear_dyn(self):
        for it in self._dyn_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._dyn_items.clear()
        if self._cbar is not None:
            try:
                self.glw.removeItem(self._cbar)
            except Exception:
                pass
            self._cbar = None

    def _render_main(self):
        self._clear_dyn()
        lo = float(self.contrast_lo.value()) if hasattr(self, "contrast_lo") else 0.0
        hi = float(self.contrast_hi.value()) if hasattr(self, "contrast_hi") else 100.0
        overlay, global_max, vmin, vmax = build_density_overlay(
            self._tth_map, self._chi_map, self._by_ref,
            self._active_refs, self._band_tol, self._cmap_name,
            low_pct=lo, high_pct=hi)
        self.img_item.setImage((overlay * 255).astype(np.ubyte), autoLevels=False)

        h, w = self._tth_map.shape
        if self._show_arcs:
            for idx, lab in enumerate(DEG_LABELS):
                if lab not in self._active_refs:
                    continue
                color = ARC_COLORS[idx % len(ARC_COLORS)]
                iso = pg.IsocurveItem(data=self._tth_map, level=LABELED_DEGS[lab],
                                      pen=pg.mkPen(color, width=1.2))
                iso.setZValue(3)
                self.plot.addItem(iso); self._dyn_items.append(iso)
                mask = np.abs(self._tth_map - LABELED_DEGS[lab]) < 0.12
                ys, xs = np.where(mask)
                if len(ys):
                    mid = len(ys) // 2
                    t = pg.TextItem(lab, color=color, anchor=(0.5, 0.5))
                    t.setPos(xs[mid], ys[mid]); t.setZValue(8)
                    self.plot.addItem(t); self._dyn_items.append(t)

        if self._show_boundaries:
            self._draw_boundaries()
        if self._show_labels:
            self._draw_cluster_labels()
        if self._show_markers:
            for i, ref in enumerate(DEG_LABELS):
                if ref not in self._active_refs or ref not in self._by_ref:
                    continue
                xs = [f["detector_x"] for f in self._by_ref[ref]]
                ys = [f["detector_y"] for f in self._by_ref[ref]]
                color = ARC_COLORS[i % len(ARC_COLORS)]
                sc = pg.ScatterPlotItem(x=xs, y=ys, size=7,
                                        brush=pg.mkBrush(color),
                                        pen=pg.mkPen("w", width=0.3))
                sc.setZValue(5)
                self.plot.addItem(sc); self._dyn_items.append(sc)

        if global_max > 0:
            self._cbar = pg.ColorBarItem(values=(vmin, vmax), interactive=False,
                                         colorMap=_get_cmap(self._cmap_name),
                                         label="Feature density")
            self.glw.addItem(self._cbar, row=0, col=1)
        self._current_hover = None

    def _draw_boundaries(self):
        by, bx = self._beam_center
        h, w = self._tth_map.shape
        length = max(h, w) * 2
        drawn = set()
        pen = pg.mkPen("#555555", width=0.8, style=Qt.DashLine)
        for ref in self._active_refs:
            for edge in self._valleys_by_ref.get(ref, []):
                key = round(edge, 1)
                if key in drawn:
                    continue
                drawn.add(key)
                rad = np.radians(edge)
                x1 = bx + np.cos(rad) * length
                y1 = by + np.sin(rad) * length
                ts = np.linspace(0, 1, 300)
                px = bx + ts * (x1 - bx)
                py = by + ts * (y1 - by)
                inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                if inside.sum() >= 2:
                    line = pg.PlotDataItem(px[inside], py[inside], pen=pen)
                    line.setZValue(4)
                    self.plot.addItem(line); self._dyn_items.append(line)

    def _draw_cluster_labels(self):
        for ref in self._active_refs:
            ref_idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
            color = ARC_COLORS[ref_idx % len(ARC_COLORS)]
            for cl in self._clusters_by_ref.get(ref, []):
                feats = cl["features"]
                if not feats:
                    continue
                cx = np.mean([f["detector_x"] for f in feats])
                cy = np.mean([f["detector_y"] for f in feats])
                t = pg.TextItem(f"{cl['pct']:.0f}%  {cl['chi_span']:.0f}°",
                                color=color, anchor=(0.5, 1.0))
                t.setPos(cx, cy - 14); t.setZValue(9)
                self.plot.addItem(t); self._dyn_items.append(t)

    # ── hover / click ──
    def _scene_to_pixel(self, scene_pos):
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return None, None
        pt = self.plot.getViewBox().mapSceneToView(scene_pos)
        return int(pt.x() + 0.5), int(pt.y() + 0.5)

    def _on_scene_move(self, scene_pos):
        col, row = self._scene_to_pixel(scene_pos)
        if col is None:
            if self._locked_sector is not None:
                self._restore_locked()
            else:
                self.hover_label.setText("Hover over a sector to see details")
                if self._current_hover is not None:
                    self._current_hover = None
                    self._clear_histograms()
            return
        self._show_sector(col, row)

    def _on_scene_click(self, ev):
        col, row = self._scene_to_pixel(ev.scenePos())
        h, w = self._tth_map.shape
        if col is None or not (0 <= row < h and 0 <= col < w):
            return
        ri, ci = int(self._sector_map[row, col, 0]), int(self._sector_map[row, col, 1])
        if ri == -1:
            return
        self._locked_sector = (ri, ci)
        self._show_sector(col, row)

    def _restore_locked(self):
        ri, ci = self._locked_sector
        if ri >= len(self._ref_list):
            return
        ref = self._ref_list[ri]
        clusters = self._clusters_by_ref.get(ref, [])
        if ci >= len(clusters):
            return
        cluster = clusters[ci]
        self.hover_label.setText(
            f"{ref}  cluster {ci+1}/{len(clusters)}  "
            f"{cluster['pct']:.0f}%  {cluster['n']} features  [pinned]")
        hover_key = (ri, ci)
        if hover_key != self._current_hover:
            self._current_hover = hover_key
            self._draw_az_histogram(ref, cluster)
            self._draw_arc_histogram(ref, cluster)

    def _unlock_histograms(self):
        self._locked_sector = None
        self._current_hover = None
        self._clear_histograms()
        self.hover_label.setText("Hover over a sector to see details")

    def _show_sector(self, col, row):
        h, w = self._tth_map.shape
        if not (0 <= row < h and 0 <= col < w):
            return
        tth = self._tth_map[row, col]
        chi = self._chi_map[row, col]
        ri, ci = int(self._sector_map[row, col, 0]), int(self._sector_map[row, col, 1])
        if ri == -1:
            if self._locked_sector is not None:
                self._restore_locked()
            else:
                self.hover_label.setText(f"pixel ({col}, {row})  2θ={tth:.3f}°  χ={chi:.1f}°")
                if self._current_hover is not None:
                    self._current_hover = None
                    self._clear_histograms()
            return
        ref = self._ref_list[ri]
        cluster = self._clusters_by_ref[ref][ci]
        pinned_tag = "  [pinned]" if self._locked_sector is not None else ""
        self.hover_label.setText(
            f"pixel ({col}, {row})  2θ={tth:.3f}°  χ={chi:.1f}°  │  "
            f"{ref}  cluster {ci+1}/{len(self._clusters_by_ref[ref])}  "
            f"{cluster['pct']:.0f}%  {cluster['n']} features{pinned_tag}")
        hover_key = (ri, ci)
        if hover_key != self._current_hover:
            self._current_hover = hover_key
            self._draw_az_histogram(ref, cluster)
            self._draw_arc_histogram(ref, cluster)

    def _draw_az_histogram(self, ref, cluster):
        self.az_hist.clear()
        ref_idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
        color = ARC_COLORS[ref_idx % len(ARC_COLORS)]
        all_pairs = [(f["chi_deg"], feature_weight(f))
                     for f in self._by_ref.get(ref, []) if f.get("chi_deg") is not None]
        cl_pairs = [(f["chi_deg"], feature_weight(f))
                    for f in cluster["features"] if f.get("chi_deg") is not None]
        if not all_pairs:
            return
        all_chis = [c for c, _ in all_pairs]
        _wrap, x_lo, x_hi = _circular_frame(all_chis)
        all_plot = [_wrap(c) for c, _ in all_pairs]; all_w = [w for _, w in all_pairs]
        cl_plot = [_wrap(c) for c, _ in cl_pairs]; cl_w = [w for _, w in cl_pairs]
        lo_plot, hi_plot = _wrap(cluster["chi_lo"]), _wrap(cluster["chi_hi"])

        edges = np.arange(x_lo - 5, x_hi + 10, 5)
        centers = (edges[:-1] + edges[1:]) / 2
        h_all, _ = np.histogram(all_plot, bins=edges, weights=all_w)
        h_cl, _ = np.histogram(cl_plot, bins=edges, weights=cl_w or None)
        self.az_hist.addItem(pg.BarGraphItem(x=centers, height=h_all, width=4.5,
                                             brush=(204, 204, 204, 130), pen=None))
        self.az_hist.addItem(pg.BarGraphItem(x=centers, height=h_cl, width=4.5,
                                             brush=pg.mkBrush(color), pen=None))
        for v in (lo_plot, hi_plot):
            self.az_hist.addItem(pg.InfiniteLine(pos=v, angle=90,
                                 pen=pg.mkPen(color, width=0.8, style=Qt.DashLine)))
        self.az_hist.setLabel("bottom", "χ (°) — bottom→top")
        self.az_hist.setLabel("left", WEIGHT_LABELS.get(WEIGHT_MODE, "Count"))
        self.az_hist.setTitle(f"{ref} — azimuthal ({cluster['pct']:.0f}%, "
                              f"{cluster['chi_span']:.0f}° span)")
        # Frame tightly to the data (plus the cluster markers, now unwrapped onto the
        # same arc) so multimodal inner rings don't render acres of empty χ.
        lo_bound = min(x_lo, lo_plot, hi_plot)
        hi_bound = max(x_hi, lo_plot, hi_plot)
        self.az_hist.setXRange(lo_bound - 5, hi_bound + 5, padding=0)

    def _draw_arc_histogram(self, ref, cluster):
        self.arc_hist.clear()
        ref_idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
        color = ARC_COLORS[ref_idx % len(ARC_COLORS)]
        ref_tth = LABELED_DEGS.get(ref)
        delta_tths, d_weights = [], []
        for f in cluster["features"]:
            w = feature_weight(f)
            for entry in f.get("intensity_profile", {}).values():
                if isinstance(entry, dict) and "tth" in entry and ref_tth is not None:
                    delta_tths.append(entry["tth"] - ref_tth)
                    d_weights.append(w)
        if not delta_tths:
            t = pg.TextItem("No Δ2θ data", color="#aaa", anchor=(0.5, 0.5))
            self.arc_hist.addItem(t); t.setPos(0, 0)
            return
        d_arr = np.array(delta_tths)
        n_bins = min(25, max(8, len(delta_tths) // 3))
        edges = np.linspace(d_arr.min() - 0.005, d_arr.max() + 0.005, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        hist, _ = np.histogram(d_arr, bins=edges, weights=np.array(d_weights))
        bw = (edges[1] - edges[0]) * 0.85
        if hist.max() > 0:
            cmap = _mpl_cmap(self._cmap_name)
            normed = hist / hist.max()
            brushes = [pg.mkBrush(*[int(ch * 255) for ch in cmap(v)[:3]]) for v in normed]
        else:
            brushes = pg.mkBrush(color)
        self.arc_hist.addItem(pg.BarGraphItem(x=centers, height=hist, width=bw,
                                              brushes=brushes if isinstance(brushes, list) else None,
                                              brush=None if isinstance(brushes, list) else brushes,
                                              pen=pg.mkPen("#999")))
        self.arc_hist.addItem(pg.InfiniteLine(pos=0, angle=90,
                              pen=pg.mkPen("#aaa", width=0.8, style=Qt.DotLine)))
        mean_d = float(np.mean(d_arr))
        self.arc_hist.addItem(pg.InfiniteLine(pos=mean_d, angle=90,
                              pen=pg.mkPen(color, width=1.2)))
        self.arc_hist.getViewBox().invertX(True)
        self.arc_hist.setLabel("bottom", "Δ2θ (°)")
        self.arc_hist.setLabel("left", WEIGHT_LABELS.get(WEIGHT_MODE, "Count"))
        self.arc_hist.setTitle(f"{ref} — Δ2θ ({cluster['n']} feats, "
                               f"{len(delta_tths)} meas, mean {mean_d:+.4f}°)")

    def _clear_histograms(self):
        for hw in (self.az_hist, self.arc_hist):
            hw.clear()
            t = pg.TextItem("Hover over a sector", color="#aaa", anchor=(0.5, 0.5))
            hw.addItem(t); t.setPos(0, 0)


# ── Entry point ───────────────────────────────────────────────────
def build_window(project_root=".", scan=None, bin_size=3, catalog=None):
    """Construct the orientation map without an event loop (for embedding)."""
    configure(project_root=project_root, bin_size=bin_size, scan=scan, catalog=catalog)
    return OrientationMapWindow()


def launch_gui(project_root=".", bin_size=3, scan=None):
    """Configure paths and launch the orientation map (used by the CLI)."""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    win = build_window(project_root=project_root, scan=scan, bin_size=bin_size)
    app = QApplication.instance() or QApplication(sys.argv)
    win.show()
    sys.exit(app.exec_())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XRD orientation map")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--bin-size", type=int, default=3)
    parser.add_argument("--scan", default=None)
    args = parser.parse_args()
    launch_gui(project_root=args.project_root, bin_size=args.bin_size, scan=args.scan)


if __name__ == "__main__":
    main()
