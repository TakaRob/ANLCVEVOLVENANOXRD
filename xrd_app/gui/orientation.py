"""
Orientation map — adaptive sector chart on detector geometry (pyqtgraph).

Finds natural clusters of features along each Bragg ring using Gaussian KDE with
valley-finding, paints each ring with a smooth density gradient, and labels
clusters with percentage and angular span. Hovering over a sector shows detailed
azimuthal and Δ2θ histograms.

pyqtgraph rewrite of the matplotlib version (full feature parity). Clustering /
density logic is framework-agnostic and unchanged; rendering is pyqtgraph.
"""

import json
import sys
from collections import defaultdict

import numpy as np
import tifffile
import matplotlib.cm as mcm
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QPushButton,
    QHBoxLayout, QLabel, QMainWindow, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from ..config import DataManager

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


def estimate_beam_center(tth_map):
    step = 10
    ys, xs = np.mgrid[0:tth_map.shape[0]:step, 0:tth_map.shape[1]:step]
    ts = tth_map[::step, ::step]

    def objective(params):
        y0, x0 = params
        dist = np.sqrt((ys - y0)**2 + (xs - x0)**2)
        dist = np.maximum(dist, 1e-6)
        k = np.sum(ts * dist) / np.sum(dist**2)
        return np.sum((ts - k * dist)**2)

    res = minimize(objective, [tth_map.shape[0] // 2, tth_map.shape[1] + 200],
                   method='Nelder-Mead')
    return res.x[0], res.x[1]


def compute_chi_map(shape, beam_center):
    by, bx = beam_center
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return np.degrees(np.arctan2(ys - by, xs - bx))


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


def _feature_area(f):
    return float(f.get("n_bins") or len(f.get("intensity_profile") or {}) or 1)


def _feature_spread(f):
    """Angular/strain spread of a feature (χ FWHM + Δ2θ FWHM), floored to 1."""
    s = float(f.get("rocking_fwhm") or 0.0) + float(f.get("strain_breadth") or 0.0)
    return s if s > 0 else 1.0


def feature_weight(f, mode=None):
    """Weight a feature contributes to a clump / histogram under ``mode``.

    count      → 1 (head-count, the original behaviour)
    area       → n_bins, so physically large features dominate
    intensity  → peak intensity, so a small bright spot can outweigh a big faint one
    bright_big → peak × spread × area, so bright *and* large *and* spread-out win
    """
    mode = mode or WEIGHT_MODE
    if mode == "area":
        return max(_feature_area(f), 0.0)
    if mode == "intensity":
        return max(float(f.get("peak_intensity") or 0.0), 0.0)
    if mode == "bright_big":
        inten = max(float(f.get("peak_intensity") or 0.0), 0.0)
        return inten * _feature_spread(f) * max(_feature_area(f), 1.0)
    return 1.0   # count (default / unknown)


# ── Clustering ────────────────────────────────────────────────────
def cluster_features_by_chi(features, bandwidth=5.0):
    """Group features by KDE over chi, split at valleys. Returns (clusters, valleys)."""
    from scipy.signal import find_peaks

    items = [(f.get("chi_deg"), f) for f in features if f.get("chi_deg") is not None]
    if not items:
        return [], []
    items.sort(key=lambda x: x[0])
    n = len(items)
    if n < 3:
        return [_make_cluster([x[1] for x in items], [x[0] for x in items])], []

    chis = np.array([x[0] for x in items])
    ws = np.array([feature_weight(x[1]) for x in items], dtype=float)
    total_w = float(ws.sum())
    grid = np.linspace(-180, 179, 360)
    kde = np.zeros(360)
    for c, w in zip(chis, ws):
        diff = (grid - c + 180) % 360 - 180
        kde += w * np.exp(-0.5 * (diff / bandwidth) ** 2)

    pad = max(4, int(bandwidth * 2))
    kde_ext = np.concatenate([kde[-pad:], kde, kde[:pad]])
    valley_idx, _ = find_peaks(-kde_ext, distance=max(4, int(bandwidth * 1.5)),
                               prominence=0.3 * kde.max())
    valley_idx = valley_idx - pad
    valley_idx = valley_idx[(valley_idx >= 0) & (valley_idx < 360)]
    if len(valley_idx) < 2 or kde.max() == 0:
        return [_make_cluster([x[1] for x in items], chis.tolist())], []

    valley_angles = grid[valley_idx]
    v_norm = np.sort((valley_angles + 180) % 360)
    n_segs = len(v_norm)

    groups = defaultdict(list)
    for chi_val, feat in items:
        c_norm = (chi_val + 180) % 360
        idx = int(np.searchsorted(v_norm, c_norm, side="right")) % n_segs
        groups[idx].append((chi_val, feat))

    clusters = []
    for idx in sorted(groups.keys()):
        g = groups[idx]
        if not g:
            continue
        cl = _make_cluster([x[1] for x in g], [x[0] for x in g], total_w)
        cl["chi_lo"] = float(v_norm[idx - 1] - 180)
        cl["chi_hi"] = float(v_norm[idx] - 180)
        cl["wraps"] = v_norm[idx - 1] > v_norm[idx]
        clusters.append(cl)
    return clusters, valley_angles.tolist()


def _make_cluster(feats, chi_vals, total=None):
    # ``total`` is the reflection's total weight (for the weighted percentage);
    # falls back to this clump's own weight (→ 100%) for single-cluster cases.
    cl_w = float(sum(feature_weight(f) for f in feats))
    total = total if total else cl_w
    chi_min, chi_max = min(chi_vals), max(chi_vals)
    if chi_max - chi_min > 180:
        shifted = [c + 360 if c < 0 else c for c in chi_vals]
        s_min, s_max = min(shifted), max(shifted)
        center = (s_min + s_max) / 2
        if center > 180:
            center -= 360
        span = s_max - s_min
        margin = max(3.0, span * 0.12)
        lo, hi = s_min - margin, s_max + margin
        chi_lo = lo if lo <= 180 else lo - 360
        chi_hi = hi if hi <= 180 else hi - 360
        wraps = True
    else:
        center = (chi_min + chi_max) / 2
        span = chi_max - chi_min
        margin = max(3.0, span * 0.12)
        chi_lo, chi_hi = chi_min - margin, chi_max + margin
        wraps = False
    return {
        "chi_center": round(center, 1), "chi_span": round(span, 1),
        "chi_lo": chi_lo, "chi_hi": chi_hi, "wraps": wraps,
        "pct": round(100.0 * cl_w / total, 1) if total else 0.0,
        "features": feats, "n": len(feats), "weight": round(cl_w, 1),
    }


def _chi_mask(chi_map, chi_lo, chi_hi, wraps):
    if wraps:
        return (chi_map >= chi_lo) | (chi_map <= chi_hi)
    return (chi_map >= chi_lo) & (chi_map <= chi_hi)


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
    cmap = mcm.get_cmap(cmap_name)

    global_max = 0
    densities = {}
    for ref in active_refs:
        if ref not in LABELED_DEGS:
            continue
        pairs = [(f["chi_deg"], feature_weight(f))
                 for f in features_by_ref.get(ref, []) if f.get("chi_deg") is not None]
        if not pairs:
            continue
        chi_arr = np.array([p[0] for p in pairs])
        w_arr = np.array([p[1] for p in pairs], dtype=float)
        hist, _ = np.histogram(chi_arr, bins=np.arange(-180, 181, 1), weights=w_arr)
        smooth = gaussian_filter1d(hist.astype(float), sigma=sigma, mode="wrap")
        densities[ref] = smooth
        global_max = max(global_max, smooth.max())

    if global_max == 0:
        return overlay, 0, 0.0, 1.0

    # Contrast window from percentiles of the (non-empty) density values.
    pool = np.concatenate([d for d in densities.values()])
    pool = pool[pool > 0]
    if pool.size:
        if high_pct <= low_pct:
            high_pct = min(100.0, low_pct + 1.0)
        vmin = float(np.percentile(pool, low_pct))
        vmax = float(np.percentile(pool, high_pct))
    else:
        vmin, vmax = 0.0, float(global_max)
    if vmax <= vmin:
        vmax = vmin + 1e-9
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
        chi_min, chi_max = min(all_chis), max(all_chis)
        wraps = (chi_max - chi_min) > 180
        _wrap = (lambda c: c + 360 if c < 0 else c) if wraps else (lambda c: c)
        all_plot = [_wrap(c) for c, _ in all_pairs]; all_w = [w for _, w in all_pairs]
        cl_plot = [_wrap(c) for c, _ in cl_pairs]; cl_w = [w for _, w in cl_pairs]
        if wraps:
            lo_plot = cluster["chi_lo"] + (360 if cluster["chi_lo"] < 0 else 0)
            hi_plot = cluster["chi_hi"] + (360 if cluster["chi_hi"] < 0 else 0)
        else:
            lo_plot, hi_plot = cluster["chi_lo"], cluster["chi_hi"]

        edges = np.arange(min(all_plot) - 5, max(all_plot) + 10, 5)
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
            cmap = mcm.get_cmap(self._cmap_name)
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
