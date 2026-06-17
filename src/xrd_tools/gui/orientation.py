"""
Orientation map — adaptive sector chart on detector geometry.

Finds natural clusters of features along each Bragg ring using
Gaussian KDE with valley-finding, paints each ring with a smooth
density gradient, and labels clusters with percentage and angular
span.  Hovering over a sector shows detailed azimuthal and Δ2θ
histograms.

Usage:
    python3 analysis/orientation_map.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QGroupBox, QPushButton,
    QHBoxLayout, QLabel, QMainWindow, QSlider,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from ..config import DataManager

# Resolved at runtime by configure(); see launch_gui().
_DM = None
_BIN_SIZE = 3
RESULTS_DIR = None
HOLDOUT_DIR = None


def configure(project_root=".", bin_size=3, scan=None):
    global _DM, _BIN_SIZE, RESULTS_DIR, HOLDOUT_DIR
    _DM = DataManager(project_root, scan=scan)
    _BIN_SIZE = bin_size
    RESULTS_DIR = _DM.results_dir()
    HOLDOUT_DIR = _DM.holdout_dir

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
    with open(RESULTS_DIR / f"feature_catalog_{_BIN_SIZE}x{_BIN_SIZE}.json") as f:
        return json.load(f)


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


# ── Clustering ────────────────────────────────────────────────────

def cluster_features_by_chi(features, bandwidth=5.0):
    """Group features by fitting a Gaussian KDE to the chi distribution
    and splitting at valleys (local minima) between peaks.

    Returns (clusters, valley_angles) where valley_angles is a list of
    chi values (degrees) where boundaries were placed.
    """
    from scipy.signal import find_peaks

    items = [(f.get("chi_deg"), f) for f in features
             if f.get("chi_deg") is not None]
    if not items:
        return [], []
    items.sort(key=lambda x: x[0])
    n = len(items)

    if n < 3:
        return [_make_cluster([x[1] for x in items],
                              [x[0] for x in items])], []

    chis = np.array([x[0] for x in items])

    # Circular KDE on 1° grid
    grid = np.linspace(-180, 179, 360)
    kde = np.zeros(360)
    for c in chis:
        diff = (grid - c + 180) % 360 - 180
        kde += np.exp(-0.5 * (diff / bandwidth) ** 2)

    # Find valleys in the circular KDE.
    # Extend array for wrap-around, then find minima of the inverted KDE.
    pad = max(4, int(bandwidth * 2))
    kde_ext = np.concatenate([kde[-pad:], kde, kde[:pad]])
    valley_idx, props = find_peaks(-kde_ext,
                                   distance=max(4, int(bandwidth * 1.5)),
                                   prominence=0.3 * kde.max())
    valley_idx = valley_idx - pad
    valley_idx = valley_idx[(valley_idx >= 0) & (valley_idx < 360)]

    if len(valley_idx) < 2 or kde.max() == 0:
        return [_make_cluster([x[1] for x in items], chis.tolist())], []

    valley_angles = grid[valley_idx]

    # Assign features to segments between consecutive valleys.
    # Work in [0, 360) space for clean searchsorted.
    v_norm = np.sort((valley_angles + 180) % 360)
    n_segs = len(v_norm)

    groups = defaultdict(list)
    for chi_val, feat in items:
        c_norm = (chi_val + 180) % 360
        idx = int(np.searchsorted(v_norm, c_norm, side="right")) % n_segs
        groups[idx].append((chi_val, feat))

    total = n
    clusters = []
    for idx in sorted(groups.keys()):
        g = groups[idx]
        if not g:
            continue
        cl = _make_cluster([x[1] for x in g], [x[0] for x in g], total)
        # Override bounds with valley boundaries
        seg_lo = float(v_norm[idx - 1] - 180)
        seg_hi = float(v_norm[idx] - 180)
        cl["chi_lo"] = seg_lo
        cl["chi_hi"] = seg_hi
        cl["wraps"] = v_norm[idx - 1] > v_norm[idx]
        clusters.append(cl)

    return clusters, valley_angles.tolist()


def _make_cluster(feats, chi_vals, total=None):
    total = total or len(feats)
    chi_min, chi_max = min(chi_vals), max(chi_vals)

    if chi_max - chi_min > 180:
        shifted = [c + 360 if c < 0 else c for c in chi_vals]
        s_min, s_max = min(shifted), max(shifted)
        center = (s_min + s_max) / 2
        if center > 180:
            center -= 360
        span = s_max - s_min
        margin = max(3.0, span * 0.12)
        lo = s_min - margin
        hi = s_max + margin
        chi_lo = lo if lo <= 180 else lo - 360
        chi_hi = hi if hi <= 180 else hi - 360
        wraps = True
    else:
        center = (chi_min + chi_max) / 2
        span = chi_max - chi_min
        margin = max(3.0, span * 0.12)
        chi_lo = chi_min - margin
        chi_hi = chi_max + margin
        wraps = False

    return {
        "chi_center": round(center, 1),
        "chi_span": round(span, 1),
        "chi_lo": chi_lo,
        "chi_hi": chi_hi,
        "wraps": wraps,
        "pct": round(100.0 * len(feats) / total, 1),
        "features": feats,
        "n": len(feats),
    }


def _chi_mask(chi_map, chi_lo, chi_hi, wraps):
    if wraps:
        return (chi_map >= chi_lo) | (chi_map <= chi_hi)
    return (chi_map >= chi_lo) & (chi_map <= chi_hi)


# ── Rendering ─────────────────────────────────────────────────────

def build_density_overlay(tth_map, chi_map, features_by_ref, active_refs,
                          band_tol, cmap_name, sigma=3.0):
    """Paint each Bragg ring with a smooth chi-density gradient."""
    h, w = tth_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    chi_idx = np.clip(((chi_map + 180)).astype(int), 0, 359)
    cmap = mcm.get_cmap(cmap_name)

    global_max = 0
    densities = {}
    for ref in active_refs:
        if ref not in LABELED_DEGS:
            continue
        chi_vals = [f["chi_deg"] for f in features_by_ref.get(ref, [])
                    if f.get("chi_deg") is not None]
        if not chi_vals:
            continue
        hist, _ = np.histogram(chi_vals, bins=np.arange(-180, 181, 1))
        smooth = gaussian_filter1d(hist.astype(float), sigma=sigma, mode="wrap")
        densities[ref] = smooth
        global_max = max(global_max, smooth.max())

    if global_max == 0:
        return overlay, None

    for ref, density in densities.items():
        if ref not in active_refs:
            continue
        ref_tth = LABELED_DEGS[ref]
        band = np.abs(tth_map - ref_tth) < band_tol
        normed = density / global_max
        pixel_d = normed[chi_idx]
        paint = band & (pixel_d > 0.01)
        if not np.any(paint):
            continue
        rgba = cmap(pixel_d[paint])
        overlay[paint, :3] = rgba[:, :3]
        overlay[paint, 3] = np.clip(pixel_d[paint] * 1.2, 0.05, 0.85)

    norm = mcolors.Normalize(vmin=0, vmax=global_max)
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return overlay, sm


def draw_arc_boundaries(ax, tth_map, active_refs, line_tol=0.12):
    for idx, lab in enumerate(DEG_LABELS):
        if lab not in active_refs:
            continue
        d = LABELED_DEGS[lab]
        mask = np.abs(tth_map - d) < line_tol
        ov = np.where(mask, 1.0, np.nan)
        color = ARC_COLORS[idx % len(ARC_COLORS)]
        cm = mcolors.ListedColormap([color])
        ax.imshow(ov, cmap=cm, alpha=0.5, interpolation="nearest")
        ys, xs = np.where(mask)
        if len(ys) > 0:
            mid = len(ys) // 2
            ax.text(xs[mid], ys[mid], lab, color=color, fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#ccc", alpha=0.85))


def draw_cluster_boundaries(ax, valleys_by_ref, beam_center, shape, active_refs):
    by, bx = beam_center
    h, w = shape
    length = max(h, w) * 2
    drawn = set()
    for ref in active_refs:
        for edge in valleys_by_ref.get(ref, []):
            key = round(edge, 1)
            if key in drawn:
                continue
            drawn.add(key)
            rad = np.radians(edge)
            x1 = bx + np.cos(rad) * length
            y1 = by + np.sin(rad) * length
            pts = [(bx + t * (x1 - bx), by + t * (y1 - by))
                   for t in np.linspace(0, 1, 300)]
            pts = [(px, py) for px, py in pts if 0 <= px < w and 0 <= py < h]
            if len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color="#555", linewidth=0.8, alpha=0.6, linestyle="--")


def draw_cluster_labels(ax, clusters_by_ref, active_refs):
    for ref in active_refs:
        ref_idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
        color = ARC_COLORS[ref_idx % len(ARC_COLORS)]
        for cl in clusters_by_ref.get(ref, []):
            feats = cl["features"]
            if not feats:
                continue
            cx = np.mean([f["detector_x"] for f in feats])
            cy = np.mean([f["detector_y"] for f in feats])
            txt = f"{cl['pct']:.0f}%  {cl['chi_span']:.0f}°"
            ax.text(cx, cy - 14, txt, color=color, fontsize=7,
                    fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.9, linewidth=0.5))


# ── Canvas classes ────────────────────────────────────────────────

class OrientationCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(7, 7))
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#f0f0f0")
        self.fig.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.03)
        super().__init__(self.fig)
        self.setMinimumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hover_cb = None
        self._click_cb = None
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_press_event", self._on_click)

    def set_hover_callback(self, cb):
        self._hover_cb = cb

    def set_click_callback(self, cb):
        self._click_cb = cb

    def _on_motion(self, event):
        if self._hover_cb and event.inaxes == self.ax and event.xdata is not None:
            self._hover_cb(int(event.xdata + 0.5), int(event.ydata + 0.5))
        elif self._hover_cb:
            self._hover_cb(None, None)

    def _on_click(self, event):
        if self._click_cb and event.inaxes == self.ax and event.xdata is not None:
            self._click_cb(int(event.xdata + 0.5), int(event.ydata + 0.5))


class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(4, 2.5))
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("white")
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.22)
        super().__init__(self.fig)
        self.setFixedHeight(230)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._unlock_cb = None
        self.mpl_connect("button_press_event", self._on_click)

    def set_unlock_callback(self, cb):
        self._unlock_cb = cb

    def _on_click(self, event):
        if self._unlock_cb and event.inaxes == self.ax:
            self._unlock_cb()


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
                cell = band & _chi_mask(self._chi_map, cl["chi_lo"],
                                        cl["chi_hi"], cl["wraps"])
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

        # Left: canvas + hover label
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self.canvas = OrientationCanvas()
        self.canvas.set_hover_callback(self._on_hover)
        self.canvas.set_click_callback(self._on_click)
        left_lay.addWidget(self.canvas, stretch=1)

        self.hover_label = QLabel("Hover over a sector to see details")
        self.hover_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #555; "
            "padding: 4px; background: #f0f0f0;")
        self.hover_label.setFixedHeight(22)
        left_lay.addWidget(self.hover_label)
        splitter.addWidget(left)

        # Right: controls + histograms
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)

        # Reflection filter
        rg = QGroupBox("Reflections")
        rgl = QVBoxLayout(rg)
        btn_row = QHBoxLayout()
        ab = QPushButton("All"); ab.setFixedWidth(45)
        ab.clicked.connect(self._check_all)
        nb = QPushButton("None"); nb.setFixedWidth(45)
        nb.clicked.connect(self._uncheck_all)
        btn_row.addWidget(ab); btn_row.addWidget(nb); btn_row.addStretch()
        rgl.addLayout(btn_row)

        self._ref_checks = {}
        row = QHBoxLayout()
        for i, ref in enumerate(self._all_reflections):
            cb = QCheckBox(ref)
            cb.setChecked(True)
            idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
            cb.setStyleSheet(
                f"QCheckBox {{ color: {ARC_COLORS[idx % len(ARC_COLORS)]}; }}")
            row.addWidget(cb)
            self._ref_checks[ref] = cb
            if (i + 1) % 4 == 0:
                rgl.addLayout(row)
                row = QHBoxLayout()
        if row.count():
            rgl.addLayout(row)
        rl.addWidget(rg)

        # Settings
        sg = QGroupBox("Settings")
        sl = QVBoxLayout(sg)

        gr = QHBoxLayout()
        gr.addWidget(QLabel("KDE bandwidth"))
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setRange(1, 30)
        self.bw_slider.setValue(int(self._bandwidth))
        gr.addWidget(self.bw_slider)
        self.bw_label = QLabel(f"{self._bandwidth:.0f}°")
        self.bw_label.setFixedWidth(32)
        gr.addWidget(self.bw_label)
        sl.addLayout(gr)

        tr = QHBoxLayout()
        tr.addWidget(QLabel("Band tol"))
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setRange(10, 100)
        self.tol_slider.setValue(int(self._band_tol * 100))
        tr.addWidget(self.tol_slider)
        self.tol_label = QLabel(f"{self._band_tol:.1f}°")
        self.tol_label.setFixedWidth(32)
        tr.addWidget(self.tol_label)
        sl.addLayout(tr)

        cr = QHBoxLayout()
        cr.addWidget(QLabel("Colormap"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        cr.addWidget(self.cmap_combo)
        sl.addLayout(cr)

        tg = QHBoxLayout()
        self.arcs_cb = QCheckBox("Arcs"); self.arcs_cb.setChecked(True)
        self.bounds_cb = QCheckBox("Bounds"); self.bounds_cb.setChecked(True)
        self.markers_cb = QCheckBox("Markers"); self.markers_cb.setChecked(True)
        self.labels_cb = QCheckBox("Labels"); self.labels_cb.setChecked(True)
        tg.addWidget(self.arcs_cb); tg.addWidget(self.bounds_cb)
        tg.addWidget(self.markers_cb); tg.addWidget(self.labels_cb)
        sl.addLayout(tg)
        rl.addWidget(sg)

        # Histograms
        rl.addWidget(QLabel("  Azimuthal distribution (χ)"))
        self.az_hist = HistogramCanvas()
        self.az_hist.set_unlock_callback(self._unlock_histograms)
        rl.addWidget(self.az_hist)

        rl.addWidget(QLabel("  Along-arc distribution (Δ2θ)"))
        self.arc_hist = HistogramCanvas()
        self.arc_hist.set_unlock_callback(self._unlock_histograms)
        rl.addWidget(self.arc_hist)

        rl.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([950, 450])

    # ── signals ──

    def _connect_signals(self):
        for cb in self._ref_checks.values():
            cb.toggled.connect(self._on_filter_changed)
        self.bw_slider.valueChanged.connect(self._on_bw_changed)
        self.tol_slider.valueChanged.connect(self._on_tol_changed)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.arcs_cb.toggled.connect(self._on_toggle)
        self.bounds_cb.toggled.connect(self._on_toggle)
        self.markers_cb.toggled.connect(self._on_toggle)
        self.labels_cb.toggled.connect(self._on_toggle)

    def _check_all(self):
        for cb in self._ref_checks.values():
            cb.setChecked(True)

    def _uncheck_all(self):
        for cb in self._ref_checks.values():
            cb.setChecked(False)

    def _on_filter_changed(self):
        self._active_refs = {r for r, cb in self._ref_checks.items()
                             if cb.isChecked()}
        self._build_sector_id_map()
        self._render_main()

    def _on_bw_changed(self, value):
        self._bandwidth = float(value)
        self.bw_label.setText(f"{value}°")
        self._cluster_all()
        self._render_main()

    def _on_tol_changed(self, value):
        self._band_tol = value / 100.0
        self.tol_label.setText(f"{self._band_tol:.2f}°")
        self._build_sector_id_map()
        self._render_main()

    def _on_cmap_changed(self, name):
        self._cmap_name = name
        self._render_main()

    def _on_toggle(self):
        self._show_arcs = self.arcs_cb.isChecked()
        self._show_boundaries = self.bounds_cb.isChecked()
        self._show_markers = self.markers_cb.isChecked()
        self._show_labels = self.labels_cb.isChecked()
        self._render_main()

    # ── rendering ──

    def _render_main(self):
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor("#f0f0f0")

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        overlay, sm = build_density_overlay(
            self._tth_map, self._chi_map, self._by_ref,
            self._active_refs, self._band_tol, self._cmap_name)

        ax.imshow(overlay, origin="upper", aspect="equal",
                  interpolation="nearest")

        if self._show_arcs:
            draw_arc_boundaries(ax, self._tth_map, self._active_refs)

        if self._show_boundaries:
            draw_cluster_boundaries(ax, self._valleys_by_ref,
                                    self._beam_center, self._tth_map.shape,
                                    self._active_refs)

        if self._show_labels:
            draw_cluster_labels(ax, self._clusters_by_ref, self._active_refs)

        if self._show_markers:
            for i, ref in enumerate(DEG_LABELS):
                if ref not in self._active_refs or ref not in self._by_ref:
                    continue
                xs = [f["detector_x"] for f in self._by_ref[ref]]
                ys = [f["detector_y"] for f in self._by_ref[ref]]
                color = ARC_COLORS[i % len(ARC_COLORS)]
                ax.scatter(xs, ys, s=15, c=color, edgecolors="white",
                           linewidths=0.3, zorder=5)

        if sm is not None:
            self._cbar = self.canvas.fig.colorbar(
                sm, ax=ax, fraction=0.03, pad=0.01)
            self._cbar.set_label("Feature density", color="#222", fontsize=9)
            self._cbar.ax.yaxis.set_tick_params(color="#222", labelcolor="#222")

        ax.set_xlim(-0.5, self._tth_map.shape[1] - 0.5)
        ax.set_ylim(self._tth_map.shape[0] - 0.5, -0.5)
        ax.set_title("Orientation Map — Adaptive Sectors",
                      color="#222", fontsize=12, pad=8)
        ax.tick_params(colors="#222", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#bbb")
        self.canvas.draw_idle()
        self._current_hover = None

    # ── hover / click ──

    def _on_hover(self, col, row):
        if col is None or row is None:
            if self._locked_sector is not None:
                self._restore_locked()
            else:
                self.hover_label.setText("Hover over a sector to see details")
                if self._current_hover is not None:
                    self._current_hover = None
                    self._clear_histograms()
            return

        self._show_sector(col, row)

    def _on_click(self, col, row):
        h, w = self._tth_map.shape
        if not (0 <= row < h and 0 <= col < w):
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
                self.hover_label.setText(
                    f"pixel ({col}, {row})  2θ={tth:.3f}°  χ={chi:.1f}°")
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
        ax = self.az_hist.ax
        ax.clear()
        ax.set_facecolor("white")

        ref_idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
        color = ARC_COLORS[ref_idx % len(ARC_COLORS)]

        all_chis = [f["chi_deg"] for f in self._by_ref.get(ref, [])
                    if f.get("chi_deg") is not None]
        cl_chis = [f["chi_deg"] for f in cluster["features"]
                   if f.get("chi_deg") is not None]

        if not all_chis:
            self.az_hist.draw_idle()
            return

        # Unwrap chi so the histogram is continuous along the arc.
        # Features span ±180° boundary; shift negatives by +360 so
        # bottom-of-detector (positive chi) is on the left and
        # top/right-of-detector (negative chi) is on the right.
        chi_min, chi_max = min(all_chis), max(all_chis)
        wraps = (chi_max - chi_min) > 180
        if wraps:
            all_plot = [c + 360 if c < 0 else c for c in all_chis]
            cl_plot = [c + 360 if c < 0 else c for c in cl_chis]
            lo_plot = cluster["chi_lo"] + (360 if cluster["chi_lo"] < 0 else 0)
            hi_plot = cluster["chi_hi"] + (360 if cluster["chi_hi"] < 0 else 0)
        else:
            all_plot = list(all_chis)
            cl_plot = list(cl_chis)
            lo_plot = cluster["chi_lo"]
            hi_plot = cluster["chi_hi"]

        bin_lo = min(all_plot) - 5
        bin_hi = max(all_plot) + 5
        edges = np.arange(bin_lo, bin_hi + 5, 5)
        centers = (edges[:-1] + edges[1:]) / 2
        h_all, _ = np.histogram(all_plot, bins=edges)
        h_cl, _ = np.histogram(cl_plot, bins=edges)

        ax.bar(centers, h_all, width=4.5, color="#ccc", alpha=0.5, label="all")
        ax.bar(centers, h_cl, width=4.5, color=color, alpha=0.85,
               label=f"{cluster['pct']:.0f}%")

        ax.axvline(lo_plot, color=color, ls="--", lw=0.8, alpha=0.6)
        ax.axvline(hi_plot, color=color, ls="--", lw=0.8, alpha=0.6)

        ax.set_xlabel("χ (°)  —  bottom of detector → top", color="#222",
                       fontsize=8)
        ax.set_ylabel("Count", color="#222", fontsize=8)
        ax.set_title(
            f"{ref} — azimuthal  ({cluster['pct']:.0f}% highlighted, "
            f"{cluster['chi_span']:.0f}° span)",
            color="#222", fontsize=9, pad=4)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7,
                  labelcolor="#222")
        ax.tick_params(colors="#222", labelsize=7)

        if wraps:
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: f"{x - 360:.0f}" if x > 180 else f"{x:.0f}"))

        for sp in ax.spines.values():
            sp.set_color("#bbb")
        self.az_hist.draw_idle()

    def _draw_arc_histogram(self, ref, cluster):
        ax = self.arc_hist.ax
        ax.clear()
        ax.set_facecolor("white")

        ref_idx = DEG_LABELS.index(ref) if ref in DEG_LABELS else 0
        color = ARC_COLORS[ref_idx % len(ARC_COLORS)]
        ref_tth = LABELED_DEGS.get(ref)

        # Collect Δ2θ from per-bin intensity profiles
        delta_tths = []
        for f in cluster["features"]:
            profile = f.get("intensity_profile", {})
            for entry in profile.values():
                if isinstance(entry, dict) and "tth" in entry and ref_tth is not None:
                    delta_tths.append(entry["tth"] - ref_tth)

        if not delta_tths:
            ax.text(0.5, 0.5, "No Δ2θ data", transform=ax.transAxes,
                    ha="center", va="center", color="#aaa", fontsize=10)
            self.arc_hist.draw_idle()
            return

        d_arr = np.array(delta_tths)
        n_bins = min(25, max(8, len(delta_tths) // 3))
        edges = np.linspace(d_arr.min() - 0.005, d_arr.max() + 0.005, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        hist, _ = np.histogram(d_arr, bins=edges)
        bw = (edges[1] - edges[0]) * 0.85

        # Color bars by density: hotter at center of distribution
        if hist.max() > 0:
            normed = hist / hist.max()
            bar_cmap = mcm.get_cmap(self._cmap_name)
            bar_colors = [bar_cmap(v) for v in normed]
        else:
            bar_colors = color

        ax.bar(centers, hist, width=bw, color=bar_colors, edgecolor="#999",
               alpha=0.85)

        ax.axvline(0, color="#aaa", ls=":", lw=0.8, alpha=0.7, label="ref 2θ")

        mean_d = np.mean(d_arr)
        ax.axvline(mean_d, color=color, ls="-", lw=1.2, alpha=0.8,
                   label=f"mean {mean_d:+.4f}°")

        ax.set_xlabel("Δ2θ (°)", color="#222", fontsize=8)
        ax.set_ylabel("Count", color="#222", fontsize=8)
        ax.set_title(
            f"{ref} cluster — Δ2θ distribution  ({cluster['n']} features, "
            f"{len(delta_tths)} measurements)",
            color="#222", fontsize=9, pad=4)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7,
                  labelcolor="#222")
        ax.invert_xaxis()
        ax.tick_params(colors="#222", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#bbb")
        self.arc_hist.draw_idle()

    def _clear_histograms(self):
        for canvas in [self.az_hist, self.arc_hist]:
            ax = canvas.ax
            ax.clear()
            ax.set_facecolor("white")
            ax.text(0.5, 0.5, "Hover over a sector", transform=ax.transAxes,
                    ha="center", va="center", color="#aaa", fontsize=10)
            ax.tick_params(colors="#bbb", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#ddd")
            canvas.draw_idle()


# ── Entry point ───────────────────────────────────────────────────

def launch_gui(project_root=".", bin_size=3, scan=None):
    """Configure paths and launch the orientation map (used by the CLI)."""
    configure(project_root=project_root, bin_size=bin_size, scan=scan)
    _run_app()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XRD orientation map")
    parser.add_argument("--project-root", type=str, default=".",
                        help="Path to the xrd-tools project root")
    parser.add_argument("--bin-size", type=int, default=3, help="Bin size to view")
    parser.add_argument("--scan", default=None, help="Scan number/name")
    args = parser.parse_args()
    launch_gui(project_root=args.project_root, bin_size=args.bin_size, scan=args.scan)


def _run_app():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(30, 30, 30))
    palette.setColor(QPalette.Button, QColor(230, 230, 230))
    palette.setColor(QPalette.ButtonText, QColor(30, 30, 30))
    palette.setColor(QPalette.BrightText, QColor(200, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = OrientationMapWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
