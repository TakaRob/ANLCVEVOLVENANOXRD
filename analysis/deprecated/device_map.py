"""
Device-level feature map for perovskite reflections.

Shows spatial profiles across the full 52x74 bin grid with switchable
metrics: intensity, lattice strain, crystallographic orientation, and
mosaicity / domain structure.  Chi-angle range filter with interactive
histogram and slider controls.

Usage:
    python3 analysis/device_map.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QCheckBox, QPushButton, QGroupBox, QComboBox, QSplitter,
    QSizePolicy, QSpinBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QBrush

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results" / "scan203"
HOLDOUT_DIR = BASE / "cvevolve_3x3" / "holdout_data"

REFLECTIONS = []
REF_COLORS = {}

_PALETTE = [
    "#4363d8", "#e6194b", "#3cb44b", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9a6324",
    "#800000", "#008080", "#fabebe", "#aaffc3",
]


def _assign_ref_colors(reflections):
    colors = {}
    for i, ref in enumerate(reflections):
        colors[ref] = _PALETTE[i % len(_PALETTE)]
    return colors


METRICS = [
    ("none",      "None (outlines only)"),
    ("intensity", "Intensity"),
    ("chi",       "χ angle"),
    ("strain",    "Lattice strain"),
    ("rocking",   "Rocking width"),
    ("strain_bw", "Strain breadth"),
]

METRIC_ZLABELS = {
    "intensity":  "Integrated intensity",
    "chi":        "χ (°)",
    "strain":     "Δ2θ (°)",
    "rocking":    "FWHM χ (°)",
    "strain_bw":  "FWHM Δ2θ (°)",
}

METRIC_CMAPS = {
    "chi": "twilight",
}

METRIC_2D_TITLES = {
    "none":       "Crystal Segmentation — feature outlines by reflection",
    "intensity":  "Integrated Intensity — peak area per bin",
    "chi":        "χ Angle — azimuthal orientation on Debye ring",
    "strain":     "Lattice Strain — d-spacing deviation (Δ2θ from reference)",
    "rocking":    "Rocking Width — crystal plane curvature / mosaic spread per feature",
    "strain_bw":  "Strain Breadth — lattice parameter gradient across feature",
}

def load_features():
    with open(RESULTS_DIR / "feature_catalog_3x3.json") as f:
        return json.load(f)


def load_grid_info():
    with open(HOLDOUT_DIR / "grid_mapping.json") as f:
        gm = json.load(f)
    return gm["n_bin_rows"], gm["n_bin_cols"]


PER_FEATURE_METRICS = {
    "rocking": "rocking_fwhm", "strain_bw": "strain_breadth", "chi": "chi_deg",
}


def _extract_metric(entry, feat, metric):
    if not isinstance(entry, dict):
        return float(entry) if metric == "intensity" else None
    if metric == "intensity":
        return entry.get("integrated", entry.get("intensity", 0))
    if metric == "strain":
        tth = entry.get("tth")
        ref_tth = feat.get("ref_tth")
        if tth is not None and ref_tth is not None:
            return tth - ref_tth
        return None
    return None


def build_device_grids(features, n_rows, n_cols, metric="intensity"):
    reflections = sorted(set(f["reflection"] for f in features))
    grids = {}
    is_per_feature = metric in PER_FEATURE_METRICS
    feat_key = PER_FEATURE_METRICS.get(metric)

    for ref in reflections:
        grid = np.full((n_rows, n_cols), np.nan)
        ref_feats = [f for f in features if f["reflection"] == ref]
        for feat in ref_feats:
            profile = feat.get("intensity_profile", {})

            if is_per_feature:
                val = feat.get(feat_key)
                if val is None:
                    continue
                for bk in profile:
                    parts = bk.split("_")
                    if len(parts) != 2:
                        continue
                    r, c = int(parts[0]), int(parts[1])
                    if 0 <= r < n_rows and 0 <= c < n_cols:
                        grid[r, c] = val
            else:
                for bk, entry in profile.items():
                    parts = bk.split("_")
                    if len(parts) != 2:
                        continue
                    r, c = int(parts[0]), int(parts[1])
                    if 0 <= r < n_rows and 0 <= c < n_cols:
                        val = _extract_metric(entry, feat, metric)
                        if val is None:
                            continue
                        if metric == "strain":
                            if np.isnan(grid[r, c]) or abs(val) > abs(grid[r, c]):
                                grid[r, c] = val
                        else:
                            grid[r, c] = np.nanmax([grid[r, c], val])
        grids[ref] = grid
    return grids


def _feat_in_chi_range(feat, chi_range):
    if chi_range is None:
        return True
    lo, hi = chi_range
    chi = feat.get("chi_deg")
    if chi is None:
        return True
    return lo <= chi <= hi


def _build_outline_groups(features, n_rows, n_cols, visible_refs,
                          chi_range=None):
    """Merge all feature masks per reflection for outline drawing."""
    groups = []
    for ref in visible_refs:
        merged = np.zeros((n_rows, n_cols), dtype=bool)
        for f in features:
            if f["reflection"] == ref and f.get("_mask") is not None:
                if _feat_in_chi_range(f, chi_range):
                    merged |= f["_mask"]
        if merged.any():
            groups.append((ref, merged))
    return groups


def _draw_2d_heatmap(ax, features, grids, n_rows, n_cols, metric, cb_axes,
                     visible_refs=None, show_labels=False, chi_range=None):
    """Draw the 2D heatmap with merged outlines colored by reflection."""
    if visible_refs is None:
        visible_refs = REFLECTIONS
    ax.clear()
    ax.set_facecolor("white")
    for cb_ax in cb_axes:
        cb_ax.set_visible(False)

    outline_groups = _build_outline_groups(features, n_rows, n_cols,
                                           visible_refs, chi_range=chi_range)

    chi_valid = {}
    if chi_range is not None:
        for ref in visible_refs:
            mask = np.zeros((n_rows, n_cols), dtype=bool)
            for f in features:
                if f["reflection"] == ref and _feat_in_chi_range(f, chi_range):
                    m = f.get("_mask")
                    if m is not None:
                        mask |= m
            chi_valid[ref] = mask

    if metric == "none":
        rgba = np.ones((n_rows, n_cols, 4))
        for ref, merged in outline_groups:
            rgb = mcolors.to_rgb(REF_COLORS[ref])
            pastel = tuple(c * 0.35 + 0.65 for c in rgb)
            for ch in range(3):
                rgba[:, :, ch] = np.where(merged, pastel[ch], rgba[:, :, ch])
        ax.imshow(rgba, origin="upper", aspect="equal", interpolation="nearest")

    else:
        combined = np.full((n_rows, n_cols), np.nan)
        for ref in visible_refs:
            if ref not in grids:
                continue
            Z = grids[ref]
            valid = np.isfinite(Z)
            if metric == "intensity":
                valid = valid & (Z > 0)
            if ref in chi_valid:
                valid = valid & chi_valid[ref]
            if not valid.any():
                continue
            first_fill = valid & np.isnan(combined)
            if metric == "strain":
                better = valid & np.isfinite(combined) & (np.abs(Z) > np.abs(combined))
            else:
                better = valid & np.isfinite(combined) & (Z > combined)
            combined = np.where(first_fill | better, Z, combined)

        has_data = np.isfinite(combined)
        if has_data.any():
            finite = combined[has_data]
            if metric == "intensity":
                vmin, vmax = 0.0, float(finite.max())
            else:
                vmin, vmax = float(finite.min()), float(finite.max())
            if abs(vmax - vmin) < 1e-8:
                vmax = vmin + 1
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(METRIC_CMAPS.get(metric, "viridis")).copy()
            cmap.set_bad("white")
            display = np.ma.array(combined, mask=~has_data)
            ax.imshow(display, origin="upper", aspect="equal",
                      interpolation="nearest", cmap=cmap, norm=norm)

            if cb_axes:
                cb_ax = cb_axes[0]
                cb_ax.set_visible(True)
                cb_ax.clear()
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cb = ax.figure.colorbar(sm, cax=cb_ax)
                cb.ax.tick_params(labelsize=6)
                cb.set_label(METRIC_ZLABELS.get(metric, ""), fontsize=7)
        else:
            ax.imshow(np.ones((n_rows, n_cols, 3)), origin="upper",
                      aspect="equal", interpolation="nearest")

    for ref, merged in outline_groups:
        color = REF_COLORS[ref]
        ax.contour(merged.astype(float), levels=[0.5], colors=[color],
                   linewidths=1.5)

    if show_labels:
        for feat in features:
            ref = feat["reflection"]
            if ref not in visible_refs:
                continue
            if not _feat_in_chi_range(feat, chi_range):
                continue
            r, c = feat["center_row"], feat["center_col"]
            chi = feat.get("chi_deg", 0)
            fid = feat.get("feature_id", "?")
            color = REF_COLORS.get(ref, "black")
            ax.plot(c, r, "o", color=color, markersize=3)
            ax.annotate(
                f"#{fid}\nχ={chi:.0f}°",
                xy=(c, r), xytext=(c + 2, r - 2),
                color=color, fontsize=5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.85,
                          ec=color, linewidth=0.3),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.3),
            )

    legend_patches = [Patch(facecolor=REF_COLORS[r], edgecolor=REF_COLORS[r],
                            label=r, linewidth=2)
                      for r in visible_refs if r in REF_COLORS]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    ax.set_xlabel("Col (scan x)", fontsize=9)
    ax.set_ylabel("Row (scan y)", fontsize=9)
    ax.set_title(METRIC_2D_TITLES.get(metric, metric), fontsize=10)
    ax.tick_params(labelsize=7)


class QRangeSlider(QWidget):
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, lo=-180, hi=180, parent=None):
        super().__init__(parent)
        self._min = lo
        self._max = hi
        self._lo = lo
        self._hi = hi
        self._dragging = None
        self.setFixedHeight(28)
        self.setMinimumWidth(100)

    def setRange(self, lo, hi):
        self._min = lo
        self._max = hi
        self._lo = max(self._lo, lo)
        self._hi = min(self._hi, hi)
        self.update()

    def setLow(self, v):
        self._lo = max(self._min, min(v, self._hi))
        self.update()

    def setHigh(self, v):
        self._hi = min(self._max, max(v, self._lo))
        self.update()

    def low(self):
        return self._lo

    def high(self):
        return self._hi

    def _val_to_x(self, v):
        margin = 8
        w = self.width() - 2 * margin
        frac = (v - self._min) / max(self._max - self._min, 1)
        return int(margin + frac * w)

    def _x_to_val(self, x):
        margin = 8
        w = self.width() - 2 * margin
        frac = (x - margin) / max(w, 1)
        return int(round(self._min + frac * (self._max - self._min)))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        track_y = h // 2 - 2
        margin = 8
        w = self.width() - 2 * margin

        p.setBrush(QBrush(QColor("#ddd")))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(margin, track_y, w, 5, 2, 2)

        x_lo = self._val_to_x(self._lo)
        x_hi = self._val_to_x(self._hi)
        p.setBrush(QBrush(QColor("#4363d8")))
        p.drawRect(x_lo, track_y, x_hi - x_lo, 5)

        for x in (x_lo, x_hi):
            p.setBrush(QBrush(QColor("white")))
            p.setPen(QColor("#4363d8"))
            p.drawEllipse(x - 7, track_y - 5, 14, 14)
        p.end()

    def mousePressEvent(self, event):
        x = event.x()
        x_lo = self._val_to_x(self._lo)
        x_hi = self._val_to_x(self._hi)
        if abs(x - x_lo) <= 10:
            self._dragging = "lo"
        elif abs(x - x_hi) <= 10:
            self._dragging = "hi"
        elif x_lo < x < x_hi:
            if abs(x - x_lo) < abs(x - x_hi):
                self._dragging = "lo"
            else:
                self._dragging = "hi"

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        v = self._x_to_val(event.x())
        v = max(self._min, min(self._max, v))
        if self._dragging == "lo":
            v = min(v, self._hi - 1)
            self._lo = v
        else:
            v = max(v, self._lo + 1)
            self._hi = v
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)

    def mouseReleaseEvent(self, event):
        self._dragging = None


class DeviceMapWindow(QMainWindow):
    def __init__(self, features, grids, n_rows, n_cols):
        super().__init__()
        self.setWindowTitle("Perovskite Device Map — Bragg Peak Analysis")
        self.setGeometry(50, 30, 1450, 900)

        self.features = features
        self.grids = grids
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.metric = "none"
        self.current_grids = grids
        self.visible_refs = list(REFLECTIONS)
        self.show_labels = False
        self._highlight_artists = []
        self._highlighted_idx = None
        self._locked_idx = None
        self._chi_hist_ref = "All"

        all_chi = [f["chi_deg"] for f in features if f.get("chi_deg") is not None]
        if all_chi:
            self._chi_data_min = int(np.floor(min(all_chi)))
            self._chi_data_max = int(np.ceil(max(all_chi)))
        else:
            self._chi_data_min, self._chi_data_max = -180, 180
        self._chi_lo = float(self._chi_data_min)
        self._chi_hi = float(self._chi_data_max)

        self._build_ui()
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._update_chi_visuals()
        self._redraw()

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

        self.fig = plt.figure(figsize=(10, 7), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, left)
        left_lay.addWidget(self.toolbar)
        left_lay.addWidget(self.canvas, stretch=1)

        self.hover_label = QLabel("Hover over a feature to see details")
        self.hover_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #555; "
            "padding: 4px; background: #f0f0f0;")
        self.hover_label.setFixedHeight(22)
        left_lay.addWidget(self.hover_label)
        splitter.addWidget(left)

        # Right: controls + histogram
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)

        # --- Layers ---
        rg = QGroupBox("Layers")
        rgl = QVBoxLayout(rg)
        btn_row = QHBoxLayout()
        ab = QPushButton("All"); ab.setFixedWidth(45)
        ab.clicked.connect(self._check_all_layers)
        nb = QPushButton("None"); nb.setFixedWidth(45)
        nb.clicked.connect(self._uncheck_all_layers)
        btn_row.addWidget(ab); btn_row.addWidget(nb); btn_row.addStretch()
        rgl.addLayout(btn_row)

        self.layer_cbs = {}
        row = QHBoxLayout()
        for i, ref in enumerate(REFLECTIONS):
            cb = QCheckBox(ref)
            cb.setChecked(True)
            cb.setStyleSheet(
                f"QCheckBox {{ color: {REF_COLORS[ref]}; }}")
            cb.toggled.connect(self._on_layer_toggle)
            row.addWidget(cb)
            self.layer_cbs[ref] = cb
            if (i + 1) % 4 == 0:
                rgl.addLayout(row)
                row = QHBoxLayout()
        if row.count():
            rgl.addLayout(row)

        tog = QHBoxLayout()
        self.labels_cb = QCheckBox("Points")
        self.labels_cb.setChecked(False)
        self.labels_cb.toggled.connect(self._on_labels_toggle)
        tog.addWidget(self.labels_cb)
        tog.addStretch()
        rgl.addLayout(tog)
        rl.addWidget(rg)

        # --- Metric ---
        sg = QGroupBox("Metric")
        sl = QVBoxLayout(sg)
        self.metric_combo = QComboBox()
        for key, label in METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        sl.addWidget(self.metric_combo)
        rl.addWidget(sg)

        # --- Azimuthal distribution (χ) ---
        rl.addWidget(QLabel("  Azimuthal distribution (χ)"))

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reflection:"))
        self.chi_ref_combo = QComboBox()
        self.chi_ref_combo.addItem("All")
        for ref in REFLECTIONS:
            self.chi_ref_combo.addItem(ref)
        self.chi_ref_combo.currentIndexChanged.connect(self._on_chi_ref_changed)
        ref_row.addWidget(self.chi_ref_combo)
        rl.addLayout(ref_row)

        self.chi_hist_fig = Figure(figsize=(4, 2.5))
        self.chi_hist_fig.patch.set_facecolor("white")
        self.chi_hist_ax = self.chi_hist_fig.add_subplot(111)
        self.chi_hist_ax.set_facecolor("white")
        self.chi_hist_fig.subplots_adjust(
            left=0.15, right=0.95, top=0.88, bottom=0.22)
        self.chi_hist_canvas = FigureCanvasQTAgg(self.chi_hist_fig)
        self.chi_hist_canvas.setFixedHeight(230)
        self.chi_hist_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        rl.addWidget(self.chi_hist_canvas)

        self.chi_range_slider = QRangeSlider(
            self._chi_data_min, self._chi_data_max)
        self.chi_range_slider.rangeChanged.connect(self._on_range_slider)
        rl.addWidget(self.chi_range_slider)

        entry_row = QHBoxLayout()
        entry_row.addWidget(QLabel("Min:"))
        self.chi_min_spin = QSpinBox()
        self.chi_min_spin.setRange(self._chi_data_min, self._chi_data_max)
        self.chi_min_spin.setValue(self._chi_data_min)
        self.chi_min_spin.setSuffix("°")
        self.chi_min_spin.setFixedWidth(72)
        self.chi_min_spin.valueChanged.connect(self._on_chi_min_spin)
        entry_row.addWidget(self.chi_min_spin)
        entry_row.addStretch()
        entry_row.addWidget(QLabel("Max:"))
        self.chi_max_spin = QSpinBox()
        self.chi_max_spin.setRange(self._chi_data_min, self._chi_data_max)
        self.chi_max_spin.setValue(self._chi_data_max)
        self.chi_max_spin.setSuffix("°")
        self.chi_max_spin.setFixedWidth(72)
        self.chi_max_spin.valueChanged.connect(self._on_chi_max_spin)
        entry_row.addWidget(self.chi_max_spin)
        rl.addLayout(entry_row)

        self.chi_range_label = QLabel(
            f"χ: {self._chi_data_min}° to {self._chi_data_max}°")
        self.chi_range_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #555; "
            "padding: 2px;")
        rl.addWidget(self.chi_range_label)

        self.arc_fig = Figure(figsize=(4, 1.5))
        self.arc_fig.patch.set_facecolor("white")
        self.arc_ax = self.arc_fig.add_subplot(111, polar=True)
        self.arc_fig.subplots_adjust(left=0.05, right=0.95, top=0.95,
                                     bottom=0.05)
        self.arc_canvas = FigureCanvasQTAgg(self.arc_fig)
        self.arc_canvas.setFixedHeight(120)
        self.arc_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        rl.addWidget(self.arc_canvas)

        # --- Feature info ---
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "QLabel { font-size: 10pt; padding: 6px; }")
        rl.addWidget(self.info_label)

        rl.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([950, 450])

    def _setup_axes(self):
        self.fig.clear()
        self.fig.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.05)
        self.ax_2d = self.fig.add_subplot(111)
        self.ax_cb1 = self.fig.add_axes([0.93, 0.25, 0.015, 0.55])
        self.ax_cb1.set_visible(False)

    def _chi_range(self):
        if (self._chi_lo <= self._chi_data_min
                and self._chi_hi >= self._chi_data_max):
            return None
        return (self._chi_lo, self._chi_hi)

    def _redraw(self):
        self._highlight_artists.clear()
        self._highlighted_idx = None
        if self._locked_idx is None:
            self.info_label.setText("")
            self.hover_label.setText("Hover over a feature to see details")
        self._setup_axes()
        _draw_2d_heatmap(self.ax_2d, self.features, self.current_grids,
                         self.n_rows, self.n_cols, self.metric,
                         [self.ax_cb1],
                         visible_refs=self.visible_refs,
                         show_labels=self.show_labels,
                         chi_range=self._chi_range())
        if self._locked_idx is not None:
            feat = self.features[self._locked_idx]
            if (feat["reflection"] in self.visible_refs
                    and _feat_in_chi_range(feat, self._chi_range())):
                self._draw_highlight(self._locked_idx)
            else:
                self._locked_idx = None
                self.info_label.setText("")
                self.hover_label.setText(
                    "Hover over a feature to see details")
        self.canvas.draw_idle()

    def _check_all_layers(self):
        for cb in self.layer_cbs.values():
            cb.setChecked(True)

    def _uncheck_all_layers(self):
        for cb in self.layer_cbs.values():
            cb.setChecked(False)

    def _on_layer_toggle(self):
        self.visible_refs = [ref for ref in REFLECTIONS
                             if self.layer_cbs[ref].isChecked()]
        self._redraw()

    def _on_labels_toggle(self, checked):
        self.show_labels = checked
        self._redraw()

    def _on_metric_changed(self):
        key = self.metric_combo.currentData()
        if key == self.metric:
            return
        self.metric = key
        if key != "none":
            self.current_grids = build_device_grids(
                self.features, self.n_rows, self.n_cols, metric=key)
        self._redraw()

    # --- Chi range filter ---

    def _update_chi_visuals(self):
        self._draw_chi_histogram()
        self._draw_chi_arc()
        self.chi_range_label.setText(
            f"χ: {self._chi_lo:.0f}° to {self._chi_hi:.0f}°")

    def _draw_chi_histogram(self):
        ax = self.chi_hist_ax
        ax.clear()
        ax.set_facecolor("white")

        ref_filter = self._chi_hist_ref
        all_chis = []
        for f in self.features:
            if ref_filter != "All" and f["reflection"] != ref_filter:
                continue
            chi = f.get("chi_deg")
            if chi is not None:
                all_chis.append(chi)

        if not all_chis:
            ax.text(0.5, 0.5, "No χ data", transform=ax.transAxes,
                    ha="center", va="center", color="#aaa", fontsize=10)
            self.chi_hist_canvas.draw_idle()
            return

        in_range = [c for c in all_chis if self._chi_lo <= c <= self._chi_hi]

        chi_min, chi_max = min(all_chis), max(all_chis)
        wraps = (chi_max - chi_min) > 180
        if wraps:
            all_plot = [c + 360 if c < 0 else c for c in all_chis]
            in_plot = [c + 360 if c < 0 else c for c in in_range]
            lo_plot = self._chi_lo + (360 if self._chi_lo < 0 else 0)
            hi_plot = self._chi_hi + (360 if self._chi_hi < 0 else 0)
        else:
            all_plot = list(all_chis)
            in_plot = list(in_range)
            lo_plot = self._chi_lo
            hi_plot = self._chi_hi

        bin_lo = min(all_plot) - 5
        bin_hi = max(all_plot) + 5
        edges = np.arange(bin_lo, bin_hi + 5, 5)
        centers = (edges[:-1] + edges[1:]) / 2
        h_all, _ = np.histogram(all_plot, bins=edges)
        h_in, _ = np.histogram(in_plot, bins=edges)

        n_in = len(in_range)
        n_all = len(all_chis)
        pct = 100 * n_in / n_all if n_all else 0
        span = self._chi_hi - self._chi_lo

        ax.bar(centers, h_all, width=4.5, color="#ccc", alpha=0.5,
               label="all")
        ax.bar(centers, h_in, width=4.5, color="#4363d8", alpha=0.85,
               label=f"{pct:.0f}%")

        ax.axvline(lo_plot, color="red", ls="--", lw=1.2, alpha=0.8)
        ax.axvline(hi_plot, color="red", ls="--", lw=1.2, alpha=0.8)

        ref_label = ref_filter if ref_filter != "All" else "all refs"
        ax.set_xlabel("χ (°)", color="#222", fontsize=8)
        ax.set_ylabel("Count", color="#222", fontsize=8)
        ax.set_title(
            f"{ref_label} — azimuthal  ({pct:.0f}% selected, "
            f"{span:.0f}° span)",
            color="#222", fontsize=9, pad=4)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7,
                  labelcolor="#222")
        ax.tick_params(colors="#222", labelsize=7)

        if wraps:
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: f"{x - 360:.0f}" if x > 180
                    else f"{x:.0f}"))

        for sp in ax.spines.values():
            sp.set_color("#bbb")
        self.chi_hist_canvas.draw_idle()

    def _on_chi_ref_changed(self):
        idx = self.chi_ref_combo.currentIndex()
        if idx == 0:
            self._chi_hist_ref = "All"
        else:
            self._chi_hist_ref = REFLECTIONS[idx - 1]
        self._draw_chi_histogram()

    def _sync_chi_widgets(self):
        for w in (self.chi_range_slider, self.chi_min_spin, self.chi_max_spin):
            w.blockSignals(True)
        self.chi_range_slider.setLow(int(self._chi_lo))
        self.chi_range_slider.setHigh(int(self._chi_hi))
        self.chi_min_spin.setValue(int(self._chi_lo))
        self.chi_max_spin.setValue(int(self._chi_hi))
        for w in (self.chi_range_slider, self.chi_min_spin, self.chi_max_spin):
            w.blockSignals(False)

    def _on_range_slider(self, lo, hi):
        self._chi_lo = float(lo)
        self._chi_hi = float(hi)
        self._sync_chi_widgets()
        self._update_chi_visuals()
        self._redraw()

    def _on_chi_min_spin(self, val):
        if val >= self._chi_hi:
            val = int(self._chi_hi) - 1
            self.chi_min_spin.blockSignals(True)
            self.chi_min_spin.setValue(val)
            self.chi_min_spin.blockSignals(False)
        self._chi_lo = float(val)
        self._sync_chi_widgets()
        self._update_chi_visuals()
        self._redraw()

    def _on_chi_max_spin(self, val):
        if val <= self._chi_lo:
            val = int(self._chi_lo) + 1
            self.chi_max_spin.blockSignals(True)
            self.chi_max_spin.setValue(val)
            self.chi_max_spin.blockSignals(False)
        self._chi_hi = float(val)
        self._sync_chi_widgets()
        self._update_chi_visuals()
        self._redraw()

    def _draw_chi_arc(self):
        ax = self.arc_ax
        ax.clear()
        ax.set_facecolor("white")

        theta_lo = np.radians(self._chi_data_min)
        theta_hi = np.radians(self._chi_data_max)
        sel_lo = np.radians(self._chi_lo)
        sel_hi = np.radians(self._chi_hi)

        bg_theta = np.linspace(theta_lo, theta_hi, 200)
        ax.fill_between(bg_theta, 0.6, 1.0, color="#e0e0e0", alpha=0.5)

        sel_theta = np.linspace(sel_lo, sel_hi, 200)
        ax.fill_between(sel_theta, 0.6, 1.0, color="#4363d8", alpha=0.6)

        for angle in (sel_lo, sel_hi):
            ax.plot([angle, angle], [0.5, 1.05], color="red",
                    lw=1.5, ls="--", alpha=0.9)

        ax.set_ylim(0, 1.1)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        ax.grid(False)
        self.arc_canvas.draw_idle()

    # --- Click-to-lock highlight ---

    def _on_click(self, event):
        if event.inaxes != self.ax_2d or event.xdata is None:
            return
        if not self.show_labels:
            return

        mx, my = event.xdata, event.ydata
        best_idx, best_dist = None, float("inf")
        chi_r = self._chi_range()
        for i, feat in enumerate(self.features):
            if feat["reflection"] not in self.visible_refs:
                continue
            if not _feat_in_chi_range(feat, chi_r):
                continue
            dc = mx - feat["center_col"]
            dr = my - feat["center_row"]
            d = (dc * dc + dr * dr) ** 0.5
            if d < best_dist:
                best_dist, best_idx = d, i

        if best_dist > 3.0 or best_idx is None:
            if self._locked_idx is not None:
                self._locked_idx = None
                self._clear_highlight()
            return

        if best_idx == self._locked_idx:
            self._locked_idx = None
            self._clear_highlight()
        else:
            self._locked_idx = best_idx
            self._clear_highlight(draw=False)
            self._draw_highlight(best_idx)

    def _draw_highlight(self, idx):
        self._highlighted_idx = idx
        feat = self.features[idx]
        mask = feat.get("_mask")
        if mask is None or not mask.any():
            return

        ref = feat["reflection"]
        color_rgb = mcolors.to_rgb(REF_COLORS[ref])
        rgba = np.zeros((self.n_rows, self.n_cols, 4))
        rgba[mask, 0] = color_rgb[0]
        rgba[mask, 1] = color_rgb[1]
        rgba[mask, 2] = color_rgb[2]
        rgba[mask, 3] = 0.45
        img = self.ax_2d.imshow(rgba, origin="upper", aspect="equal",
                                interpolation="nearest", zorder=10)
        self._highlight_artists.append(img)

        cs = self.ax_2d.contour(mask.astype(float), levels=[0.5],
                                colors=["white"], linewidths=3, zorder=11)
        self._highlight_artists.append(cs)
        cs2 = self.ax_2d.contour(mask.astype(float), levels=[0.5],
                                 colors=[REF_COLORS[ref]],
                                 linewidths=1.8, zorder=12)
        self._highlight_artists.append(cs2)

        fid = feat.get("feature_id", "?")
        ref_tth = feat.get("ref_tth")
        chi = feat.get("chi_deg", 0)
        nbins = int(mask.sum())
        lines = [f"#{fid}  {ref}"]
        if ref_tth is not None:
            lines.append(f"2θ={ref_tth:.3f}°  χ={chi:.1f}°")
        else:
            lines.append(f"χ={chi:.1f}°")
        lines.append(f"{nbins} bins")
        self.info_label.setText("\n".join(lines))

        pinned = "  [pinned]" if self._locked_idx is not None else ""
        tth_str = f"2θ={ref_tth:.3f}°  " if ref_tth is not None else ""
        self.hover_label.setText(
            f"#{fid}  {ref}  {tth_str}χ={chi:.1f}°  "
            f"{nbins} bins{pinned}")
        self.canvas.draw_idle()

    # --- Hover ---

    def _on_mouse_move(self, event):
        if self._locked_idx is not None:
            return
        if not self.show_labels:
            if self._highlighted_idx is not None:
                self._clear_highlight()
            return
        if event.inaxes != self.ax_2d or event.xdata is None:
            if self._highlighted_idx is not None:
                self._clear_highlight()
            return

        mx, my = event.xdata, event.ydata
        best_idx, best_dist = None, float("inf")
        chi_r = self._chi_range()
        for i, feat in enumerate(self.features):
            if feat["reflection"] not in self.visible_refs:
                continue
            if not _feat_in_chi_range(feat, chi_r):
                continue
            dc = mx - feat["center_col"]
            dr = my - feat["center_row"]
            d = (dc * dc + dr * dr) ** 0.5
            if d < best_dist:
                best_dist, best_idx = d, i

        if best_dist > 3.0 or best_idx is None:
            if self._highlighted_idx is not None:
                self._clear_highlight()
            return

        if best_idx == self._highlighted_idx:
            return

        self._clear_highlight(draw=False)
        self._draw_highlight(best_idx)

    def _clear_highlight(self, draw=True):
        for artist in self._highlight_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._highlight_artists.clear()
        self._highlighted_idx = None
        if self._locked_idx is None:
            self.info_label.setText("")
            self.hover_label.setText(
                "Hover over a feature to see details")
        if draw:
            self.canvas.draw_idle()


def main():
    global REFLECTIONS, REF_COLORS

    features = load_features()
    REFLECTIONS = sorted(set(f["reflection"] for f in features))
    REF_COLORS = _assign_ref_colors(REFLECTIONS)

    n_rows, n_cols = load_grid_info()

    for feat in features:
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        for bk in feat.get("intensity_profile", {}):
            parts = bk.split("_")
            if len(parts) == 2:
                r, c = int(parts[0]), int(parts[1])
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    mask[r, c] = True
        feat["_mask"] = mask

    grids = build_device_grids(features, n_rows, n_cols)

    print(f"Device grid: {n_rows} x {n_cols}")
    for ref in REFLECTIONS:
        n = sum(1 for f in features if f["reflection"] == ref)
        nz = int(np.sum(np.isfinite(grids[ref]) & (grids[ref] > 0)))
        print(f"  {ref}: {n} features, {nz} non-zero bins")

    print("\nOpening device map...")
    app = QApplication.instance() or QApplication(sys.argv)
    win = DeviceMapWindow(features, grids, n_rows, n_cols)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
