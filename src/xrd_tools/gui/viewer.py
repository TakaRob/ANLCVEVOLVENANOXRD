"""
Spatial Feature Viewer — interactive GUI for reviewing kept/filtered
Bragg peak features from spatial_feature_analysis.py.

Three-panel layout:
  Left:   Device heatmap (52×74 bin grid, zoomed to feature, one reflection layer)
  Center: Detector image (1062×1028, with peak circled)
  Right:  Controls (category, reflection, feature index, info, display settings)

Usage:
    python3 analysis/feature_viewer.py
"""

import csv
import importlib.util
import json
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget, QCheckBox,
)

# These are resolved at runtime by configure(); see launch_gui().
_DM = None
_BIN_SIZE = 3
RESULTS_DIR = None
HOLDOUT_DIR = None
DETECTOR_PATH = None
H5_PATH = None

COLORMAPS = [
    "inferno", "viridis", "plasma", "magma", "cividis",
    "hot", "coolwarm", "gray", "jet", "turbo",
]

ARC_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6",
]

from ..core.algorithms import (
    ALGORITHM_NAMES, ALGORITHM_DISPLAY,
    compute_radial_profile, fit_all_models, build_background_image,
    subtract_background,
)
from ..config import DataManager


def configure(project_root=".", bin_size=3, scan=None):
    """Resolve all data paths for the viewer from the project config."""
    global _DM, _BIN_SIZE, RESULTS_DIR, HOLDOUT_DIR, DETECTOR_PATH, H5_PATH
    _DM = DataManager(project_root, scan=scan)
    _BIN_SIZE = bin_size
    RESULTS_DIR = _DM.results_dir()
    HOLDOUT_DIR = _DM.holdout_dir
    DETECTOR_PATH = _DM.detector_script()
    H5_PATH = _DM.bins_h5(bin_size)


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Data loading ───────────────────────────────────────────────────

def load_features():
    suffix = f"{_BIN_SIZE}x{_BIN_SIZE}"
    catalog_path = RESULTS_DIR / f"feature_catalog_{suffix}.json"
    filtered_csv = RESULTS_DIR / f"filtered_peaks_{suffix}.csv"

    with open(catalog_path) as f:
        kept = json.load(f)

    filtered = []
    with open(filtered_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            extent_str = row["spatial_extent"].strip()
            extent_list = extent_str.split() if extent_str else [row["bin_key"]]
            intensity = float(row["peak_intensity"])
            profile = {}
            for bk in extent_list:
                profile[bk] = intensity
            filtered.append({
                "feature_id": int(row["feature_id"]),
                "reflection": row["reflection"],
                "detector_x": int(row["detector_x"]),
                "detector_y": int(row["detector_y"]),
                "peak_intensity": intensity,
                "mean_snr": float(row["mean_snr"]),
                "n_bins": int(row["n_bins"]),
                "spatial_extent": extent_list,
                "center_bin": row["bin_key"],
                "center_row": int(row["center_row"]),
                "center_col": int(row["center_col"]),
                "reason": row["reason"],
                "intensity_profile": profile,
            })

    return kept, filtered


def group_by_reflection(features):
    groups = defaultdict(list)
    for feat in features:
        groups[feat["reflection"]].append(feat)
    return dict(groups)


# ── Canvases ───────────────────────────────────────────────────────

class HeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(5, 6))
        self.fig.patch.set_facecolor("#1a1a1a")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.fig.subplots_adjust(left=0.12, right=0.88, top=0.93, bottom=0.07)
        super().__init__(self.fig)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._click_cb = None
        self._hover_cb = None
        self._grid_data = None
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_click_callback(self, cb):
        self._click_cb = cb

    def set_hover_callback(self, cb):
        self._hover_cb = cb

    def _on_click(self, event):
        if event.inaxes == self.ax and self._click_cb and event.xdata is not None:
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            self._click_cb(row, col)

    def _on_motion(self, event):
        if self._hover_cb and event.inaxes == self.ax and event.xdata is not None:
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            intensity = None
            if self._grid_data is not None:
                r_off = getattr(self, '_grid_r_lo', 0)
                c_off = getattr(self, '_grid_c_lo', 0)
                ri = row - r_off
                ci = col - c_off
                if 0 <= ri < self._grid_data.shape[0] and 0 <= ci < self._grid_data.shape[1]:
                    val = self._grid_data[ri, ci]
                    if not np.isnan(val):
                        intensity = val
            self._hover_cb(row, col, intensity)
        elif self._hover_cb:
            self._hover_cb(None, None, None)


class IsometricCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(5, 4))
        self.fig.patch.set_facecolor("#1a1a1a")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#1a1a1a")
        self.fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        super().__init__(self.fig)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class DetectorCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6))
        self.fig.patch.set_facecolor("#1a1a1a")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.03)
        super().__init__(self.fig)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hover_cb = None
        self._click_cb = None
        self._drag_cb = None
        self._display_data = None
        self._drag_start = None
        self._dragging = False
        self._drag_rect = None
        self.drag_enabled = False
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)

    def set_hover_callback(self, cb):
        self._hover_cb = cb

    def set_click_callback(self, cb):
        self._click_cb = cb

    def set_drag_callback(self, cb):
        self._drag_cb = cb

    def _on_press(self, event):
        if event.button != 1 or event.inaxes != self.ax or event.xdata is None:
            return
        if self.drag_enabled:
            self._dragging = True
            self._drag_start = (event.xdata, event.ydata)
            return
        if self._click_cb:
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            self._click_cb(col, row)

    def _on_release(self, event):
        if not self._dragging or not self._drag_start:
            return
        # Remove the transient drag preview rect
        if self._drag_rect:
            try:
                self._drag_rect.remove()
            except Exception:
                pass
            self._drag_rect = None

        if event.inaxes == self.ax and event.xdata is not None:
            x0, y0 = self._drag_start
            x1, y1 = event.xdata, event.ydata
            if abs(x1 - x0) > 3 and abs(y1 - y0) > 3 and self._drag_cb:
                rx0, rx1 = int(min(x0, x1)), int(max(x0, x1))
                ry0, ry1 = int(min(y0, y1)), int(max(y0, y1))
                # Draw a persistent rectangle for this selection
                from matplotlib.patches import Rectangle
                persist = Rectangle(
                    (rx0, ry0), rx1 - rx0, ry1 - ry0,
                    linewidth=2, edgecolor="#f0a030", facecolor="#f0a030",
                    alpha=0.2, linestyle="-")
                self.ax.add_patch(persist)
                self._drag_cb(rx0, ry0, rx1, ry1, persist)

        self._dragging = False
        self._drag_start = None
        self.draw_idle()

    def _on_motion(self, event):
        if self._dragging and self._drag_start and event.inaxes == self.ax:
            from matplotlib.patches import Rectangle
            if self._drag_rect:
                try:
                    self._drag_rect.remove()
                except Exception:
                    pass
            x0, y0 = self._drag_start
            x1, y1 = event.xdata, event.ydata
            self._drag_rect = Rectangle(
                (min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
                linewidth=2, edgecolor="lime", facecolor="lime",
                alpha=0.15, linestyle=":")
            self.ax.add_patch(self._drag_rect)
            self.draw_idle()
            return

        if self._hover_cb and event.inaxes == self.ax and event.xdata is not None:
            col = int(event.xdata + 0.5)
            row = int(event.ydata + 0.5)
            intensity = None
            if (self._display_data is not None and
                    0 <= row < self._display_data.shape[0] and
                    0 <= col < self._display_data.shape[1]):
                intensity = float(self._display_data[row, col])
            self._hover_cb(col, row, intensity)
        elif self._hover_cb:
            self._hover_cb(None, None, None)


# ── Background expansion worker ───────────────────────────────────

class _ExpansionWorker(QThread):
    finished = pyqtSignal(int, object, object)

    LINK_TOLERANCE = 5

    def __init__(self, job_id, viewer, bin_key, seed_peak):
        super().__init__()
        self._job_id = job_id
        self._viewer = viewer
        self._bin_key = bin_key
        self._seed_peak = seed_peak

    def run(self):
        from ..core.processing import detect_peaks_with_intensity, _best_per_bin

        v = self._viewer
        seed = self._seed_peak
        parts = self._bin_key.split("_")
        center_row, center_col = int(parts[0]), int(parts[1])
        center_bk = self._bin_key
        target_x, target_y = seed["x"], seed["y"]

        h5 = h5py.File(str(H5_PATH), "r")
        try:
            visited = {center_bk}
            queue = [center_bk]
            members = [(center_bk, 0, center_row, center_col,
                        seed["x"], seed["y"], seed)]
            max_radius = 10

            while queue:
                bk = queue.pop(0)
                br, bc = int(bk.split("_")[0]), int(bk.split("_")[1])
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = br + dr, bc + dc
                        nbk = f"{nr}_{nc}"
                        if nbk in visited or nbk not in h5:
                            continue
                        dist = max(abs(nr - center_row), abs(nc - center_col))
                        if dist > max_radius:
                            continue
                        visited.add(nbk)

                        image = np.clip(h5[nbk][:].astype(np.float64), 0, 1e9)
                        peaks, cleaned = detect_peaks_with_intensity(
                            image, v._tth_map, v._ref_degs,
                            v._ref_labels, v._tth_data, v._det)

                        for p in peaks:
                            r = 3
                            py0 = max(0, p['y'] - r)
                            py1 = min(cleaned.shape[0], p['y'] + r + 1)
                            px0 = max(0, p['x'] - r)
                            px1 = min(cleaned.shape[1], p['x'] + r + 1)
                            p['cleaned_intensity'] = float(
                                np.max(cleaned[py0:py1, px0:px1]))

                        match = None
                        for p in peaks:
                            d = ((p["x"] - target_x)**2 +
                                 (p["y"] - target_y)**2) ** 0.5
                            if d <= self.LINK_TOLERANCE and p["label"] == seed["label"]:
                                if match is None or p["snr"] > match["snr"]:
                                    match = p
                        if match:
                            members.append((nbk, 0, nr, nc,
                                            match["x"], match["y"], match))
                            queue.append(nbk)
        finally:
            h5.close()

        feat = v._build_explore_feature(members, seed)
        feat["_members"] = members
        self.finished.emit(self._job_id, feat, members)


# ── Main Window ────────────────────────────────────────────────────

class FeatureViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatial Feature Viewer — 3x3 Bins")
        self.setGeometry(50, 30, 1700, 950)

        self._cmap_name = "inferno"
        self._log_scale = False
        self._selected_bin = None
        self._highlight_bin = None
        self._expand_boundary = True
        self._fill_surface = False
        self._iso_bar_info = None
        self._h5f = None
        self._image_cache = OrderedDict()
        self._raw_image_cache = OrderedDict()
        self._cache_max = 50
        self._noise_cache = {}
        self._noise_enabled = False
        self._noise_algo = "gaussian"
        self._noise_strength = 1.0
        self._noise_shift = 0.0
        self._vmin_pct = 2.0
        self._vmax_pct = 98.5
        self._display_metric = "integrated"
        self._detector_other_features = []

        self._explore_mode = False
        self._explore_peaks = []
        self._explore_feature = None
        self._explore_bin = None
        self._pending_features = []
        self._explore_workers = []
        self._explore_rects = []
        self._next_job_id = 0
        self._region_shown = False

        self._load_data()
        self._build_ui()
        self._connect_signals()

        self._on_category_changed()

    # ── Data ───────────────────────────────────────────────────────

    def _load_data(self):
        kept, filtered = load_features()
        self._features = {
            "kept": group_by_reflection(kept),
            "filtered": group_by_reflection(filtered),
        }
        self._kept_total = len(kept)
        self._filtered_total = len(filtered)

        self._bin_to_features = defaultdict(list)
        for feat in kept:
            for bk in feat.get("spatial_extent", []):
                self._bin_to_features[bk].append(feat)

        self._bin_to_all_features = defaultdict(list)
        for feat in kept + filtered:
            for bk in feat.get("spatial_extent", []):
                self._bin_to_all_features[bk].append(feat)

        with open(_DM.grid_mapping(bin_size=_BIN_SIZE)) as f:
            gm = json.load(f)
        self._n_rows = gm["n_bin_rows"]
        self._n_cols = gm["n_bin_cols"]

        import tifffile
        self._tth_map = tifffile.imread(str(_DM.tth_map())).astype(np.float64)

        det = load_module(DETECTOR_PATH)
        self._det = det
        self._tth_data = det.precompute_tth(self._tth_map)
        self._radial_median_subtract = det.radial_median_subtract

        ref_mod = load_module(_DM.reflections())
        self._ref_degs = ref_mod.degs
        self._ref_labels = ref_mod.deg_labels
        self._show_tth_overlay = False

        from ..core.processing import estimate_beam_center
        self._beam_center = estimate_beam_center(self._tth_map)

        from ..core.algorithms import compute_tth_binning
        self._tth_edges, self._tth_centers, self._n_tth_bins, \
            self._tth_bin_indices, self._tth_radial_counts = compute_tth_binning(self._tth_map)

    def _get_h5(self):
        if self._h5f is None:
            self._h5f = h5py.File(str(H5_PATH), "r")
        return self._h5f

    def _load_raw_image(self, bin_key):
        if bin_key in self._raw_image_cache:
            self._raw_image_cache.move_to_end(bin_key)
            return self._raw_image_cache[bin_key]
        h5 = self._get_h5()
        if bin_key not in h5:
            return None
        image = np.clip(h5[bin_key][:].astype(np.float64), 0, 1e9)
        self._raw_image_cache[bin_key] = image
        if len(self._raw_image_cache) > self._cache_max:
            self._raw_image_cache.popitem(last=False)
        return image

    def _get_display_image(self, bin_key):
        raw = self._load_raw_image(bin_key)
        if raw is None:
            return None

        if self._noise_enabled:
            algo = self._noise_algo
            cache_key = (bin_key, algo)
            if cache_key not in self._noise_cache:
                valid = self._tth_radial_counts > 50
                profile = compute_radial_profile(raw, self._tth_bin_indices, self._n_tth_bins)
                fits = fit_all_models(self._tth_centers, profile, valid,
                                      self._tth_edges[0], self._tth_edges[-1])
                self._noise_cache[cache_key] = fits
            fits = self._noise_cache[cache_key]
            if algo in fits:
                bg = build_background_image(
                    self._tth_map, self._tth_centers, fits[algo]["profile"],
                    self._tth_bin_indices)
                cleaned = subtract_background(raw, bg,
                                              strength=self._noise_strength,
                                              shift=self._noise_shift)
                return np.clip(cleaned, 0, None)

        return raw

    # ── UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # Left: checkbox + heatmap (top) + isometric 3D (bottom)
        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(2)

        cb_bar = QHBoxLayout()
        cb_bar.setContentsMargins(0, 0, 0, 0)
        cb_bar.setSpacing(12)
        self.peak_mode_cb = QCheckBox("Peak intensity")
        self.peak_mode_cb.setToolTip("Unchecked = integrated (area under peak)\nChecked = peak pixel intensity")
        cb_bar.addWidget(self.peak_mode_cb)
        self.expand_cb = QCheckBox("Expand boundary")
        self.expand_cb.setToolTip("Show interpolated border around data bins")
        self.expand_cb.setChecked(True)
        self.expand_cb.stateChanged.connect(self._on_expand_changed)
        cb_bar.addWidget(self.expand_cb)
        self.fill_cb = QCheckBox("Fill surface")
        self.fill_cb.setToolTip("Interpolate between bins and render\nas a continuous surface")
        self.fill_cb.stateChanged.connect(self._on_fill_changed)
        cb_bar.addWidget(self.fill_cb)
        self.explore_cb = QCheckBox("Explore new points")
        self.explore_cb.setToolTip("Drag-select peaks on detector image\nthat algorithms missed")
        self.explore_cb.setStyleSheet("QCheckBox { color: #f0a030; }")
        self.explore_cb.stateChanged.connect(self._on_explore_toggled)
        cb_bar.addWidget(self.explore_cb)
        cb_bar.addStretch()
        left_vbox.addLayout(cb_bar)

        left_splitter = QSplitter(Qt.Vertical)
        self.heatmap_canvas = HeatmapCanvas()
        self.heatmap_canvas.set_click_callback(self._on_heatmap_click)
        self.heatmap_canvas.set_hover_callback(self._on_heatmap_hover)
        left_splitter.addWidget(self.heatmap_canvas)

        self.iso_canvas = IsometricCanvas()
        left_splitter.addWidget(self.iso_canvas)
        left_splitter.setSizes([400, 300])
        left_vbox.addWidget(left_splitter)
        splitter.addWidget(left_container)

        # Center: detector image
        self.detector_canvas = DetectorCanvas()
        self.detector_canvas.set_hover_callback(self._on_detector_hover)
        self.detector_canvas.set_click_callback(self._on_detector_click)
        self.detector_canvas.set_drag_callback(self._on_explore_drag)
        splitter.addWidget(self.detector_canvas)

        # Right: controls
        right_scroll = QWidget()
        right_scroll.setMaximumWidth(400)
        right_scroll.setMinimumWidth(330)
        right_layout = QVBoxLayout(right_scroll)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self._build_navigation(right_layout)
        self._build_info_panel(right_layout)
        self._build_pending_panel(right_layout)
        self._build_visualization_controls(right_layout)
        self._build_noise_reduction(right_layout)

        # Device location mini-map (clickable in explore mode)
        self.loc_canvas = FigureCanvasQTAgg(Figure(figsize=(2, 1.5)))
        self.loc_canvas.figure.patch.set_facecolor("#1a1a1a")
        self.loc_ax = self.loc_canvas.figure.add_subplot(111)
        self.loc_canvas.setFixedHeight(120)
        self.loc_canvas.figure.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)
        self.loc_canvas.mpl_connect("button_press_event", self._on_minimap_click)
        right_layout.addWidget(self.loc_canvas)

        # Status bar for hover info
        self.hover_label = QLabel("")
        self.hover_label.setWordWrap(True)
        self.hover_label.setMinimumHeight(36)
        self.hover_label.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #aaa; padding: 2px;")
        right_layout.addWidget(self.hover_label)

        right_layout.addStretch()
        splitter.addWidget(right_scroll)

        splitter.setSizes([450, 800, 400])
        layout.addWidget(splitter)

    def _build_navigation(self, parent_layout):
        grp = QGroupBox("Feature Selection")
        lay = QVBoxLayout(grp)

        h = QHBoxLayout()
        h.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        h.addWidget(self.category_combo)
        lay.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Reflection:"))
        self.reflection_combo = QComboBox()
        h.addWidget(self.reflection_combo)
        lay.addLayout(h)

        h = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.setFixedWidth(70)
        h.addWidget(self.prev_btn)
        self.feat_spin = QSpinBox()
        self.feat_spin.setMinimum(0)
        h.addWidget(self.feat_spin)
        self.next_btn = QPushButton("Next >")
        self.next_btn.setFixedWidth(70)
        h.addWidget(self.next_btn)
        lay.addLayout(h)

        h = QHBoxLayout()
        h.addWidget(QLabel("Go to #:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("feature ID")
        self.search_edit.returnPressed.connect(self._on_search_feature)
        h.addWidget(self.search_edit)
        self.search_btn = QPushButton("Go")
        self.search_btn.setFixedWidth(40)
        self.search_btn.clicked.connect(self._on_search_feature)
        h.addWidget(self.search_btn)
        lay.addLayout(h)

        parent_layout.addWidget(grp)

    def _build_info_panel(self, parent_layout):
        grp = QGroupBox("Feature Info")
        lay = QVBoxLayout(grp)
        self.info_label = QLabel("No feature selected")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(self.info_label)
        parent_layout.addWidget(grp)

    def _build_pending_panel(self, parent_layout):
        grp = QGroupBox("Pending Features")
        grp.setStyleSheet(
            "QGroupBox { color: #f0a030; border: 1px solid #555; "
            "border-radius: 4px; margin-top: 6px; padding-top: 10px; }"
            "QGroupBox::title { subcontrol-position: top left; padding: 2px 6px; }")
        lay = QVBoxLayout(grp)

        self.pending_list = QListWidget()
        self.pending_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pending_list.setStyleSheet(
            "QListWidget { background: #1a1a1a; color: #ccc; font-family: monospace; "
            "font-size: 10px; border: 1px solid #444; }"
            "QListWidget::item:selected { background: #333; }")
        self.pending_list.setMaximumHeight(150)
        self.pending_list.itemSelectionChanged.connect(self._on_pending_selected)
        lay.addWidget(self.pending_list)

        self.score_label = QLabel("")
        self.score_label.setWordWrap(True)
        self.score_label.setStyleSheet(
            "font-family: monospace; font-size: 10px; color: #aaa; padding: 2px;")
        lay.addWidget(self.score_label)

        self.region_btn = QPushButton("Show region peaks")
        self.region_btn.setToolTip("Sum ~10 nearby bins and overlay all\n"
                                   "detected peaks (kept + filtered)")
        self.region_btn.setStyleSheet(
            "QPushButton { background: #2a4a5a; color: white; font-weight: bold; "
            "padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background: #3a6a7a; }")
        self.region_btn.clicked.connect(self._on_show_region)
        lay.addWidget(self.region_btn)

        btn_row = QHBoxLayout()
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setStyleSheet(
            "QPushButton { background: #2a5a2a; color: white; font-weight: bold; "
            "padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background: #3a7a3a; }"
            "QPushButton:disabled { background: #333; color: #666; }")
        self.accept_btn.clicked.connect(self._on_accept_pending)
        self.accept_btn.setEnabled(False)
        btn_row.addWidget(self.accept_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setStyleSheet(
            "QPushButton { background: #5a2a2a; color: white; font-weight: bold; "
            "padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background: #7a3a3a; }"
            "QPushButton:disabled { background: #333; color: #666; }")
        self.remove_btn.clicked.connect(self._on_remove_pending)
        self.remove_btn.setEnabled(False)
        btn_row.addWidget(self.remove_btn)
        lay.addLayout(btn_row)

        parent_layout.addWidget(grp)
        self._pending_group = grp
        grp.setVisible(False)

    def _build_visualization_controls(self, parent_layout):
        grp = QGroupBox("Visualization")
        lay = QGridLayout(grp)

        lay.addWidget(QLabel("Colormap:"), 0, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText(self._cmap_name)
        lay.addWidget(self.cmap_combo, 0, 1, 1, 2)

        self.reverse_cb = QCheckBox("Reverse")
        lay.addWidget(self.reverse_cb, 0, 3)

        self.log_cb = QCheckBox("Log scale")
        lay.addWidget(self.log_cb, 1, 0, 1, 2)

        self.tth_cb = QCheckBox("2θ overlay")
        self.tth_cb.setToolTip("Show 2-theta reflection rings on detector image")
        self.tth_cb.stateChanged.connect(self._on_tth_overlay_changed)
        lay.addWidget(self.tth_cb, 1, 2, 1, 2)

        lay.addWidget(QLabel("Min %:"), 2, 0)
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 1000)
        self.vmin_slider.setValue(20)
        lay.addWidget(self.vmin_slider, 2, 1, 1, 2)
        self.vmin_val = QLineEdit("2.0")
        self.vmin_val.setFixedWidth(45)
        lay.addWidget(self.vmin_val, 2, 3)

        lay.addWidget(QLabel("Max %:"), 3, 0)
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setRange(0, 1000)
        self.vmax_slider.setValue(985)
        lay.addWidget(self.vmax_slider, 3, 1, 1, 2)
        self.vmax_val = QLineEdit("98.5")
        self.vmax_val.setFixedWidth(45)
        lay.addWidget(self.vmax_val, 3, 3)

        preset_layout = QHBoxLayout()
        for name, lo, hi in [("Full", 0, 1000), ("Auto", 20, 985),
                              ("Tight", 50, 995), ("High", 100, 999)]:
            btn = QPushButton(name)
            btn.setFixedWidth(55)
            btn.clicked.connect(lambda _, l=lo, h=hi: self._set_contrast_preset(l, h))
            preset_layout.addWidget(btn)
        lay.addLayout(preset_layout, 4, 0, 1, 4)

        parent_layout.addWidget(grp)

    def _build_noise_reduction(self, parent_layout):
        grp = QGroupBox("Noise Reduction")
        lay = QGridLayout(grp)

        self.noise_cb = QCheckBox("Enable noise reduction")
        lay.addWidget(self.noise_cb, 0, 0, 1, 3)

        self.noise_algo_label = QLabel("Algorithm:")
        lay.addWidget(self.noise_algo_label, 1, 0)
        self.noise_algo_combo = QComboBox()
        for key in ALGORITHM_NAMES:
            self.noise_algo_combo.addItem(ALGORITHM_DISPLAY[key], key)
        lay.addWidget(self.noise_algo_combo, 1, 1, 1, 2)

        self.strength_label = QLabel("Strength:")
        lay.addWidget(self.strength_label, 2, 0)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        lay.addWidget(self.strength_slider, 2, 1)
        self.strength_val = QLineEdit("1.00")
        self.strength_val.setFixedWidth(45)
        lay.addWidget(self.strength_val, 2, 2)

        self.shift_label = QLabel("Shift:")
        lay.addWidget(self.shift_label, 3, 0)
        self.shift_slider = QSlider(Qt.Horizontal)
        self.shift_slider.setRange(-500, 500)
        self.shift_slider.setValue(0)
        lay.addWidget(self.shift_slider, 3, 1)
        self.shift_val = QLineEdit("0.0")
        self.shift_val.setFixedWidth(45)
        lay.addWidget(self.shift_val, 3, 2)

        self._noise_widgets = [
            self.noise_algo_label, self.noise_algo_combo,
            self.strength_label, self.strength_slider, self.strength_val,
            self.shift_label, self.shift_slider, self.shift_val,
        ]
        for w in self._noise_widgets:
            w.setVisible(False)

        parent_layout.addWidget(grp)

    # ── Signals ────────────────────────────────────────────────────

    def _connect_signals(self):
        self.category_combo.addItems([
            f"Kept Features ({self._kept_total})",
            f"Filtered Features ({self._filtered_total})",
        ])
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        self.reflection_combo.currentIndexChanged.connect(self._on_reflection_changed)
        self.feat_spin.valueChanged.connect(self._on_feature_changed)
        self.prev_btn.clicked.connect(lambda: self.feat_spin.setValue(self.feat_spin.value() - 1))
        self.next_btn.clicked.connect(lambda: self.feat_spin.setValue(self.feat_spin.value() + 1))

        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.reverse_cb.toggled.connect(self._on_cmap_changed)
        self.log_cb.toggled.connect(self._on_log_changed)
        self.peak_mode_cb.toggled.connect(self._on_metric_changed)
        self.vmin_slider.valueChanged.connect(self._on_contrast_changed)
        self.vmax_slider.valueChanged.connect(self._on_contrast_changed)
        self.vmin_val.editingFinished.connect(self._on_vmin_text)
        self.vmax_val.editingFinished.connect(self._on_vmax_text)

        self.noise_cb.toggled.connect(self._on_noise_toggle)
        self.noise_algo_combo.currentIndexChanged.connect(self._on_noise_algo_changed)
        self.strength_slider.valueChanged.connect(self._on_noise_param_changed)
        self.shift_slider.valueChanged.connect(self._on_noise_param_changed)
        self.strength_val.editingFinished.connect(self._on_strength_text)
        self.shift_val.editingFinished.connect(self._on_shift_text)

    # ── Navigation callbacks ───────────────────────────────────────

    def _current_category_key(self):
        return "kept" if self.category_combo.currentIndex() == 0 else "filtered"

    def _current_reflection_groups(self):
        return self._features.get(self._current_category_key(), {})

    def _current_features_list(self):
        groups = self._current_reflection_groups()
        ref = self.reflection_combo.currentData()
        if ref and ref in groups:
            return groups[ref]
        return []

    def _current_feature(self):
        feats = self._current_features_list()
        idx = self.feat_spin.value()
        if 0 <= idx < len(feats):
            return feats[idx]
        return None

    def _on_category_changed(self):
        self.reflection_combo.blockSignals(True)
        self.reflection_combo.clear()
        groups = self._current_reflection_groups()
        for ref in sorted(groups.keys()):
            count = len(groups[ref])
            self.reflection_combo.addItem(f"{ref} ({count})", ref)
        self.reflection_combo.blockSignals(False)
        self._on_reflection_changed()

    def _on_reflection_changed(self):
        feats = self._current_features_list()
        self.feat_spin.blockSignals(True)
        self.feat_spin.setMaximum(max(0, len(feats) - 1))
        self.feat_spin.setValue(0)
        self.feat_spin.blockSignals(False)
        self._on_feature_changed()

    def _on_search_feature(self):
        text = self.search_edit.text().strip().lstrip("#")
        if not text.isdigit():
            return
        target_id = int(text)
        for cat_idx, cat_key in enumerate(("kept", "filtered")):
            groups = self._features.get(cat_key, {})
            for ref in sorted(groups.keys()):
                for feat_idx, feat in enumerate(groups[ref]):
                    if feat.get("feature_id") == target_id:
                        self.category_combo.blockSignals(True)
                        self.category_combo.setCurrentIndex(cat_idx)
                        self.category_combo.blockSignals(False)
                        self._on_category_changed.__wrapped__(self) if hasattr(self._on_category_changed, '__wrapped__') else None
                        self.reflection_combo.blockSignals(True)
                        self.reflection_combo.clear()
                        for r in sorted(groups.keys()):
                            self.reflection_combo.addItem(
                                f"{r} ({len(groups[r])})", r)
                        ref_idx = sorted(groups.keys()).index(ref)
                        self.reflection_combo.setCurrentIndex(ref_idx)
                        self.reflection_combo.blockSignals(False)
                        feats = groups[ref]
                        self.feat_spin.blockSignals(True)
                        self.feat_spin.setMaximum(max(0, len(feats) - 1))
                        self.feat_spin.setValue(feat_idx)
                        self.feat_spin.blockSignals(False)
                        self._on_feature_changed()
                        self.search_edit.clear()
                        return
        self.search_edit.setStyleSheet("QLineEdit { background: #ffe0e0; }")
        QTimer.singleShot(800, lambda: self.search_edit.setStyleSheet(""))

    def _on_feature_changed(self):
        feat = self._current_feature()
        if feat is None:
            self.info_label.setText("No feature selected")
            return
        self._highlight_bin = None
        self._update_info(feat)
        self._render_heatmap(feat)
        self._render_isometric(feat)
        self._render_location(feat)
        self._selected_bin = feat.get("center_bin")
        if self._explore_mode:
            self._explore_bin = feat.get("center_bin")
            self._explore_rects = []
        self._region_shown = False
        self.region_btn.setText("Show region peaks")
        self._load_detector_image(feat["center_bin"], feat)

    # ── Device location mini-map ──────────────────────────────────────

    def _render_location(self, feat):
        ax = self.loc_ax
        ax.clear()
        ax.set_facecolor("#222")
        nr, nc = self._n_rows, self._n_cols
        ax.add_patch(plt.Rectangle((0, 0), nc, nr, linewidth=1.5,
                                    edgecolor="gray", facecolor="#333"))
        cr = feat.get("center_row", 0)
        cc = feat.get("center_col", 0)
        ax.plot(cc, cr, "o", color="red", markersize=7,
                markeredgecolor="white", markeredgewidth=1)
        ax.set_xlim(-2, nc + 2)
        ax.set_ylim(nr + 2, -2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Device ({cr}, {cc})", color="white", fontsize=8, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.loc_canvas.draw_idle()

    # ── Profile value helper ─────────────────────────────────────────

    def _get_profile_value(self, entry):
        if isinstance(entry, dict):
            if self._display_metric == "integrated":
                return entry.get("integrated", entry.get("intensity", 0))
            return entry.get("intensity", 0)
        return float(entry)

    def _on_metric_changed(self, *_):
        self._display_metric = "intensity" if self.peak_mode_cb.isChecked() else "integrated"
        self._refresh_display()

    # ── Info panel ─────────────────────────────────────────────────

    def _update_info(self, feat):
        lines = [
            f"Feature ID:     {feat.get('feature_id', '?')}",
            f"Reflection:     {feat['reflection']}",
            f"Bins:           {feat['n_bins']}",
            f"Peak intensity: {feat['peak_intensity']:.1f}",
            f"Mean SNR:       {feat['mean_snr']:.1f}",
            f"Center bin:     {feat.get('center_bin', '?')}",
            f"Detector pos:   ({feat['detector_x']}, {feat['detector_y']})",
            f"",
            f"Reason:",
            f"  {feat.get('reason', 'N/A')}",
        ]
        self.info_label.setText("\n".join(lines))

    # ── Hover ──────────────────────────────────────────────────────

    def _on_heatmap_hover(self, row, col, intensity):
        if row is None:
            self.hover_label.setText("")
            return
        if intensity is not None:
            self.hover_label.setText(
                f"Heatmap  bin ({row}, {col})  intensity: {intensity:.1f}")
        else:
            self.hover_label.setText(f"Heatmap  bin ({row}, {col})  [empty]")

    def _on_detector_click(self, x, y):
        if self._explore_mode:
            self._on_explore_detector_click(x, y)

    def _on_detector_hover(self, x, y, intensity):
        if x is None:
            self.hover_label.setText("")
            return
        if intensity is not None:
            base = f"Detector  ({x}, {y})  intensity: {intensity:.1f}"
        else:
            base = f"Detector  ({x}, {y})"

        near = self._find_nearby_feature(x, y, radius=25)
        if near:
            fid = near.get("feature_id", "?")
            ref = near.get("reflection", "")
            snr = near.get("mean_snr", 0)
            nb = near.get("n_bins", 0)
            pi = near.get("peak_intensity", 0)
            base += (f"\n  ▸ Feature #{fid}  {ref}  "
                     f"peak={pi:.0f}  SNR={snr:.1f}  bins={nb}")

        self.hover_label.setText(base)

    def _find_nearby_feature(self, x, y, radius=25):
        best, best_d = None, radius
        for other in self._detector_other_features:
            dx = other["detector_x"] - x
            dy = other["detector_y"] - y
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best, best_d = other, d
        return best

    # ── Shared bounds ──────────────────────────────────────────────

    def _feature_bounds(self, feat):
        """Padded bounding box for a feature — used by both heatmap and 3D.

        Always guarantees at least 1 bin of empty border on every side,
        even for features touching the grid edge (extends past grid bounds).
        """
        extent = feat.get("spatial_extent", [])
        rows, cols = [], []
        for bk in extent:
            parts = bk.split("_")
            if len(parts) == 2:
                rows.append(int(parts[0]))
                cols.append(int(parts[1]))
        if not rows:
            return None
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        pad = 3
        r_lo = r_min - pad
        r_hi = r_max + pad
        c_lo = c_min - pad
        c_hi = c_max + pad
        return r_lo, r_hi, c_lo, c_hi

    # ── Shared Z grid for heatmap + 3D ──────────────────────────────

    def _build_feature_grid(self, feat):
        """Build the intensity grid for a feature within its padded bounds.

        Returns (Z_display, z_max, bounds) or (None, None, None).
        Z_display includes expansion only when self._expand_boundary is True,
        but z_max is always based on the expanded grid so both views share
        a consistent color scale regardless of the checkbox.
        """
        profile = feat.get("intensity_profile", {})
        if not profile:
            return None, None, None
        bounds = self._feature_bounds(feat)
        if not bounds:
            return None, None, None

        r_lo, r_hi, c_lo, c_hi = bounds
        nr = r_hi - r_lo + 1
        nc = c_hi - c_lo + 1

        Z_raw = np.zeros((nr, nc))
        for bk, entry in profile.items():
            parts = bk.split("_")
            if len(parts) == 2:
                r, c = int(parts[0]), int(parts[1])
                ri = r - r_lo
                ci = c - c_lo
                if 0 <= ri < nr and 0 <= ci < nc:
                    Z_raw[ri, ci] = self._get_profile_value(entry)

        expanded = np.zeros_like(Z_raw)
        for ri in range(nr):
            for ci in range(nc):
                if Z_raw[ri, ci] > 0:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nri, nci = ri + dr, ci + dc
                            if 0 <= nri < nr and 0 <= nci < nc and Z_raw[nri, nci] == 0:
                                neighbors = []
                                for dr2 in [-1, 0, 1]:
                                    for dc2 in [-1, 0, 1]:
                                        nnr, nnc = nri + dr2, nci + dc2
                                        if 0 <= nnr < nr and 0 <= nnc < nc and Z_raw[nnr, nnc] > 0:
                                            neighbors.append(Z_raw[nnr, nnc])
                                if neighbors:
                                    expanded[nri, nci] = max(expanded[nri, nci],
                                                              np.mean(neighbors) * 0.3)
        Z_expanded = Z_raw + expanded
        z_max = float(Z_expanded.max()) if Z_expanded.max() > 0 else 1.0

        Z_display = Z_expanded if self._expand_boundary else Z_raw
        return Z_display, z_max, bounds

    # ── Heatmap ────────────────────────────────────────────────────

    def _render_heatmap(self, feat):
        self.heatmap_canvas.fig.clear()
        ax = self.heatmap_canvas.fig.add_subplot(111)
        self.heatmap_canvas.ax = ax
        ax.set_facecolor("black")
        self.heatmap_canvas.fig.patch.set_facecolor("#1a1a1a")
        self.heatmap_canvas.fig.subplots_adjust(left=0.12, right=0.88, top=0.93, bottom=0.07)

        Z, z_max, bounds = self._build_feature_grid(feat)
        if Z is None:
            self.heatmap_canvas.draw_idle()
            return
        r_lo, r_hi, c_lo, c_hi = bounds

        grid = np.where(Z > 0, Z, np.nan)
        self.heatmap_canvas._grid_data = grid
        self.heatmap_canvas._grid_r_lo = r_lo
        self.heatmap_canvas._grid_c_lo = c_lo

        cmap = plt.get_cmap(self._cmap_name).copy()
        cmap.set_bad(color="black")

        im = ax.imshow(grid, origin="upper", cmap=cmap, interpolation="nearest",
                       aspect="equal", vmin=0, vmax=z_max,
                       extent=[c_lo - 0.5, c_hi + 0.5, r_hi + 0.5, r_lo - 0.5])

        cb = self.heatmap_canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7)
        cb_label = "Integrated" if self._display_metric == "integrated" else "Intensity"
        cb.set_label(cb_label, color="white", fontsize=8)

        # Mark center bin
        center = feat.get("center_bin", "")
        parts = center.split("_")
        if len(parts) == 2:
            cr, cc = int(parts[0]), int(parts[1])
            ax.plot(cc, cr, "s", markersize=10, markerfacecolor="none",
                    markeredgecolor="white", markeredgewidth=2)

        # Mark highlighted (selected) bin
        if self._highlight_bin is not None:
            hp = self._highlight_bin.split("_")
            if len(hp) == 2:
                hr, hc = int(hp[0]), int(hp[1])
                ax.plot(hc, hr, "s", markersize=10, markerfacecolor="none",
                        markeredgecolor="cyan", markeredgewidth=2)

        cat = "Kept" if self._current_category_key() == "kept" else "Filtered"
        mode_str = "peak pixel" if self._display_metric == "intensity" else "area under curve"
        ref = feat["reflection"]
        ax.set_title(f"Spatial heatmap — intensity = {mode_str}\n"
                     f"{ref} #{feat.get('feature_id', '?')}, {cat}, {feat['n_bins']} bins",
                     color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)

        self.heatmap_canvas.draw_idle()

    # ── Isometric 3D ─────────────────────────────────────────────────

    def _render_isometric(self, feat):
        ax = self.iso_canvas.ax
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        self.iso_canvas.fig.patch.set_facecolor("#1a1a1a")

        Z, z_max, bounds = self._build_feature_grid(feat)
        if Z is None:
            ax.set_title("No profile data", color="white", fontsize=9)
            self.iso_canvas.draw_idle()
            return

        r_lo, r_hi, c_lo, c_hi = bounds
        nr = r_hi - r_lo + 1
        nc = c_hi - c_lo + 1

        cmap = plt.get_cmap(self._cmap_name)

        # Determine which bin (if any) is highlighted
        sel_ri, sel_ci = None, None
        has_selection = False
        if self._highlight_bin is not None:
            parts = self._highlight_bin.split("_")
            if len(parts) == 2:
                sr, sc = int(parts[0]), int(parts[1])
                sel_ri = sr - r_lo
                sel_ci = sc - c_lo
                if 0 <= sel_ri < nr and 0 <= sel_ci < nc:
                    has_selection = True

        if self._fill_surface:
            self._iso_bar_info = None
            import scipy.ndimage as _ndi
            mask = Z > 0
            if mask.sum() >= 2:
                Z_surf = Z.copy()
                indices = _ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
                Z_filled = Z[tuple(indices)]
                blend = np.exp(-_ndi.distance_transform_edt(~mask) * 0.8)
                Z_surf = Z_filled * blend
                Z_surf[mask] = Z[mask]
            else:
                Z_surf = Z.copy()
            Z_surf = np.clip(Z_surf, 0, None)
            C_grid, R_grid = np.meshgrid(np.arange(nc), np.arange(nr))
            X_plot = C_grid + c_lo
            Y_plot = R_grid + r_lo
            facecolors = cmap(Z_surf / z_max)
            if has_selection and 0 <= sel_ri < nr and 0 <= sel_ci < nc:
                for ri in range(nr):
                    for ci in range(nc):
                        if ri != sel_ri or ci != sel_ci:
                            facecolors[ri, ci, 3] = 0.25
            ax.plot_surface(X_plot, Y_plot, Z_surf, facecolors=facecolors,
                            edgecolor="gray", linewidth=0.2, shade=True)
        else:
            bar_x, bar_y, bar_z, bar_dx, bar_dy, bar_dz, bar_colors = [], [], [], [], [], [], []
            for ri in range(nr):
                for ci in range(nc):
                    val = Z[ri, ci]
                    if val > 0:
                        bar_x.append(c_lo + ci - 0.5)
                        bar_y.append(r_lo + ri - 0.5)
                        bar_z.append(0)
                        bar_dx.append(1)
                        bar_dy.append(1)
                        bar_dz.append(val)
                        rgba = list(cmap(val / z_max))
                        if has_selection:
                            if ri == sel_ri and ci == sel_ci:
                                rgba[3] = 1.0
                            else:
                                rgba[3] = 0.25
                        else:
                            rgba[3] = 0.85
                        bar_colors.append(rgba)

            if bar_x:
                ax.bar3d(bar_x, bar_y, bar_z, bar_dx, bar_dy, bar_dz,
                         color=bar_colors, edgecolor="gray", linewidth=0.3,
                         shade=True)

            self._iso_bar_info = {
                'cells': [(ri, ci, Z[ri, ci]) for ri in range(nr) for ci in range(nc) if Z[ri, ci] > 0],
                'z_max': z_max,
            }

        ax.set_xlim(c_lo - 0.5, c_hi + 0.5)
        ax.set_ylim(r_lo - 0.5, r_hi + 0.5)
        ax.set_zlim(0, z_max * 1.15)
        ax.set_xlabel("Col", color="white", fontsize=7, labelpad=1)
        ax.set_ylabel("Row", color="white", fontsize=7, labelpad=1)
        z_label = "Integrated" if self._display_metric == "integrated" else "Intensity"
        ax.set_zlabel(z_label, color="white", fontsize=7, labelpad=1)
        ax.tick_params(colors="white", labelsize=6, pad=0)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333")
        ax.yaxis.pane.set_edgecolor("#333")
        ax.zaxis.pane.set_edgecolor("#333")

        # Rotate view from the closest corner to the selected bin
        if has_selection:
            c_mid = (c_lo + c_hi) / 2.0
            r_mid = (r_lo + r_hi) / 2.0
            sel_c = c_lo + sel_ci
            sel_r = r_lo + sel_ri
            angle = np.degrees(np.arctan2(sel_r - r_mid, sel_c - c_mid))
            corners = [-135, -45, 45, 135]
            azim = min(corners, key=lambda c: abs((angle - c + 180) % 360 - 180))
            ax.view_init(elev=35, azim=azim)
        else:
            ax.view_init(elev=35, azim=-60)

        mode_str = "peak pixel" if self._display_metric == "intensity" else "area under curve"
        ax.set_title(f"3D profile — intensity = {mode_str}, {feat['n_bins']} bins",
                     color="white", fontsize=9, pad=2)

        self.iso_canvas.draw_idle()

    # ── Detector image ─────────────────────────────────────────────

    def _load_detector_image(self, bin_key, feat=None):
        if feat is None:
            feat = self._current_feature()
        if feat is None:
            return

        cleaned = self._get_display_image(bin_key)
        if cleaned is None:
            self.detector_canvas.ax.clear()
            self.detector_canvas.ax.set_title(f"Bin {bin_key} not found", color="white")
            self.detector_canvas.draw_idle()
            return

        ax = self.detector_canvas.ax
        ax.clear()
        ax.set_facecolor("black")

        display = cleaned.copy()
        if self._log_scale:
            display = np.log1p(np.clip(display, 0, None))

        self.detector_canvas._display_data = display

        finite = display[np.isfinite(display)]
        if len(finite) == 0:
            vmin, vmax = 0, 1
        else:
            vmin = float(np.percentile(finite, self._vmin_pct))
            vmax = float(np.percentile(finite, self._vmax_pct))

        cmap_name = self._cmap_name
        if self.reverse_cb.isChecked():
            cmap_name += "_r"
        cmap = plt.get_cmap(cmap_name)
        ax.imshow(display, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", aspect="equal")

        # Circle other features visible in this bin
        self._detector_other_features = []
        for other in self._bin_to_features.get(bin_key, []):
            if other.get("feature_id") == feat.get("feature_id"):
                continue
            ox, oy = other["detector_x"], other["detector_y"]
            self._detector_other_features.append(other)
            c = plt.Circle((ox, oy), 18, fill=False, color="#7fff00",
                            linewidth=1.5, linestyle="--")
            ax.add_patch(c)
            ax.text(ox + 22, oy - 12,
                    f"#{other.get('feature_id','')} {other['reflection']}",
                    color="#7fff00", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black",
                              ec="#7fff00", alpha=0.8, linewidth=0.5))

        # Circle the current feature's peak
        det_x, det_y = feat["detector_x"], feat["detector_y"]
        circle = plt.Circle((det_x, det_y), 15, fill=False, color="lime",
                            linewidth=2, linestyle="--")
        ax.add_patch(circle)

        # 2-theta band overlay (same style as labeling_tool.show_arcs)
        if self._show_tth_overlay:
            tth = self._tth_map
            for idx, (lab, d) in enumerate(zip(self._ref_labels, self._ref_degs)):
                mask = np.abs(tth - d) < 0.3
                overlay = np.where(mask, 1.0, np.nan)
                color = ARC_COLORS[idx % len(ARC_COLORS)]
                cmap_band = mcolors.ListedColormap([color])
                ax.imshow(overlay, cmap=cmap_band, alpha=0.25,
                          interpolation="nearest")
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    mid = len(ys) // 2
                    ax.text(xs[mid], ys[mid], lab, color=color, fontsize=7,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="black", alpha=0.7))

        self._selected_bin = bin_key

        ax.set_title(f"Detector image — bin {bin_key}, peak at ({det_x}, {det_y})",
                     color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)

        self.detector_canvas.draw_idle()

    # ── Heatmap click ──────────────────────────────────────────────

    def _update_heatmap_marker(self):
        """Move the cyan selection marker without redrawing the heatmap."""
        ax = self.heatmap_canvas.ax
        if ax is None:
            return
        while len(ax.lines) > 1:
            ax.lines[-1].remove()
        if self._highlight_bin is not None:
            hp = self._highlight_bin.split("_")
            if len(hp) == 2:
                hr, hc = int(hp[0]), int(hp[1])
                ax.plot(hc, hr, "s", markersize=10, markerfacecolor="none",
                        markeredgecolor="cyan", markeredgewidth=2)
        self.heatmap_canvas.draw_idle()

    def _update_iso_alpha(self):
        """Update bar transparency without rebuilding the 3D geometry."""
        ax = self.iso_canvas.ax
        if not ax.collections:
            return
        feat = self._current_feature()
        if feat is None:
            return
        bounds = self._feature_bounds(feat)
        if not bounds:
            return
        r_lo, r_hi, c_lo, c_hi = bounds
        nr = r_hi - r_lo + 1
        nc = c_hi - c_lo + 1

        sel_ri, sel_ci = None, None
        has_selection = False
        if self._highlight_bin is not None:
            parts = self._highlight_bin.split("_")
            if len(parts) == 2:
                sr, sc = int(parts[0]), int(parts[1])
                sel_ri = sr - r_lo
                sel_ci = sc - c_lo
                if 0 <= sel_ri < nr and 0 <= sel_ci < nc:
                    has_selection = True

        bar_info = getattr(self, '_iso_bar_info', None)
        if bar_info is None:
            self._render_isometric(feat)
            return

        colors = []
        cmap = plt.get_cmap(self._cmap_name)
        z_max = bar_info['z_max']
        for ri, ci, val in bar_info['cells']:
            rgba = list(cmap(val / z_max))
            if has_selection:
                rgba[3] = 1.0 if (ri == sel_ri and ci == sel_ci) else 0.25
            else:
                rgba[3] = 0.85
            colors.extend([rgba] * 6)

        poly = ax.collections[0] if ax.collections else None
        if poly is not None:
            poly.set_facecolors(colors)

        if has_selection:
            c_mid = (c_lo + c_hi) / 2.0
            r_mid = (r_lo + r_hi) / 2.0
            sel_c = c_lo + sel_ci
            sel_r = r_lo + sel_ri
            angle = np.degrees(np.arctan2(sel_r - r_mid, sel_c - c_mid))
            corners = [-135, -45, 45, 135]
            azim = min(corners, key=lambda c: abs((angle - c + 180) % 360 - 180))
            ax.view_init(elev=35, azim=azim)
        else:
            ax.view_init(elev=35, azim=-60)

        self.iso_canvas.draw_idle()

    def _on_heatmap_click(self, row, col):
        feat = self._explore_feature if self._explore_mode else self._current_feature()
        if feat is None:
            return
        if 0 <= row < self._n_rows and 0 <= col < self._n_cols:
            bin_key = f"{row}_{col}"
            # Toggle: clicking same bin deselects (no highlight, all solid)
            if bin_key == self._highlight_bin:
                self._highlight_bin = None
                center = feat.get("center_bin")
                self._selected_bin = center
                self._load_detector_image(center, feat)
                self._update_iso_alpha()
                self._update_heatmap_marker()
                return
            self._highlight_bin = bin_key
            self._selected_bin = bin_key
            self._load_detector_image(bin_key, feat)
            self._update_iso_alpha()
            self._update_heatmap_marker()

    # ── Visualization controls ─────────────────────────────────────

    def _on_cmap_changed(self, *_):
        name = self.cmap_combo.currentText()
        if self.reverse_cb.isChecked():
            name += "_r"
        self._cmap_name = self.cmap_combo.currentText()
        self._refresh_display()

    def _on_expand_changed(self, state):
        self._expand_boundary = bool(state)
        feat = self._current_feature()
        if feat:
            self._render_heatmap(feat)
            self._render_isometric(feat)

    def _on_fill_changed(self, state):
        self._fill_surface = bool(state)
        feat = self._current_feature()
        if feat:
            self._render_isometric(feat)

    # ── Pending features & scoring ────────────────────────────────

    def _score_explore_feature(self, feat, members):
        checks = []
        score = 0

        if len(members) >= 2:
            checks.append(("bins >= 2", True, f"{len(members)} bins"))
            score += 25
        else:
            checks.append(("bins >= 2", False, "isolated: single-bin"))

        peak_int = feat["peak_intensity"]
        if peak_int > 0:
            checks.append(("intensity > 0", True, f"{peak_int:.1f}"))
            score += 25
        else:
            checks.append(("intensity > 0", False, f"{peak_int:.1f}"))

        intensities = [m[6]["cleaned_intensity"] for m in members]
        i_arr = np.array(intensities)
        cv = float(np.std(i_arr) / np.mean(i_arr)) if np.mean(i_arr) > 0 else 0
        if cv >= 0.05:
            checks.append(("CV >= 0.05", True, f"CV={cv:.3f}"))
            score += 25
        else:
            checks.append(("CV >= 0.05", False, f"CV={cv:.3f} (flat)"))

        if len(members) >= 2:
            positions = [(m[2], m[3]) for m in members]
            imax = int(np.argmax(intensities))
            cr, cc = positions[imax]
            distances = [np.sqrt((r - cr)**2 + (c - cc)**2) for r, c in positions]
            n_mono = 0
            n_comp = 0
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if abs(distances[i] - distances[j]) < 0.1:
                        continue
                    n_comp += 1
                    if (distances[i] < distances[j] and intensities[i] > intensities[j]) or \
                       (distances[i] > distances[j] and intensities[i] < intensities[j]):
                        n_mono += 1
            mono_frac = n_mono / n_comp if n_comp > 0 else 1.0
            if mono_frac >= 0.4:
                checks.append(("monotonic >= 40%", True, f"{mono_frac:.0%}"))
                score += 25
            else:
                checks.append(("monotonic >= 40%", False, f"{mono_frac:.0%}"))
        else:
            checks.append(("monotonic >= 40%", False, "n/a (1 bin)"))

        return score, checks

    def _update_pending_list(self):
        self.pending_list.clear()
        for i, pf in enumerate(self._pending_features):
            if pf["status"] == "processing":
                label = pf.get("label", "?")
                px, py = pf.get("peak_x", 0), pf.get("peak_y", 0)
                bk = pf.get("bin_key", "?")
                text = f"#{i+1}  ⏳ {label} ({px},{py}) bin {bk}..."
                item = QListWidgetItem(text)
                item.setForeground(QColor("#888"))
            else:
                feat = pf["feature"]
                sc = pf["score"]
                ref = feat["reflection"]
                chi = feat.get("chi_deg", 0)
                nb = feat["n_bins"]
                color = "#3cb44b" if sc == 100 else "#f0a030"
                item = QListWidgetItem(
                    f"#{i+1}  {ref}  χ={chi:.0f}°  {nb}bins  [{sc}%]")
                item.setForeground(QColor(color))
            self.pending_list.addItem(item)

    def _on_pending_selected(self):
        rows = [idx.row() for idx in self.pending_list.selectedIndexes()]
        has_sel = len(rows) > 0
        if not has_sel:
            self.accept_btn.setEnabled(False)
            self.remove_btn.setEnabled(False)
            self.score_label.setText("")
            return
        idx = rows[0]
        if idx >= len(self._pending_features):
            return
        pf = self._pending_features[idx]

        if pf["status"] == "processing":
            self.accept_btn.setEnabled(False)
            self.remove_btn.setEnabled(True)
            self.score_label.setText("⏳ Expansion in progress...")
            return

        self.accept_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)

        feat = pf["feature"]
        self._explore_feature = feat

        self._render_heatmap(feat)
        self._render_isometric(feat)
        self._load_detector_image(feat.get("center_bin"), feat)
        self._update_info(feat)

        lines = [f"Score: {pf['score']}%"]
        for name, passed, detail in pf["checks"]:
            mark = "✔" if passed else "✘"
            lines.append(f"  {mark} {name}: {detail}")
        if pf["score"] == 100:
            lines.append("\nWould be ACCEPTED by pipeline")
        else:
            lines.append(f"\nWould be FILTERED ({pf['score']}% < 100%)")
        self.score_label.setText("\n".join(lines))

    def _remove_rect_for_job(self, job_id):
        for i, (rid, rect_patch) in enumerate(self._explore_rects):
            if rid == job_id:
                try:
                    rect_patch.remove()
                except Exception:
                    pass
                self._explore_rects.pop(i)
                self.detector_canvas.draw_idle()
                break

    def _on_accept_pending(self):
        rows = [idx.row() for idx in self.pending_list.selectedIndexes()]
        if not rows:
            return
        idx = rows[0]
        if idx >= len(self._pending_features):
            return
        pf = self._pending_features.pop(idx)
        if pf["status"] == "processing":
            return
        feat = pf["feature"]
        feat.pop("_members", None)

        self._remove_rect_for_job(pf.get("job_id", -1))

        catalog_path = RESULTS_DIR / f"feature_catalog_{_BIN_SIZE}x{_BIN_SIZE}.json"
        with open(catalog_path) as f:
            catalog = json.load(f)
        next_id = max((f.get("feature_id", 0) for f in catalog), default=0) + 1
        feat["feature_id"] = next_id
        catalog.append(feat)
        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

        ref = feat["reflection"]
        if "kept" not in self._features:
            self._features["kept"] = {}
        if ref not in self._features["kept"]:
            self._features["kept"][ref] = []
        self._features["kept"][ref].append(feat)
        for bk in feat.get("spatial_extent", []):
            self._bin_to_features[bk].append(feat)

        self._update_pending_list()
        self._render_explore_minimap()
        self.info_label.setText(
            f"Saved as feature #{next_id}\n"
            f"{ref}  χ={feat['chi_deg']:.0f}°  {feat['n_bins']} bins")
        self.score_label.setText("")
        self.accept_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)

    def _on_remove_pending(self):
        rows = [idx.row() for idx in self.pending_list.selectedIndexes()]
        if not rows:
            return
        idx = rows[0]
        if idx >= len(self._pending_features):
            return
        pf = self._pending_features.pop(idx)
        self._remove_rect_for_job(pf.get("job_id", -1))
        self._update_pending_list()
        self._render_explore_minimap()
        self.score_label.setText("")
        self.accept_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)

    def _on_show_region(self):
        if self._region_shown:
            self._region_shown = False
            self.region_btn.setText("Show region peaks")
            bin_key = self._explore_bin or self._selected_bin
            if bin_key:
                feat = self._current_feature() if not self._explore_feature else self._explore_feature
                self._load_detector_image(bin_key, feat)
                self.info_label.setText(f"Bin {bin_key} — single view")
            return

        bin_key = self._explore_bin or self._selected_bin
        if not bin_key:
            feat = self._current_feature()
            if feat:
                bin_key = feat.get("center_bin")
        if not bin_key:
            self.info_label.setText("No bin in view")
            return

        parts = bin_key.split("_")
        cr, cc = int(parts[0]), int(parts[1])
        h5 = self._get_h5()
        radius = 1

        summed = None
        bin_keys = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                bk = f"{cr + dr}_{cc + dc}"
                if bk in h5:
                    image = self._load_raw_image(bk)
                    if image is not None:
                        if summed is None:
                            summed = image.copy()
                        else:
                            summed += image
                        bin_keys.append(bk)

        if summed is None:
            self.info_label.setText("No data in region")
            return

        cleaned = self._radial_median_subtract(summed, self._tth_data)

        seen_ids = set()
        region_feats = []
        for bk in bin_keys:
            for feat in self._bin_to_all_features.get(bk, []):
                fid = feat.get("feature_id", id(feat))
                if fid not in seen_ids:
                    seen_ids.add(fid)
                    region_feats.append(feat)

        ax = self.detector_canvas.ax
        ax.clear()
        ax.set_facecolor("black")

        display = cleaned.copy()
        if self._log_scale:
            display = np.log1p(np.clip(display, 0, None))
        self.detector_canvas._display_data = display

        finite = display[np.isfinite(display)]
        if len(finite) == 0:
            vmin, vmax = 0, 1
        else:
            vmin = float(np.percentile(finite, self._vmin_pct))
            vmax = float(np.percentile(finite, self._vmax_pct))

        cmap_name = self._cmap_name
        if self.reverse_cb.isChecked():
            cmap_name += "_r"
        ax.imshow(display, origin="upper", cmap=plt.get_cmap(cmap_name),
                  vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")

        ref_color_map = {}
        for idx, lab in enumerate(self._ref_labels):
            ref_color_map[lab] = ARC_COLORS[idx % len(ARC_COLORS)]

        n_kept = 0
        n_filtered = 0
        for feat in region_feats:
            dx = feat.get("detector_x", 0)
            dy = feat.get("detector_y", 0)
            ref = feat.get("reflection", "?")
            fid = feat.get("feature_id", "?")
            reason = feat.get("reason", "")
            color = ref_color_map.get(ref, "#aaa")

            is_kept = "non-Gaussian" not in reason and "isolated" not in reason \
                and "flat" not in reason and "non-positive" not in reason
            if is_kept:
                n_kept += 1
                style = "-"
                alpha = 0.9
                lw = 2
            else:
                n_filtered += 1
                style = ":"
                alpha = 0.6
                lw = 1.5

            circle = plt.Circle((dx, dy), 18, fill=False,
                                color=color, linewidth=lw,
                                linestyle=style, alpha=alpha)
            ax.add_patch(circle)

            tag = f"#{fid} {ref}"
            if not is_kept:
                short = reason.split(":")[0] if ":" in reason else reason[:15]
                tag += f"\n({short})"
            ax.text(dx + 22, dy - 12, tag,
                    color=color, fontsize=7, fontweight="bold",
                    alpha=alpha,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black",
                              ec=color, alpha=0.6, linewidth=0.4))

        ax.set_title(
            f"Region sum — {len(bin_keys)} bins around {bin_key}  "
            f"({n_kept} kept, {n_filtered} filtered)",
            color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)
        self.detector_canvas.draw_idle()

        self._region_shown = True
        self.region_btn.setText("Back to single bin")
        self.info_label.setText(
            f"Region: {len(bin_keys)} bins summed\n"
            f"{n_kept} kept + {n_filtered} filtered peaks shown\n"
            f"Solid = kept, dotted = filtered")

    # ── Explore mode ──────────────────────────────────────────────

    def _on_explore_toggled(self, state):
        self._explore_mode = bool(state)
        self._explore_peaks = []
        self._explore_feature = None
        self._explore_rects = []
        self.detector_canvas.drag_enabled = self._explore_mode

        feat = self._current_feature()
        if self._explore_mode and not self._explore_bin and feat:
            self._explore_bin = feat.get("center_bin")

        self._pending_group.setVisible(self._explore_mode)

        if self._explore_mode:
            self._render_explore_minimap()
            self._update_pending_list()
            self.info_label.setText("EXPLORE MODE\n\nDrag a rectangle on the detector\n"
                                   "image to select a peak region.\n\n"
                                   "Use nav or mini-map to change bins.")
        else:
            self._explore_bin = None
            self._on_feature_changed()

    def _on_minimap_click(self, event):
        if not self._explore_mode:
            return
        if event.inaxes != self.loc_ax or event.xdata is None:
            return
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        if 0 <= row < self._n_rows and 0 <= col < self._n_cols:
            self._on_explore_navigate(row, col)

    def _render_explore_minimap(self):
        ax = self.loc_ax
        ax.clear()
        ax.set_facecolor("#222")
        nr, nc = self._n_rows, self._n_cols

        h5 = self._get_h5()
        grid = np.full((nr, nc), np.nan)
        for bk in h5.keys():
            parts = bk.split("_")
            if len(parts) == 2:
                r, c = int(parts[0]), int(parts[1])
                if 0 <= r < nr and 0 <= c < nc:
                    grid[r, c] = 0.15

        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad(color="#111")
        ax.imshow(grid, origin="upper", cmap=cmap, interpolation="nearest",
                  aspect="equal", vmin=0, vmax=1,
                  extent=[-0.5, nc - 0.5, nr - 0.5, -0.5])

        for feat_list in self._features.get("kept", {}).values():
            for f in feat_list:
                ax.plot(f["center_col"], f["center_row"], ".",
                        color="#666", markersize=1.5)

        if self._explore_bin:
            parts = self._explore_bin.split("_")
            if len(parts) == 2:
                er, ec = int(parts[0]), int(parts[1])
                ax.plot(ec, er, "s", markersize=5, markerfacecolor="none",
                        markeredgecolor="cyan", markeredgewidth=1.5)

        for pf in self._pending_features:
            feat = pf["feature"]
            color = "#3cb44b" if pf["score"] == 100 else "#f0a030"
            ax.plot(feat["center_col"], feat["center_row"], "o",
                    markersize=3, color=color, alpha=0.8)

        if self._explore_feature:
            for bk in self._explore_feature.get("spatial_extent", []):
                parts = bk.split("_")
                if len(parts) == 2:
                    ax.plot(int(parts[1]), int(parts[0]), "s",
                            markersize=3, color="lime", alpha=0.7)

        ax.set_xlim(-1, nc)
        ax.set_ylim(nr, -1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Click to navigate", color="#f0a030", fontsize=7, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.loc_canvas.draw_idle()

    def _on_explore_navigate(self, row, col):
        bin_key = f"{row}_{col}"
        h5 = self._get_h5()
        if bin_key not in h5:
            self.info_label.setText(f"Bin {bin_key} — no data")
            return

        self._explore_bin = bin_key
        self._explore_peaks = []
        self._explore_rects = []

        self._load_detector_image(bin_key)
        self._render_explore_minimap()
        self.info_label.setText(f"Bin {bin_key} loaded.\n\n"
                                "Drag a rectangle around a peak\n"
                                "on the detector to select it.")

    def _render_explore_detector(self, bin_key, cleaned, peaks):
        ax = self.detector_canvas.ax
        ax.clear()
        ax.set_facecolor("black")

        display = cleaned.copy()
        if self._log_scale:
            display = np.log1p(np.clip(display, 0, None))
        self.detector_canvas._display_data = display

        finite = display[np.isfinite(display)]
        if len(finite) == 0:
            vmin, vmax = 0, 1
        else:
            vmin = float(np.percentile(finite, self._vmin_pct))
            vmax = float(np.percentile(finite, self._vmax_pct))

        cmap_name = self._cmap_name
        if self.reverse_cb.isChecked():
            cmap_name += "_r"
        ax.imshow(display, origin="upper", cmap=plt.get_cmap(cmap_name),
                  vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")

        ref_color_map = {}
        for idx, lab in enumerate(self._ref_labels):
            ref_color_map[lab] = ARC_COLORS[idx % len(ARC_COLORS)]

        for i, p in enumerate(peaks):
            color = ref_color_map.get(p["label"], "#7fff00")
            circle = plt.Circle((p["x"], p["y"]), 18, fill=False,
                                color=color, linewidth=2, linestyle="--")
            ax.add_patch(circle)
            ax.text(p["x"] + 22, p["y"] - 12,
                    f"{p['label']}  SNR={p['snr']:.1f}",
                    color=color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black",
                              ec=color, alpha=0.8, linewidth=0.5))

        if self._show_tth_overlay:
            tth = self._tth_map
            for idx, (lab, d) in enumerate(zip(self._ref_labels, self._ref_degs)):
                mask = np.abs(tth - d) < 0.3
                overlay = np.where(mask, 1.0, np.nan)
                color = ARC_COLORS[idx % len(ARC_COLORS)]
                cmap_band = mcolors.ListedColormap([color])
                ax.imshow(overlay, cmap=cmap_band, alpha=0.25,
                          interpolation="nearest")

        ax.set_title(f"Explore — bin {bin_key}, {len(peaks)} peaks",
                     color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)
        self.detector_canvas.draw_idle()

    def _on_explore_detector_click(self, x, y):
        pass

    def _on_explore_drag(self, x0, y0, x1, y1, rect_patch=None):
        bin_key = self._selected_bin or self._explore_bin
        if not bin_key:
            return

        image = self._load_raw_image(bin_key)
        if image is None:
            return
        cleaned = self._radial_median_subtract(image, self._tth_data)

        ry0 = max(0, y0)
        ry1 = min(image.shape[0], y1)
        rx0 = max(0, x0)
        rx1 = min(image.shape[1], x1)
        patch = cleaned[ry0:ry1, rx0:rx1]
        if patch.size == 0:
            self.info_label.setText("Empty selection")
            return

        local_y, local_x = np.unravel_index(np.argmax(patch), patch.shape)
        peak_x = rx0 + int(local_x)
        peak_y = ry0 + int(local_y)
        peak_val = float(cleaned[peak_y, peak_x])

        tth_val = float(self._tth_map[peak_y, peak_x])
        best_label, best_dist = "unknown", 999
        for deg, lab in zip(self._ref_degs, self._ref_labels):
            d = abs(tth_val - deg)
            if d < best_dist:
                best_label, best_dist = lab, d

        seed_peak = {
            "x": peak_x, "y": peak_y,
            "label": best_label,
            "snr": peak_val / max(float(np.median(cleaned[cleaned > 0])), 1),
            "npix": int(patch.size),
            "compactness": 1.0,
            "integrated_intensity": float(np.sum(patch[patch > 0])),
            "cleaned_intensity": peak_val,
        }

        job_id = self._next_job_id
        self._next_job_id += 1

        if rect_patch:
            self._explore_rects.append((job_id, rect_patch))

        self._pending_features.append({
            "feature": None, "members": None,
            "score": -1, "checks": [],
            "job_id": job_id, "status": "processing",
            "label": best_label, "peak_x": peak_x, "peak_y": peak_y,
            "bin_key": bin_key,
        })
        self._update_pending_list()

        n_running = sum(1 for pf in self._pending_features if pf["status"] == "processing")
        self.info_label.setText(
            f"Expanding {best_label} at ({peak_x}, {peak_y})\n"
            f"{n_running} expansion(s) running...")

        worker = _ExpansionWorker(job_id, self, bin_key, seed_peak)
        worker.finished.connect(self._on_expansion_done)
        self._explore_workers.append(worker)
        worker.start()

    def _on_expansion_done(self, job_id, feat, members):
        for pf in self._pending_features:
            if pf.get("job_id") == job_id and pf["status"] == "processing":
                score, checks = self._score_explore_feature(feat, members)
                pf["feature"] = feat
                pf["members"] = members
                pf["score"] = score
                pf["checks"] = checks
                pf["status"] = "ready"
                break

        for i, (rid, rect_patch) in enumerate(self._explore_rects):
            if rid == job_id:
                rect_patch.set_edgecolor("#3cb44b")
                rect_patch.set_facecolor("#3cb44b")
                self.detector_canvas.draw_idle()
                break

        self._explore_workers = [w for w in self._explore_workers if w.isRunning()]
        self._update_pending_list()

        n_running = sum(1 for pf in self._pending_features if pf["status"] == "processing")
        if n_running > 0:
            self.info_label.setText(f"{n_running} expansion(s) still running...")
        else:
            self.info_label.setText("All expansions complete.")

    LINK_TOLERANCE = 5

    def _expand_peak_spatially(self, center_row, center_col, seed_peak):
        from ..core.processing import detect_peaks_with_intensity

        center_bk = f"{center_row}_{center_col}"
        h5 = self._get_h5()
        visited = {center_bk}
        queue = [center_bk]
        members = [(center_bk, 0, center_row, center_col,
                     seed_peak["x"], seed_peak["y"], seed_peak)]
        target_x, target_y = seed_peak["x"], seed_peak["y"]
        max_radius = 10

        while queue:
            bk = queue.pop(0)
            br, bc = int(bk.split("_")[0]), int(bk.split("_")[1])

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = br + dr, bc + dc
                    nbk = f"{nr}_{nc}"
                    if nbk in visited or nbk not in h5:
                        continue
                    dist = max(abs(nr - center_row), abs(nc - center_col))
                    if dist > max_radius:
                        continue
                    visited.add(nbk)

                    image = self._load_raw_image(nbk)
                    peaks, cleaned = detect_peaks_with_intensity(
                        image, self._tth_map, self._ref_degs,
                        self._ref_labels, self._tth_data, self._det
                    )

                    for p in peaks:
                        r = 3
                        py0 = max(0, p['y'] - r)
                        py1 = min(cleaned.shape[0], p['y'] + r + 1)
                        px0 = max(0, p['x'] - r)
                        px1 = min(cleaned.shape[1], p['x'] + r + 1)
                        p['cleaned_intensity'] = float(
                            np.max(cleaned[py0:py1, px0:px1]))

                    match = None
                    for p in peaks:
                        d = ((p["x"] - target_x)**2 +
                             (p["y"] - target_y)**2) ** 0.5
                        if d <= self.LINK_TOLERANCE and p["label"] == seed_peak["label"]:
                            if match is None or p["snr"] > match["snr"]:
                                match = p
                    if match:
                        members.append((nbk, 0, nr, nc,
                                        match["x"], match["y"], match))
                        queue.append(nbk)

        return members

    def _build_explore_feature(self, members, seed_peak):
        from ..core.processing import _best_per_bin

        bins_in_feature = set(m[0] for m in members)
        intensities = [m[6]["cleaned_intensity"] for m in members]
        snrs = [m[6]["snr"] for m in members]
        xs = [m[4] for m in members]
        ys = [m[5] for m in members]
        rows = [m[2] for m in members]
        cols = [m[3] for m in members]

        imax = int(np.argmax(intensities))
        det_x = int(np.mean(xs))
        det_y = int(np.mean(ys))

        by, bx = self._beam_center
        chi = float(np.degrees(np.arctan2(det_y - by, det_x - bx)))

        feat = {
            "reflection": seed_peak["label"],
            "detector_x": det_x,
            "detector_y": det_y,
            "peak_intensity": float(max(intensities)),
            "mean_snr": float(np.mean(snrs)),
            "n_bins": len(bins_in_feature),
            "spatial_extent": sorted(bins_in_feature),
            "center_bin": members[imax][0],
            "center_row": rows[imax],
            "center_col": cols[imax],
            "intensity_profile": _best_per_bin(members),
            "chi_deg": round(chi, 1),
            "reason": f"explore: {len(bins_in_feature)} bins from manual selection",
            "feature_id": -1,
        }
        return feat

    def _on_tth_overlay_changed(self, state):
        self._show_tth_overlay = bool(state)
        feat = self._current_feature()
        if feat:
            bin_key = self._selected_bin or feat.get("center_bin")
            self._load_detector_image(bin_key, feat)

    def _on_log_changed(self, state):
        self._log_scale = bool(state)
        self._refresh_display()

    def _on_contrast_changed(self, *_):
        lo = self.vmin_slider.value() / 10.0
        hi = self.vmax_slider.value() / 10.0
        self._vmin_pct = lo
        self._vmax_pct = hi
        self.vmin_val.setText(f"{lo:.1f}")
        self.vmax_val.setText(f"{hi:.1f}")
        self._refresh_detector_only()

    def _set_contrast_preset(self, lo, hi):
        self.vmin_slider.blockSignals(True)
        self.vmax_slider.blockSignals(True)
        self.vmin_slider.setValue(lo)
        self.vmax_slider.setValue(hi)
        self.vmin_slider.blockSignals(False)
        self.vmax_slider.blockSignals(False)
        self._on_contrast_changed()

    def _on_vmin_text(self):
        try:
            val = float(self.vmin_val.text())
            self.vmin_slider.setValue(int(val * 10))
        except ValueError:
            pass

    def _on_vmax_text(self):
        try:
            val = float(self.vmax_val.text())
            self.vmax_slider.setValue(int(val * 10))
        except ValueError:
            pass

    # ── Noise reduction ────────────────────────────────────────────

    def _on_noise_toggle(self, checked):
        self._noise_enabled = checked
        for w in self._noise_widgets:
            w.setVisible(checked)
        self._refresh_detector_only()

    def _on_noise_algo_changed(self, *_):
        self._noise_algo = self.noise_algo_combo.currentData() or "gaussian"
        if self._noise_enabled:
            self._refresh_detector_only()

    def _on_noise_param_changed(self, *_):
        self._noise_strength = self.strength_slider.value() / 100.0
        self.strength_val.setText(f"{self._noise_strength:.2f}")
        self._noise_shift = self.shift_slider.value() / 10.0
        self.shift_val.setText(f"{self._noise_shift:.1f}")
        if self._noise_enabled:
            self._refresh_detector_only()

    def _on_strength_text(self):
        try:
            val = float(self.strength_val.text())
            self.strength_slider.setValue(int(val * 100))
        except ValueError:
            pass

    def _on_shift_text(self):
        try:
            val = float(self.shift_val.text())
            self.shift_slider.setValue(int(val * 10))
        except ValueError:
            pass

    # ── Refresh helpers ────────────────────────────────────────────

    def _refresh_display(self):
        feat = self._current_feature()
        if feat is None:
            return
        self._render_heatmap(feat)
        self._render_isometric(feat)
        self._refresh_detector_only()

    def _refresh_detector_only(self):
        feat = self._current_feature()
        if feat is None:
            return
        bin_key = self._selected_bin or feat.get("center_bin")
        if bin_key:
            self._load_detector_image(bin_key, feat)

    # ── Cleanup ────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._h5f is not None:
            self._h5f.close()
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────────────

def launch_gui(project_root=".", bin_size=3, scan=None):
    """Configure paths and launch the feature viewer (used by the CLI)."""
    configure(project_root=project_root, bin_size=bin_size, scan=scan)
    _run_app()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XRD feature viewer")
    parser.add_argument("--project-root", type=str, default=".",
                        help="Path to the xrd-tools project root")
    parser.add_argument("--bin-size", type=int, default=3, help="Bin size to view")
    parser.add_argument("--scan", default=None, help="Scan number/name")
    args = parser.parse_args()
    launch_gui(project_root=args.project_root, bin_size=args.bin_size, scan=args.scan)


def _run_app():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt5.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    viewer = FeatureViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
