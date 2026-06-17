#!/usr/bin/env python3
"""
Multi-bin-size XRD image analysis and labeling tool.

Loads per-bin summed detector images (from pre-built HDF5 or raw H5 scans)
and provides interactive viewing, peak detection overlay, and point
annotation for Bragg peak locations.

Features:
  - Bin size selector (3x3, 4x4, 5x5) with pre-built HDF5 loading
  - Navigate bins by index with < > buttons
  - Noise reduction toggle with algorithm dropdown and strength/shift sliders
  - Peak detection algorithm selector (base + CVEvolve-evolved algorithms)
  - Toggleable labeling mode for annotation
  - Colormap selector with reverse toggle, contrast percentile sliders, log scale
  - Annotations persist per bin size in CVEvolve JSON format

Usage:
    python labeling_tool.py
    python labeling_tool.py --project-root /path/to/project
"""

import sys
import os
import json
import csv
import time
import copy
import inspect
import importlib.util
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import h5py
import tifffile
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.patches import Rectangle, Circle
from matplotlib import colors as mcolors
from scipy import ndimage as ndi
from scipy.signal import argrelextrema

try:
    import hdf5plugin  # noqa: F401 — registers LZ4/etc filters for h5py
except ImportError:
    pass

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFileDialog, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QGridLayout, QSplitter,
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu,
    QSizePolicy, QMessageBox, QAction,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

from ..core.algorithms import (
    ALGORITHM_NAMES, ALGORITHM_DISPLAY,
    compute_tth_binning, compute_radial_profile, fit_all_models,
    build_background_image, subtract_background,
)
from ..config import DataManager


# ===== Constants =====
H5_DATASET = "entry/data/data"

COLORMAPS = [
    "inferno", "viridis", "plasma", "magma", "cividis",
    "hot", "coolwarm", "gray", "jet", "turbo",
    "cubehelix", "bone", "copper", "pink", "seismic",
]

DEGS = [6.81319, 7.51422, 10.61748, 13.00831, 15.01266,
        16.07224, 16.79944, 18.42549, 21.29655, 22.59817, 26.16205]
DEG_LABELS = ["PbI2", "(001)", "(011)", "(111)", "(002)",
              "ITO", "(012)", "(112)"]

ARC_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6",
]

BIN_SIZES = [1, 3, 4, 5]

# H5 paths are resolved at runtime via DataManager.bins_h5(); labels only here.
BIN_CONFIGS = {
    5: {"label": "5×5"},
    4: {"label": "4×4"},
    3: {"label": "3×3"},
    1: {"label": "1×1 (raw)"},
}

IMAGE_CACHE_MAX = 50


def discover_algorithms(project_root):
    """Find available peak detection algorithms from CVEvolve directories."""
    algos = [{"name": "Base (percentile threshold)", "source": "built-in",
              "func_type": "base", "file": None, "holdout_f1": None}]

    for bs in BIN_SIZES:
        cvdir = project_root / f"cvevolve_{bs}x{bs}" / "test_data"
        manifest = cvdir / "top_algorithms.json"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            entries = json.load(f)
        for entry in entries:
            algo_file = cvdir / entry["file"]
            if not algo_file.exists():
                continue
            f1 = entry.get("holdout_f1")
            f1_str = f", F1={f1:.2f}" if f1 else ""
            algos.append({
                "name": f"{entry['name']} ({entry['source']}{f1_str})",
                "source": entry.get("source", f"{bs}x{bs}"),
                "func_type": "cvevolve",
                "file": str(algo_file),
                "data_dir": str(cvdir),
                "holdout_f1": f1,
            })
    return algos


def run_cvevolve_algorithm(algo_info, image, tth_map, degs, deg_labels, tth_data,
                           **overrides):
    """Dynamically import and run a CVEvolve-generated detection algorithm."""
    algo_path = algo_info["file"]
    data_dir = algo_info.get("data_dir", str(Path(algo_path).parent))

    saved_path = sys.path[:]
    try:
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        spec = importlib.util.spec_from_file_location("_peak_algo", algo_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if hasattr(mod, "detect_peaks"):
            sig = inspect.signature(mod.detect_peaks)
            kwargs = dict(image=image, tth_map=tth_map, degs=degs,
                          deg_labels=deg_labels, tth_data=tth_data)
            for k, v in overrides.items():
                if k in sig.parameters:
                    kwargs[k] = v
            return mod.detect_peaks(**kwargs)
        elif hasattr(mod, "find_peaks"):
            return mod.find_peaks(image, tth_map, degs, deg_labels)
    finally:
        sys.path[:] = saved_path

    return {}


# ===== Data loading =====

def load_xrd_metadata(xrd_dir, scan_number=203):
    xrd_files = sorted(Path(xrd_dir).glob(f"scan_{scan_number:04d}_*.h5"))
    xrd_files = [f for f in xrd_files if f.stat().st_size > 0]
    xrd_file_map = []
    for fp in xrd_files:
        with h5py.File(fp, "r") as f:
            n_frames = f[H5_DATASET].shape[0]
        for j in range(n_frames):
            xrd_file_map.append((xrd_files.index(fp), j))
    return xrd_files, xrd_file_map, len(xrd_file_map)


def load_positions(csv_path, n_total):
    x, y = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["X_Position"]))
            y.append(float(row["Y_Position"]))
    frame_x = np.full(n_total, np.nan)
    n = min(len(x), n_total)
    frame_x[:n] = x[:n]
    return frame_x


def build_scan_grid(frame_x, n_total, kernel=20, order=50):
    valid = ~np.isnan(frame_x)
    x = frame_x.copy()
    if np.any(~valid):
        x[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], frame_x[valid])
    x_smooth = np.convolve(x, np.ones(kernel) / kernel, mode="same")
    x_max = argrelextrema(x_smooth, np.greater, order=order)[0]
    x_min = argrelextrema(x_smooth, np.less, order=order)[0]
    turns = np.sort(np.concatenate([x_max, x_min]))
    starts = np.concatenate([[0], turns])
    ends = np.concatenate([turns, [n_total]])
    row = np.zeros(n_total, dtype=int)
    col = np.zeros(n_total, dtype=int)
    for i in range(len(starts)):
        s, e = starts[i], ends[i]
        row[s:e] = i
        c = np.arange(e - s)
        if i % 2 == 1:
            c = c[::-1]
        col[s:e] = c
    return row, col, row.max() + 1, col.max() + 1


def build_bin_mapping(n_rows, n_cols, bin_size, grid_to_frames):
    n_br = (n_rows + bin_size - 1) // bin_size
    n_bc = (n_cols + bin_size - 1) // bin_size
    mapping = {}
    for br in range(n_br):
        for bc in range(n_bc):
            frames = []
            for dr in range(bin_size):
                for dc in range(bin_size):
                    r, c = br * bin_size + dr, bc * bin_size + dc
                    if r < n_rows and c < n_cols:
                        frames.extend(grid_to_frames.get((r, c), []))
            if frames:
                mapping[(br, bc)] = frames
    return mapping, n_br, n_bc


def load_and_sum_frames(frame_indices, xrd_files, xrd_file_map):
    summed = None
    by_file = {}
    for gi in frame_indices:
        fi, fj = xrd_file_map[gi]
        by_file.setdefault(fi, []).append(fj)
    for fi, frame_list in by_file.items():
        with h5py.File(xrd_files[fi], "r") as f:
            ds = f[H5_DATASET]
            for fj in frame_list:
                frame = ds[fj].astype(np.float64)
                if summed is None:
                    summed = frame
                else:
                    summed += frame
    if summed is not None:
        summed[summed < 0] = 0
        summed[summed > 1e9] = 0
    return summed


# ===== Peak detection =====

def find_peaks_on_image(image, tth_map=None, degs=None, deg_labels=None,
                        percentile=97.0, min_pixels=3, pad=10, ignore_edge=2):
    """Detect bright peaks anywhere on the image via global thresholding.

    Returns {"peak": [(y0,y1,x0,x1,cx,cy), ...]} with reflection labels
    assigned by nearest 2-theta arc when tth_map is provided.
    """
    finite = image[np.isfinite(image)]
    if len(finite) == 0:
        return {}

    thr = np.percentile(finite, percentile)
    hotspot = image >= thr

    if ignore_edge > 0:
        hotspot[:ignore_edge, :] = False
        hotspot[-ignore_edge:, :] = False
        hotspot[:, :ignore_edge] = False
        hotspot[:, -ignore_edge:] = False

    cc, n_comp = ndi.label(hotspot)
    peaks = []
    for comp_id in range(1, n_comp + 1):
        ys, xs = np.where(cc == comp_id)
        if len(ys) < min_pixels:
            continue
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, image.shape[0])
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, image.shape[1])
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        peaks.append((y0, y1, x0, x1, cx, cy))

    peaks.sort(key=lambda r: (r[0], r[2]))
    dedup = []
    for p in peaks:
        keep = True
        for q in dedup:
            if p[0] >= q[0] and p[1] <= q[1] and p[2] >= q[2] and p[3] <= q[3]:
                keep = False
                break
        if keep:
            dedup.append(p)

    # Assign reflection labels by nearest 2-theta arc
    results = {}
    if tth_map is not None and degs is not None and deg_labels is not None:
        for y0, y1, x0, x1, cx, cy in dedup:
            tth_val = tth_map[cy, cx] if 0 <= cy < tth_map.shape[0] and 0 <= cx < tth_map.shape[1] else None
            label = "unknown"
            if tth_val is not None:
                best_dist = float("inf")
                for lab, d in zip(deg_labels, degs):
                    dist = abs(tth_val - d)
                    if dist < best_dist:
                        best_dist = dist
                        label = lab
            results.setdefault(label, []).append((y0, y1, x0, x1, cx, cy))
    else:
        results["peak"] = dedup

    return results


def find_local_maximum(image, click_x, click_y, radius=30):
    """Find the brightest pixel within radius of the click point."""
    h, w = image.shape
    y_lo = max(0, click_y - radius)
    y_hi = min(h, click_y + radius + 1)
    x_lo = max(0, click_x - radius)
    x_hi = min(w, click_x + radius + 1)
    patch = image[y_lo:y_hi, x_lo:x_hi]
    if patch.size == 0:
        return click_x, click_y
    local_y, local_x = np.unravel_index(np.argmax(patch), patch.shape)
    return x_lo + int(local_x), y_lo + int(local_y)


# ===== Canvas =====

class LabelCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 9))
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)

        self.image_data = None
        self.display_data = None
        self.im = None
        self.cmap = "inferno"
        self.vmin = None
        self.vmax = None
        self.log_scale = False

        # Annotations: list of {"x": int, "y": int, "reflection": str}
        self.annotations = []
        self._ann_markers = []
        self._ann_labels = []

        # Outlines from auto-detect: list of Rectangle patches
        self._outline_patches = []
        self._outline_data = {}  # label -> [(y0,y1,x0,x1,cx,cy), ...]

        # Arc overlay patches
        self._arc_patches = []

        # Detection overlay markers (from peak algorithms)
        self._detection_markers = []

        # Drag selection state
        self._drag_rect = None
        self._drag_start = None
        self._dragging = False

        # Mode flags
        self.mode = "click"  # "click", "drag", "outline_select", "select_delete"
        self.selected_annotations = set()

        # Callbacks
        self._on_hover_cb = None
        self._on_annotation_change_cb = None
        self._on_before_modify_cb = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_hover_callback(self, cb):
        self._on_hover_cb = cb

    def set_annotation_change_callback(self, cb):
        self._on_annotation_change_cb = cb

    def set_before_modify_callback(self, cb):
        self._on_before_modify_cb = cb

    def display_image(self, data, cmap=None, vmin=None, vmax=None, title=None):
        self.image_data = data
        if cmap:
            self.cmap = cmap
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax

        self._apply_display_transform()

        if self.im is None:
            self.im = self.ax.imshow(
                self.display_data, origin="upper", aspect="equal",
                cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                interpolation="nearest",
            )
            self.ax.set_xlabel("Detector X (px)")
            self.ax.set_ylabel("Detector Y (px)")
        else:
            self.im.set_data(self.display_data)
            self.im.set_clim(self.vmin, self.vmax)
            self.im.set_cmap(self.cmap)

        if title:
            self.ax.set_title(title, fontsize=10)

        self._redraw_annotations()
        self.draw_idle()

    def _apply_display_transform(self):
        if self.image_data is None:
            return
        data = self.image_data
        if self.log_scale:
            data = data.copy()
            data[data < 0] = 0
            data = np.log1p(data)
        self.display_data = data

        finite = data[np.isfinite(data)]
        if self.vmin is None:
            self.vmin = float(np.percentile(finite, 2)) if len(finite) else 0
        if self.vmax is None:
            self.vmax = float(np.percentile(finite, 98)) if len(finite) else 1

    def update_contrast(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        if self.im is not None:
            self.im.set_clim(vmin, vmax)
            self.draw_idle()

    def update_cmap(self, cmap):
        self.cmap = cmap
        if self.im is not None:
            self.im.set_cmap(cmap)
            self.draw_idle()

    def set_log_scale(self, enabled):
        self.log_scale = enabled
        if self.image_data is not None:
            self.vmin = None
            self.vmax = None
            self._apply_display_transform()
            if self.im is not None:
                self.im.set_data(self.display_data)
                self.im.set_clim(self.vmin, self.vmax)
                self.draw_idle()

    # --- Annotation rendering ---

    def _redraw_annotations(self):
        for m in self._ann_markers:
            try: m.remove()
            except Exception: pass
        for t in self._ann_labels:
            try: t.remove()
            except Exception: pass
        self._ann_markers.clear()
        self._ann_labels.clear()

        for i, ann in enumerate(self.annotations):
            is_sel = (i in self.selected_annotations)
            marker_size = 12 if is_sel else 8
            edge_width = 2.5 if is_sel else 1.5
            color = "lime" if is_sel else "cyan"

            sc = self.ax.plot(
                ann["x"], ann["y"], "o",
                markersize=marker_size, markeredgewidth=edge_width,
                markeredgecolor=color, markerfacecolor="none",
            )
            self._ann_markers.extend(sc)

            txt = self.ax.text(
                ann["x"] + 5, ann["y"] - 5, ann.get("reflection", ""),
                color=color, fontsize=7, fontweight="bold" if is_sel else "normal",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6),
            )
            self._ann_labels.append(txt)

    def refresh_display(self):
        self._redraw_annotations()
        self.draw_idle()

    # --- Outline rendering ---

    def show_outlines(self, detected_peaks):
        self.clear_outlines()
        self._outline_data = detected_peaks
        for lab_idx, (lab, peaks) in enumerate(detected_peaks.items()):
            color = ARC_COLORS[lab_idx % len(ARC_COLORS)]
            for y0, y1, x0, x1, cx, cy in peaks:
                rect = Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=1.5, edgecolor=color, facecolor=color,
                    alpha=0.15, linestyle="--",
                )
                self.ax.add_patch(rect)
                self._outline_patches.append(rect)
                txt = self.ax.text(
                    x0, y0 - 2, lab, color=color, fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5),
                )
                self._outline_patches.append(txt)
        self.draw_idle()

    def clear_outlines(self):
        for p in self._outline_patches:
            try: p.remove()
            except Exception: pass
        self._outline_patches.clear()
        self._outline_data.clear()

    # --- Arc overlay ---

    def show_arcs(self, tth_map, degs, deg_labels, line_tol=0.3):
        self.clear_arcs()
        for idx, (lab, d) in enumerate(zip(deg_labels, degs)):
            mask = np.abs(tth_map - d) < line_tol
            overlay = np.where(mask, 1.0, np.nan)
            color = ARC_COLORS[idx % len(ARC_COLORS)]
            cmap = mcolors.ListedColormap([color])
            im = self.ax.imshow(overlay, cmap=cmap, alpha=0.25,
                                interpolation="nearest")
            self._arc_patches.append(im)

            ys, xs = np.where(mask)
            if len(ys) > 0:
                mid = len(ys) // 2
                txt = self.ax.text(
                    xs[mid], ys[mid], lab, color=color, fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.7),
                )
                self._arc_patches.append(txt)
        self.draw_idle()

    def clear_arcs(self):
        for p in self._arc_patches:
            try: p.remove()
            except Exception: pass
        self._arc_patches.clear()

    # --- Detection point overlay ---

    def show_detection_points(self, detections):
        """Show algorithm detections as X markers. detections: {label: [[x,y], ...]}"""
        self.clear_detection_points()
        for lab_idx, (lab, points) in enumerate(detections.items()):
            color = ARC_COLORS[lab_idx % len(ARC_COLORS)]
            for x, y in points:
                marker = self.ax.plot(
                    x, y, "x", markersize=10, markeredgewidth=2, color=color)
                self._detection_markers.extend(marker)
                txt = self.ax.text(
                    x + 6, y + 6, lab, color=color, fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
                self._detection_markers.append(txt)
        self.draw_idle()

    def clear_detection_points(self):
        for m in self._detection_markers:
            try: m.remove()
            except Exception: pass
        self._detection_markers.clear()

    # --- Mouse interaction ---

    def _toolbar_active(self):
        toolbar = self.parent().findChild(NavigationToolbar2QT) if self.parent() else None
        return toolbar and toolbar.mode

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if self._toolbar_active():
            return
        if self.mode == "view":
            return

        x, y = event.xdata, event.ydata

        if self.mode == "drag":
            self._dragging = True
            self._drag_start = (x, y)
            return

        if self.mode == "select_delete":
            mod = None
            if event.key == "control":
                mod = "ctrl"
            elif event.key == "shift":
                mod = "shift"
            self._try_select_annotation(x, y, modifier=mod)
            return

        if self.mode == "outline_select":
            self._try_select_outline(x, y)
            return

        # Default click mode: find local max
        if self.mode == "click" and self.image_data is not None:
            ix, iy = int(round(x)), int(round(y))
            mx, my = find_local_maximum(self.image_data, ix, iy, radius=30)
            ref = self._guess_reflection(mx, my)
            if self._on_before_modify_cb:
                self._on_before_modify_cb()
            self.annotations.append({"x": mx, "y": my, "reflection": ref})
            self._redraw_annotations()
            self.draw_idle()
            if self._on_annotation_change_cb:
                self._on_annotation_change_cb()

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            if self._on_hover_cb:
                self._on_hover_cb(None, None, None)
            return

        x, y = event.xdata, event.ydata

        if self._on_hover_cb and self.image_data is not None:
            col, row = int(round(x)), int(round(y))
            if 0 <= row < self.image_data.shape[0] and 0 <= col < self.image_data.shape[1]:
                self._on_hover_cb(col, row, float(self.image_data[row, col]))
            else:
                self._on_hover_cb(col, row, None)

        if self._dragging and self._drag_start:
            if self._drag_rect:
                try: self._drag_rect.remove()
                except Exception: pass
            x0, y0 = self._drag_start
            self._drag_rect = Rectangle(
                (min(x0, x), min(y0, y)), abs(x - x0), abs(y - y0),
                linewidth=2, edgecolor="lime", facecolor="lime",
                alpha=0.15, linestyle=":",
            )
            self.ax.add_patch(self._drag_rect)
            self.draw_idle()

    def _on_release(self, event):
        if event.button != 1:
            return

        if self._dragging and self._drag_start and event.inaxes == self.ax:
            if self._drag_rect:
                try: self._drag_rect.remove()
                except Exception: pass
                self._drag_rect = None

            x0, y0 = self._drag_start
            x1, y1 = event.xdata, event.ydata
            if abs(x1 - x0) > 5 and abs(y1 - y0) > 5 and self.image_data is not None:
                rx0, rx1 = int(min(x0, x1)), int(max(x0, x1))
                ry0, ry1 = int(min(y0, y1)), int(max(y0, y1))
                rx0 = max(0, rx0)
                ry0 = max(0, ry0)
                rx1 = min(self.image_data.shape[1], rx1)
                ry1 = min(self.image_data.shape[0], ry1)
                patch = self.image_data[ry0:ry1, rx0:rx1]
                if patch.size > 0:
                    local_y, local_x = np.unravel_index(np.argmax(patch), patch.shape)
                    mx, my = rx0 + int(local_x), ry0 + int(local_y)
                    ref = self._guess_reflection(mx, my)
                    if self._on_before_modify_cb:
                        self._on_before_modify_cb()
                    self.annotations.append({"x": mx, "y": my, "reflection": ref})
                    self._redraw_annotations()
                    self.draw_idle()
                    if self._on_annotation_change_cb:
                        self._on_annotation_change_cb()

            self._dragging = False
            self._drag_start = None
            self.draw_idle()

        elif self._dragging:
            if self._drag_rect:
                try: self._drag_rect.remove()
                except Exception: pass
                self._drag_rect = None
            self._dragging = False
            self._drag_start = None
            self.draw_idle()

    def _guess_reflection(self, x, y):
        """Guess which reflection family this point belongs to based on 2-theta."""
        if not hasattr(self, "_tth_map") or self._tth_map is None:
            return ""
        h, w = self._tth_map.shape
        if 0 <= y < h and 0 <= x < w:
            tth_val = self._tth_map[y, x]
            best_lab = ""
            best_dist = float("inf")
            for lab, d in zip(DEG_LABELS, DEGS):
                dist = abs(tth_val - d)
                if dist < best_dist:
                    best_dist = dist
                    best_lab = lab
            if best_dist < 1.0:
                return best_lab
        return ""

    def _try_select_annotation(self, x, y, modifier=None):
        best_idx = None
        best_dist = 15.0
        for i, ann in enumerate(self.annotations):
            dist = np.sqrt((ann["x"] - x) ** 2 + (ann["y"] - y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if modifier == "ctrl" and best_idx is not None:
            self.selected_annotations ^= {best_idx}
        elif modifier == "shift" and best_idx is not None and self.selected_annotations:
            anchor = min(self.selected_annotations)
            lo, hi = min(anchor, best_idx), max(anchor, best_idx)
            self.selected_annotations = set(range(lo, hi + 1))
        else:
            self.selected_annotations = {best_idx} if best_idx is not None else set()
        self._redraw_annotations()
        self.draw_idle()
        if self._on_annotation_change_cb:
            self._on_annotation_change_cb()

    def _try_select_outline(self, x, y):
        for lab, peaks in self._outline_data.items():
            for y0, y1, x0, x1, cx, cy in peaks:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    already = any(
                        abs(a["x"] - cx) < 5 and abs(a["y"] - cy) < 5
                        for a in self.annotations
                    )
                    if not already:
                        if self._on_before_modify_cb:
                            self._on_before_modify_cb()
                        self.annotations.append({"x": cx, "y": cy, "reflection": lab})
                        self._redraw_annotations()
                        self.draw_idle()
                        if self._on_annotation_change_cb:
                            self._on_annotation_change_cb()
                    return


# ===== Main Window =====

class LabelingTool(QMainWindow):

    def __init__(self, project_root=None, data_manager=None, scan=None):
        super().__init__()
        self.setWindowTitle("XRD Bin Analysis & Labeling Tool")
        self.setGeometry(60, 30, 1600, 1000)

        self.dm = data_manager or DataManager(project_root or ".", scan=scan)
        self.project_root = self.dm.root
        self._scan_number = self.dm.config.get("scan", "number") or 203

        self.bin_size = 5
        self._labeling_enabled = False
        self._init_done = False

        self._load_project_data()
        self._current_bin_idx = 0

        self._undo_stack = []
        self._redo_stack = []

        self._available_algorithms = discover_algorithms(self.project_root)

        self._build_ui()

        self.canvas.set_hover_callback(self._on_hover)
        self.canvas.set_annotation_change_callback(self._on_annotations_changed)
        self.canvas.set_before_modify_callback(self._push_undo)
        self.canvas._tth_map = self.tth
        self.canvas.mode = "view"

        self.canvas.annotations = self._load_annotations_for_bin()
        self._init_peak_controls()
        QTimer.singleShot(200, self._initial_load)

    # ----- Data loading -----

    def _load_project_data(self):
        tth_path = self.dm.tth_map()
        self.tth = tifffile.imread(str(tth_path)).astype(np.float64)

        self.tth_edges, self.tth_centers, self.n_tth_bins, \
            self.tth_bin_indices, self.tth_radial_counts = compute_tth_binning(self.tth)

        order = np.argsort(self.tth_bin_indices)
        sorted_idx = self.tth_bin_indices[order]
        boundaries = np.searchsorted(sorted_idx, np.arange(self.n_tth_bins + 1))
        self.tth_data = {
            "edges": self.tth_edges, "centers": self.tth_centers,
            "n_bins": self.n_tth_bins, "indices": self.tth_bin_indices,
            "counts": self.tth_radial_counts, "valid_mask": self.tth_radial_counts > 50,
            "order": order, "boundaries": boundaries,
        }

        self._load_bin_data_for_size(self.bin_size)

    def _ensure_raw_grid(self):
        """Lazily load raw scan grid data (shared across all fallback modes)."""
        if hasattr(self, "xrd_files"):
            return
        xrd_dir = self.dm.xrd_frames_dir()
        pos_csv = self.dm.position_csv()
        self.xrd_files, self.xrd_file_map, self.n_total = load_xrd_metadata(
            xrd_dir, scan_number=self._scan_number)
        frame_x = load_positions(pos_csv, self.n_total)
        grid_row, grid_col, self._n_rows, self._n_cols = build_scan_grid(frame_x, self.n_total)
        self._grid_to_frames = {}
        for gi in range(self.n_total):
            key = (grid_row[gi], grid_col[gi])
            self._grid_to_frames.setdefault(key, []).append(gi)

    def _load_bin_data_for_size(self, bin_size):
        """Load bin keys and mapping for the given bin size."""
        self.bin_size = bin_size
        resolved = self.dm.bins_h5(bin_size) if bin_size != 1 else None
        self.bins_h5_path = str(resolved) if resolved else None

        if self.bins_h5_path and os.path.exists(self.bins_h5_path):
            with h5py.File(self.bins_h5_path, "r") as f:
                raw_keys = list(f.keys())
            self.bin_keys = sorted(raw_keys, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1])))
            self.n_bins = len(self.bin_keys)
            self.bin_mapping = None
        elif bin_size == 1:
            self._ensure_raw_grid()
            self.bin_mapping = {}
            for (r, c), frames in self._grid_to_frames.items():
                self.bin_mapping[(r, c)] = frames
            self.bin_keys = sorted(self.bin_mapping.keys())
            self.n_bins = len(self.bin_keys)
        else:
            self._ensure_raw_grid()
            mapping, n_br, n_bc = build_bin_mapping(
                self._n_rows, self._n_cols, bin_size, self._grid_to_frames)
            self.bin_mapping = mapping
            self.bin_keys = sorted(mapping.keys())
            self.n_bins = len(self.bin_keys)

        results_dir = self.dm.results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        self._annotations_path = results_dir / f"bin_annotations_{bin_size}x{bin_size}.json"
        self._all_annotations = {}
        if self._annotations_path.exists():
            with open(self._annotations_path) as f:
                self._all_annotations = json.load(f)
        else:
            old_path = results_dir / "bin_annotations.json"
            if bin_size == 5 and old_path.exists():
                with open(old_path) as f:
                    self._all_annotations = json.load(f)

        for key, ref_dict in self._all_annotations.items():
            has_points = any(
                k != "__reviewed__" and isinstance(v, list) and len(v) > 0
                for k, v in ref_dict.items()
            )
            if has_points and not ref_dict.get("__reviewed__"):
                ref_dict["__reviewed__"] = True

        self._image_cache = OrderedDict()
        self._noise_cache = {}

    # ----- UI construction -----

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: canvas
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = LabelCanvas(left)
        self.toolbar = NavigationToolbar2QT(self.canvas, left)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 4px; font-family: monospace;")
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        # Right: controls
        right = QWidget()
        right.setMaximumWidth(400)
        right.setMinimumWidth(320)
        right_layout = QVBoxLayout(right)

        # --- Bin Size Selector ---
        bin_group = QGroupBox("Bin Size")
        bin_layout = QHBoxLayout()
        self.bin_size_combo = QComboBox()
        for bs in BIN_SIZES:
            self.bin_size_combo.addItem(BIN_CONFIGS[bs]["label"], bs)
        self.bin_size_combo.setCurrentIndex(BIN_SIZES.index(self.bin_size))
        self.bin_size_combo.currentIndexChanged.connect(self._on_bin_size_changed)
        bin_layout.addWidget(self.bin_size_combo)
        self.bin_size_status = QLabel("")
        bin_layout.addWidget(self.bin_size_status)
        bin_group.setLayout(bin_layout)
        right_layout.addWidget(bin_group)

        # --- Navigation ---
        nav_group = QGroupBox("Image Navigation")
        nav_layout = QGridLayout()

        nav_layout.addWidget(QLabel("Index:"), 0, 0)
        self.idx_spin = QSpinBox()
        self.idx_spin.setRange(0, self.n_bins - 1)
        self.idx_spin.setValue(0)
        self.idx_spin.valueChanged.connect(self._on_idx_changed)
        nav_layout.addWidget(self.idx_spin, 0, 1)

        self.bin_label = QLabel(f"of {self.n_bins} bins")
        nav_layout.addWidget(self.bin_label, 0, 2)

        btn_layout = QHBoxLayout()
        prev_btn = QPushButton("< Prev")
        prev_btn.clicked.connect(self._prev_image)
        btn_layout.addWidget(prev_btn)
        next_btn = QPushButton("Next >")
        next_btn.clicked.connect(self._next_image)
        btn_layout.addWidget(next_btn)
        nav_layout.addLayout(btn_layout, 1, 0, 1, 3)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("font-size: 10px; color: #aaa;")
        nav_layout.addWidget(self.info_label, 2, 0, 1, 3)

        review_layout = QHBoxLayout()
        self.mark_review_btn = QPushButton("Mark Reviewed")
        self.mark_review_btn.clicked.connect(self._toggle_reviewed)
        review_layout.addWidget(self.mark_review_btn)
        next_unrev_btn = QPushButton("Next Reviewed")
        next_unrev_btn.clicked.connect(self._next_reviewed)
        review_layout.addWidget(next_unrev_btn)
        nav_layout.addLayout(review_layout, 3, 0, 1, 3)

        self.review_label = QLabel("")
        nav_layout.addWidget(self.review_label, 4, 0, 1, 3)

        nav_group.setLayout(nav_layout)
        right_layout.addWidget(nav_group)

        # --- Colormap & Contrast ---
        vis_group = QGroupBox("Visualization")
        vis_layout = QGridLayout()

        vis_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText("inferno")
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        vis_layout.addWidget(self.cmap_combo, 0, 1, 1, 2)

        self.reverse_cb = QCheckBox("Reverse")
        self.reverse_cb.toggled.connect(self._on_cmap_changed)
        vis_layout.addWidget(self.reverse_cb, 0, 3)

        self.log_cb = QCheckBox("Log scale")
        self.log_cb.toggled.connect(self._on_log_toggle)
        vis_layout.addWidget(self.log_cb, 1, 0, 1, 2)

        vis_layout.addWidget(QLabel("Min %:"), 2, 0)
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 1000)
        self.vmin_slider.setValue(20)
        self.vmin_slider.valueChanged.connect(self._on_contrast_changed)
        vis_layout.addWidget(self.vmin_slider, 2, 1, 1, 2)
        self.vmin_val = QLineEdit("2.0")
        self.vmin_val.setFixedWidth(45)
        self.vmin_val.editingFinished.connect(self._on_vmin_text)
        vis_layout.addWidget(self.vmin_val, 2, 3)

        vis_layout.addWidget(QLabel("Max %:"), 3, 0)
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setRange(0, 1000)
        self.vmax_slider.setValue(995)
        self.vmax_slider.valueChanged.connect(self._on_contrast_changed)
        vis_layout.addWidget(self.vmax_slider, 3, 1, 1, 2)
        self.vmax_val = QLineEdit("99.5")
        self.vmax_val.setFixedWidth(45)
        self.vmax_val.editingFinished.connect(self._on_vmax_text)
        vis_layout.addWidget(self.vmax_val, 3, 3)

        preset_layout = QHBoxLayout()
        for name, lo, hi in [("Full", 0, 1000), ("Auto", 20, 985),
                              ("Tight", 50, 995), ("High", 100, 999)]:
            btn = QPushButton(name)
            btn.setFixedWidth(55)
            lo_val, hi_val = lo, hi
            btn.clicked.connect(lambda _, l=lo_val, h=hi_val: self._set_contrast_preset(l, h))
            preset_layout.addWidget(btn)
        vis_layout.addLayout(preset_layout, 4, 0, 1, 4)

        vis_group.setLayout(vis_layout)
        right_layout.addWidget(vis_group)

        # --- Noise Reduction ---
        noise_group = QGroupBox("Noise Reduction")
        noise_layout = QGridLayout()

        self.noise_cb = QCheckBox("Enable noise reduction")
        self.noise_cb.toggled.connect(self._on_noise_toggle)
        noise_layout.addWidget(self.noise_cb, 0, 0, 1, 3)

        self.noise_algo_label = QLabel("Algorithm:")
        noise_layout.addWidget(self.noise_algo_label, 1, 0)
        self.noise_algo_combo = QComboBox()
        for key in ALGORITHM_NAMES:
            self.noise_algo_combo.addItem(ALGORITHM_DISPLAY[key], key)
        self.noise_algo_combo.currentIndexChanged.connect(self._on_noise_algo_changed)
        noise_layout.addWidget(self.noise_algo_combo, 1, 1, 1, 2)

        self.strength_label = QLabel("Strength:")
        noise_layout.addWidget(self.strength_label, 2, 0)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.valueChanged.connect(self._on_noise_param_changed)
        noise_layout.addWidget(self.strength_slider, 2, 1)
        self.strength_val = QLineEdit("1.00")
        self.strength_val.setFixedWidth(45)
        self.strength_val.editingFinished.connect(self._on_strength_text)
        noise_layout.addWidget(self.strength_val, 2, 2)

        self.shift_label = QLabel("Shift:")
        noise_layout.addWidget(self.shift_label, 3, 0)
        self.shift_slider = QSlider(Qt.Horizontal)
        self.shift_slider.setRange(-500, 500)
        self.shift_slider.setValue(0)
        self.shift_slider.valueChanged.connect(self._on_noise_param_changed)
        noise_layout.addWidget(self.shift_slider, 3, 1)
        self.shift_val = QLineEdit("0.0")
        self.shift_val.setFixedWidth(45)
        self.shift_val.editingFinished.connect(self._on_shift_text)
        noise_layout.addWidget(self.shift_val, 3, 2)

        self._noise_widgets = [
            self.noise_algo_label, self.noise_algo_combo,
            self.strength_label, self.strength_slider, self.strength_val,
            self.shift_label, self.shift_slider, self.shift_val,
        ]
        for w in self._noise_widgets:
            w.setVisible(False)

        noise_group.setLayout(noise_layout)
        right_layout.addWidget(noise_group)

        # --- Peak Detection ---
        peak_group = QGroupBox("Peak Detection")
        peak_layout = QGridLayout()
        row = 0

        peak_layout.addWidget(QLabel("Algorithm:"), row, 0)
        self.peak_algo_combo = QComboBox()
        for algo in self._available_algorithms:
            self.peak_algo_combo.addItem(algo["name"])
        self.peak_algo_combo.currentIndexChanged.connect(self._on_peak_algo_changed)
        peak_layout.addWidget(self.peak_algo_combo, row, 1, 1, 2)
        row += 1

        self.peak_source_label = QLabel("")
        self.peak_source_label.setStyleSheet("font-size: 10px; color: #aaa; font-style: italic;")
        peak_layout.addWidget(self.peak_source_label, row, 0, 1, 3)
        row += 1

        self.peak_use_filtered_cb = QCheckBox("Run on displayed image (filtered/clipped)")
        self.peak_use_filtered_cb.setChecked(False)
        self.peak_use_filtered_cb.toggled.connect(lambda: None)
        peak_layout.addWidget(self.peak_use_filtered_cb, row, 0, 1, 3)
        row += 1

        # Sensitivity slider (percentile for base, SNR for CVEvolve)
        self.peak_sens_label = QLabel("Sensitivity:")
        peak_layout.addWidget(self.peak_sens_label, row, 0)
        self.peak_sens_slider = QSlider(Qt.Horizontal)
        self.peak_sens_slider.setRange(0, 100)
        self.peak_sens_slider.setValue(50)
        self.peak_sens_slider.setTickInterval(10)
        self.peak_sens_slider.valueChanged.connect(self._on_peak_param_changed)
        peak_layout.addWidget(self.peak_sens_slider, row, 1)
        self.peak_sens_val = QLineEdit("50")
        self.peak_sens_val.setFixedWidth(45)
        self.peak_sens_val.editingFinished.connect(self._on_peak_sens_text)
        peak_layout.addWidget(self.peak_sens_val, row, 2)
        row += 1

        # 2-theta band width slider
        self.peak_band_label = QLabel("Band width:")
        peak_layout.addWidget(self.peak_band_label, row, 0)
        self.peak_band_slider = QSlider(Qt.Horizontal)
        self.peak_band_slider.setRange(5, 100)
        self.peak_band_slider.setValue(40)
        self.peak_band_slider.setTickInterval(10)
        self.peak_band_slider.valueChanged.connect(self._on_peak_param_changed)
        peak_layout.addWidget(self.peak_band_slider, row, 1)
        self.peak_band_val = QLineEdit("0.40")
        self.peak_band_val.setFixedWidth(45)
        self.peak_band_val.editingFinished.connect(self._on_peak_band_text)
        peak_layout.addWidget(self.peak_band_val, row, 2)
        row += 1

        # Restrict to 2-theta bands checkbox (base algorithm only)
        self.peak_bands_only_cb = QCheckBox("2θ bands only")
        self.peak_bands_only_cb.setChecked(False)
        self.peak_bands_only_cb.toggled.connect(lambda: None)
        peak_layout.addWidget(self.peak_bands_only_cb, row, 0, 1, 3)
        row += 1

        peak_btn_layout = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Peaks")
        self.detect_btn.clicked.connect(self._run_peak_detection)
        peak_btn_layout.addWidget(self.detect_btn)
        self.arc_btn = QPushButton("Show 2θ Arcs")
        self.arc_btn.setCheckable(True)
        self.arc_btn.toggled.connect(self._toggle_arcs)
        peak_btn_layout.addWidget(self.arc_btn)
        self.clear_outlines_btn = QPushButton("Clear")
        self.clear_outlines_btn.clicked.connect(self._clear_detections)
        peak_btn_layout.addWidget(self.clear_outlines_btn)
        peak_layout.addLayout(peak_btn_layout, row, 0, 1, 3)
        row += 1

        self.det_count_label = QLabel("")
        peak_layout.addWidget(self.det_count_label, row, 0, 1, 3)

        peak_group.setLayout(peak_layout)
        right_layout.addWidget(peak_group)

        # --- Labeling Mode Toggle ---
        self.labeling_btn = QPushButton("Enable Labeling")
        self.labeling_btn.setCheckable(True)
        self.labeling_btn.setStyleSheet(
            "font-weight: bold; padding: 8px; background-color: #2563eb; color: white;")
        self.labeling_btn.toggled.connect(self._toggle_labeling_mode)
        right_layout.addWidget(self.labeling_btn)

        # --- Tools ---
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout()

        # Mode buttons
        mode_layout = QHBoxLayout()
        self.click_btn = QPushButton("Click (Local Max)")
        self.click_btn.setCheckable(True)
        self.click_btn.setChecked(True)
        self.click_btn.clicked.connect(lambda: self._set_mode("click"))
        mode_layout.addWidget(self.click_btn)

        self.drag_btn = QPushButton("Drag Select")
        self.drag_btn.setCheckable(True)
        self.drag_btn.clicked.connect(lambda: self._set_mode("drag"))
        mode_layout.addWidget(self.drag_btn)
        tools_layout.addLayout(mode_layout)

        # Outline select + arc overlay
        action_layout = QHBoxLayout()
        self.outline_select_btn = QPushButton("Select Outline")
        self.outline_select_btn.setCheckable(True)
        self.outline_select_btn.clicked.connect(lambda: self._set_mode("outline_select"))
        action_layout.addWidget(self.outline_select_btn)
        tools_layout.addLayout(action_layout)


        # Select & Delete
        del_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select Mode")
        self.select_btn.setCheckable(True)
        self.select_btn.clicked.connect(lambda: self._set_mode("select_delete"))
        del_layout.addWidget(self.select_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        del_layout.addWidget(self.delete_btn)
        tools_layout.addLayout(del_layout)

        # Undo/Redo
        undo_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self._undo)
        undo_layout.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self._redo)
        undo_layout.addWidget(self.redo_btn)
        tools_layout.addLayout(undo_layout)

        tools_group.setLayout(tools_layout)
        right_layout.addWidget(tools_group)

        # --- Annotations list ---
        ann_group = QGroupBox("Annotations")
        ann_layout = QVBoxLayout()

        self.ann_list = QListWidget()
        self.ann_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.ann_list.itemSelectionChanged.connect(self._on_ann_list_selection)
        ann_layout.addWidget(self.ann_list)

        self.ann_count_label = QLabel("0 annotations")
        ann_layout.addWidget(self.ann_count_label)

        save_layout = QHBoxLayout()
        save_btn = QPushButton("Save All")
        save_btn.clicked.connect(self._save_annotations)
        save_layout.addWidget(save_btn)
        export_btn = QPushButton("Export CVEvolve JSON")
        export_btn.clicked.connect(self._export_cvevolve)
        save_layout.addWidget(export_btn)
        holdout_btn = QPushButton("Update Holdout Set")
        holdout_btn.setToolTip("Copy reviewed annotations to CVEvolve holdout_data/bin_annotations.json")
        holdout_btn.clicked.connect(self._update_holdout_set)
        save_layout.addWidget(holdout_btn)
        ann_layout.addLayout(save_layout)

        ann_group.setLayout(ann_layout)
        right_layout.addWidget(ann_group)

        # Hide labeling widgets by default
        self._tools_group = tools_group
        self._ann_group = ann_group
        tools_group.setVisible(False)
        ann_group.setVisible(False)

        right_layout.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([1200, 400])

    # ----- Mode management -----

    def _set_mode(self, mode):
        if not self._labeling_enabled and mode != "view":
            return
        self.canvas.mode = mode
        self.click_btn.setChecked(mode == "click")
        self.drag_btn.setChecked(mode == "drag")
        self.outline_select_btn.setChecked(mode == "outline_select")
        self.select_btn.setChecked(mode == "select_delete")
        mode_names = {
            "click": "Click to find local peak",
            "drag": "Click and drag to select region",
            "outline_select": "Click an outline to add annotation",
            "select_delete": "Click an annotation to select, then Delete",
            "view": "View only (labeling disabled)",
        }
        self.status_label.setText(f"Mode: {mode_names.get(mode, mode)}")

    # ----- Bin size change -----

    def _current_grid_center(self):
        """Return the grid-space center (row, col) of the current bin."""
        bk = self.bin_keys[self._current_bin_idx]
        br, bc = self._bin_key_parts(bk)
        bs = self.bin_size
        return br * bs + bs / 2.0, bc * bs + bs / 2.0

    def _find_closest_bin_idx(self, grid_row, grid_col):
        """Find the bin index whose center is closest to (grid_row, grid_col)."""
        bs = self.bin_size
        best_idx = 0
        best_dist = float("inf")
        for i, bk in enumerate(self.bin_keys):
            br, bc = self._bin_key_parts(bk)
            cr = br * bs + bs / 2.0
            cc = bc * bs + bs / 2.0
            dist = (cr - grid_row) ** 2 + (cc - grid_col) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _on_bin_size_changed(self, idx):
        new_size = self.bin_size_combo.currentData()
        if new_size is None or new_size == self.bin_size or not self._init_done:
            return

        gr, gc = self._current_grid_center()

        self._save_current_bin_annotations()
        self._save_annotations()
        self.canvas.clear_outlines()

        self._load_bin_data_for_size(new_size)

        target_idx = self._find_closest_bin_idx(gr, gc)

        self.idx_spin.blockSignals(True)
        self.idx_spin.setRange(0, max(0, self.n_bins - 1))
        self.idx_spin.setValue(target_idx)
        self.idx_spin.blockSignals(False)
        self.bin_label.setText(f"of {self.n_bins} bins")
        self._current_bin_idx = target_idx

        h5_exists = self.bins_h5_path and os.path.exists(self.bins_h5_path)
        label = BIN_CONFIGS[new_size]["label"]
        self.bin_size_status.setText(
            f"{self.n_bins} bins" + ("" if h5_exists else " (raw frames)"))

        self.setWindowTitle(f"XRD {label} Bin Analysis & Labeling Tool")
        self.canvas.annotations = self._load_annotations_for_bin()
        self.canvas.selected_annotations = set()
        self._load_and_display()

    # ----- Labeling mode -----

    def _toggle_labeling_mode(self, enabled):
        self._labeling_enabled = enabled
        self._tools_group.setVisible(enabled)
        self._ann_group.setVisible(enabled)
        self.labeling_btn.setText(
            "Disable Labeling" if enabled else "Enable Labeling")
        if not enabled:
            self.canvas.mode = "view"
            self.status_label.setText("Mode: View only (labeling disabled)")
        else:
            self._set_mode("click")

    # ----- Peak detection -----

    def _init_peak_controls(self):
        """Set initial visibility of peak controls without running detection."""
        idx = self.peak_algo_combo.currentIndex()
        if idx < 0 or idx >= len(self._available_algorithms):
            return
        algo = self._available_algorithms[idx]
        is_base = algo["func_type"] == "base"
        self.peak_bands_only_cb.setVisible(is_base)
        self.peak_band_label.setVisible(not is_base)
        self.peak_band_slider.setVisible(not is_base)
        self.peak_band_val.setVisible(not is_base)
        if is_base:
            self.peak_sens_label.setText("Percentile:")
            self.peak_sens_slider.setRange(800, 999)
            self.peak_sens_slider.setValue(970)
            self.peak_sens_val.setText("97.0")
        else:
            self.peak_sens_label.setText("SNR threshold:")
            self.peak_sens_slider.setRange(10, 100)
            self.peak_sens_slider.setValue(40)
            self.peak_sens_val.setText("4.0")

    def _initial_load(self):
        self._init_done = True
        self._load_and_display()

    def _on_peak_algo_changed(self, idx):
        if idx < 0 or idx >= len(self._available_algorithms):
            self.peak_source_label.setText("")
            return
        algo = self._available_algorithms[idx]
        is_base = algo["func_type"] == "base"
        self.peak_source_label.setText("" if is_base else f"Tuned from {algo['source']}")

        # Base: sensitivity = percentile (90-99.5), band restriction optional
        # CVEvolve: sensitivity = SNR threshold (1-10), band width always used
        self.peak_bands_only_cb.setVisible(is_base)
        if is_base:
            self.peak_sens_label.setText("Percentile:")
            self.peak_sens_slider.setRange(800, 999)
            self.peak_sens_slider.setValue(970)
            self.peak_sens_val.setText("97.0")
            self.peak_band_label.setVisible(False)
            self.peak_band_slider.setVisible(False)
            self.peak_band_val.setVisible(False)
        else:
            self.peak_sens_label.setText("SNR threshold:")
            self.peak_sens_slider.setRange(10, 100)
            self.peak_sens_slider.setValue(40)
            self.peak_sens_val.setText("4.0")
            self.peak_band_label.setVisible(True)
            self.peak_band_slider.setVisible(True)
            self.peak_band_val.setVisible(True)
            self.peak_band_slider.setValue(40)
            self.peak_band_val.setText("0.40")

    def _on_peak_param_changed(self):
        idx = self.peak_algo_combo.currentIndex()
        if idx < 0 or idx >= len(self._available_algorithms):
            return
        algo = self._available_algorithms[idx]
        if algo["func_type"] == "base":
            val = self.peak_sens_slider.value() / 10.0
            self.peak_sens_val.setText(f"{val:.1f}")
        else:
            val = self.peak_sens_slider.value() / 10.0
            self.peak_sens_val.setText(f"{val:.1f}")
            band = self.peak_band_slider.value() / 100.0
            self.peak_band_val.setText(f"{band:.2f}")

    def _on_peak_sens_text(self):
        try:
            val = float(self.peak_sens_val.text())
            self.peak_sens_slider.setValue(int(val * 10))
        except ValueError:
            pass

    def _on_peak_band_text(self):
        try:
            val = float(self.peak_band_val.text())
            self.peak_band_slider.setValue(int(val * 100))
        except ValueError:
            pass

    def _run_peak_detection(self):
        if not self._init_done:
            return
        idx = self.peak_algo_combo.currentIndex()
        if idx < 0 or idx >= len(self._available_algorithms):
            return

        algo = self._available_algorithms[idx]

        if self.peak_use_filtered_cb.isChecked():
            img = self._get_display_image()
            if self.canvas.log_scale:
                img = img.copy()
                img[img < 0] = 0
                img = np.log1p(img)
            if self.canvas.vmin is not None and self.canvas.vmax is not None:
                img = np.clip(img, self.canvas.vmin, self.canvas.vmax)
        else:
            img = self._load_current_image()
        if img is None:
            return

        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting...")
        self.status_label.setText("Running peak detection...")
        QApplication.processEvents()

        pad = 10
        try:
            if algo["func_type"] == "base":
                percentile = self.peak_sens_slider.value() / 10.0
                detected = find_peaks_on_image(
                    img, self.tth, DEGS, DEG_LABELS, percentile=percentile)
                if self.peak_bands_only_cb.isChecked():
                    band_tol = 0.5
                    filtered = {}
                    for lab, peaks in detected.items():
                        kept = []
                        for peak in peaks:
                            y0, y1, x0, x1, cx, cy = peak
                            tth_val = self.tth[cy, cx] if 0 <= cy < self.tth.shape[0] and 0 <= cx < self.tth.shape[1] else None
                            if tth_val is not None and any(abs(tth_val - d) < band_tol for d in DEGS):
                                kept.append(peak)
                        if kept:
                            filtered[lab] = kept
                    detected = filtered
            else:
                snr = self.peak_sens_slider.value() / 10.0
                band_width = self.peak_band_slider.value() / 100.0
                raw = run_cvevolve_algorithm(
                    algo, img, self.tth, DEGS, DEG_LABELS, self.tth_data,
                    snr_threshold=snr, tth_tolerance=band_width,
                    band_half_width=band_width)
                h, w = img.shape[:2]
                detected = {}
                for lab, points in (raw or {}).items():
                    boxes = []
                    for pt in points:
                        cx, cy = int(pt[0]), int(pt[1])
                        y0 = max(cy - pad, 0)
                        y1 = min(cy + pad + 1, h)
                        x0 = max(cx - pad, 0)
                        x1 = min(cx + pad + 1, w)
                        boxes.append((y0, y1, x0, x1, cx, cy))
                    detected[lab] = boxes

            self.canvas.clear_outlines()
            if detected:
                self.canvas.show_outlines(detected)

            total = sum(len(pts) for pts in detected.values())
            self.det_count_label.setText(f"{total} peaks detected")
            self.status_label.setText(f"Detection complete: {total} peaks")
        except Exception as e:
            self.status_label.setText(f"Detection error: {e}")
            self.det_count_label.setText("")
        finally:
            self.detect_btn.setEnabled(True)
            self.detect_btn.setText("Detect Peaks")

    def _clear_detections(self):
        self.canvas.clear_outlines()
        self.canvas.draw_idle()
        self.det_count_label.setText("")

    # ----- Image loading -----

    def _load_current_image(self):
        bk = self.bin_keys[self._current_bin_idx]
        bin_key_str = bk if isinstance(bk, str) else f"{bk[0]}_{bk[1]}"

        if bin_key_str in self._image_cache:
            self._image_cache.move_to_end(bin_key_str)
            return self._image_cache[bin_key_str]

        img = None
        if self.bins_h5_path and os.path.exists(self.bins_h5_path):
            with h5py.File(self.bins_h5_path, "r") as f:
                if bin_key_str in f:
                    img = f[bin_key_str][:].astype(np.float64)
        elif self.bin_mapping is not None:
            frames = self.bin_mapping[bk]
            img = load_and_sum_frames(frames, self.xrd_files, self.xrd_file_map)

        self._image_cache[bin_key_str] = img
        while len(self._image_cache) > IMAGE_CACHE_MAX:
            self._image_cache.popitem(last=False)

        return img

    def _get_display_image(self):
        raw = self._load_current_image()
        if raw is None:
            return np.zeros((100, 100))

        if self.noise_cb.isChecked():
            algo = self.noise_algo_combo.currentData()
            strength = self.strength_slider.value() / 100.0
            shift = self.shift_slider.value() / 10.0
            bk = self.bin_keys[self._current_bin_idx]
            cache_key = (self._bin_key_str(bk), algo)

            if cache_key not in self._noise_cache:
                valid = self.tth_radial_counts > 50
                profile = compute_radial_profile(raw, self.tth_bin_indices, self.n_tth_bins)
                fits = fit_all_models(self.tth_centers, profile, valid,
                                      self.tth_edges[0], self.tth_edges[-1])
                self._noise_cache[cache_key] = fits

            fits = self._noise_cache[cache_key]
            if algo in fits:
                bg = build_background_image(
                    self.tth, self.tth_centers, fits[algo]["profile"],
                    self.tth_bin_indices)
                cleaned = subtract_background(raw, bg, strength=strength, shift=shift)
                return np.clip(cleaned, 0, None)

        return raw

    def _bin_key_str(self, bk):
        return bk if isinstance(bk, str) else f"{bk[0]}_{bk[1]}"

    def _bin_key_parts(self, bk):
        if isinstance(bk, str):
            parts = bk.split("_")
            return int(parts[0]), int(parts[1])
        return bk[0], bk[1]

    def _load_and_display(self):
        bk = self.bin_keys[self._current_bin_idx]
        bk_str = self._bin_key_str(bk)
        br, bc = self._bin_key_parts(bk)
        img = self._get_display_image()

        self.canvas.vmin = None
        self.canvas.vmax = None

        n_frames = len(self.bin_mapping[bk]) if self.bin_mapping else 0
        frames_str = f"  |  {n_frames} frames" if n_frames else ""
        label = BIN_CONFIGS[self.bin_size]["label"]
        if self.bin_size == 1:
            title = f"Raw ({br}, {bc})  |  Index {self._current_bin_idx}{frames_str}"
        else:
            title = f"Bin ({br}, {bc})  |  {label}  |  Index {self._current_bin_idx}{frames_str}"
        self.canvas.display_image(img, title=title)

        self.info_label.setText(f"Bin ({br},{bc})  row={br}  col={bc}")

        self._apply_contrast()
        self._update_ann_list()
        self._update_review_status()

        if self.arc_btn.isChecked():
            self.canvas.show_arcs(self.tth, DEGS, DEG_LABELS)

    def _load_annotations_for_bin(self):
        bk = self.bin_keys[self._current_bin_idx]
        key = self._bin_key_str(bk)
        if key in self._all_annotations:
            ann_data = self._all_annotations[key]
            anns = []
            for ref, points in ann_data.items():
                if ref == "__reviewed__":
                    continue
                for pt in points:
                    anns.append({"x": pt[0], "y": pt[1], "reflection": ref})
            return anns
        return []

    def _save_current_bin_annotations(self):
        bk = self.bin_keys[self._current_bin_idx]
        key = self._bin_key_str(bk)
        by_ref = {}
        for ann in self.canvas.annotations:
            ref = ann.get("reflection", "unknown") or "unknown"
            by_ref.setdefault(ref, []).append([ann["x"], ann["y"]])

        was_reviewed = (key in self._all_annotations and
                        self._all_annotations.get(key, {}).get("__reviewed__"))

        if by_ref or was_reviewed:
            existing = self._all_annotations.get(key, {})
            reviewed = existing.get("__reviewed__", False)
            by_ref["__reviewed__"] = reviewed or bool(by_ref)
            self._all_annotations[key] = by_ref
        elif key in self._all_annotations:
            del self._all_annotations[key]

    def _is_bin_reviewed(self, bin_idx):
        bk = self.bin_keys[bin_idx]
        key = self._bin_key_str(bk)
        return key in self._all_annotations and self._all_annotations[key].get("__reviewed__", False)

    def _toggle_reviewed(self):
        bk = self.bin_keys[self._current_bin_idx]
        key = self._bin_key_str(bk)
        if key not in self._all_annotations:
            self._all_annotations[key] = {}
        currently_reviewed = self._all_annotations[key].get("__reviewed__", False)
        self._all_annotations[key]["__reviewed__"] = not currently_reviewed
        self._save_current_bin_annotations()
        self._update_review_status()

    def _update_review_status(self):
        reviewed = self._is_bin_reviewed(self._current_bin_idx)
        n_reviewed = sum(1 for i in range(self.n_bins) if self._is_bin_reviewed(i))
        status = "REVIEWED" if reviewed else "unreviewed"
        self.review_label.setText(f"{status}  |  {n_reviewed}/{self.n_bins} reviewed")
        self.review_label.setStyleSheet(
            "font-weight: bold; color: #2ecc71;" if reviewed
            else "color: #e74c3c;"
        )
        if reviewed:
            self.mark_review_btn.setText("Mark Unreviewed")
            self.mark_review_btn.setStyleSheet(
                "background-color: #e74c3c; color: white; font-weight: bold;")
        else:
            self.mark_review_btn.setText("Mark Reviewed")
            self.mark_review_btn.setStyleSheet(
                "background-color: #2ecc71; color: white; font-weight: bold;")

    def _next_reviewed(self):
        for offset in range(1, self.n_bins):
            idx = (self._current_bin_idx + offset) % self.n_bins
            if self._is_bin_reviewed(idx):
                self.idx_spin.setValue(idx)
                return
        self.status_label.setText("No other reviewed bins found.")

    # ----- Navigation -----

    def _on_idx_changed(self, val):
        self._save_current_bin_annotations()
        self._current_bin_idx = val
        self.canvas.annotations = self._load_annotations_for_bin()
        self.canvas.selected_annotations = set()
        self.canvas.clear_outlines()
        self.det_count_label.setText("")
        self._load_and_display()

    def _prev_image(self):
        if self._current_bin_idx > 0:
            self.idx_spin.setValue(self._current_bin_idx - 1)

    def _next_image(self):
        if self._current_bin_idx < self.n_bins - 1:
            self.idx_spin.setValue(self._current_bin_idx + 1)

    # ----- Visualization controls -----

    def _on_cmap_changed(self, *_):
        name = self.cmap_combo.currentText()
        if self.reverse_cb.isChecked():
            name += "_r"
        self.canvas.update_cmap(name)

    def _on_log_toggle(self, checked):
        self.canvas.set_log_scale(checked)
        self._apply_contrast()

    def _on_contrast_changed(self, *_):
        self._apply_contrast()

    def _apply_contrast(self):
        lo = self.vmin_slider.value() / 10.0
        hi = self.vmax_slider.value() / 10.0
        self.vmin_val.setText(f"{lo:.1f}")
        self.vmax_val.setText(f"{hi:.1f}")

        data = self.canvas.display_data
        if data is not None:
            finite = data[np.isfinite(data)]
            if len(finite) > 0:
                vmin = float(np.percentile(finite, lo))
                vmax = float(np.percentile(finite, hi))
                self.canvas.update_contrast(vmin, vmax)

    def _set_contrast_preset(self, lo, hi):
        self.vmin_slider.setValue(lo)
        self.vmax_slider.setValue(hi)

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

    # ----- Noise reduction -----

    def _on_noise_toggle(self, checked):
        for w in self._noise_widgets:
            w.setVisible(checked)
        self._load_and_display()

    def _on_noise_algo_changed(self, *_):
        if self.noise_cb.isChecked():
            self._load_and_display()

    def _on_noise_param_changed(self, *_):
        s = self.strength_slider.value() / 100.0
        self.strength_val.setText(f"{s:.2f}")
        sh = self.shift_slider.value() / 10.0
        self.shift_val.setText(f"{sh:.1f}")
        if self.noise_cb.isChecked():
            self._load_and_display()

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

    # ----- Tools -----

    def _auto_outline(self):
        img = self._get_display_image()
        if img is None:
            return
        detected = find_peaks_on_image(img, self.tth, DEGS, DEG_LABELS)
        total = sum(len(v) for v in detected.values())
        self.canvas.show_outlines(detected)
        self.status_label.setText(
            f"Auto outline: {total} peaks detected. "
            f"Click 'Select Outline' then click a box to add it."
        )

    def _toggle_arcs(self, checked):
        if checked:
            self.canvas.show_arcs(self.tth, DEGS, DEG_LABELS)
        else:
            self.canvas.clear_arcs()
        self.canvas.draw_idle()

    def _delete_selected(self):
        sel = self.canvas.selected_annotations
        if not sel:
            return
        self._push_undo()
        self.canvas.annotations = [
            a for i, a in enumerate(self.canvas.annotations) if i not in sel]
        self.canvas.selected_annotations = set()
        self.canvas.refresh_display()
        self._on_annotations_changed()

    def _push_undo(self):
        snapshot = copy.deepcopy(self.canvas.annotations)
        self._undo_stack.append(snapshot)
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(copy.deepcopy(self.canvas.annotations))
        self.canvas.annotations = self._undo_stack.pop()
        self.canvas.selected_annotations = set()
        self.canvas.refresh_display()
        self._on_annotations_changed()

    def _redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(copy.deepcopy(self.canvas.annotations))
        self.canvas.annotations = self._redo_stack.pop()
        self.canvas.selected_annotations = set()
        self.canvas.refresh_display()
        self._on_annotations_changed()

    # ----- Annotation list -----

    def _update_ann_list(self):
        self.ann_list.clear()
        for i, ann in enumerate(self.canvas.annotations):
            ref = ann.get("reflection", "?")
            item = QListWidgetItem(f"[{i}] ({ann['x']}, {ann['y']})  {ref}")
            self.ann_list.addItem(item)
        self.ann_count_label.setText(f"{len(self.canvas.annotations)} annotations")

    def _on_ann_list_selection(self):
        rows = {idx.row() for idx in self.ann_list.selectedIndexes()}
        self.canvas.selected_annotations = rows
        self.canvas._redraw_annotations()
        self.canvas.draw_idle()

    def _on_annotations_changed(self):
        self._update_ann_list()
        if self.canvas.annotations:
            bk = self.bin_keys[self._current_bin_idx]
            key = self._bin_key_str(bk)
            if key not in self._all_annotations:
                self._all_annotations[key] = {}
            self._all_annotations[key]["__reviewed__"] = True
            self._save_current_bin_annotations()
            self._update_review_status()

    def _on_hover(self, x, y, val):
        if x is None:
            self.status_label.setText("")
            return
        if val is not None:
            self.status_label.setText(f"x={x}  y={y}  I={val:.1f}")
        else:
            self.status_label.setText(f"x={x}  y={y}")

    # ----- Save / Export -----

    def _save_annotations(self):
        self._save_current_bin_annotations()
        self._annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._annotations_path, "w") as f:
            json.dump(self._all_annotations, f, indent=2)
        n = sum(
            sum(len(pts) for k, pts in ref_dict.items() if k != "__reviewed__")
            for ref_dict in self._all_annotations.values()
        )
        self.status_label.setText(
            f"Saved {n} annotations across {len(self._all_annotations)} bins "
            f"to {self._annotations_path.name}"
        )

    def _export_cvevolve(self):
        self._save_current_bin_annotations()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CVEvolve Annotations", str(self._annotations_path.parent),
            "JSON Files (*.json)")
        if not path:
            return

        merged = {}
        for _bin_key, ref_dict in self._all_annotations.items():
            for ref, points in ref_dict.items():
                if ref == "__reviewed__":
                    continue
                merged.setdefault(ref, []).extend(points)

        for lab in DEG_LABELS:
            if lab not in merged:
                merged[lab] = []

        with open(path, "w") as f:
            json.dump(merged, f, indent=2)

        n = sum(len(v) for v in merged.values())
        self.status_label.setText(f"Exported {n} points to {Path(path).name}")

    def _update_holdout_set(self):
        """Copy reviewed annotations + empty bins to CVEvolve holdout_data."""
        self._save_current_bin_annotations()

        holdout_dir = self.dm.holdout_dir
        holdout_dir.mkdir(parents=True, exist_ok=True)

        reviewed_ann = {}
        empty_bins = []
        for bk, ref_dict in self._all_annotations.items():
            if not ref_dict.get("__reviewed__", False):
                continue
            peaks = {r: pts for r, pts in ref_dict.items() if r != "__reviewed__" and pts}
            if peaks:
                reviewed_ann[bk] = peaks
            else:
                empty_bins.append(bk)

        n_ann = len(reviewed_ann)
        n_pts = sum(len(pts) for refs in reviewed_ann.values() for pts in refs.values())
        n_empty = len(empty_bins)

        reply = QMessageBox.question(
            self, "Update Holdout Set",
            f"Write to {holdout_dir.name}/:\n\n"
            f"  bin_annotations.json: {n_ann} bins, {n_pts} peaks\n"
            f"  empty_bins.json: {n_empty} empty bins\n\n"
            f"This will overwrite existing holdout labels. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        ann_path = holdout_dir / "bin_annotations.json"
        with open(ann_path, "w") as f:
            json.dump(reviewed_ann, f, indent=2)

        empty_path = holdout_dir / "empty_bins.json"
        with open(empty_path, "w") as f:
            json.dump(empty_bins, f, indent=2)

        self.status_label.setText(
            f"Holdout updated: {n_ann} bins ({n_pts} pts) + {n_empty} empty → {holdout_dir.name}/"
        )

    # ----- Cleanup -----

    def closeEvent(self, event):
        self._save_current_bin_annotations()
        if self._all_annotations:
            reply = QMessageBox.question(
                self, "Save annotations?",
                "Save annotations before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self._save_annotations()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()


# ===== Entry point =====

def launch_gui(project_root=".", scan=None):
    """Launch the labeling GUI for the given project root (used by the CLI)."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LabelingTool(project_root=project_root, scan=scan)
    window.show()
    sys.exit(app.exec_())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XRD bin analysis and labeling tool")
    parser.add_argument("--project-root", type=str, default=".",
                        help="Path to the xrd-tools project root")
    parser.add_argument("--scan", default=None, help="Scan number/name")
    args = parser.parse_args()
    launch_gui(project_root=args.project_root, scan=args.scan)


if __name__ == "__main__":
    main()
