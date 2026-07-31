"""
Device-level feature map for perovskite reflections (pyqtgraph).

Shows spatial profiles across the full bin grid with switchable metrics:
intensity, 2θ deviation (Δ2θ), crystallographic orientation, and azimuthal /
radial breadth. Chi-angle range filter with interactive histogram, arc, and slider.

pyqtgraph rewrite of the original matplotlib version (full feature parity). All
data-prep logic is framework-agnostic and unchanged; only the rendering layer is
pyqtgraph now (fast ImageItem heatmap, IsocurveItem outlines, BarGraphItem
histogram, hover/click highlight).
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
    QSizePolicy, QSpinBox, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont

from ..config import DataManager
from ..core import catalogs
from .palette import element_colors as _element_colors, hex_to_rgba as _hex_to_rgba

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

_ISO_GHOST_ALPHA = 0    # opacity of everything but the pick while isolating;
                        # 0 = hide the rest entirely so the clicked feature reads
                        # alone, like a single feature in Shape/Verify.

# Resolved at runtime by configure(); see launch_gui().
_DM = None
_BIN_SIZE = 3
RESULTS_DIR = None
HOLDOUT_DIR = None
CATALOG_PATH = None   # selected feature-map JSON; None → canonical per-bin file
GRID_PATH = None      # grid mapping matching the selected catalog's bins


def configure(project_root=".", bin_size=3, scan=None, catalog=None):
    global _DM, _BIN_SIZE, RESULTS_DIR, HOLDOUT_DIR, CATALOG_PATH, GRID_PATH
    _DM = DataManager(project_root, scan=scan)
    _BIN_SIZE = bin_size
    RESULTS_DIR = _DM.results_dir()
    HOLDOUT_DIR = _DM.holdout_dir
    CATALOG_PATH = catalog
    GRID_PATH = _resolve_grid_mapping()


def _resolve_grid_mapping():
    """Grid mapping whose bins match the selected catalog (so a catalog built on
    a non-default coordinate grid uses the right grid dimensions instead of
    clipping its out-of-range bins). Falls back to the per-bin default."""
    default = _DM.grid_mapping(bin_size=_BIN_SIZE)
    if not CATALOG_PATH:
        return default
    try:
        cand_dir = _DM.metadata_scan_dir()
        tagged = sorted(p for p in cand_dir.glob(
            f"grid_mapping_{_BIN_SIZE}x{_BIN_SIZE}*.json") if Path(p) != Path(default))
        return catalogs.best_grid_mapping([default] + tagged, CATALOG_PATH, default=default)
    except Exception:
        return default


REFLECTIONS = []
REF_COLORS = {}

_PALETTE = [
    "#4363d8", "#e6194b", "#3cb44b", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9a6324",
    "#800000", "#008080", "#fabebe", "#aaffc3",
]


def _assign_ref_colors(reflections):
    return {ref: _PALETTE[i % len(_PALETTE)] for i, ref in enumerate(reflections)}


METRICS = [
    ("none",      "None (outlines only)"),
    ("intensity", "Intensity"),
    ("chi",       "χ angle"),
    ("tth_dev",   "2θ deviation (Δ2θ)"),
    ("chi_breadth", "Azimuthal breadth (χ FWHM)"),
    ("tth_breadth", "Radial breadth (Δ2θ FWHM)"),
]

METRIC_ZLABELS = {
    "intensity":  "Integrated intensity",
    "chi":        "χ (°)",
    "tth_dev":    "Δ2θ (°)",
    "chi_breadth": "FWHM χ (°)",
    "tth_breadth": "FWHM Δ2θ (°)",
}

METRIC_CMAPS = {"chi": "twilight"}

# Colormap choices for the device heatmap. "auto" defers to METRIC_CMAPS (so χ
# stays on the cyclic twilight map); the rest force a specific map.
CMAP_CHOICES = [
    ("auto",     "Auto (per metric)"),
    ("viridis",  "Viridis"),
    ("inferno",  "Inferno"),
    ("magma",    "Magma"),
    ("plasma",   "Plasma"),
    ("cividis",  "Cividis"),
    ("turbo",    "Turbo"),
    ("twilight", "Twilight (cyclic)"),
]

METRIC_DESCRIPTIONS = {
    "none":       "Feature outlines colored by reflection",
    "intensity":  "Integrated peak area (summed counts) per bin",
    "chi":        "Azimuthal angle χ around the Debye ring",
    "tth_dev":    "Δ2θ — deviation of the measured 2θ from the reference Bragg angle per bin (not a calibrated strain)",
    "chi_breadth": "χ-breadth — FWHM of azimuthal angle across the feature's bins (no rocking data involved)",
    "tth_breadth": "Radial breadth — FWHM of Δ2θ across the feature's bins (not a calibrated strain gradient)",
}

METRIC_2D_TITLES = {
    "none":       "Crystal Segmentation — feature outlines by reflection",
    "intensity":  "Integrated Intensity — peak area per bin",
    "chi":        "χ Angle — azimuthal orientation on Debye ring",
    "tth_dev":    "2θ Deviation — Δ2θ from the reference Bragg angle per bin",
    "chi_breadth": "Azimuthal Breadth — χ FWHM per feature",
    "tth_breadth": "Radial Breadth — Δ2θ FWHM per feature",
}


def load_features():
    """Kept-feature list from the selected feature map (or the newest shapes/
    combined catalog for the bin when none is selected).

    Reads any catalog kind via ``load_features_any`` (shapes ``kept``, combined
    ``features``, or a legacy plain list), so a shapes catalog renders here as
    its kept features.
    """
    path = CATALOG_PATH or catalogs.default_feature_source(RESULTS_DIR, _BIN_SIZE)
    if not path:
        return []
    kept, _ = catalogs.load_features_any(path)
    return kept


def load_grid_info():
    with open(GRID_PATH or _DM.grid_mapping(bin_size=_BIN_SIZE)) as f:
        gm = json.load(f)
    return gm["n_bin_rows"], gm["n_bin_cols"]


PER_FEATURE_METRICS = {
    "chi_breadth": "chi_fwhm", "tth_breadth": "tth_fwhm", "chi": "chi_deg",
}


def _extract_metric(entry, feat, metric):
    if not isinstance(entry, dict):
        return float(entry) if metric == "intensity" else None
    if metric == "intensity":
        return entry.get("integrated", entry.get("intensity", 0))
    if metric == "tth_dev":
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
                if val is None and feat_key == "chi_fwhm":
                    val = feat.get("rocking_fwhm")  # accept legacy field
                if val is None and feat_key == "tth_fwhm":
                    val = feat.get("strain_breadth")  # accept legacy field
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
                        if metric == "tth_dev":
                            if np.isnan(grid[r, c]) or abs(val) > abs(grid[r, c]):
                                grid[r, c] = val
                        else:
                            grid[r, c] = np.nanmax([grid[r, c], val])
        grids[ref] = grid
    return grids


def _feat_grid_rc(feat):
    """Feature center as (row, col) in *grid* space, matching the mask / heatmap.

    ``center_bin`` ("row_col") is always the strongest bin in grid coordinates,
    the same space the ImageItem fill and outlines live in. ``center_row`` /
    ``center_col`` are NOT reliable here: coordinate-based catalogs
    (``coord_shapes``) store them in physical XY, which would scatter the markers
    far off the features. Fall back to them only when ``center_bin`` is absent.
    """
    cb = feat.get("center_bin")
    if isinstance(cb, str) and "_" in cb:
        parts = cb.split("_")
        try:
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass
    return float(feat.get("center_row", 0)), float(feat.get("center_col", 0))


def _feat_in_chi_range(feat, chi_range):
    if chi_range is None:
        return True
    lo, hi = chi_range
    chi = feat.get("chi_deg")
    if chi is None:
        return True
    if hi > 180 and chi < lo:
        chi += 360
    return lo <= chi <= hi


def _build_outline_groups(features, n_rows, n_cols, visible_refs, chi_range=None):
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


# ─────────────────────────────────────────────────────────────────────
# pyqtgraph helpers
# ─────────────────────────────────────────────────────────────────────
def _get_cmap(name):
    """pyqtgraph ColorMap by name, falling back through matplotlib."""
    try:
        return pg.colormap.get(name)
    except Exception:
        try:
            return pg.colormap.get(name, source="matplotlib")
        except Exception:
            return pg.colormap.get("viridis")


def _scalar_to_rgba(arr, vmin, vmax, cmap):
    """Map a 2-D scalar array → (H, W, 4) uint8 RGBA, NaN → transparent."""
    h, w = arr.shape
    out = np.zeros((h, w, 4), dtype=np.ubyte)
    finite = np.isfinite(arr)
    if finite.any():
        lut = cmap.getLookupTable(0.0, 1.0, 256)  # (256, 3) uint8
        norm = np.clip((arr - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        idx = np.zeros_like(arr, dtype=int)
        idx[finite] = (norm[finite] * 255).astype(int)
        out[..., :3] = lut[idx]
        out[..., 3] = np.where(finite, 255, 0).astype(np.ubyte)
    return out


def _hex_rgb(hex_color):
    c = QColor(hex_color)
    return c.red(), c.green(), c.blue()


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
            self._dragging = "lo" if abs(x - x_lo) < abs(x - x_hi) else "hi"

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        v = self._x_to_val(event.x())
        v = max(self._min, min(self._max, v))
        if self._dragging == "lo":
            self._lo = min(v, self._hi - 1)
        else:
            self._hi = max(v, self._lo + 1)
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)

    def mouseReleaseEvent(self, event):
        self._dragging = None


class _WrapSpinBox(QSpinBox):
    """Spin box for a circular angle (internal unwrapped, displays wrapped)."""

    @staticmethod
    def _wrap(v):
        return v - 360 if v >= 180 else v

    def textFromValue(self, v):
        return str(self._wrap(v))

    def valueFromText(self, text):
        m = re.search(r"-?\d+", text)
        if not m:
            return self.value()
        v = int(m.group())
        lo, hi = self.minimum(), self.maximum()
        if v < lo:
            v += 360
        return max(lo, min(hi, v))


class _ChiAxisItem(pg.AxisItem):
    """χ histogram axis: keep unwrapped coordinates, display wrapped labels."""

    @staticmethod
    def _wrap(v):
        return v - 360 if v >= 180 else v

    def tickStrings(self, values, scale, spacing):
        labels = []
        for v in values:
            wrapped = self._wrap(v * scale)
            if abs(wrapped) < 1e-9:
                wrapped = 0.0
            if abs(wrapped - round(wrapped)) < 1e-6:
                labels.append(str(int(round(wrapped))))
            else:
                labels.append(f"{wrapped:g}")
        return labels


def _draw_feature_point_labels(plot, entries, colors, font_pt, label_offset=(0.5, -0.5)):
    """Draw the shared Device View feature-center markers and labels."""
    font = QFont()
    font.setPointSize(font_pt)
    spots, items = [], []
    dx, dy = label_offset
    for feat, pos in entries:
        ref = feat.get("reflection")
        color = colors.get(ref, "k")
        spots.append({"pos": pos, "brush": pg.mkBrush(color), "size": 6, "pen": None})
        fid = feat.get("feature_id", "?")
        chi = feat.get("chi_deg")
        label = f"#{fid}" + (f" χ={chi:.0f}°" if chi is not None else "")
        text = pg.TextItem(label, color=color, anchor=(0, 1))
        text.setFont(font)
        text.setPos(pos[0] + dx, pos[1] + dy)
        text.setZValue(15)
        plot.addItem(text)
        items.append(text)
    if spots:
        scatter = pg.ScatterPlotItem(spots=spots)
        scatter.setZValue(14)
        plot.addItem(scatter)
        items.append(scatter)
    return items


class DeviceMapWindow(QMainWindow):
    def __init__(self, features, grids, n_rows, n_cols, xrf=None):
        super().__init__()
        self.setWindowTitle("Perovskite Device Map — Bragg Peak Analysis")
        self.setGeometry(50, 30, 1450, 900)

        self.features = features
        self.grids = grids
        self.n_rows = n_rows
        self.n_cols = n_cols
        # XRF underlay state (element fluorescence beneath the device features)
        self.xrf = xrf
        self.xrf_elements = list(xrf["elements"]) if xrf else []
        self.xrf_colors = _element_colors(self.xrf_elements)
        self.visible_xrf = list(self.xrf_elements)   # checked candidate elements
        self.xrf_on = False
        self.xrf_mode = "dominant"    # "dominant" (argmax element) | "total" (sum)
        self.xrf_normalize = True     # per-element normalize before argmax
        self.xrf_opacity = 0.75
        self.metric = "none"
        self.cmap_name = "auto"    # colormap override; "auto" → METRIC_CMAPS
        self.current_grids = grids
        self.visible_refs = list(REFLECTIONS)
        self.show_labels = False
        self._highlight_items = []
        self._point_items = []
        self._outline_items = []
        self._points_state = None      # (chi_range, only_idx) of the last _draw_points
        self._highlighted_idx = None
        self._locked_idx = None
        # Debounce resizes so the labels re-scale once the drag settles, not on
        # every intermediate resize event.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._relayout_labels)
        self._isolate = True   # clicked feature dims the rest (Photoshop-style)
        self._chi_hist_ref = None
        self._chi_weight = "count"

        all_chi = [f["chi_deg"] for f in features if f.get("chi_deg") is not None]
        self._chi_wraps = bool(all_chi) and (max(all_chi) - min(all_chi)) > 180
        unwrapped = [self._unwrap_chi(c) for c in all_chi]
        if unwrapped:
            self._chi_data_min = int(np.floor(min(unwrapped)))
            self._chi_data_max = int(np.ceil(max(unwrapped)))
        else:
            self._chi_data_min, self._chi_data_max = -180, 180
        self._chi_lo = float(self._chi_data_min)
        self._chi_hi = float(self._chi_data_max)

        self._build_ui()
        self._update_chi_visuals()
        self._redraw()

    # ----- UI ---------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: main heatmap + hover label
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "Col (scan x)")
        self.plot.setLabel("left", "Row (scan y)")
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)            # origin='upper': row 0 at top
        self.plot.getViewBox().setBackgroundColor("w")
        # XRF underlay sits *below* the base feature image (which is transparent
        # wherever there is no feature, so the fluorescence shows through).
        self.xrf_img = pg.ImageItem()
        self.xrf_img.setZValue(-10)
        self.xrf_img.setVisible(False)
        self.plot.addItem(self.xrf_img)
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.colorbar = None
        self.legend = self.plot.addLegend(offset=(-10, 10))
        left_lay.addWidget(self.glw, 1)

        self.plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        self.hover_label = QLabel("Hover over a feature to see details")
        self.hover_label.setStyleSheet(
            "font-family: monospace; font-size: 0.9em; color: #555; "
            "padding: 4px; background: #f0f0f0;")
        self.hover_label.setFixedHeight(22)
        left_lay.addWidget(self.hover_label)
        splitter.addWidget(left)

        # Right: controls + histogram + arc
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)

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
            cb.setStyleSheet(f"QCheckBox {{ color: {REF_COLORS[ref]}; }}")
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
        self.labels_cb.toggled.connect(self._on_labels_toggle)
        tog.addWidget(self.labels_cb)
        self.isolate_cb = QCheckBox("Isolate selection")
        self.isolate_cb.setChecked(self._isolate)
        self.isolate_cb.setToolTip(
            "When a feature is clicked, hide everything else and rescale the "
            "color map to that feature's own value range — shows the clicked "
            "feature alone, like a single feature in Shape/Verify. Uncheck to "
            "keep the full map visible behind the selection.")
        self.isolate_cb.toggled.connect(self._on_isolate_toggle)
        tog.addWidget(self.isolate_cb)
        tog.addStretch()
        rgl.addLayout(tog)
        rl.addWidget(rg)

        sg = QGroupBox("Metric")
        sl = QVBoxLayout(sg)
        self.metric_combo = QComboBox()
        for key, label in METRICS:
            self.metric_combo.addItem(label, key)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        sl.addWidget(self.metric_combo)
        self.metric_desc_label = QLabel(METRIC_DESCRIPTIONS.get(self.metric, ""))
        self.metric_desc_label.setWordWrap(True)
        self.metric_desc_label.setStyleSheet(
            "color: #666; font-size: 0.9em; font-style: italic; padding: 2px;")
        sl.addWidget(self.metric_desc_label)
        # Colormap picker
        cm = QHBoxLayout()
        cm.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        for key, label in CMAP_CHOICES:
            self.cmap_combo.addItem(label, key)
        self.cmap_combo.setToolTip(
            "Colormap for the heatmap. 'Auto' uses the per-metric default "
            "(cyclic twilight for χ, viridis otherwise).")
        self.cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        cm.addWidget(self.cmap_combo)
        cm.addStretch()
        sl.addLayout(cm)
        # Contrast: the colormap spans [min%, max%] percentiles of the metric
        # values (the colorbar on the right is a legend for that range).
        ch = QHBoxLayout()
        ch.addWidget(QLabel("Contrast %:"))
        self.lo_spin = QSpinBox()
        self.lo_spin.setRange(0, 100); self.lo_spin.setValue(0)
        self.lo_spin.setToolTip("Lower percentile — colormap bottom maps here")
        self.lo_spin.valueChanged.connect(self._redraw)
        ch.addWidget(self.lo_spin)
        self.hi_spin = QSpinBox()
        self.hi_spin.setRange(0, 100); self.hi_spin.setValue(100)
        self.hi_spin.setToolTip("Upper percentile — colormap top maps here")
        self.hi_spin.valueChanged.connect(self._redraw)
        ch.addWidget(self.hi_spin)
        ch.addStretch()
        sl.addLayout(ch)
        rl.addWidget(sg)

        self._build_xrf_group(rl)

        rl.addWidget(QLabel("  Azimuthal distribution (χ)"))
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reflection:"))
        self.chi_ref_combo = QComboBox()
        self.chi_ref_combo.addItem("Checked layers")
        for ref in REFLECTIONS:
            self.chi_ref_combo.addItem(ref)
        self.chi_ref_combo.currentIndexChanged.connect(self._on_chi_ref_changed)
        ref_row.addWidget(self.chi_ref_combo)
        ref_row.addStretch()
        ref_row.addWidget(QLabel("Weight:"))
        self.chi_weight_combo = QComboBox()
        self.chi_weight_combo.addItem("Counts", "count")
        self.chi_weight_combo.addItem("Area (bins)", "area")
        self.chi_weight_combo.currentIndexChanged.connect(self._on_chi_weight_changed)
        ref_row.addWidget(self.chi_weight_combo)
        rl.addLayout(ref_row)

        self.chi_hist = pg.PlotWidget(
            plotItem=pg.PlotItem(axisItems={"bottom": _ChiAxisItem("bottom")}))
        self.chi_hist.setBackground("w")
        self.chi_hist.setFixedHeight(230)
        self.chi_hist.setLabel("bottom", "χ (°)")
        self.chi_hist.setLabel("left", "Count")
        rl.addWidget(self.chi_hist)

        self.chi_range_slider = QRangeSlider(self._chi_data_min, self._chi_data_max)
        self.chi_range_slider.rangeChanged.connect(self._on_range_slider)
        rl.addWidget(self.chi_range_slider)

        entry_row = QHBoxLayout()
        entry_row.addWidget(QLabel("Min:"))
        # Internal values stay continuous for filtering; display wraps past 180°
        # so the histogram reads like the feature labels (e.g. 235° -> -125°).
        self.chi_min_spin = _WrapSpinBox()
        self.chi_min_spin.setRange(self._chi_data_min, self._chi_data_max)
        self.chi_min_spin.setValue(self._chi_data_min)
        self.chi_min_spin.setSuffix("°")
        self.chi_min_spin.setFixedWidth(72)
        self.chi_min_spin.valueChanged.connect(self._on_chi_min_spin)
        entry_row.addWidget(self.chi_min_spin)
        entry_row.addStretch()
        entry_row.addWidget(QLabel("Max:"))
        self.chi_max_spin = _WrapSpinBox()
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
            "font-family: monospace; font-size: 0.9em; color: #555; padding: 2px;")
        rl.addWidget(self.chi_range_label)

        self.arc_plot = pg.PlotWidget()
        self.arc_plot.setBackground("w")
        self.arc_plot.setFixedHeight(120)
        self.arc_plot.setAspectLocked(True)
        self.arc_plot.hideAxis("bottom")
        self.arc_plot.hideAxis("left")
        self.arc_plot.setMouseEnabled(False, False)
        rl.addWidget(self.arc_plot)

        # Feature/bin details. At 1×1 this text can get long; keep it inside a
        # scroll area so it fills the leftover sidebar height and scrolls, rather
        # than inflating the panel's minimum height and stretching the whole view.
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop)
        self.info_label.setStyleSheet("QLabel { font-size: 10pt; padding: 6px; }")
        self._info_scroll = QScrollArea()
        self._info_scroll.setWidgetResizable(True)
        self._info_scroll.setWidget(self.info_label)
        self._info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._info_scroll.setMinimumHeight(80)
        self._info_scroll.setFrameShape(QScrollArea.NoFrame)
        rl.addWidget(self._info_scroll, 1)  # stretch: take remaining space, scroll inside

        splitter.addWidget(right)
        splitter.setSizes([950, 450])

    # ----- chi coordinate helpers -------------------------------------
    def _unwrap_chi(self, c):
        return c + 360 if (self._chi_wraps and c < 0) else c

    def _wrap_chi(self, u):
        return u - 360 if u >= 180 else u

    def _chi_range(self):
        if (self._chi_lo <= self._chi_data_min and self._chi_hi >= self._chi_data_max):
            return None
        return (self._chi_lo, self._chi_hi)

    # ----- main heatmap -----------------------------------------------
    def _compute_combined(self, metric, visible_refs, chi_range):
        """Return (combined, vmin, vmax) over visible refs with chi filtering."""
        chi_valid = {}
        if chi_range is not None:
            for ref in visible_refs:
                mask = np.zeros((self.n_rows, self.n_cols), dtype=bool)
                for f in self.features:
                    if f["reflection"] == ref and _feat_in_chi_range(f, chi_range):
                        m = f.get("_mask")
                        if m is not None:
                            mask |= m
                chi_valid[ref] = mask

        combined = np.full((self.n_rows, self.n_cols), np.nan)
        for ref in visible_refs:
            if ref not in self.current_grids:
                continue
            Z = self.current_grids[ref]
            valid = np.isfinite(Z)
            if metric == "intensity":
                valid = valid & (Z > 0)
            if ref in chi_valid:
                valid = valid & chi_valid[ref]
            if not valid.any():
                continue
            first_fill = valid & np.isnan(combined)
            if metric == "tth_dev":
                better = valid & np.isfinite(combined) & (np.abs(Z) > np.abs(combined))
            else:
                better = valid & np.isfinite(combined) & (Z > combined)
            combined = np.where(first_fill | better, Z, combined)

        has = np.isfinite(combined)
        if not has.any():
            return combined, 0.0, 1.0
        finite = combined[has]
        # Colormap spans the [min%, max%] percentile window of the metric values.
        lo = float(self.lo_spin.value()) if hasattr(self, "lo_spin") else 0.0
        hi = float(self.hi_spin.value()) if hasattr(self, "hi_spin") else 100.0
        if hi <= lo:
            hi = min(100.0, lo + 1.0)
        vmin = float(np.percentile(finite, lo))
        vmax = float(np.percentile(finite, hi))
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1
        return combined, vmin, vmax

    def _compute_isolated(self, feat):
        """(grid, vmin, vmax) for a single feature's bins only, scaled to its
        own value range so its internal metric variation is visible. Bins
        outside the feature stay NaN → rendered transparent by _scalar_to_rgba.
        """
        ref = feat["reflection"]
        mask = feat.get("_mask")
        Z = self.current_grids.get(ref)
        grid = np.full((self.n_rows, self.n_cols), np.nan)
        if Z is None or mask is None:
            return grid, 0.0, 1.0
        valid = np.isfinite(Z) & mask
        if self.metric == "intensity":
            valid = valid & (Z > 0)
        grid[valid] = Z[valid]
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            return grid, 0.0, 1.0
        lo = float(self.lo_spin.value()) if hasattr(self, "lo_spin") else 0.0
        hi = float(self.hi_spin.value()) if hasattr(self, "hi_spin") else 100.0
        if hi <= lo:
            hi = min(100.0, lo + 1.0)
        vmin = float(np.percentile(finite, lo))
        vmax = float(np.percentile(finite, hi))
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1
        return grid, vmin, vmax

    def _current_cmap(self):
        """Resolved colormap for the current metric, honoring the picker
        ("auto" → the per-metric default in METRIC_CMAPS)."""
        name = self.cmap_name
        if name == "auto":
            name = METRIC_CMAPS.get(self.metric, "viridis")
        return _get_cmap(name)

    def _on_cmap_changed(self):
        self.cmap_name = self.cmap_combo.currentData()
        self._redraw()

    def _on_isolate_toggle(self, checked):
        self._isolate = bool(checked)
        self._redraw()

    def _clear_items(self, items):
        for it in items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        items.clear()

    def _redraw(self):
        self._clear_highlight(draw=False)
        self._clear_items(self._outline_items)
        self._clear_items(self._point_items)
        if self._locked_idx is None:
            self.info_label.setText("")
            self.hover_label.setText("Hover over a feature to see details")

        chi_range = self._chi_range()

        # Is a locked feature being isolated this redraw? (only if still visible)
        isolate = (self._isolate and self._locked_idx is not None
                   and 0 <= self._locked_idx < len(self.features))
        iso_feat = self.features[self._locked_idx] if isolate else None
        if isolate and not (iso_feat["reflection"] in self.visible_refs
                            and _feat_in_chi_range(iso_feat, chi_range)):
            isolate, iso_feat = False, None
            self._locked_idx = None

        outline_groups = _build_outline_groups(
            self.features, self.n_rows, self.n_cols, self.visible_refs, chi_range)

        # Base image
        if self.metric == "none":
            rgba = np.zeros((self.n_rows, self.n_cols, 4), dtype=np.ubyte)
            for ref, merged in outline_groups:
                r, g, b = _hex_rgb(REF_COLORS[ref])
                pastel = tuple(int(ch * 0.35 + 0.65 * 255) for ch in (r, g, b))
                rgba[merged, 0] = pastel[0]
                rgba[merged, 1] = pastel[1]
                rgba[merged, 2] = pastel[2]
                rgba[merged, 3] = 255
            if isolate and iso_feat.get("_mask") is not None:
                rgba[~iso_feat["_mask"], 3] = _ISO_GHOST_ALPHA  # hide the rest
            self.img_item.setImage(rgba, autoLevels=False)
            self._update_colorbar(None, None, None)
        elif isolate:
            # Hide the full map (ghost alpha 0), then paint the selected
            # feature (its own value range) at full strength on top.
            cmap = self._current_cmap()
            full, fvmin, fvmax = self._compute_combined(
                self.metric, self.visible_refs, chi_range)
            rgba = _scalar_to_rgba(full, fvmin, fvmax, cmap)
            a = rgba[..., 3].astype(np.uint16) * _ISO_GHOST_ALPHA // 255
            rgba[..., 3] = a.astype(np.ubyte)
            sel, vmin, vmax = self._compute_isolated(iso_feat)
            sel_rgba = _scalar_to_rgba(sel, vmin, vmax, cmap)
            on = sel_rgba[..., 3] > 0
            rgba[on] = sel_rgba[on]
            self.img_item.setImage(rgba, autoLevels=False)
            if np.isfinite(sel).any():
                self._update_colorbar(cmap, vmin, vmax)
            else:
                self._update_colorbar(None, None, None)
        else:
            combined, vmin, vmax = self._compute_combined(
                self.metric, self.visible_refs, chi_range)
            cmap = self._current_cmap()
            rgba = _scalar_to_rgba(combined, vmin, vmax, cmap)
            self.img_item.setImage(rgba, autoLevels=False)
            if np.isfinite(combined).any():
                self._update_colorbar(cmap, vmin, vmax)
            else:
                self._update_colorbar(None, None, None)

        # Outlines (IsocurveItem at 0.5 on each merged mask). While isolating
        # the other outlines are hidden (ghost alpha 0); the selected
        # feature's own outline is drawn at full strength by _draw_highlight.
        for ref, merged in outline_groups:
            if isolate:
                col = QColor(REF_COLORS[ref])
                col.setAlpha(_ISO_GHOST_ALPHA)
                pen = pg.mkPen(col, width=1.5)
            else:
                pen = pg.mkPen(REF_COLORS[ref], width=1.5)
            iso = pg.IsocurveItem(data=merged.astype(float), level=0.5, pen=pen)
            iso.setZValue(5)
            self.plot.addItem(iso)
            self._outline_items.append(iso)

        # Legend (rebuild)
        self.legend.clear()
        for ref in self.visible_refs:
            if ref in REF_COLORS:
                s = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(REF_COLORS[ref]), size=10)
                self.legend.addItem(s, ref)

        # Points / labels (only the selected feature's label while isolating)
        if self.show_labels:
            self._draw_points(chi_range,
                              only_idx=self._locked_idx if isolate else None)

        self._update_xrf_underlay()

        title = METRIC_2D_TITLES.get(self.metric, self.metric)
        self.plot.setTitle(f"{title}  —  {self.n_cols}×{self.n_rows} (col×row)")

        if self._locked_idx is not None:
            feat = self.features[self._locked_idx]
            if (feat["reflection"] in self.visible_refs
                    and _feat_in_chi_range(feat, chi_range)):
                # While isolating, the base image already shows the feature in
                # full color — draw just its crisp outline (no translucent fill).
                self._draw_highlight(self._locked_idx, fill=not isolate)
            else:
                self._locked_idx = None

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

    def _label_font_pt(self):
        """Index-label point size, scaled to the plot's on-screen height so labels
        stay legible on a large window and don't dominate a small one."""
        h = self.glw.height() or 600
        return max(8, min(24, int(round(h / 45))))

    def _draw_points(self, chi_range, only_idx=None):
        # Cache args so a window resize can re-lay the labels at the new font size.
        self._points_state = (chi_range, only_idx)
        entries = []
        for i, feat in enumerate(self.features):
            if only_idx is not None and i != only_idx:
                continue
            ref = feat["reflection"]
            if ref not in self.visible_refs or not _feat_in_chi_range(feat, chi_range):
                continue
            # Grid (row, col) → pixel center. Use grid space, not center_row/col,
            # which are physical XY in coordinate-based catalogs.
            gr, gc = _feat_grid_rc(feat)
            entries.append((feat, (gc + 0.5, gr + 0.5)))
        self._point_items.extend(_draw_feature_point_labels(
            self.plot, entries, REF_COLORS, self._label_font_pt()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale the index labels to the new window size (debounced).
        if self.show_labels and self._points_state is not None:
            self._resize_timer.start()

    def _relayout_labels(self):
        """Redraw only the point/label layer at the current font size."""
        if not self.show_labels or self._points_state is None:
            return
        self._clear_items(self._point_items)
        chi_range, only_idx = self._points_state
        self._draw_points(chi_range, only_idx=only_idx)

    # ----- layer / metric controls ------------------------------------
    # ----- XRF underlay ----------------------------------------------
    def _build_xrf_group(self, parent_layout):
        """Element-fluorescence underlay controls (mirrors the reflection Layers
        group but for XRF elements)."""
        grp = QGroupBox("XRF underlay")
        gl = QVBoxLayout(grp)

        self.xrf_on_cb = QCheckBox("Underlay XRF map")
        have = bool(self.xrf)
        self.xrf_on_cb.setEnabled(have)
        self.xrf_on_cb.setToolTip(
            "Show the ME7 fluorescence element map beneath the device features."
            if have else
            "No XRF product for this scan/bin.\nRun:  xrd-app xrf --scans <scan> --refine-roi")
        self.xrf_on_cb.toggled.connect(self._on_xrf_toggle)
        gl.addWidget(self.xrf_on_cb)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Show:"))
        self.xrf_mode_combo = QComboBox()
        self.xrf_mode_combo.addItem("Dominant element (ranked)", "dominant")
        self.xrf_mode_combo.addItem("Total intensity", "total")
        self.xrf_mode_combo.setToolTip(
            "Dominant: each pixel is coloured by whichever checked element is "
            "strongest there (the ranked-material map).\n"
            "Total: summed intensity of the checked elements as one heatmap.")
        self.xrf_mode_combo.setEnabled(have)
        self.xrf_mode_combo.currentIndexChanged.connect(self._on_xrf_mode_changed)
        mode_row.addWidget(self.xrf_mode_combo)
        mode_row.addStretch()
        gl.addLayout(mode_row)

        opt_row = QHBoxLayout()
        self.xrf_norm_cb = QCheckBox("Normalize per element")
        self.xrf_norm_cb.setChecked(self.xrf_normalize)
        self.xrf_norm_cb.setEnabled(have)
        self.xrf_norm_cb.setToolTip(
            "Scale each element to its own range before ranking, so the dominant-"
            "element map reflects relative presence rather than raw fluorescence "
            "yield (Pb/I would otherwise always win).")
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

        # element checkboxes (candidate set for the ranking / sum)
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

    def _compute_xrf_rgba(self):
        """(n_rows, n_cols, 4) uint8 RGBA for the XRF underlay, or None."""
        return compute_xrf_rgba(self.xrf, self.visible_xrf, self.xrf_mode,
                                self.xrf_normalize, self.xrf_opacity, self.xrf_colors)

    def _update_xrf_underlay(self):
        """Refresh the underlay image + visibility from the current XRF state."""
        if not hasattr(self, "xrf_img"):
            return
        if not (self.xrf_on and self.xrf):
            self.xrf_img.setVisible(False)
            return
        rgba = self._compute_xrf_rgba()
        if rgba is None:
            self.xrf_img.setVisible(False)
            return
        self.xrf_img.setImage(rgba, autoLevels=False)
        # The XRF maps are 1×1; stretch them across the (coarser) bin grid so the
        # underlay lines up with the device features regardless of bin size.
        self.xrf_img.setRect(QRectF(0.0, 0.0, float(self.n_cols), float(self.n_rows)))
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

    def _check_all_layers(self):
        for cb in self.layer_cbs.values():
            cb.setChecked(True)

    def _uncheck_all_layers(self):
        for cb in self.layer_cbs.values():
            cb.setChecked(False)

    def _on_layer_toggle(self):
        self.visible_refs = [ref for ref in REFLECTIONS if self.layer_cbs[ref].isChecked()]
        if self._chi_hist_ref is None:
            self._draw_chi_histogram()
        self._redraw()

    def _on_labels_toggle(self, checked):
        self.show_labels = checked
        self._redraw()

    def _on_metric_changed(self):
        key = self.metric_combo.currentData()
        self.metric_desc_label.setText(METRIC_DESCRIPTIONS.get(key, ""))
        if key == self.metric:
            return
        self.metric = key
        if key != "none":
            self.current_grids = build_device_grids(
                self.features, self.n_rows, self.n_cols, metric=key)
        self._redraw()

    # ----- view-state carry-over (across a feature-catalog switch) -----
    def get_view_state(self):
        """Selected layers + outline metric, so a catalog switch keeps them."""
        return {
            "metric": self.metric,
            "cmap": self.cmap_name,
            "hidden_layers": [r for r, cb in self.layer_cbs.items()
                              if not cb.isChecked()],
            "points": self.labels_cb.isChecked(),
            "isolate": self._isolate,
            "xrf_on": self.xrf_on,
            "xrf_mode": self.xrf_mode,
            "xrf_normalize": self.xrf_normalize,
            "xrf_opacity": self.xrf_opacity,
            "xrf_hidden": [e for e, cb in self.xrf_cbs.items()
                           if not cb.isChecked()],
        }

    def apply_view_state(self, state):
        """Re-apply a saved view state. Reflections not in this catalog and
        unknown metric keys are ignored; new reflections default to visible."""
        if not state:
            return
        hidden = set(state.get("hidden_layers", []))
        for ref, cb in self.layer_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(ref not in hidden)
            cb.blockSignals(False)
        self.visible_refs = [r for r in REFLECTIONS if self.layer_cbs[r].isChecked()]
        if "points" in state:
            self.labels_cb.blockSignals(True)
            self.labels_cb.setChecked(bool(state["points"]))
            self.labels_cb.blockSignals(False)
            self.show_labels = bool(state["points"])
        if "isolate" in state:
            self._isolate = bool(state["isolate"])
            self.isolate_cb.blockSignals(True)
            self.isolate_cb.setChecked(self._isolate)
            self.isolate_cb.blockSignals(False)
        cmap = state.get("cmap")
        if cmap and self.cmap_combo.findData(cmap) >= 0:
            self.cmap_name = cmap
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentIndex(self.cmap_combo.findData(cmap))
            self.cmap_combo.blockSignals(False)
        m = state.get("metric")
        if m and self.metric_combo.findData(m) >= 0:
            self.metric_combo.blockSignals(True)
            self.metric_combo.setCurrentIndex(self.metric_combo.findData(m))
            self.metric_combo.blockSignals(False)
            self.metric_desc_label.setText(METRIC_DESCRIPTIONS.get(m, ""))
            self.metric = m
            if m != "none":
                self.current_grids = build_device_grids(
                    self.features, self.n_rows, self.n_cols, metric=m)
        self._apply_xrf_view_state(state)
        self._draw_chi_histogram()
        self._redraw()

    def _apply_xrf_view_state(self, state):
        """Re-apply saved XRF underlay settings (elements absent from this scan's
        product are ignored; the checkbox stays disabled if there's no product)."""
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

    # ----- chi range filter -------------------------------------------
    def _update_chi_visuals(self):
        self._draw_chi_histogram()
        self._draw_chi_arc()
        # Display wraps past 180° while internal values stay continuous.
        self.chi_range_label.setText(
            f"χ: {self._wrap_chi(self._chi_lo):.0f}° to "
            f"{self._wrap_chi(self._chi_hi):.0f}°")

    def _draw_chi_histogram(self):
        self.chi_hist.clear()
        ref_filter = self._chi_hist_ref
        area_mode = self._chi_weight == "area"
        chis, weights = [], []
        for f in self.features:
            if ref_filter is None:
                if f["reflection"] not in self.visible_refs:
                    continue
            elif f["reflection"] != ref_filter:
                continue
            chi = f.get("chi_deg")
            if chi is None:
                continue
            chis.append(chi)
            weights.append(float(f.get("n_bins", 0)) if area_mode else 1.0)

        if not chis or sum(weights) == 0:
            t = pg.TextItem("No χ data", color="#aaa", anchor=(0.5, 0.5))
            self.chi_hist.addItem(t)
            t.setPos(0, 0)
            return

        all_plot = [self._unwrap_chi(c) for c in chis]
        selected = [self._chi_lo <= u <= self._chi_hi for u in all_plot]
        in_plot = [u for u, s in zip(all_plot, selected) if s]
        in_w = [w for w, s in zip(weights, selected) if s]

        bin_lo = min(all_plot) - 5
        bin_hi = max(all_plot) + 5
        edges = np.arange(bin_lo, bin_hi + 5, 5)
        centers = (edges[:-1] + edges[1:]) / 2
        h_all, _ = np.histogram(all_plot, bins=edges, weights=weights)
        h_in, _ = np.histogram(in_plot, bins=edges, weights=in_w) if in_plot \
            else (np.zeros(len(centers)), None)

        total = sum(weights)
        pct = 100 * sum(in_w) / total if total else 0
        span = self._chi_hi - self._chi_lo

        self.chi_hist.addItem(pg.BarGraphItem(
            x=centers, height=h_all, width=4.5, brush=(204, 204, 204, 130), pen=None))
        self.chi_hist.addItem(pg.BarGraphItem(
            x=centers, height=h_in, width=4.5, brush=(67, 99, 216, 220), pen=None))
        for v in (self._chi_lo, self._chi_hi):
            self.chi_hist.addItem(pg.InfiniteLine(
                pos=v, angle=90, pen=pg.mkPen("r", width=1.2, style=Qt.DashLine)))

        ref_label = "checked layers" if ref_filter is None else ref_filter
        self.chi_hist.setLabel("left", "Total bins" if area_mode else "Count")
        self.chi_hist.setTitle(f"{ref_label} — {pct:.0f}% selected, {span:.0f}° span")

    def _on_chi_ref_changed(self):
        idx = self.chi_ref_combo.currentIndex()
        self._chi_hist_ref = None if idx == 0 else REFLECTIONS[idx - 1]
        self._draw_chi_histogram()

    def _on_chi_weight_changed(self):
        self._chi_weight = self.chi_weight_combo.currentData()
        self._draw_chi_histogram()
        self.chi_hist.autoRange()

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
        self._chi_lo, self._chi_hi = float(lo), float(hi)
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
        self.arc_plot.clear()

        def arc_xy(a_lo, a_hi, r):
            ang = np.radians(np.linspace(a_lo, a_hi, 200))
            return r * np.cos(ang), -r * np.sin(ang)  # E=0, clockwise

        # background band
        for (lo, hi, col) in [
            (self._chi_data_min, self._chi_data_max, (224, 224, 224)),
            (self._chi_lo, self._chi_hi, (67, 99, 216))]:
            xo, yo = arc_xy(lo, hi, 1.0)
            xi, yi = arc_xy(lo, hi, 0.6)
            outer = pg.PlotDataItem(xo, yo)
            inner = pg.PlotDataItem(xi, yi)
            fill = pg.FillBetweenItem(outer, inner, brush=pg.mkBrush(*col, 150))
            self.arc_plot.addItem(fill)
        for a in (self._chi_lo, self._chi_hi):
            x, y = arc_xy(a, a, 1.0)
            x0, y0 = arc_xy(a, a, 0.5)
            self.arc_plot.addItem(pg.PlotDataItem(
                [x0[0], x[0]], [y0[0], y[0]],
                pen=pg.mkPen("r", width=1.5, style=Qt.DashLine)))

    # ----- hover / click ----------------------------------------------
    def _nearest_feature(self, mx, my):
        best_idx, best_dist = None, float("inf")
        chi_r = self._chi_range()
        for i, feat in enumerate(self.features):
            if feat["reflection"] not in self.visible_refs:
                continue
            if not _feat_in_chi_range(feat, chi_r):
                continue
            gr, gc = _feat_grid_rc(feat)
            dc = mx - (gc + 0.5)
            dr = my - (gr + 0.5)
            d = (dc * dc + dr * dr) ** 0.5
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx, best_dist

    def _scene_to_view(self, scene_pos):
        vb = self.plot.getViewBox()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        pt = vb.mapSceneToView(scene_pos)
        return pt.x(), pt.y()

    def _on_click(self, ev):
        if not self.show_labels:
            return
        pos = self._scene_to_view(ev.scenePos())
        if pos is None:
            return
        idx, dist = self._nearest_feature(*pos)
        if dist > 3.0 or idx is None:
            if self._locked_idx is not None:
                self._locked_idx = None
                self._redraw() if self._isolate else self._clear_highlight()
            return
        if idx == self._locked_idx:
            self._locked_idx = None
            self._redraw() if self._isolate else self._clear_highlight()
        elif self._isolate:
            # Isolate path rebuilds the base image around the new selection.
            self._locked_idx = idx
            self._redraw()
        else:
            self._locked_idx = idx
            self._clear_highlight(draw=False)
            self._draw_highlight(idx)

    def _on_mouse_move(self, scene_pos):
        if self._locked_idx is not None:
            return
        if not self.show_labels:
            if self._highlighted_idx is not None:
                self._clear_highlight()
            return
        pos = self._scene_to_view(scene_pos)
        if pos is None:
            if self._highlighted_idx is not None:
                self._clear_highlight()
            return
        idx, dist = self._nearest_feature(*pos)
        if dist > 3.0 or idx is None:
            if self._highlighted_idx is not None:
                self._clear_highlight()
            return
        if idx == self._highlighted_idx:
            return
        self._clear_highlight(draw=False)
        self._draw_highlight(idx)

    def _draw_highlight(self, idx, fill=True):
        self._highlighted_idx = idx
        feat = self.features[idx]
        mask = feat.get("_mask")
        if mask is None or not mask.any():
            return
        ref = feat["reflection"]
        r, g, b = _hex_rgb(REF_COLORS[ref])
        if fill:
            rgba = np.zeros((self.n_rows, self.n_cols, 4), dtype=np.ubyte)
            rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = r, g, b
            rgba[mask, 3] = 115
            hl = pg.ImageItem(rgba)
            hl.setZValue(10)
            self.plot.addItem(hl)
            self._highlight_items.append(hl)
        for pen in (pg.mkPen("w", width=3), pg.mkPen(REF_COLORS[ref], width=1.8)):
            iso = pg.IsocurveItem(data=mask.astype(float), level=0.5, pen=pen)
            iso.setZValue(11)
            self.plot.addItem(iso)
            self._highlight_items.append(iso)

        fid = feat.get("feature_id", "?")
        ref_tth = feat.get("ref_tth")
        chi = feat.get("chi_deg", 0)
        nbins = int(mask.sum())
        lines = [f"#{fid}  {ref}"]
        lines.append(f"2θ={ref_tth:.3f}°  χ={chi:.1f}°" if ref_tth is not None
                     else f"χ={chi:.1f}°")
        lines.append(f"{nbins} bins")
        self.info_label.setText("\n".join(lines))
        pinned = "  [pinned]" if self._locked_idx is not None else ""
        tth_str = f"2θ={ref_tth:.3f}°  " if ref_tth is not None else ""
        self.hover_label.setText(f"#{fid}  {ref}  {tth_str}χ={chi:.1f}°  {nbins} bins{pinned}")

    def _clear_highlight(self, draw=True):
        self._clear_items(self._highlight_items)
        self._highlighted_idx = None
        if self._locked_idx is None:
            self.info_label.setText("")
            self.hover_label.setText("Hover over a feature to see details")


def load_xrf_underlay():
    """Load the current scan's XRF element maps for the underlay, or ``None``.

    Reads the single per-scan ``<scan>_xrf.npz`` (via ``core.xrf.load_product``),
    computed at 1×1. Returns ``{elements, maps, norm, shape}`` where ``maps[name]``
    is the raw ``(nr, nc)`` intensity and ``norm[name]`` is each element scaled to
    [0,1] by its own 99th percentile (so the "dominant element" argmax is a fair
    per-element comparison, not just the highest-yield line). The 1×1 map is scaled
    onto whatever bin grid the device view uses via ``setRect``, so no shape match
    is required. Returns ``None`` if there is no product.
    """
    try:
        from ..core import xrf as xrf_core
        path = _DM.xrf_product()
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
            "norm": norm, "shape": prod.get("shape")}


def compute_xrf_rgba(xrf, visible, mode, normalize, opacity, colors):
    """Shared XRF-underlay compositor (used by both Device View and HD Device View).

    ``xrf`` is a ``load_xrf_underlay`` dict; ``visible`` the checked element names;
    ``mode`` "dominant" (argmax element, coloured by element) or "total" (summed
    intensity heatmap); ``normalize`` uses per-element-normalized maps; ``colors``
    the ``{element: hex}`` map. Returns ``(H, W, 4)`` uint8 RGBA or ``None``.
    """
    if not xrf:
        return None
    els = [e for e in visible if e in xrf["maps"]]
    if not els:
        return None
    src = xrf["norm"] if normalize else xrf["maps"]
    shape = src[els[0]].shape
    alpha = int(np.clip(opacity, 0, 1) * 255)

    if mode == "total":
        total = np.zeros(shape, dtype=float)
        for e in els:
            total += np.asarray(src[e], dtype=float)
        pos = total[total > 0]
        vmax = np.percentile(pos, 99.0) if pos.size else 0.0
        if vmax <= 0:
            return None
        rgba = _scalar_to_rgba(total, 0.0, vmax, _get_cmap("inferno"))
        rgba[..., 3] = (rgba[..., 3].astype(np.uint16) * alpha // 255).astype(np.ubyte)
        return rgba

    stack = np.stack([np.asarray(src[e], dtype=float) for e in els], axis=0)
    winner = np.argmax(stack, axis=0)
    strength = np.max(stack, axis=0)
    has = strength > 0
    smax = np.percentile(strength[has], 99.0) if has.any() else 0.0
    if smax <= 0:
        return None
    bright = np.clip(strength / smax, 0.0, 1.0)
    rgba = np.zeros((shape[0], shape[1], 4), dtype=np.ubyte)
    for wi, e in enumerate(els):
        sel = has & (winner == wi)
        if not sel.any():
            continue
        r, g, b = _hex_rgb(colors[e])
        rgba[sel, 0] = (r * bright[sel]).astype(np.ubyte)
        rgba[sel, 1] = (g * bright[sel]).astype(np.ubyte)
        rgba[sel, 2] = (b * bright[sel]).astype(np.ubyte)
        rgba[sel, 3] = alpha
    return rgba


def build_window(project_root=".", scan=None, bin_size=3, catalog=None):
    """Construct the device map without an event loop (for embedding as a tab)."""
    global REFLECTIONS, REF_COLORS
    configure(project_root=project_root, bin_size=bin_size, scan=scan, catalog=catalog)

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
    xrf = load_xrf_underlay()
    return DeviceMapWindow(features, grids, n_rows, n_cols, xrf=xrf)


def launch_gui(project_root=".", bin_size=3, scan=None):
    """Configure paths and launch the device map (used by the CLI)."""
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    win = build_window(project_root=project_root, scan=scan, bin_size=bin_size)
    app = QApplication.instance() or QApplication(sys.argv)
    win.show()
    app.exec_()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XRD device feature map")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--bin-size", type=int, default=3)
    parser.add_argument("--scan", default=None)
    args = parser.parse_args()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    win = build_window(project_root=args.project_root, bin_size=args.bin_size, scan=args.scan)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
