"""Territorial (cell-model) device map + shape inspector (pyqtgraph).

Renders the **variable-footprint** territories of the skew-free reference
binning (``core/territory.py``) to-scale: each territory is drawn as its true
(X, Y) polygon, coloured by the linked shape's per-territory peak intensity.
Selecting a shape in the list highlights the territories it spans.

This is a focused, self-contained viewer (not a clone of the 4k-line grid
viewer): the grid device-map assumes one pixel per ``"r_c"`` bin, which cannot
represent irregular territories, so this draws real polygons instead. It reads
the territorial grid mapping (``..._territory.h5``, which carries each cell's
polygon / centroid / area / count) and the territorial shapes catalog.

``build_window(project_root, scan, bin_size, catalog=None)`` mirrors
``gui.device_map`` so the tab wrappers and standalone runner work unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt, QPointF, QProcess, QTimer
from PyQt5.QtGui import QColor, QPolygonF, QBrush, QFont
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QGroupBox, QSplitter, QGraphicsPolygonItem,
    QCheckBox, QPushButton, QSpinBox, QDoubleSpinBox, QProgressBar,
    QPlainTextEdit,
)

from ..config import DataManager
from ..core import catalogs, scan_table as st
from . import palette
from .lifecycle import start_process, stop_process

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
    from ..core import io
    gm = io.load_grid_mapping(gm_path)
    return gm if gm.get("territories") else None


def _load_shapes(dm: DataManager, scan, catalog=None) -> list:
    """Kept features from the territorial shapes catalog (empty if none yet)."""
    if catalog and Path(catalog).exists():
        path = Path(catalog)
    else:
        path = dm.shapes_path("territory", 1, scan, variant="territory")
        if not Path(path).exists():
            ldir = dm.labels_dir(scan)
            hits = [candidate for candidate in catalogs.list_catalogs(ldir, "shapes", 1)
                    if catalogs.catalog_variant(candidate, ldir) == "territory"]
            if not hits:
                return []
            path = hits[-1]
    data = catalogs.load_result(path) or {}
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

    def set_white_background(self, white: bool):
        self.setBackground("w" if white else "#101014")

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
        bl.addWidget(QLabel("Shape peak intensity"))
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
        self.white_bg = QCheckBox("White background")
        self.white_bg.setToolTip("Toggle the canvas between the dark and a white background.")
        bl.addWidget(self.white_bg)
        lyt.addWidget(box)

        # ── χ angle range filter ──────────────────────────────────────
        # Two plain min/max spin boxes (no histogram/slider), auto-set to the
        # χ range actually present in the kept shapes.
        chis = [float(s["chi_deg"]) for s in self.shapes
                if s.get("chi_deg") is not None]
        chi_lo = float(np.floor(min(chis))) if chis else -180.0
        chi_hi = float(np.ceil(max(chis))) if chis else 180.0
        cbox = QGroupBox("χ angle range")
        cl = QHBoxLayout(cbox)
        self.chi_min = QDoubleSpinBox()
        self.chi_max = QDoubleSpinBox()
        for sp in (self.chi_min, self.chi_max):
            sp.setRange(-360.0, 360.0)
            sp.setDecimals(1)
            sp.setSuffix("°")
            sp.setSingleStep(1.0)
        self.chi_min.setValue(chi_lo)
        self.chi_max.setValue(chi_hi)
        self.chi_min.setToolTip("Hide shapes with χ below this angle.")
        self.chi_max.setToolTip("Hide shapes with χ above this angle.")
        cl.addWidget(QLabel("min"))
        cl.addWidget(self.chi_min)
        cl.addWidget(QLabel("max"))
        cl.addWidget(self.chi_max)
        lyt.addWidget(cbox)

        self.sbox = QGroupBox(f"Shapes ({len(self.shapes)} kept)")
        sbox = self.sbox
        sl = QVBoxLayout(sbox)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Feature ID",
            "Size (largest first)",
            "Size (smallest first)",
            "Fill % (highest first)",
            "Reflection, then size (largest)",
            "Reflection, then size (smallest)",
        ])
        self.sort_combo.setToolTip("Order the shape list by cell count and/or reflection")
        self.sort_combo.currentIndexChanged.connect(self._populate_shape_list)
        sl.addWidget(QLabel("Sort by:"))
        sl.addWidget(self.sort_combo)
        self.flist = QListWidget()
        sl.addWidget(self.flist)
        lyt.addWidget(sbox, 1)
        self._populate_shape_list()

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

        self.reflection.currentIndexChanged.connect(self._refresh_metric)
        self.reflection.currentIndexChanged.connect(self._update_sort_options)
        self.reflection.currentIndexChanged.connect(self._populate_shape_list)
        self.cmap.currentIndexChanged.connect(self._refresh_metric)
        self.white_bg.toggled.connect(self.canvas.set_white_background)
        self.chi_min.valueChanged.connect(self._refresh_metric)
        self.chi_min.valueChanged.connect(self._populate_shape_list)
        self.chi_max.valueChanged.connect(self._refresh_metric)
        self.chi_max.valueChanged.connect(self._populate_shape_list)
        self.flist.currentItemChanged.connect(self._on_select)

    # ── per-shape bounding area + fill % (solidity) ───────────────────
    def _shape_fill(self, s):
        """(bounding area in CSV units, fill fraction 0..1) for a shape, cached.

        ``bounding area`` outlines the shape's territories' outer points (convex
        hull); ``fill`` is the summed territory area ÷ that hull — how solidly the
        shape fills its footprint (holes / missing territories lower it). Shared
        with the Scan Summary table's Fill % column via ``core.scan_table``.
        """
        if "_fill" not in s:
            bounding, fill = st.territory_fill(
                self.territories, s.get("spatial_extent", []))
            s["_bounding_area"], s["_fill"] = bounding, fill
        return s.get("_bounding_area", 0.0), s["_fill"]

    # ── shape list ordering ──────────────────────────────────────────
    def _update_sort_options(self):
        """Hide the reflection-grouping sort modes when a single reflection is
        selected — with the list already filtered to one reflection they
        collapse to plain size ordering, so they'd be redundant."""
        single = self.reflection.currentText() != "(all reflections)"
        view = self.sort_combo.view()
        idx_large = self.sort_combo.findText("Reflection, then size (largest)")
        idx_small = self.sort_combo.findText("Reflection, then size (smallest)")
        for i in (idx_large, idx_small):
            if i >= 0:
                view.setRowHidden(i, single)
        cur = self.sort_combo.currentIndex()
        if single and cur in (idx_large, idx_small):
            # Fall back to the equivalent plain-size ordering.
            self.sort_combo.setCurrentText(
                "Size (largest first)" if cur == idx_large else "Size (smallest first)")

    def _populate_shape_list(self):
        """Rebuild the shape list in the order chosen by the sort combo.

        The list is first filtered to the reflection picked in the reflection
        combo above (``(all reflections)`` keeps everything), then ordered.
        Size is the shape's cell count (``n_bins``); reflection is a secondary
        alphabetical key so shapes group by reflection then descend/ascend by
        size within each group.
        """
        def size_of(s):
            try:
                return float(s.get("n_bins", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        def refl_of(s):
            return str(s.get("reflection", "") or "")

        # Filter to the selected reflection (matches the combo's own key), then
        # to the χ angle range.
        want = self.reflection.currentText()
        shapes = list(self.shapes)
        if want != "(all reflections)":
            shapes = [s for s in shapes if s.get("reflection", "?") == want]
        shapes = [s for s in shapes if self._chi_pass(s)]
        self.sbox.setTitle(
            f"Shapes ({len(shapes)} of {len(self.shapes)} kept)"
            if len(shapes) != len(self.shapes)
            else f"Shapes ({len(self.shapes)} kept)")

        mode = self.sort_combo.currentText()
        if mode == "Size (largest first)":
            shapes.sort(key=size_of, reverse=True)
        elif mode == "Size (smallest first)":
            shapes.sort(key=size_of)
        elif mode == "Fill % (highest first)":
            shapes.sort(key=lambda s: self._shape_fill(s)[1], reverse=True)
        elif mode == "Reflection, then size (largest)":
            shapes.sort(key=lambda s: (refl_of(s), -size_of(s)))
        elif mode == "Reflection, then size (smallest)":
            shapes.sort(key=lambda s: (refl_of(s), size_of(s)))
        else:  # "Feature ID"
            def fid_of(s):
                try:
                    return (0, float(s.get("feature_id")))
                except (TypeError, ValueError):
                    return (1, str(s.get("feature_id", "")))
            shapes.sort(key=fid_of)

        self.flist.blockSignals(True)
        self.flist.clear()
        for s in shapes:
            bounding, fill = self._shape_fill(s)
            label = (f"#{s.get('feature_id','?')} {s.get('reflection','?')} · "
                     f"{s.get('n_bins','?')} cells · area={bounding:,.0f} · "
                     f"χ={s.get('chi_deg','?')}° · {fill * 100:.0f}%")
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, s)
            self.flist.addItem(it)
        self.flist.blockSignals(False)

    # ── metric colouring ─────────────────────────────────────────────
    def _refresh_metric(self):
        cmap = self.cmap.currentText()
        self.canvas.color_by_values(self._shape_intensity_by_territory(), cmap,
                                    log_scale=True)

    def _chi_pass(self, s) -> bool:
        """True if the shape's χ is within the min/max range (None χ always shown)."""
        c = s.get("chi_deg")
        if c is None:
            return True
        return self.chi_min.value() <= float(c) <= self.chi_max.value()

    def _shape_intensity_by_territory(self) -> dict:
        """Max per-territory peak intensity over kept shapes (reflection + χ filtered)."""
        want = self.reflection.currentText()
        out: dict = {}
        for s in self.shapes:
            if want != "(all reflections)" and s.get("reflection") != want:
                continue
            if not self._chi_pass(s):
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
        bounding, fill = self._shape_fill(s)
        self.info.setText(
            f"<b>#{s.get('feature_id','?')} {s.get('reflection','?')}</b><br>"
            f"cells: {s.get('n_bins','?')} · peak I: {s.get('peak_intensity','?')} · "
            f"SNR: {s.get('mean_snr','?')}<br>"
            f"bounding area: {bounding:,.0f} (CSV²) · fill: {fill * 100:.0f}%<br>"
            f"χ = {s.get('chi_deg','?')}° · χ FWHM = {s.get('chi_fwhm','?')} · "
            f"Δ2θ FWHM = {s.get('tth_fwhm','?')}<br>"
            f"detector: ({s.get('detector_x','?')}, {s.get('detector_y','?')})<br>"
            f"<i>{s.get('reason','')}</i>")


# ─────────────────────────────────────────────────────────────────────
# In-tab builder (shown when the territorial reference isn't built yet)
# ─────────────────────────────────────────────────────────────────────
_PROGRESS_RE = re.compile(r"PROGRESS\s+(\d+)\s*/\s*(\d+)")


class TerritoryBuilder(QWidget):
    """Shown in place of the map when no territorial reference exists yet.

    A target-size spinbox + Build button that runs the whole skew-free chain
    (``xrd-app territory-build`` = territory-grid → bin → peaks → shapes, all
    ``--variant territory``) in-app — same CLI-is-the-engine path the Programs
    tab uses — with an ``(i/n)`` progress status and a Cancel button. On success
    ``on_built`` is called so the embedding tab swaps in the real map.
    """

    def __init__(self, project_root, scan, on_built=None):
        super().__init__()
        self._project_root = project_root
        self._scan = scan
        self._on_built = on_built
        self._proc = None
        self._cancelled = False

        lay = QVBoxLayout(self)
        lay.addStretch()

        msg = QLabel(f"No territorial reference for {scan or 'this scan'} yet.")
        msg.setAlignment(Qt.AlignCenter); msg.setWordWrap(True)
        msg.setStyleSheet("font-size: 1.15em;")
        lay.addWidget(msg)
        detail = QLabel(
            "Bin frames by true (X, Y) stage position into skew-free territories, "
            "then run peaks + shapes over them. Reads raw frames (heavy) and needs "
            "a real position CSV — runs once, then the mapping is cached.")
        detail.setAlignment(Qt.AlignCenter); detail.setWordWrap(True)
        detail.setStyleSheet("color:#999; font-size:0.9em;")
        lay.addWidget(detail)

        # Target-size + Build + Cancel + (i/n) status.
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(QLabel("Target frames/territory:"))
        self._target = QSpinBox()
        self._target.setRange(1, 999)
        self._target.setValue(1)
        self._target.setToolTip(
            "Frames grouped per territory before it stops growing. 1 ≈ true 1×1 "
            "resolution (cells drawn as boxes; the default); larger = higher "
            "per-cell SNR and to-scale hull footprints.")
        row.addWidget(self._target)
        self._run_btn = QPushButton("Build territorial reference")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setToolTip(
            f"Run  xrd-app territory-build --scan {scan or ''} --target-size N")
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

        # Compact output log so failures (e.g. no position CSV) are visible.
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(4000)
        self._log.setMaximumHeight(180)
        self._log.setFont(QFont("monospace", 8))
        self._log.setVisible(False)
        lay.addWidget(self._log)

        lay.addStretch()

    def _run(self):
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            return
        self._cancelled = False
        args = ["territory-build", "--root", str(self._project_root),
                "--target-size", str(self._target.value())]
        if self._scan:
            args += ["--scan", str(self._scan)]
        cmd = [sys.executable, "-m", "xrd_app.cli", *args]
        self._run_btn.setEnabled(False)
        self._target.setEnabled(False)
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
        self._target.setEnabled(True)
        self._run_btn.setEnabled(True)
        if self._cancelled:
            self._status.setText("cancelled")
            self._log.appendPlainText("\n[cancelled]")
            return
        if code == 0:
            self._status.setText("done ✓")
            self._progress.setValue(100)
            # Defer the swap-in so we don't rebuild the parent (which deletes this
            # widget) while still inside the QProcess.finished handler.
            if self._on_built is not None:
                QTimer.singleShot(0, self._on_built)
            return
        self._status.setText(f"failed (exit {code})")

    def closeEvent(self, event):  # noqa: N802 (Qt signature)
        stop_process(self._proc)
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────
# Window / tab entry points
# ─────────────────────────────────────────────────────────────────────
def build_window(project_root=".", scan=None, bin_size=1, catalog=None,
                 on_built=None) -> QWidget:
    """Build the territorial device-map widget for the current scan.

    When the territorial grid mapping is missing, returns a
    :class:`TerritoryBuilder` with a Build button that runs the chain in-app;
    ``on_built`` (called on a successful build) lets the embedding tab swap in
    the real map. Never crashes the window.
    """
    dm = DataManager(project_root, scan=scan)
    gm = _load_territory_mapping(dm, scan, bin_size)
    if gm is None:
        return TerritoryBuilder(project_root, scan, on_built=on_built)
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
