"""Helpers for embedding legacy QMainWindow GUIs as tab content."""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from ..config import DataManager
from ..core import catalogs

# Standard bins offered when a scan has no catalogs yet; views otherwise show
# whatever bins actually have data (incl. 2×2) via catalogs.available_bins.
_BIN_SIZES = [1, 2, 3, 4, 5]


def existing_bins(dm, scan=None) -> list:
    """Bin sizes (ints, ascending) that already exist for a scan.

    Always includes 1 (raw per-frame frames) and adds every NxN that has a
    built ``xrd_NxN_bins.h5`` under the scan's Binned directory.
    """
    sizes = {1}
    if dm is not None:
        try:
            bdir = dm.binned_dir(scan)
            if bdir.is_dir():
                for p in bdir.glob("xrd_*x*_bins.h5"):
                    m = re.match(r"xrd_(\d+)x(\d+)_bins", p.name)
                    if m:
                        sizes.add(int(m.group(1)))
        except Exception:
            pass
    return sorted(sizes)


def bins_status_text(dm, bin_size, scan=None) -> str:
    """Human-readable status of the binned HDF5 for a scan/bin size.

    Mirrors the labeling tool's hint: says whether the bins were built and, if
    so, when they were computed from raw — otherwise notes raw-frame fallback.
    """
    if dm is None:
        return ""
    if bin_size == 1:
        return "1×1: raw frames (per-frame, no binning)"
    try:
        path = dm.binned_h5(bin_size, scan=scan)
    except Exception:
        path = None
    if not path:
        return f"{bin_size}×{bin_size}: raw frames (no binned file)"
    if os.path.exists(path):
        ts = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
        return f"{bin_size}×{bin_size}: built {ts:%Y-%m-%d %H:%M}"
    return f"{bin_size}×{bin_size}: not built — run Programs → Create bins"


def embed_window(win) -> QWidget:
    """Wrap a constructed QMainWindow in a container suitable for a tab.

    A QMainWindow embeds fine as a child widget; all four legacy GUIs keep their
    content in the central widget, so nothing is lost. We retain a reference on
    the container so the window is not garbage-collected.
    """
    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(win)
    container._embedded_window = win
    return container


def embed_with_toolbar(win, buttons) -> QWidget:
    """Embed a window with a thin toolbar of buttons above it.

    ``buttons`` is a list of ``(label, callback)`` tuples. Keeps the window from
    being garbage-collected.
    """
    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    bar = QHBoxLayout()
    for label, cb in buttons:
        b = QPushButton(label)
        b.clicked.connect(cb)
        bar.addWidget(b)
    bar.addStretch()
    lay.addLayout(bar)
    lay.addWidget(win)
    container._embedded_window = win
    return container


class BinnedTab(QWidget):
    """Embed a ``build_window(project_root, scan, bin_size)`` GUI with its own
    per-tab Bin selector + a bins-built status, rebuilding on bin change.

    This replaces the old global header bin-size selector: each bin-dependent
    tab now owns its bin choice, so it is unambiguous which view it applies to.
    """

    def __init__(self, build_window, project_root, scan=None, bin_size=3,
                 extra_buttons=None):
        super().__init__()
        self._build_window = build_window
        self._project_root = project_root
        self._scan = scan
        self._bin_size = bin_size if bin_size in _BIN_SIZES else _BIN_SIZES[0]
        self._win = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>Bin:</b>"))
        self._bin_combo = QComboBox()
        self._bin_combo.addItems([f"{b}x{b}" for b in _BIN_SIZES])
        self._bin_combo.setCurrentText(f"{self._bin_size}x{self._bin_size}")
        self._bin_combo.currentTextChanged.connect(self._on_bin_changed)
        bar.addWidget(self._bin_combo)
        self._status = QLabel("")
        self._status.setStyleSheet(
            "color:#888; font-size:0.9em; padding-left:8px;")
        bar.addWidget(self._status)
        for label, cb in (extra_buttons or []):
            b = QPushButton(label)
            b.clicked.connect(cb)
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)

        self._rebuild()

    def current_bin_size(self):
        return self._bin_size

    def current_window(self):
        return self._win

    def _on_bin_changed(self, text):
        try:
            self._bin_size = int(text.split("x")[0])
        except ValueError:
            return
        self._rebuild()

    def _rebuild(self):
        if self._win is not None:
            self._win.setParent(None)
            self._win.deleteLater()
            self._win = None
        try:
            self._win = self._build_window(
                self._project_root, scan=self._scan, bin_size=self._bin_size)
        except Exception as e:
            self._win = placeholder("Could not load this view.",
                                    f"{type(e).__name__}: {e}")
        self.layout().addWidget(self._win)
        self._embedded_window = self._win
        self._update_status()

    def _update_status(self):
        try:
            dm = DataManager(self._project_root, scan=self._scan)
            self._status.setText(bins_status_text(dm, self._bin_size, self._scan))
        except Exception:
            self._status.setText("")


class LineageCatalogTab(QWidget):
    """Embed a ``build_window(project_root, scan, bin_size, catalog)`` GUI with a
    **Bin → Feature catalog** selector.

    You search by feature catalog: pick a bin (only bins that have catalogs are
    listed — any size), then a feature catalog for that bin. The status line
    prints the catalog's provenance read from its lineage — the **scan** it came
    from and the peak set it was derived from. ``Browse…`` loads any catalog JSON
    not auto-discovered and reads its bin/lineage from the file.
    """

    _BROWSE = "__browse__"

    def __init__(self, build_window, project_root, scan=None, bin_size=3):
        super().__init__()
        self._build_window = build_window
        self._project_root = project_root
        self._scan = scan
        self._bin_size = bin_size if bin_size in _BIN_SIZES else _BIN_SIZES[0]
        self._feature = None    # selected feature-catalog path str
        self._win = None
        self._view_state = None  # carried layers/metric across catalog switches

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>Bin:</b>"))
        self._bin_combo = QComboBox()
        self._bin_combo.activated.connect(self._on_bin_changed)
        bar.addWidget(self._bin_combo)
        bar.addWidget(QLabel("<b>Feature catalog:</b>"))
        self._feat_combo = QComboBox()
        self._feat_combo.setMinimumWidth(260)
        self._feat_combo.setToolTip(
            "Feature catalog to display (shapes / combined / feature list) for "
            "this bin. Browse… for a catalog that isn't listed.")
        self._feat_combo.activated.connect(self._on_feature_changed)
        bar.addWidget(self._feat_combo)
        self._status = QLabel("")
        self._status.setStyleSheet("color:#888; font-size:0.9em; padding-left:8px;")
        bar.addWidget(self._status)
        bar.addStretch()
        lay.addLayout(bar)

        self._populate_bins()
        self._populate_features()
        self._rebuild()

    def current_bin_size(self):
        return self._bin_size

    def current_window(self):
        return self._win

    def _results_dir(self):
        try:
            return DataManager(self._project_root,
                               scan=self._scan).results_dir(self._scan)
        except Exception:
            return None

    def _populate_bins(self):
        rd = self._results_dir()
        bins = catalogs.available_bins(rd) if rd else []
        if not bins:                       # no catalogs yet → offer the standard set
            bins = _BIN_SIZES
        if self._bin_size not in bins:
            self._bin_size = bins[0]
        self._bin_combo.blockSignals(True)
        self._bin_combo.clear()
        for b in bins:
            self._bin_combo.addItem(f"{b}x{b}", b)
        i = self._bin_combo.findData(self._bin_size)
        self._bin_combo.setCurrentIndex(i if i >= 0 else 0)
        self._bin_combo.blockSignals(False)

    def _populate_features(self):
        rd = self._results_dir()
        feats = catalogs.feature_sources(rd, self._bin_size) if rd else []
        self._feat_combo.blockSignals(True)
        self._feat_combo.clear()
        if not feats:
            self._feat_combo.addItem("(no feature catalog)", None)
            self._feature = None
        else:
            for p in feats:
                self._feat_combo.addItem(p.name, str(p))
            i = self._feat_combo.findData(self._feature)
            if i < 0:
                self._feature = str(feats[0])
                i = self._feat_combo.findData(self._feature)
            self._feat_combo.setCurrentIndex(i if i >= 0 else 0)
        self._feat_combo.addItem("Browse…", self._BROWSE)
        self._feat_combo.blockSignals(False)

    def _on_bin_changed(self, _i):
        new_bin = self._bin_combo.currentData()
        rd = self._results_dir()
        # Carry the chosen catalog's lineage to the new bin; blank if no match.
        if rd and self._feature:
            kind = (catalogs.parse_name(Path(self._feature).name) or {}).get("kind")
            m = catalogs.match_across_bin(rd, kind, self._feature, new_bin) if kind else None
            self._feature = str(m) if m else None
        else:
            self._feature = None
        self._bin_size = new_bin
        self._populate_features()
        self._rebuild()

    def _on_feature_changed(self, _i):
        data = self._feat_combo.currentData()
        if data == self._BROWSE:
            self._browse()
            return
        self._feature = data
        self._rebuild()

    def _browse(self):
        rd = self._results_dir()
        start = str(rd) if rd else (self._project_root or "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a feature catalog JSON", start,
            "Catalog JSON (*_shapes_*.json *_combined_*.json feature_catalog_*.json *.json)")
        if not path:   # cancelled → revert the combo to the current selection
            i = self._feat_combo.findData(self._feature)
            self._feat_combo.setCurrentIndex(max(0, i))
            return
        # Load strictly by the browsed file: bin comes from its own lineage/name.
        b = catalogs.catalog_bin(path, rd)
        if b is not None:
            self._bin_size = b
            self._populate_bins()
        self._populate_features()
        if self._feat_combo.findData(path) < 0:
            at = max(0, self._feat_combo.count() - 1)  # before Browse…
            self._feat_combo.insertItem(at, Path(path).name, path)
        self._feat_combo.setCurrentIndex(self._feat_combo.findData(path))
        self._feature = path
        self._rebuild()

    def _rebuild(self):
        if self._win is not None:
            # Remember the current layers/metric so the next catalog inherits them.
            if hasattr(self._win, "get_view_state"):
                try:
                    self._view_state = self._win.get_view_state()
                except Exception:
                    pass
            self._win.setParent(None)
            self._win.deleteLater()
            self._win = None
        if not self._feature:
            self._win = placeholder(
                "Select a feature catalog to display.",
                "Choose a bin and a feature catalog above.")
        else:
            rd = self._results_dir()
            bin_size = catalogs.catalog_bin(self._feature, rd) or self._bin_size
            try:
                self._win = self._build_window(
                    self._project_root, scan=self._scan,
                    bin_size=bin_size, catalog=self._feature)
            except Exception as e:
                self._win = placeholder("Could not load this view.",
                                        f"{type(e).__name__}: {e}")
        self.layout().addWidget(self._win)
        self._embedded_window = self._win
        # Re-apply the carried layers/metric to the freshly built view.
        if self._view_state and hasattr(self._win, "apply_view_state"):
            try:
                self._win.apply_view_state(self._view_state)
            except Exception:
                pass
        self._update_status()

    def _update_status(self):
        if not self._feature:
            self._status.setText("no feature catalog selected — blank view")
            return
        rd = self._results_dir()
        b = catalogs.catalog_bin(self._feature, rd) or self._bin_size
        lin = catalogs.read_lineage(self._feature, rd)
        if isinstance(lin, dict):
            scan = lin.get("scan") or self._scan or "?"
            src = lin.get("peak_source_file")
            prov = f"scan: {scan}" + (f" · from peaks: {src}" if src else "")
        else:
            prov = "⚠ no lineage (manual)"
        self._status.setText(
            f"bin {b}×{b} · {Path(self._feature).name} · {prov}")


def placeholder(message: str, detail: str = "") -> QWidget:
    """A simple centered message widget shown when a tab can't be built yet."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addStretch()
    lbl = QLabel(message)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("font-size: 1.2em;")
    lay.addWidget(lbl)
    if detail:
        d = QLabel(detail)
        d.setAlignment(Qt.AlignCenter)
        d.setWordWrap(True)
        d.setStyleSheet("color: #999; font-size: 0.9em;")
        lay.addWidget(d)
    lay.addStretch()
    return w
