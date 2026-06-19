"""Helpers for embedding legacy QMainWindow GUIs as tab content."""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..config import DataManager

_BIN_SIZES = [1, 3, 4, 5]


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
            "color:#888; font-size:11px; padding-left:8px;")
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


def placeholder(message: str, detail: str = "") -> QWidget:
    """A simple centered message widget shown when a tab can't be built yet."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addStretch()
    lbl = QLabel(message)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("font-size: 14px;")
    lay.addWidget(lbl)
    if detail:
        d = QLabel(detail)
        d.setAlignment(Qt.AlignCenter)
        d.setWordWrap(True)
        d.setStyleSheet("color: #999; font-size: 11px;")
        lay.addWidget(d)
    lay.addStretch()
    return w
