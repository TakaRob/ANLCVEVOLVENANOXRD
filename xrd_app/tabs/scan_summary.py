"""Scan Summary tab — one comparable row per scan for the whole project.

The cross-scan companion to the per-feature Device/Orientation views: pick a
**bin size** and a **catalog type** (the JSON lineage shared across scans — e.g.
the gaussian shapes at 3×3, or a *territorial* mapping) and it prints an
SQLite-style table, one row per scan, of feature count, footprint area (sum +
union), coverage %, the preferred χ (dominant azimuthal cluster) ± range, and
shape fill % (solidity). Territorial types report areas in coordinate-CSV units.

All numbers come from :mod:`core.scan_table` (the same engine the ``xrd-app
scan-table`` CLI command and the Territory Map's per-shape fill % use), so the
GUI stays a thin face over the CLI engine.
"""

from __future__ import annotations

import math

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import scan_table as st
from ._embed import placeholder

TAB_META = {
    "title": "Scan Summary",
    "order": 58,
    "takes_bin_size": False,
    "scan_dependent": False,   # project-wide: one row per scan, not per active scan
    "general": (
        "One comparable row per scan for the whole project. Choose a bin size "
        "and a catalog type (the JSON lineage — e.g. gaussian shapes 3×3, or a "
        "territorial mapping) and read, per scan: feature count, footprint area "
        "(Σ and set-union), coverage %, the preferred χ (tip of the dominant "
        "area-weighted azimuthal cluster) ± range, and shape fill % (how solidly "
        "shapes fill their convex-hull outline, area-weighted). Territorial types "
        "measure area in coordinate-CSV units, coverage against the outline hull."
    ),
}

_BIN_SIZES = [1, 3, 4, 5]
_ALL_REFL = "(all reflections)"


class _NumItem(QTableWidgetItem):
    """Table item that sorts by its numeric ``UserRole`` value when present, so a
    formatted display ("6,342", "15.5%") still orders numerically. Falls back to
    the default string compare for text cells (the Scan column)."""

    def __lt__(self, other):
        a, b = self.data(Qt.UserRole), other.data(Qt.UserRole)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


class ScanSummaryTab(QWidget):
    bin_size_changed = pyqtSignal(int)

    def __init__(self, project_root, bin_size=3):
        super().__init__()
        self._project_root = project_root
        self._dm = DataManager(project_root)
        self._bin_size = bin_size if bin_size in _BIN_SIZES else 3
        self._types = []                # catalog_types() entries for the bin

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>Bin:</b>"))
        self._bin_combo = QComboBox()
        for b in _BIN_SIZES:
            self._bin_combo.addItem(f"{b}x{b}", b)
        self._bin_combo.setCurrentIndex(self._bin_combo.findData(self._bin_size))
        self._bin_combo.activated.connect(self._on_bin_changed)
        bar.addWidget(self._bin_combo)

        bar.addWidget(QLabel("<b>Catalog type:</b>"))
        self._type_combo = QComboBox()
        self._type_combo.setMinimumWidth(240)
        self._type_combo.setToolTip(
            "The JSON lineage compared across scans (shapes/combined variant or a "
            "territorial mapping). Only types with at least one scan are listed.")
        self._type_combo.activated.connect(self._on_type_changed)
        bar.addWidget(self._type_combo)

        bar.addWidget(QLabel("<b>Reflection:</b>"))
        self._refl_combo = QComboBox()
        self._refl_combo.setMinimumWidth(120)
        self._refl_combo.activated.connect(lambda _i: self._reload())
        bar.addWidget(self._refl_combo)

        self._show_all = QCheckBox("Show all")
        self._show_all.setToolTip(
            "Lay out every reflection: an “(all reflections)” row per scan plus "
            "one row per reflection, with a Reflection column. Disables the "
            "reflection filter.")
        self._show_all.toggled.connect(self._on_show_all)
        bar.addWidget(self._show_all)

        bar.addWidget(QLabel("χ bw"))
        self._bw = QDoubleSpinBox()
        self._bw.setRange(1.0, 30.0)
        self._bw.setSingleStep(1.0)
        self._bw.setValue(5.0)
        self._bw.setSuffix("°")
        self._bw.setToolTip("Gaussian KDE bandwidth for the χ clustering (matches "
                            "the Orientation Map default of 5°).")
        self._bw.valueChanged.connect(lambda _v: self._reload())
        bar.addWidget(self._bw)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._reload)
        bar.addWidget(self._refresh_btn)
        self._copy_btn = QPushButton("Copy CSV")
        self._copy_btn.setToolTip("Copy the current table to the clipboard as CSV.")
        self._copy_btn.clicked.connect(self._copy_csv)
        bar.addWidget(self._copy_btn)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#888; padding-left:8px;")
        bar.addWidget(self._status)
        bar.addStretch()
        lay.addLayout(bar)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table, 1)

        self._rows = []
        self._meta = {}
        self._populate_types()
        self._reload()

    # ── context (project-wide tab; scan changes don't rebuild it) ──────
    def update_context(self, scan=None, bin_size=None):
        """Called by the host when the active scan/bin changes — no-op here since
        the table spans every scan; kept so the host's persistent-tab path works."""
        return

    # ── selectors ─────────────────────────────────────────────────────
    def current_bin_size(self):
        return self._bin_size

    def _populate_types(self):
        try:
            self._types = st.catalog_types(self._dm, self._bin_size)
        except Exception:
            self._types = []
        prev = self._type_combo.currentData()
        self._type_combo.blockSignals(True)
        self._type_combo.clear()
        for i, t in enumerate(self._types):
            suffix = " · territory" if t["territory"] else ""
            self._type_combo.addItem(f"{t['label']}{suffix}  ({t['scans']})", i)
        # Keep the same lineage key selected across a bin switch when possible.
        keep = 0
        if prev is not None and 0 <= prev < len(self._types):
            keep = prev
        if self._type_combo.count():
            self._type_combo.setCurrentIndex(keep)
        self._type_combo.blockSignals(False)
        self._populate_reflections()

    def _current_type(self):
        i = self._type_combo.currentData()
        if i is None or not (0 <= i < len(self._types)):
            return None
        return self._types[i]

    def _populate_reflections(self):
        t = self._current_type()
        refs = []
        if t is not None:
            try:
                refs = st.catalog_reflections(self._dm, self._bin_size, t["key"])
            except Exception:
                refs = []
        prev = self._refl_combo.currentText()
        self._refl_combo.blockSignals(True)
        self._refl_combo.clear()
        self._refl_combo.addItem(_ALL_REFL)
        self._refl_combo.addItems(refs)
        j = self._refl_combo.findText(prev)
        self._refl_combo.setCurrentIndex(j if j >= 0 else 0)
        self._refl_combo.blockSignals(False)

    def _on_bin_changed(self, _i):
        self._bin_size = self._bin_combo.currentData()
        self._populate_types()
        self._reload()
        self.bin_size_changed.emit(self._bin_size)

    def _on_type_changed(self, _i):
        self._populate_reflections()
        self._reload()

    def _on_show_all(self, checked):
        # With every reflection broken out into its own row, the single-reflection
        # filter is meaningless — grey it out.
        self._refl_combo.setEnabled(not checked)
        self._reload()

    # ── build the table ───────────────────────────────────────────────
    def _reload(self):
        t = self._current_type()
        if t is None:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._status.setText(
                f"no catalogs at {self._bin_size}x{self._bin_size} "
                "— run peaks/shapes first")
            return
        breakdown = self._show_all.isChecked()
        want = self._refl_combo.currentText()
        refs = None if (breakdown or want == _ALL_REFL) else [want]
        try:
            self._rows, self._meta = st.scan_table_rows(
                self._dm, self._bin_size, t["key"], refs=refs,
                bandwidth=float(self._bw.value()), breakdown=breakdown)
        except Exception as e:
            self._rows, self._meta = [], {}
            self._status.setText(f"error: {type(e).__name__}: {e}")
        self._fill_table()

    def _headers(self):
        return st.column_headers(self._meta)

    def _fill_table(self):
        headers = self._headers()
        helps = st.column_help(self._meta)
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        # Explain each terse header on hover — especially "shape solidity",
        # which isn't obvious from the label alone.
        for c, tip in enumerate(helps):
            hitem = self._table.horizontalHeaderItem(c)
            if hitem is not None:
                hitem.setToolTip(tip)
        self._table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            for c, key in enumerate(st.COLUMNS):
                item = _NumItem()
                v = row.get(key)
                item.setData(Qt.DisplayRole, self._display(key, v))
                # Sort numerically where the value is numeric.
                if isinstance(v, (int, float)) and not (
                        isinstance(v, float) and math.isnan(v)):
                    item.setData(Qt.UserRole, float(v))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(r, c, item)
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()
        terr = " · territory (CSV² units)" if self._meta.get("territory") else ""
        self._status.setText(f"{len(self._rows)} scan(s){terr}")

    @staticmethod
    def _display(key, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if key in ("Coverage %", "Fill %"):
            return f"{v:.1f}%"
        if key == "Preferred χ":
            return f"{v:.0f}°"
        if key == "χ ± range":
            return f"±{v:.0f}°"
        if key in ("Area sum", "Area union", "Total") and isinstance(v, (int, float)):
            return f"{v:,.0f}"
        return str(v)

    def _copy_csv(self):
        if not self._rows:
            return
        headers = self._headers()
        lines = [",".join(headers)]
        for row in self._rows:
            cells = []
            for key in st.COLUMNS:
                v = row.get(key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("")
                elif isinstance(v, float):
                    cells.append(f"{v:.4g}")
                else:
                    cells.append(str(v))
            lines.append(",".join(cells))
        QApplication.clipboard().setText("\n".join(lines))
        self._status.setText(f"copied {len(self._rows)} rows as CSV")


def make_tab(project_root=".", scan=None, bin_size=3):
    try:
        return ScanSummaryTab(project_root, bin_size=bin_size)
    except Exception as e:
        return placeholder("Could not load the Scan Summary.",
                           f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
