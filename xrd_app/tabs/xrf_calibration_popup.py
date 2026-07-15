"""XRF calibration dialog — elements + energy calibration for the ME7 fluorescence.

The XRF analogue of the "Manual reflections" dialog (:mod:`tabs.reflection_popup`):
show the whole-scan grand-sum MCA spectrum, edit the element ROI table (name /
emission energy / window), pick lines from a bundled emission-line dictionary,
detect peaks in the spectrum, and set the linear energy calibration
(``eV = bin·ev_per_bin + offset``). Saves ``xrf_elements.json`` into the scan's
Metadata dir — exactly the config :mod:`core.xrf` / ``xrd-app xrf`` read.

All heavy logic lives in :mod:`core.xrf`; this module only edits + plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import xrf as xrf_core
from ..gui.palette import element_colors, hex_to_rgba

_COLS = ["Element", "Line (eV)", "± Window (eV)"]


def _me7_dir(dm: DataManager, scan):
    """ME7 directory for a scan (prefers the local copy), or None."""
    try:
        return dm.me7_dir(scan=scan)
    except Exception:
        return None


class XrfCalibrationDialog(QDialog):
    def __init__(self, project_root, scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.setWindowTitle("XRF calibration — elements & energy")
        self.resize(1100, 720)
        self.project_root = project_root
        self.dm = DataManager(project_root, scan=scan)
        self.scan = self.dm._scan()
        self._filling = False
        self._spectrum = None      # (N_BINS,) grand-sum counts
        self._detected = []        # detected peak dicts from find_spectrum_peaks

        self.cfg = self._load_cfg()
        cal = self.cfg.get("calibration", {})

        root = QVBoxLayout(self)

        # ---- scan selector -------------------------------------------------
        top = QHBoxLayout()
        top.addWidget(QLabel("Scan:"))
        self.scan_combo = QComboBox()
        scans = self.dm.discover_scans() if hasattr(self.dm, "discover_scans") else []
        names = [s if isinstance(s, str) else getattr(s, "name", str(s)) for s in scans]
        if self.scan and self.scan not in names:
            names = [self.scan] + names
        self.scan_combo.addItems(names or ([self.scan] if self.scan else ["(no scan)"]))
        if self.scan:
            i = self.scan_combo.findText(self.scan)
            if i >= 0:
                self.scan_combo.setCurrentIndex(i)
        self.scan_combo.currentIndexChanged.connect(self._on_scan_changed)
        top.addWidget(self.scan_combo)
        top.addStretch()
        self.status = QLabel("")
        self.status.setStyleSheet("color:#888; font-family:monospace; font-size:0.85em;")
        top.addWidget(self.status)
        root.addLayout(top)

        # ---- spectrum plot -------------------------------------------------
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "Energy", units="keV")
        self.plot.setLabel("left", "counts")
        self.plot.setLogMode(x=False, y=True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_click)
        root.addWidget(self.plot, 1)

        # ---- calibration + spectrum controls ------------------------------
        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("eV / bin:"))
        self.evbin_spin = QDoubleSpinBox()
        self.evbin_spin.setRange(0.1, 200.0)
        self.evbin_spin.setDecimals(3)
        self.evbin_spin.setValue(float(cal.get("ev_per_bin", 10.0)))
        self.evbin_spin.valueChanged.connect(self._redraw)
        cal_row.addWidget(self.evbin_spin)
        cal_row.addWidget(QLabel("offset (eV):"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-5000.0, 5000.0)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setValue(float(cal.get("offset_ev", 0.0)))
        self.offset_spin.valueChanged.connect(self._redraw)
        cal_row.addWidget(self.offset_spin)
        cal_row.addSpacing(16)
        cal_row.addWidget(QLabel("Recompute rows:"))
        self.maxfiles_spin = QSpinBox()
        self.maxfiles_spin.setRange(0, 2000)
        self.maxfiles_spin.setValue(60)
        self.maxfiles_spin.setToolTip("Subsample this many ME7 rows for a fast "
                                      "grand-sum preview (0 = all rows).")
        cal_row.addWidget(self.maxfiles_spin)
        b_recompute = QPushButton("Recompute from ME7")
        b_recompute.clicked.connect(self._recompute_spectrum)
        cal_row.addWidget(b_recompute)
        b_detect = QPushButton("Detect peaks")
        b_detect.clicked.connect(self._detect_peaks)
        cal_row.addWidget(b_detect)
        cal_row.addStretch()
        root.addLayout(cal_row)

        # ---- table + dictionary side by side ------------------------------
        mid = QHBoxLayout()

        tbox = QGroupBox("Elements (ROIs)")
        tl = QVBoxLayout(tbox)
        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        tl.addWidget(self.table)
        trow = QHBoxLayout()
        b_new = QPushButton("New row")
        b_new.clicked.connect(self._add_row)
        b_del = QPushButton("Delete selected")
        b_del.clicked.connect(self._delete_rows)
        b_add_det = QPushButton("Add detected →")
        b_add_det.setToolTip("Add every detected peak as an element, naming each "
                             "from the nearest tabulated emission line.")
        b_add_det.clicked.connect(self._add_detected)
        trow.addWidget(b_new); trow.addWidget(b_del); trow.addWidget(b_add_det)
        trow.addStretch()
        tl.addLayout(trow)
        mid.addWidget(tbox, 2)

        dbox = QGroupBox("Emission-line dictionary")
        dl = QVBoxLayout(dbox)
        self.dict_combo = QComboBox()
        for e in sorted(xrf_core.EMISSION_LINES, key=lambda x: x["energy_ev"]):
            label = f"{xrf_core.emission_line_name(e):8}  {e['energy_ev']:.0f} eV"
            self.dict_combo.addItem(label, e)
        dl.addWidget(self.dict_combo)
        b_add_line = QPushButton("Add selected line →")
        b_add_line.clicked.connect(self._add_dict_line)
        dl.addWidget(b_add_line)
        dl.addWidget(QLabel("Tip: click the spectrum to add the\nnearest "
                            "tabulated line at that energy."))
        dl.addStretch()
        mid.addWidget(dbox, 1)
        root.addLayout(mid)

        # ---- action buttons ------------------------------------------------
        btns = QHBoxLayout()
        btns.addStretch()
        b_save = QPushButton("Save calibration")
        b_save.setStyleSheet("font-weight: bold;")
        b_save.clicked.connect(self._save)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.reject)
        btns.addWidget(b_save); btns.addWidget(b_close)
        root.addLayout(btns)

        self._fill_table(self.cfg.get("elements", []))
        self._load_spectrum()
        self._redraw()

    # ----- config / spectrum IO --------------------------------------------
    def _cfg_path(self, scan=None):
        scan = scan or self.scan
        per_scan = self.dm.metadata_scan_dir(scan) / "xrf_elements.json"
        proj = self.dm.metadata_dir / "xrf_elements.json"
        return per_scan if per_scan.exists() else (proj if proj.exists() else None)

    def _load_cfg(self):
        try:
            p = self._cfg_path()
            return xrf_core.read_config(p) if p else xrf_core.default_config()
        except Exception:
            return xrf_core.default_config()

    def _load_spectrum(self):
        """Prefer the saved product's grand-sum spectrum (instant); else None."""
        self._spectrum = None
        try:
            path = self.dm.xrf_product(scan=self.scan)
            if path.exists():
                prod = xrf_core.load_product(path)
                if prod.get("spectrum") is not None:
                    self._spectrum = np.asarray(prod["spectrum"], dtype=float)
                    self.status.setText("spectrum: saved product grand-sum")
                    return
        except Exception:
            pass
        self.status.setText("no saved spectrum — click 'Recompute from ME7'")

    def _recompute_spectrum(self):
        me7 = _me7_dir(self.dm, self.scan)
        if me7 is None:
            QMessageBox.information(self, "No ME7",
                                   f"No ME7 directory found for {self.scan}.")
            return
        channels = list(self.cfg.get("channels", []))
        deadtime = bool(self.cfg.get("deadtime_correction", True))
        mf = self.maxfiles_spin.value() or None
        self.status.setText("summing ME7 …")
        self.repaint()
        try:
            self._spectrum = xrf_core.grand_sum_spectrum(
                me7, channels, deadtime, max_files=mf)
        except Exception as e:
            QMessageBox.warning(self, "Recompute failed", str(e))
            return
        self.status.setText(f"spectrum: recomputed ({mf or 'all'} rows)")
        self._redraw()

    # ----- table -----------------------------------------------------------
    def _fill_table(self, elements):
        self._filling = True
        self.table.setRowCount(0)
        for el in elements:
            self._append_row(el.get("name", ""), el.get("line_ev", 0.0),
                             el.get("half_width_ev",
                                    xrf_core.DEFAULT_HALF_WIDTH_EV))
        self._filling = False

    def _append_row(self, name, line_ev, half_width_ev):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(name)))
        self.table.setItem(r, 1, QTableWidgetItem(f"{float(line_ev):.0f}"))
        self.table.setItem(r, 2, QTableWidgetItem(f"{float(half_width_ev):.0f}"))

    def _elements_from_table(self):
        els = []
        for r in range(self.table.rowCount()):
            name = (self.table.item(r, 0).text().strip()
                    if self.table.item(r, 0) else "")
            if not name:
                continue
            try:
                line = float(self.table.item(r, 1).text())
                hw = float(self.table.item(r, 2).text())
            except (ValueError, AttributeError):
                continue
            els.append({"name": name, "line_ev": line, "half_width_ev": hw})
        return els

    def _on_item_changed(self, *_):
        if not self._filling:
            self._redraw()

    def _add_row(self):
        self._append_row("New", 0.0, xrf_core.DEFAULT_HALF_WIDTH_EV)

    def _delete_rows(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self._redraw()

    def _add_dict_line(self):
        e = self.dict_combo.currentData()
        if e:
            self._append_row(xrf_core.emission_line_name(e), e["energy_ev"],
                             xrf_core.DEFAULT_HALF_WIDTH_EV)
            self._redraw()

    def _add_detected(self):
        if not self._detected:
            QMessageBox.information(self, "No peaks",
                                   "Click 'Detect peaks' first.")
            return
        for d in self._detected:
            ev = d["energy_ev"]
            match = xrf_core.nearest_emission_line(ev, tol_ev=150.0)
            name = (xrf_core.emission_line_name(match) if match
                    else f"Peak_{ev:.0f}")
            self._append_row(name, ev, xrf_core.DEFAULT_HALF_WIDTH_EV)
        self._redraw()

    # ----- peak detection --------------------------------------------------
    def _detect_peaks(self):
        if self._spectrum is None:
            QMessageBox.information(self, "No spectrum",
                                   "Load or recompute a spectrum first.")
            return
        ev_per_bin = self.evbin_spin.value()
        offset = self.offset_spin.value()
        ref = {**xrf_core.DEFAULT_REFINEMENT, **self.cfg.get("refinement", {})}
        try:
            peaks, _ = xrf_core.find_spectrum_peaks(
                self._spectrum, ev_per_bin, offset,
                prominence_frac=ref["prominence_frac"],
                min_width_bins=ref["min_width_bins"],
                noise_floor_ev=ref["noise_floor_ev"],
                incident_ev=ref["incident_ev"],
                exclude_incident_lo_ev=ref["exclude_incident_lo_ev"],
                exclude_incident_hi_ev=ref["exclude_incident_hi_ev"])
        except Exception as e:
            QMessageBox.warning(self, "Detect failed", str(e))
            return
        self._detected = [{"bin": int(b), "energy_ev": b * ev_per_bin + offset}
                          for b in peaks]
        self.status.setText(f"detected {len(self._detected)} peak(s)")
        self._redraw()

    # ----- drawing ---------------------------------------------------------
    def _redraw(self, *_):
        self.plot.clear()
        if self._spectrum is None:
            return
        ev_per_bin = self.evbin_spin.value()
        offset = self.offset_spin.value()
        spec = self._spectrum
        e_kev = (np.arange(spec.shape[0]) * ev_per_bin + offset) / 1000.0
        disp = np.clip(spec, 1.0, None)
        self.plot.plot(e_kev, disp, pen=pg.mkPen("#333", width=1))

        elements = self._elements_from_table()
        colors = element_colors([el["name"] for el in elements])
        max_hi = 0.0
        for el in elements:
            center = float(el["line_ev"])
            _, _, lo_ev, hi_ev = xrf_core.roi_bins(center, el, ev_per_bin, offset)
            lo, hi = lo_ev / 1000.0, hi_ev / 1000.0
            max_hi = max(max_hi, hi)
            col = colors.get(el["name"], "#888888")
            reg = pg.LinearRegionItem(values=(lo, hi), movable=False,
                                      brush=pg.mkBrush(hex_to_rgba(col, 70)),
                                      pen=pg.mkPen(None))
            reg.setZValue(-10)
            self.plot.addItem(reg)
            line = pg.InfiniteLine(pos=center / 1000.0, angle=90,
                                   pen=pg.mkPen(col, width=1, style=Qt.DashLine),
                                   label=el["name"],
                                   labelOpts={"color": col, "position": 0.95})
            self.plot.addItem(line)

        # detected peaks as thin grey markers
        for d in self._detected:
            self.plot.addItem(pg.InfiniteLine(
                pos=d["energy_ev"] / 1000.0, angle=90,
                pen=pg.mkPen("#d08000", width=1, style=Qt.DotLine)))

        hi_kev = min(20.0, max(16.0, max_hi + 2)) if max_hi else 16.0
        self.plot.setXRange(0.0, hi_kev, padding=0.02)

    def _on_plot_click(self, ev):
        """Click the spectrum → add the nearest tabulated line at that energy."""
        if self._spectrum is None:
            return
        vb = self.plot.getViewBox()
        pt = vb.mapSceneToView(ev.scenePos())
        energy_ev = float(pt.x()) * 1000.0
        match = xrf_core.nearest_emission_line(energy_ev, tol_ev=200.0)
        if match:
            self._append_row(xrf_core.emission_line_name(match),
                             match["energy_ev"], xrf_core.DEFAULT_HALF_WIDTH_EV)
        else:
            self._append_row(f"Peak_{energy_ev:.0f}", energy_ev,
                             xrf_core.DEFAULT_HALF_WIDTH_EV)
        self._redraw()

    # ----- scan switch / save ----------------------------------------------
    def _on_scan_changed(self):
        self.scan = self.scan_combo.currentText()
        self.dm = DataManager(self.project_root, scan=self.scan)
        self.cfg = self._load_cfg()
        cal = self.cfg.get("calibration", {})
        self.evbin_spin.blockSignals(True)
        self.evbin_spin.setValue(float(cal.get("ev_per_bin", 10.0)))
        self.evbin_spin.blockSignals(False)
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(float(cal.get("offset_ev", 0.0)))
        self.offset_spin.blockSignals(False)
        self._detected = []
        self._fill_table(self.cfg.get("elements", []))
        self._load_spectrum()
        self._redraw()

    def _save(self):
        els = self._elements_from_table()
        if not els:
            QMessageBox.information(self, "No elements",
                                   "Add at least one element before saving.")
            return
        cfg = dict(self.cfg)
        cfg["elements"] = els
        cfg["calibration"] = {"ev_per_bin": self.evbin_spin.value(),
                              "offset_ev": self.offset_spin.value()}
        cfg.setdefault("detector", "ME7")
        cfg.setdefault("channels", list(range(xrf_core.N_CHANNELS)))
        cfg.setdefault("deadtime_correction", True)
        out = self.dm.metadata_scan_dir(self.scan) / "xrf_elements.json"
        try:
            xrf_core.write_config(cfg, out)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self.cfg = cfg
        self.status.setText(f"saved → {out}")
        QMessageBox.information(
            self, "Saved",
            f"Wrote {len(els)} elements + calibration to:\n{out}\n\n"
            "Re-run 'xrd-app xrf' (or the pipeline) to rebuild the element maps.")


def open_xrf_calibration_dialog(project_root, scan=None, bin_size=3, parent=None):
    """Factory mirroring ``reflection_popup.open_reflection_dialog``."""
    return XrfCalibrationDialog(project_root, scan=scan, bin_size=bin_size,
                                parent=parent)
