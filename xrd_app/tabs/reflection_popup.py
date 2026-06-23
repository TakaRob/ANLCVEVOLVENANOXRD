"""Manual reflection editor (pyqtgraph).

Computes a radial (2θ) histogram on the fully-summed #x# binned image, shows the
summed image / 2θ map on the left and the histogram on the right. Reflections
are edited in an Excel-like table (Name, Bragg 2θ, Width); a "2θ reflections"
checkbox loads the saved set into the table and overlays colored arcs on both
the image and the histogram (matching the View/Label GUI). On Create the set is
written (per-scan reflections.json + generated reflections.py) plus a PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..config import DataManager
from ..core import algorithms as nralgo
from ..core import io
from ..core import reflections as refl_io
from ..gui.palette import ARC_COLORS, COLORMAPS, _get_cmap, _hex_rgb

# Detector's own non-parametric radial-median subtraction (the original default),
# offered alongside the parametric models from the shared algorithm library.
_RADIAL_MEDIAN = "radial_median"

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

_OVERLAY_ALPHA = 64        # band transparency on the detector image
_LINE_TOL = 0.3            # ± deg drawn on the image when a reflection width is 0

_COLS = ["Name", "2θ (deg)", "Width ±(deg)"]


class _Cancelled(Exception):
    """Raised from the compute progress callback when the user cancels."""


def _arc_color(idx):
    return ARC_COLORS[idx % len(ARC_COLORS)]


class ReflectionDialog(QDialog):
    def __init__(self, project_root, scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self._project_root = project_root
        self.dm = DataManager(project_root, scan=scan)
        self.scan = scan or self.dm.scan_name
        self.bin_size = bin_size
        self.reflections = list(refl_io.read_json(self.dm.reflections_json(scan)))
        self._image = None          # raw summed image
        self._cleaned = None        # summed image after optional noise reduction
        self._tth = None            # currently displayed 2θ map (may be edited)
        self._tth_raw = None        # 2θ map as loaded from disk
        self._centers = None
        self._profile = None
        self._overlay_items = []    # image-overlay items (band + labels)
        self._center = None         # estimated (y0, x0) beam center
        self._det = None            # detector module (for noise reduction)
        self._tth_data = None
        # When no binned file exists, summing raw frames is slow: the first
        # Compute press warns and arms this flag; the second press runs on raw.
        self._raw_armed = False
        self._computing = False     # a Compute is in progress (button → Cancel)
        self._cancel = False        # set by a second click to abort the compute
        self._show_overlay = False  # 2θ arcs drawn on image + histogram
        # Colormap-level state: remember the intensity image's histogram region
        # so contraction sticks; auto-level once when a fresh image is loaded.
        self._summed_levels = None
        self._fresh_image = True
        self._showing_summed = False
        self._cmap_name = "inferno"
        self._filling = False       # guard against itemChanged during programmatic fill
        self._last_tth = 0.0        # last picked 2θ (seeds New row)

        self.setWindowTitle(f"Reflections — {self.dm.scan_name or project_root}")
        self.resize(1200, 760)
        lay = QVBoxLayout(self)

        # ---- scan selector (top-left, like the main GUI) ---------------
        trow = QHBoxLayout()
        trow.addWidget(QLabel("<b>Scan:</b>"))
        self.scan_combo = QComboBox()
        self.scan_combo.setMinimumWidth(150)
        self.scan_combo.currentTextChanged.connect(self._on_scan_changed)
        trow.addWidget(self.scan_combo)
        trow.addStretch()
        lay.addLayout(trow)

        # ---- figures (pyqtgraph) ---------------------------------------
        figs = QHBoxLayout()
        self.img_view = pg.ImageView()
        self.img_view.ui.roiBtn.hide()
        self.img_view.ui.menuBtn.hide()
        try:
            self.img_view.setColorMap(_get_cmap(self._cmap_name))
        except Exception:
            pass
        figs.addWidget(self.img_view, 1)

        self.hist = pg.PlotWidget()
        self.hist.setBackground("w")
        self.hist.setLabel("bottom", "2θ", units="deg")
        self.hist.setLabel("left", "mean intensity")
        for ax in ("bottom", "left"):
            self.hist.getAxis(ax).setPen("#333")
            self.hist.getAxis(ax).setTextPen("#333")
        self.hist.showGrid(x=True, y=True, alpha=0.15)
        self.hist.scene().sigMouseClicked.connect(self._on_hist_click)
        figs.addWidget(self.hist, 1)
        lay.addLayout(figs, 1)

        # ---- compute controls ------------------------------------------
        crow = QHBoxLayout()
        crow.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Summed image", "2θ map (tth.tiff)"])
        # Default to the summed image so a loaded/computed sum is shown directly.
        self.view_combo.currentIndexChanged.connect(self._redraw_image)
        crow.addWidget(self.view_combo)
        crow.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText(self._cmap_name)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        crow.addWidget(self.cmap_combo)
        self.log_cb = QCheckBox("Log scale")
        self.log_cb.setToolTip(
            "Display the summed image on a log1p intensity scale (does not affect "
            "the saved data or the 2θ histogram)")
        self.log_cb.setChecked(True)
        self.log_cb.toggled.connect(self._on_log_changed)
        crow.addWidget(self.log_cb)
        # Bin size is intentionally not exposed: every bin size yields the same
        # grand sum, so the dialog auto-picks the fastest source and keeps a
        # single saved summed image per scan.
        self.noise_cb = QCheckBox("Noise reduction")
        self.noise_cb.setToolTip(
            "Radial background subtraction before the 2θ histogram is computed")
        self.noise_cb.toggled.connect(self._on_noise_changed)
        crow.addWidget(self.noise_cb)
        self.noise_algo_combo = QComboBox()
        self.noise_algo_combo.addItem("Radial median (detector)", _RADIAL_MEDIAN)
        for key in nralgo.ALGORITHM_NAMES:
            self.noise_algo_combo.addItem(nralgo.ALGORITHM_DISPLAY[key], key)
        self.noise_algo_combo.setEnabled(False)
        self.noise_algo_combo.setToolTip(
            "Background-subtraction method (enabled when 'Noise reduction' is on)")
        self.noise_algo_combo.currentIndexChanged.connect(self._on_noise_changed)
        crow.addWidget(self.noise_algo_combo)
        crow.addWidget(QLabel("Max bins to sum (0 = all):"))
        self.max_bins = QSpinBox()
        self.max_bins.setRange(0, 1_000_000)
        crow.addWidget(self.max_bins)
        self.compute_btn = QPushButton("Compute histogram")
        self.compute_btn.setToolTip(
            "Sum the bins and compute the 2θ histogram. While running, click "
            "again to cancel.")
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        crow.addWidget(self.compute_btn)
        self.status = QLabel("click 'Compute' (then click the histogram to set 2θ)")
        crow.addWidget(self.status, 1)
        lay.addLayout(crow)

        # ---- geometry / 2θ calibration (edit the tth map) --------------
        grow = QHBoxLayout()
        self.center_label = QLabel("beam center: —")
        self.center_label.setStyleSheet("color:#555;")
        grow.addWidget(self.center_label)
        grow.addSpacing(12)
        grow.addWidget(QLabel("Detected 2θ:"))
        self.detect_combo = QComboBox()
        self.detect_combo.setMinimumWidth(110)
        self.detect_combo.activated.connect(self._on_detected)
        grow.addWidget(self.detect_combo)
        grow.addSpacing(12)
        grow.addWidget(QLabel("2θ offset:"))
        self.tth_offset = QDoubleSpinBox()
        self.tth_offset.setRange(-20, 20)
        self.tth_offset.setDecimals(4)
        self.tth_offset.setSingleStep(0.01)
        grow.addWidget(self.tth_offset)
        grow.addWidget(QLabel("2θ scale:"))
        self.tth_scale = QDoubleSpinBox()
        self.tth_scale.setRange(0.5, 2.0)
        self.tth_scale.setDecimals(4)
        self.tth_scale.setSingleStep(0.001)
        self.tth_scale.setValue(1.0)
        grow.addWidget(self.tth_scale)
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Preview tth' = scale·tth + offset")
        apply_btn.clicked.connect(self._apply_geometry)
        grow.addWidget(apply_btn)
        self.save_tth_btn = QPushButton("Save tth…")
        self.save_tth_btn.clicked.connect(self._save_tth)
        grow.addWidget(self.save_tth_btn)
        grow.addStretch()
        lay.addLayout(grow)

        # ---- reflection-set controls -----------------------------------
        srow = QHBoxLayout()
        self.tth_cb = QCheckBox("2θ reflections")
        self.tth_cb.setToolTip(
            "Load the saved reflections into the table and overlay colored arcs "
            "on the image and histogram")
        self.tth_cb.toggled.connect(self._on_overlay_toggled)
        srow.addWidget(self.tth_cb)
        srow.addSpacing(12)
        srow.addWidget(QLabel("Reflection set:"))
        self.set_combo = QComboBox()
        self.set_combo.addItems(["Current (reflections.json)", "New"])
        self.set_combo.currentIndexChanged.connect(self._on_set_changed)
        srow.addWidget(self.set_combo)
        srow.addStretch()
        add_btn = QPushButton("New row")
        add_btn.clicked.connect(lambda: self._add_row())
        srow.addWidget(add_btn)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_rows)
        srow.addWidget(del_btn)
        lay.addLayout(srow)

        # ---- editable table --------------------------------------------
        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(220)
        self.table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.table)

        # ---- actions ---------------------------------------------------
        brow = QHBoxLayout()
        brow.addStretch()
        save_refl_btn = QPushButton("Save reflections")
        save_refl_btn.setToolTip("Write reflections.json + reflections.py")
        save_refl_btn.clicked.connect(self._save_reflections)
        brow.addWidget(save_refl_btn)
        save_png_btn = QPushButton("Save PNG")
        save_png_btn.setToolTip("Save the 2θ histogram figure as a PNG")
        save_png_btn.clicked.connect(self._save_png)
        brow.addWidget(save_png_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        brow.addWidget(close_btn)
        lay.addLayout(brow)

        self._fill_table(self.reflections)
        self._redraw_hist()
        self._populate_scans()
        self._load_tth_for_display()
        # Restore the saved 2θ offset/scale for this scan and re-apply it as a
        # preview, so the dialog opens showing the value instead of 0.
        if self._load_calib() and self._tth_raw is not None:
            self._apply_geometry()
        # Bring back the saved summed image for this scan, if any
        # (the histogram is recomputed fresh from it).
        self._load_sum()

    # ----- source selection -------------------------------------------
    def _source_bin(self):
        """Bin size to sum from. All sizes give the same grand sum, so prefer the
        fastest: a prebuilt binned h5 (fewest datasets) if one exists, else raw 1×1."""
        from ._embed import existing_bins
        built = [b for b in existing_bins(self.dm, self.scan) if b != 1]
        return max(built) if built else 1

    # ----- noise reduction --------------------------------------------
    def _apply_noise_reduction(self, image):
        """Radial background subtraction using the selected method.

        ``radial_median`` uses the project detector's non-parametric median
        subtraction (the original behavior); the other choices fit a parametric
        background model (gaussian, split/skewed gaussian, fourier low-pass) —
        the same library the View/Label and Shape/Verify GUIs use.
        """
        algo = self.noise_algo_combo.currentData() or _RADIAL_MEDIAN
        if algo == _RADIAL_MEDIAN:
            if self._det is None:
                self._det = io.load_module(self.dm.detector_script())
            tth_data = self._det.precompute_tth(self._tth)
            return self._det.radial_median_subtract(image, tth_data)
        edges, centers, n_bins, indices, counts = nralgo.compute_tth_binning(self._tth)
        valid = counts > 50
        profile = nralgo.compute_radial_profile(image, indices, n_bins)
        fits = nralgo.fit_all_models(centers, profile, valid, edges[0], edges[-1])
        if algo not in fits:
            raise RuntimeError(f"{algo} background fit did not converge")
        bg = nralgo.build_background_image(self._tth, centers, fits[algo]["profile"], indices)
        return np.clip(nralgo.subtract_background(image, bg), 0, None)

    def _noise_label(self):
        """Human-readable description of the active noise-reduction method."""
        if not self.noise_cb.isChecked():
            return "off"
        return self.noise_algo_combo.currentText()

    def _on_noise_changed(self, *_):
        """Noise checkbox/method changed: recompute the histogram from the sum."""
        self.noise_algo_combo.setEnabled(self.noise_cb.isChecked())
        if self._image is None:
            return
        self._recompute_histogram()
        self.status.setText(f"histogram recomputed (noise: {self._noise_label()})")

    def _recompute_histogram(self):
        """Recompute the 2θ histogram fresh from the current summed image.

        The summed image is the persisted artifact; the histogram (and any noise
        reduction) is always derived here from the live options + 2θ map. The
        summed image is drawn whenever it is loaded, even if no 2θ map is present
        (in which case only the histogram is skipped).
        """
        if self._image is not None:
            self._cleaned = self._image
            if self.noise_cb.isChecked() and self._tth is not None:
                try:
                    self._cleaned = self._apply_noise_reduction(self._image)
                except Exception as e:
                    self.status.setText(f"noise reduction failed: {e}")
                    self._cleaned = self._image
            if self._tth is not None:
                self._centers, self._profile = io.radial_profile(self._cleaned, self._tth)
                self._populate_detected()
            else:
                self._centers = self._profile = None
        self._redraw_image()
        self._redraw_hist()

    # ----- compute ----------------------------------------------------
    def _on_compute_clicked(self):
        """Compute button: start a compute, or cancel one already running."""
        if self._computing:
            self._cancel = True
            self.status.setText("cancelling…")
            return
        self._compute()

    def _set_computing(self, on):
        """Toggle the Compute button into a Cancel button and lock controls."""
        self._computing = on
        self.compute_btn.setText("Cancel" if on else "Compute histogram")
        for w in (self.noise_cb, self.max_bins, self.view_combo):
            w.setEnabled(not on)

    def _compute(self):
        tth_path = self.dm.tth_map(scan=self.scan)
        if self._tth_raw is None and Path(tth_path).exists():
            self._tth_raw = io.load_tth_map(tth_path)
            self._tth = self._tth_raw.copy()
        if self._tth is None:
            self.status.setText("missing tth map (load tth.tiff or convert a .poni)")
            return

        bin_size = self._source_bin()
        has_h5 = bin_size != 1
        if not has_h5:
            # No binned file anywhere → raw frames. Confirm on a second press
            # first, since summing raw is much slower than reading prebuilt bins.
            if not self._raw_armed:
                self._raw_armed = True
                self.status.setText(
                    "No binned file — press 'Compute histogram' again to "
                    "calculate from raw frames (slower).")
                return

        mb = self.max_bins.value() or None
        noun = "raw frames" if not has_h5 else "bins"

        def _progress(i, n):
            if self._cancel:
                raise _Cancelled()
            if n and (i % max(1, n // 100) == 0 or i == n):
                self.status.setText(f"summing {noun}… {i}/{n}")
            QApplication.processEvents()  # keep the Cancel button responsive

        self._cancel = False
        self._set_computing(True)
        self.status.setText(f"summing {noun}…")
        self.status.repaint()
        try:
            src = io.open_bin_source(self.dm, bin_size, self.scan)
            try:
                self._image = src.sum_all(max_bins=mb, progress=_progress)
                is_raw = src.is_raw
            finally:
                src.close()
        except _Cancelled:
            self.status.setText("compute cancelled")
            return
        except Exception as e:
            self.status.setText(f"could not sum data: {e}")
            return
        finally:
            self._set_computing(False)

        self._raw_armed = False
        self._raw_active = is_raw
        self._fresh_image = True   # new image → auto-level once
        self._summed_levels = None
        # Persist only the summed image; the histogram is derived fresh from it.
        saved = self._save_sum()
        self._recompute_histogram()
        raw = f" · {io.RAW_FALLBACK_NOTE}" if self._raw_active else ""
        save_msg = f"  · saved summed image → {saved.name}" if saved else ""
        self.status.setText(
            f"summed {self._image.shape} (noise: {self._noise_label()}){raw}{save_msg}")

    # ----- saved summed-image ("sum data") store ----------------------
    # One summed image is persisted per scan; the 2θ histogram is always
    # recomputed fresh from it (see _recompute_histogram), so it reflects the
    # current noise method and 2θ map rather than a frozen result.
    def _sum_dir(self):
        return self.dm.metadata_scan_dir(self.scan) if self.scan else self.dm.metadata_dir

    def _sum_path(self):
        return Path(self._sum_dir()) / "reflection_sum.npz"

    # ----- saved 2θ offset/scale (per scan) ---------------------------
    # The offset/scale are a living calibration preview (tth' = scale·tth +
    # offset); the value is remembered per scan so reopening shows it instead
    # of 0 and re-applies the preview. Saving the tth map bakes it in and
    # clears this (so it isn't double-applied).
    def _calib_path(self):
        return Path(self._sum_dir()) / "reflection_calibration.json"

    def _save_calib(self):
        path = self._calib_path()
        try:
            import json
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump({"offset": float(self.tth_offset.value()),
                           "scale": float(self.tth_scale.value())}, f, indent=2)
        except Exception:
            pass

    def _load_calib(self):
        """Restore the saved offset/scale into the spinboxes (no apply)."""
        path = self._calib_path()
        if not path.exists():
            return False
        try:
            import json
            with open(path) as f:
                data = json.load(f)
        except Exception:
            return False
        self.tth_offset.blockSignals(True); self.tth_scale.blockSignals(True)
        self.tth_offset.setValue(float(data.get("offset", 0.0)))
        self.tth_scale.setValue(float(data.get("scale", 1.0)))
        self.tth_offset.blockSignals(False); self.tth_scale.blockSignals(False)
        return True

    def _save_sum(self):
        """Persist the current summed image for this scan. Returns path/None."""
        if self._image is None:
            return None
        path = self._sum_path()
        try:
            import os
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / (path.stem + ".tmp.npz")
            np.savez_compressed(
                str(tmp),
                image=self._image.astype(np.float32),
                is_raw=np.array(bool(getattr(self, "_raw_active", False))),
                max_bins=np.array(int(self.max_bins.value())))
            os.replace(str(tmp), str(path))
            return path
        except Exception:
            return None

    def _load_sum(self):
        """Load the saved summed image and recompute the histogram fresh from it."""
        path = self._sum_path()
        if not path.exists():
            self._clear_data("no saved sum — press 'Compute histogram'")
            return False
        try:
            with np.load(path) as data:
                self._image = data["image"].astype(np.float64)
                self._raw_active = bool(data["is_raw"]) if "is_raw" in data.files else False
                mb = int(data["max_bins"]) if "max_bins" in data.files else 0
        except Exception:
            self._clear_data("could not read saved sum — press 'Compute histogram'")
            return False
        self.max_bins.blockSignals(True); self.max_bins.setValue(mb)
        self.max_bins.blockSignals(False)
        self._fresh_image = True   # new image → auto-level once
        self._summed_levels = None
        self._recompute_histogram()      # histogram derived fresh from the sum
        import datetime
        ts = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        raw = f" · {io.RAW_FALLBACK_NOTE}" if self._raw_active else ""
        self.status.setText(
            f"loaded saved sum (computed {ts:%Y-%m-%d %H:%M}); "
            f"histogram recomputed (noise: {self._noise_label()}){raw}")
        return True

    def _clear_data(self, message):
        """Drop any loaded sum/histogram and show a prompt."""
        self._image = self._cleaned = None
        self._centers = self._profile = None
        self._raw_active = False
        self._fresh_image = True
        self._summed_levels = None
        self._showing_summed = False
        self.detect_combo.clear()
        self.detect_combo.addItem("—")
        self.status.setText(message)
        self._redraw_image()
        self._redraw_hist()

    # ----- scan selection ---------------------------------------------
    def _populate_scans(self):
        scans = self.dm.discover_scans() or ([self.scan] if self.scan else [])
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
        self.scan_combo.addItems(scans or ["(no scans)"])
        if self.scan and self.scan in scans:
            self.scan_combo.setCurrentText(self.scan)
        self.scan_combo.blockSignals(False)

    def _on_scan_changed(self, text):
        if not text or text.startswith("(") or text == self.scan:
            return
        self._switch_scan(text)

    def _switch_scan(self, name):
        """Re-point the dialog at a different scan and reload everything."""
        self.scan = name
        self.dm = DataManager(self._project_root, scan=name)
        self.setWindowTitle(f"Reflections — {self.dm.scan_name or self._project_root}")
        # Reset per-scan state.
        self._det = None
        self._tth = self._tth_raw = None
        self._image = self._cleaned = None
        self._centers = self._profile = None
        self._raw_armed = False
        self._raw_active = False
        # Reset the calibration preview to default before restoring this scan's.
        self.tth_offset.blockSignals(True); self.tth_scale.blockSignals(True)
        self.tth_offset.setValue(0.0); self.tth_scale.setValue(1.0)
        self.tth_offset.blockSignals(False); self.tth_scale.blockSignals(False)
        # Reload this scan's reflections, 2θ map, and saved sum.
        self.reflections = list(refl_io.read_json(self.dm.reflections_json(name)))
        self._fill_table(self.reflections)
        self._load_tth_for_display()
        if self._load_calib() and self._tth_raw is not None:
            self._apply_geometry()
        self._load_sum()

    def _load_tth_for_display(self):
        """Load the linked tth.tiff so it can be shown before summing bins."""
        tth_path = self.dm.tth_map(scan=self.scan)
        if tth_path and Path(tth_path).exists():
            try:
                self._tth_raw = io.load_tth_map(tth_path)
                self._tth = self._tth_raw.copy()
                self._update_center()
                self.status.setText(
                    f"tth.tiff {self._tth.shape}  "
                    f"2θ range {float(self._tth.min()):.3f}–{float(self._tth.max()):.3f}°  "
                    f"({tth_path})")
            except Exception as e:
                self.status.setText(f"could not read tth.tiff: {e}")
        else:
            self.status.setText("no tth.tiff loaded — use Setup → Load tth.tiff")
        self._redraw_image()

    # ----- detected 2θ peaks + beam center ----------------------------
    def _populate_detected(self):
        """Auto-list 2θ peaks found in the radial profile (pick → fills 2θ)."""
        self.detect_combo.clear()
        self.detect_combo.addItem("—")
        if self._profile is None:
            return
        try:
            from scipy.signal import find_peaks
            prof = np.asarray(self._profile, dtype=float)
            finite = prof[np.isfinite(prof)]
            if finite.size == 0:
                return
            prom = max((finite.max() - finite.min()) * 0.03, 1e-9)
            idx, _ = find_peaks(prof, prominence=prom, distance=3)
            order = sorted(idx, key=lambda i: prof[i], reverse=True)[:20]
            for i in sorted(order):
                self.detect_combo.addItem(f"{self._centers[i]:.3f}°", float(self._centers[i]))
        except Exception:
            pass

    def _on_detected(self, _idx):
        val = self.detect_combo.currentData()
        if val is not None:
            self._set_picked_tth(float(val))

    def _update_center(self):
        if self._tth_raw is None:
            return
        try:
            from ..core.processing import estimate_beam_center
            y0, x0 = estimate_beam_center(self._tth_raw)
            self._center = (y0, x0)
            self.center_label.setText(f"beam center (x, y): ({x0:.1f}, {y0:.1f})")
        except Exception:
            self.center_label.setText("beam center: —")

    # ----- geometry edit (2θ calibration) -----------------------------
    def _apply_geometry(self):
        if self._tth_raw is None:
            self.status.setText("no tth.tiff loaded to adjust")
            return
        scale = self.tth_scale.value()
        offset = self.tth_offset.value()
        self._tth = self._tth_raw * scale + offset
        self._save_calib()  # remember the offset/scale for this scan
        if self._cleaned is not None:
            self._centers, self._profile = io.radial_profile(self._cleaned, self._tth)
            self._populate_detected()
        self.status.setText(
            f"preview: 2θ' = {scale:.4f}·2θ + {offset:.4f}  "
            f"(range {float(self._tth.min()):.3f}–{float(self._tth.max()):.3f}°)  "
            f"— Save tth… to keep")
        self._redraw_image()
        self._redraw_hist()

    def _save_tth(self):
        if self._tth is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Save 2θ map")
        box.setText("Save the adjusted 2θ map (tth.tiff)?")
        save_btn = box.addButton("Save", QMessageBox.AcceptRole)
        new_btn = box.addButton("Save as new", QMessageBox.ActionRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is None or clicked not in (save_btn, new_btn):
            return

        import tifffile
        if clicked is save_btn:
            path = Path(self.dm.tth_map(scan=self.scan))
        else:
            start = str(self.dm.metadata_dir / "tth_edited.tiff")
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Save new 2θ map", start, "TIFF (*.tif *.tiff)")
            if not chosen:
                return
            path = Path(chosen)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(path), self._tth.astype(np.float32))
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Could not write {path}:\n{e}")
            return
        # The saved map becomes the new baseline; reset the offset/scale preview.
        self._tth_raw = self._tth.copy()
        self.tth_offset.blockSignals(True); self.tth_scale.blockSignals(True)
        self.tth_offset.setValue(0.0); self.tth_scale.setValue(1.0)
        self.tth_offset.blockSignals(False); self.tth_scale.blockSignals(False)
        self._save_calib()  # baked into tth now → clear the preview offset/scale
        self._update_center()
        self.status.setText(f"saved tth → {path}")

    # ----- drawing ----------------------------------------------------
    def _redraw_image(self):
        vb = self.img_view.getView()
        for item in self._overlay_items:
            vb.removeItem(item)
        self._overlay_items = []

        # If the summed (intensity) image is currently on screen, capture the
        # user's histogram region first — this is how "contracting the shown
        # intensities" is remembered so the colormap stays mapped across that
        # window across redraws (the 2θ map shares the same histogram widget,
        # so we only capture while the intensity image is showing).
        if self._showing_summed:
            try:
                self._summed_levels = self.img_view.getHistogramWidget().item.getLevels()
            except Exception:
                pass

        summed = self._cleaned if self._cleaned is not None else self._image
        showing_tth = self.view_combo.currentText().startswith("2θ")
        if showing_tth and self._tth is not None:
            # Raw 2θ-per-pixel map (linear, not log) — always auto-level; it is a
            # different quantity than intensity.
            self.img_view.setImage(self._tth, autoLevels=True)
            self._showing_summed = False
        elif summed is not None:
            auto = self._summed_levels is None or self._fresh_image
            disp = summed
            if self.log_cb.isChecked():
                disp = np.log1p(np.clip(summed, 0, None))
            self.img_view.setImage(disp, autoLevels=auto)
            if auto:
                try:
                    self._summed_levels = \
                        self.img_view.getHistogramWidget().item.getLevels()
                except Exception:
                    self._summed_levels = None
            else:
                self.img_view.setLevels(*self._summed_levels)
            self._fresh_image = False
            self._showing_summed = True
        elif self._tth is not None:
            self.img_view.setImage(self._tth, autoLevels=True)
            self._showing_summed = False
        try:
            self.img_view.setColorMap(_get_cmap(self._cmap_name))
        except Exception:
            pass

        if self._show_overlay and self._tth is not None and self.reflections:
            self._add_tth_overlay(vb)

    def _add_tth_overlay(self, vb):
        """Translucent colored 2θ bands + labels on the detector image."""
        tth = self._tth
        rgba = np.zeros((tth.shape[0], tth.shape[1], 4), dtype=np.ubyte)
        for idx, r in enumerate(self.reflections):
            t = float(r["two_theta"])
            w = float(r.get("width", refl_io.DEFAULT_WIDTH)) or _LINE_TOL
            mask = np.abs(tth - t) < w
            if not mask.any():
                continue
            cr, cg, cb = _hex_rgb(_arc_color(idx))
            rgba[mask, 0] = cr
            rgba[mask, 1] = cg
            rgba[mask, 2] = cb
            rgba[mask, 3] = _OVERLAY_ALPHA
            ys, xs = np.where(mask)
            mid = len(ys) // 2
            label = pg.TextItem(str(r["name"]), color=_arc_color(idx), anchor=(0, 1),
                                fill=pg.mkBrush(0, 0, 0, 180))
            label.setPos(float(xs[mid]), float(ys[mid]))
            label.setZValue(6)
            vb.addItem(label)
            self._overlay_items.append(label)
        band = pg.ImageItem(rgba)
        band.setZValue(1)
        vb.addItem(band)
        self._overlay_items.append(band)

    def _redraw_hist(self):
        self.hist.clear()
        if self._profile is not None:
            self.hist.plot(self._centers, self._profile,
                           pen=pg.mkPen("#1f6fdc", width=1))
        if not self._show_overlay:
            return
        for idx, r in enumerate(self.reflections):
            t = float(r["two_theta"])
            w = float(r.get("width", refl_io.DEFAULT_WIDTH))
            cr, cg, cb = _hex_rgb(_arc_color(idx))
            region = pg.LinearRegionItem(values=[t - w, t + w], movable=False,
                                         brush=pg.mkBrush(cr, cg, cb, 50))
            region.setZValue(-10)
            self.hist.addItem(region)
            line = pg.InfiniteLine(pos=t, angle=90,
                                   pen=pg.mkPen(cr, cg, cb, width=1),
                                   label=str(r["name"]), labelOpts={"position": 0.9})
            self.hist.addItem(line)

    # ----- table management -------------------------------------------
    def _fill_table(self, reflections):
        """Populate the table from a list of {name, two_theta, width} dicts."""
        self._filling = True
        self.table.setRowCount(0)
        for r in reflections:
            self._append_row(
                str(r.get("name", "")),
                float(r.get("two_theta", 0.0)),
                float(r.get("width", refl_io.DEFAULT_WIDTH)))
        self._filling = False
        self.reflections = list(reflections)
        self._redraw_image()
        self._redraw_hist()

    def _append_row(self, name, tth, width):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(f"{tth:.4f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{width:.3f}"))

    def _reflections_from_table(self):
        """Rebuild self.reflections from the table (skip rows with no valid 2θ)."""
        out = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            tth_item = self.table.item(row, 1)
            w_item = self.table.item(row, 2)
            try:
                tth = float(tth_item.text())
            except (AttributeError, ValueError):
                continue
            try:
                width = float(w_item.text())
            except (AttributeError, ValueError):
                width = refl_io.DEFAULT_WIDTH
            name = name_item.text().strip() if name_item else ""
            out.append({"name": name or f"{tth:.2f}",
                        "two_theta": tth, "width": width})
        self.reflections = out

    def _on_item_changed(self, _item):
        if self._filling:
            return
        self._reflections_from_table()
        self._redraw_image()
        self._redraw_hist()

    def _add_row(self, tth=None):
        if tth is None:
            tth = self._last_tth
        self._filling = True
        self._append_row(f"ref{self.table.rowCount() + 1}",
                         float(tth), refl_io.DEFAULT_WIDTH)
        self._filling = False
        self.table.selectRow(self.table.rowCount() - 1)
        self._reflections_from_table()
        self._redraw_image()
        self._redraw_hist()

    def _delete_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self._filling = True
        for row in rows:
            self.table.removeRow(row)
        self._filling = False
        self._reflections_from_table()
        self._redraw_image()
        self._redraw_hist()

    def _set_picked_tth(self, val):
        """A 2θ was picked (histogram click / detected peak): set the selected
        row's 2θ, or add a new row if none is selected."""
        self._last_tth = float(val)
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows:
            self.table.setItem(rows[0], 1, QTableWidgetItem(f"{val:.4f}"))
            self.status.setText(f"row {rows[0] + 1}: 2θ set to {val:.4f}")
        else:
            self._add_row(val)
            self.status.setText(f"added row at 2θ = {val:.4f} (edit its name)")

    # ----- set / colormap / overlay controls --------------------------
    def _load_current(self):
        self._fill_table(list(refl_io.read_json(self.dm.reflections_json(self.scan))))

    def _on_set_changed(self, idx):
        if idx == 0:
            self._load_current()
        else:
            self._fill_table([])

    def _on_cmap_changed(self, name):
        self._cmap_name = name
        try:
            self.img_view.setColorMap(_get_cmap(name))
        except Exception:
            pass

    def _on_log_changed(self, _checked):
        # log1p vs linear changes the value range entirely, so re-auto-level.
        self._fresh_image = True
        self._summed_levels = None
        self._redraw_image()

    def _on_overlay_toggled(self, checked):
        self._show_overlay = bool(checked)
        if self._show_overlay:
            # Load the saved set into the table, then show the arcs.
            if self.set_combo.currentIndex() != 0:
                self.set_combo.setCurrentIndex(0)   # triggers _load_current
            else:
                self._load_current()
        self._redraw_image()
        self._redraw_hist()

    # ----- interactivity ----------------------------------------------
    def _on_hist_click(self, ev):
        vb = self.hist.getViewBox()
        if vb is None:
            return
        pt = vb.mapSceneToView(ev.scenePos())
        self._set_picked_tth(float(pt.x()))

    # ----- save -------------------------------------------------------
    def _save_reflections(self):
        """Write reflections.json + reflections.py (does not close the dialog)."""
        self._reflections_from_table()
        if not self.reflections:
            self.status.setText("no reflections to save")
            return
        mdir = self.dm.metadata_scan_dir(self.scan) if self.scan else self.dm.metadata_dir
        json_path = mdir / "reflections.json"
        py_path = mdir / "reflections.py"
        try:
            refl_io.save(self.reflections, json_path, py_path)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Could not write reflections:\n{e}")
            return
        self.status.setText(f"saved {len(self.reflections)} reflections → {json_path}")
        QMessageBox.information(
            self, "Saved reflections",
            f"Wrote {len(self.reflections)} reflections:\n{json_path}\n{py_path}")

    def _save_png(self):
        """Save the 2θ histogram figure as a PNG.

        Uses QWidget.grab() (renders on the GUI thread) rather than pyqtgraph's
        ImageExporter, which can segfault on some setups.
        """
        if self._profile is None:
            QMessageBox.information(
                self, "Nothing to save",
                "Compute the histogram first, then Save PNG.")
            return
        fdir = self.dm.figures_dir
        try:
            fdir.mkdir(parents=True, exist_ok=True)
            path = fdir / f"{self.dm.scan_name or 'project'}_reflections.png"
            pix = self.hist.grab()
            if not pix.save(str(path), "PNG"):
                raise RuntimeError("QPixmap.save returned False")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", f"Could not write PNG:\n{e}")
            return
        self.status.setText(f"saved figure → {path}")
        QMessageBox.information(self, "Saved PNG", f"Figure written:\n{path}")


def open_reflection_dialog(project_root, scan=None, bin_size=3, parent=None):
    return ReflectionDialog(project_root, scan=scan, bin_size=bin_size, parent=parent)


if __name__ == "__main__":
    from ._standalone import run_dialog_standalone
    run_dialog_standalone(open_reflection_dialog, "Manual Reflections")
