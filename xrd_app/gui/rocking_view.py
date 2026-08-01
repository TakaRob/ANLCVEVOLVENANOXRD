"""Rocking-Study view — run the θ-series pipeline and browse its results.

One widget with a study selector on top and four sub-tabs:

* **Run** — scans/bin-size/output controls that shell out to ``xrd-app
  run-study`` (aggregate → track → rocking → predict → combined-device, and
  optionally qspace → rsm). Output streams to an embedded console; on success the
  study registers itself and the viewers refresh.
* **Rocking curves** — pick a grain track and see its measured intensity(θ)
  points with the fitted Gaussian rocking curve (θ_Bragg, FWHM/mosaicity, R²)
  overlaid.
* **Combined device** — the fused-over-θ spatial canvas (max intensity,
  recurrence count, argmax-θ orientation, or a per-reflection layer).
* **Report** — the prediction report (recall / precision / repeatability floor).

Render-only over ``core.studies`` (+ ``core.rocking`` for the fit model and
``core.tracking`` for the intensity curve). Missing artifacts degrade to a hint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QSpinBox, QSplitter, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from ..core import studies as studies_core
from .palette import ARC_COLORS, _get_cmap

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


class RockingStudyView(QWidget):
    """Study selector + Run/Rocking/Combined/Report sub-tabs over one project."""

    bin_size_changed = pyqtSignal(int)

    def __init__(self, project_root=".", scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.project_root = str(Path(project_root).resolve())
        self.scan = scan
        self.bin_size = bin_size or 3
        self._study = Path(self.project_root) / "Study"
        self._tracks = []
        self._rocking_by_id = {}

        root = QVBoxLayout(self)

        # ---- study selector (shared by all viewer sub-tabs) --------------
        top = QHBoxLayout()
        top.addWidget(QLabel("Study:"))
        self.study_cb = QComboBox()
        self.study_cb.setMinimumWidth(280)
        self.study_cb.currentIndexChanged.connect(self._on_study_changed)
        top.addWidget(self.study_cb)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._rescan_studies)
        top.addWidget(reload_btn)
        top.addStretch(1)
        self.study_info = QLabel("")
        self.study_info.setStyleSheet("color:#888;")
        top.addWidget(self.study_info)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_rocking_tab(), "Rocking curves")
        self.tabs.addTab(self._build_combined_tab(), "Combined device")
        self.tabs.addTab(self._build_report_tab(), "Report")
        root.addWidget(self.tabs, 1)

        self._rescan_studies()

    # ---- Run tab ---------------------------------------------------------
    def _build_run_tab(self) -> QWidget:
        from ..tabs._console import JobConsole
        w = QWidget()
        lay = QVBoxLayout(w)

        form = QHBoxLayout()
        form.addWidget(QLabel("Scans:"))
        self.scans_edit = QLineEdit()
        self.scans_edit.setPlaceholderText("all in Labels/  (e.g. 203,204,205,…)")
        form.addWidget(self.scans_edit, 2)

        form.addWidget(QLabel("Bin:"))
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(1, 99)
        self.bin_spin.setValue(self.bin_size)
        self.bin_spin.valueChanged.connect(self._on_bin_changed)
        form.addWidget(self.bin_spin)

        form.addWidget(QLabel("Output:"))
        self.out_edit = QLineEdit("Study")
        self.out_edit.setMaximumWidth(140)
        form.addWidget(self.out_edit)

        form.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("optional label")
        self.name_edit.setMaximumWidth(160)
        form.addWidget(self.name_edit)
        lay.addLayout(form)

        row2 = QHBoxLayout()
        self.rsm_chk = QCheckBox("also build 3D RSM (qspace + rsm)")
        self.rsm_chk.setToolTip("Runs qspace → rsm for the reciprocal-space "
                                "volume. Needs the xrd-app[qspace] extra.")
        row2.addWidget(self.rsm_chk)
        row2.addStretch(1)
        self.run_btn = QPushButton("Run study")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run_study)
        row2.addWidget(self.run_btn)
        lay.addLayout(row2)

        note = QLabel(
            "Chains aggregate → track → rocking → predict → combined-device into "
            "the output directory, then registers it so it appears in every study "
            "selector. Each Run shells out to <b>xrd-app run-study</b>.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#777; font-size:0.9em;")
        lay.addWidget(note)

        self.console = JobConsole()
        lay.addWidget(self.console, 1)
        return w

    def _run_study(self):
        if not self.project_root:
            self.console._append("[no project — open one in Setup first]\n")
            return
        args = ["run-study", "--root", self.project_root,
                "--bin-size", str(self.bin_spin.value()),
                "--out", self.out_edit.text().strip() or "Study"]
        scans = self.scans_edit.text().strip()
        if scans:
            args += ["--scans", scans]
        name = self.name_edit.text().strip()
        if name:
            args += ["--name", name]
        if self.rsm_chk.isChecked():
            args += ["--with-rsm"]
        self.run_btn.setEnabled(False)
        self.console.run(args, on_finished=self._on_run_finished)

    def _on_run_finished(self, code):
        self.run_btn.setEnabled(True)
        # Select the just-built study by its output dir, then refresh viewers.
        out = self.out_edit.text().strip() or "Study"
        self._rescan_studies(prefer=str((Path(self.project_root) / out).resolve()))

    # ---- Rocking-curves tab ---------------------------------------------
    def _build_rocking_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.recurrent_chk = QCheckBox("recurrent tracks only")
        self.recurrent_chk.setChecked(True)
        self.recurrent_chk.stateChanged.connect(self._populate_tracks)
        bar.addWidget(self.recurrent_chk)
        bar.addStretch(1)
        lay.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        self.track_list = QListWidget()
        self.track_list.setMinimumWidth(230)
        self.track_list.currentItemChanged.connect(self._on_track_selected)
        split.addWidget(self.track_list)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.rock_plot = pg.PlotWidget()
        self.rock_plot.setLabel("bottom", "θ (deg)")
        self.rock_plot.setLabel("left", "intensity")
        self.rock_plot.showGrid(x=True, y=True, alpha=0.25)
        rl.addWidget(self.rock_plot, 1)
        self.fit_label = QLabel("")
        self.fit_label.setStyleSheet("color:#555;")
        self.fit_label.setWordWrap(True)
        rl.addWidget(self.fit_label)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        return w

    def _populate_tracks(self):
        self.track_list.blockSignals(True)
        self.track_list.clear()
        recurrent_only = self.recurrent_chk.isChecked()
        for t in self._tracks:
            if recurrent_only and not t.get("is_recurrent"):
                continue
            n_theta = len({m.get("theta") for m in t.get("members", [])})
            r, c = t.get("centroid_row"), t.get("centroid_col")
            rc = f"({r:.0f},{c:.0f})" if r is not None and c is not None else ""
            label = (f"#{t.get('track_id')} · {t.get('reflection') or '?'} · "
                     f"{rc} · {n_theta}θ")
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, t.get("track_id"))
            self.track_list.addItem(it)
        self.track_list.blockSignals(False)
        if self.track_list.count():
            self.track_list.setCurrentRow(0)
        else:
            self.rock_plot.clear()
            self.fit_label.setText("No tracks in this study (run the pipeline).")

    def _on_track_selected(self, cur, _prev):
        if cur is None:
            return
        tid = cur.data(Qt.UserRole)
        track = next((t for t in self._tracks if t.get("track_id") == tid), None)
        if track is None:
            return
        self._draw_rocking(track)

    def _draw_rocking(self, track):
        from ..core import tracking, rocking
        self.rock_plot.clear()
        thetas, intens = tracking.intensity_curve(track)
        if not thetas:
            self.fit_label.setText("Track has no θ points.")
            return
        col = QColor(ARC_COLORS[(hash(track.get("reflection") or "") % len(ARC_COLORS))])
        self.rock_plot.plot(thetas, intens, pen=None, symbol="o", symbolSize=9,
                            symbolBrush=pg.mkBrush(col), symbolPen=pg.mkPen("k", width=0.4))

        row = self._rocking_by_id.get(track.get("track_id"))
        bits = [f"track #{track.get('track_id')}  ·  {track.get('reflection') or '?'}"]
        if row:
            status = row.get("status")
            tb, fwhm = row.get("theta_bragg"), row.get("fwhm")
            amp, bg = row.get("amplitude"), row.get("background")
            r2 = row.get("r_squared")
            if (isinstance(status, str) and status.startswith("fit")
                    and None not in (tb, fwhm, amp, bg)) and fwhm:
                xs = np.linspace(min(thetas), max(thetas), 200)
                ys = rocking.gaussian(xs, bg, amp, tb, fwhm)
                self.rock_plot.plot(xs, ys, pen=pg.mkPen(col, width=2))
                bits.append(f"θ_Bragg = {tb:.2f}°   FWHM = {fwhm:.2f}° (mosaicity)")
                if r2 is not None:
                    bits.append(f"R² = {r2:.3f}")
            else:
                bits.append(f"status: {status} (no Gaussian overlaid)")
            ms = row.get("microstrain")
            if isinstance(ms, (int, float)):
                bits.append(f"microstrain = {ms*100:+.3f}%")
        else:
            bits.append("(no rocking-curve fit row — run `rocking`)")
        self.fit_label.setText("     ".join(bits))

    # ---- Combined-device tab -------------------------------------------
    def _build_combined_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Metric:"))
        self.metric_cb = QComboBox()
        self.metric_cb.currentIndexChanged.connect(self._draw_combined)
        bar.addWidget(self.metric_cb)
        bar.addStretch(1)
        self.combined_info = QLabel("")
        self.combined_info.setStyleSheet("color:#888;")
        bar.addWidget(self.combined_info)
        lay.addLayout(bar)
        self.combined_view = pg.ImageView()
        try:
            self.combined_view.setColorMap(_get_cmap("inferno"))
        except Exception:
            pass
        lay.addWidget(self.combined_view, 1)
        return w

    def _populate_combined(self):
        self._combined = studies_core.load_combined_device(self._study)
        self.metric_cb.blockSignals(True)
        self.metric_cb.clear()
        if self._combined is None:
            self.metric_cb.blockSignals(False)
            self.combined_view.clear()
            self.combined_info.setText("No combined_device.npz in this study.")
            return
        self.metric_cb.addItem("max intensity (any θ)", ("max_intensity", None))
        self.metric_cb.addItem("recurrence (θ count)", ("n_theta_present", None))
        self.metric_cb.addItem("orientation (argmax θ)", ("argmax_theta", None))
        refls = [str(r) for r in self._combined.get("reflections", [])]
        for k, ref in enumerate(refls):
            self.metric_cb.addItem(f"layer: {ref} intensity", ("layer_intensity", k))
        self.metric_cb.blockSignals(False)
        self.metric_cb.setCurrentIndex(0)
        self._draw_combined()

    def _draw_combined(self):
        if getattr(self, "_combined", None) is None:
            return
        data = self.metric_cb.currentData()
        if not data:
            return
        key, idx = data
        arr = self._combined.get(key)
        if arr is None:
            return
        img = np.asarray(arr[idx] if idx is not None else arr, dtype=float)
        self.combined_view.setImage(img, autoLevels=True, autoRange=True)
        nrows, ncols = img.shape
        self.combined_info.setText(f"{nrows}×{ncols} bins")

    # ---- Report tab ------------------------------------------------------
    def _build_report_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("monospace", 10))
        lay.addWidget(self.report_text)
        return w

    def _populate_report(self):
        md = studies_core.load_prediction_report(self._study)
        if md is None:
            self.report_text.setPlainText(
                "No prediction_report.md in this study. Run the pipeline "
                "(Run tab) or `xrd-app predict`.")
            return
        try:
            self.report_text.setMarkdown(md)  # Qt ≥ 5.14
        except Exception:
            self.report_text.setPlainText(md)

    # ---- study selection / refresh --------------------------------------
    def _rescan_studies(self, prefer=None):
        """Rebuild the study dropdown from discovery; keep/prefer a selection."""
        self.study_cb.blockSignals(True)
        self.study_cb.clear()
        try:
            found = studies_core.list_studies(self.project_root)
        except Exception:
            found = []
        want = prefer or str(self._study.resolve())
        sel = 0
        if found:
            for i, e in enumerate(found):
                desc = studies_core.describe(e)
                self.study_cb.addItem(e["name"] + (f"   ({desc})" if desc else ""),
                                      e["abs_path"])
                if e["abs_path"] == want:
                    sel = i
            self.study_cb.setCurrentIndex(sel)
            self._study = Path(self.study_cb.currentData())
        else:
            self.study_cb.addItem("(no studies — run one in the Run tab)", None)
        self.study_cb.blockSignals(False)
        self._reload_viewers()

    def _on_study_changed(self, _idx):
        data = self.study_cb.currentData()
        if data:
            self._study = Path(data)
            self._reload_viewers()

    def _reload_viewers(self):
        self._tracks = studies_core.load_tracks(self._study)
        rocking_rows = studies_core.load_rocking_curves(self._study)
        self._rocking_by_id = {r.get("track_id"): r for r in rocking_rows}
        n_rec = sum(1 for t in self._tracks if t.get("is_recurrent"))
        self.study_info.setText(
            f"{len(self._tracks)} tracks ({n_rec} recurrent)  ·  "
            f"{len(rocking_rows)} rocking curves"
            if self._tracks else "no tracks yet")
        self._populate_tracks()
        self._populate_combined()
        self._populate_report()

    # ---- host hook (scan/bin changes) -----------------------------------
    def current_bin_size(self):
        return self.bin_size

    def _on_bin_changed(self, bin_size):
        self.bin_size = bin_size
        self.bin_size_changed.emit(bin_size)

    def update_context(self, scan, bin_size):
        self.scan = scan
        if bin_size:
            self.bin_size = bin_size
            self.bin_spin.blockSignals(True)
            self.bin_spin.setValue(bin_size)
            self.bin_spin.blockSignals(False)
