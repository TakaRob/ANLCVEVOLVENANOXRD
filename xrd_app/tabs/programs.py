"""Programs tab — run the pipeline. Every Run button shells out to the CLI."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QWidget, QSizePolicy,
)

from ..config import DataManager, format_detector_label
from ._console import JobConsole

TAB_META = {
    "title": "Programs",
    "order": 20,
    "takes_bin_size": True,
    "scan_dependent": False,
    "general": (
        "Run the two-stage pipeline. Data Prep builds the binned HDF5 from raw "
        "frames (grid mapping → bins). Peak Finding runs a detector over every "
        "bin (Phase 1). Shape Finding links peaks across bins and keeps "
        "gaussian-like features (Phase 2); pick “run peak algorithm above first” "
        "to chain peaks→shapes in one process. Combined runs a per-frame (1×1) "
        "algorithm that does peak+shape in one pass. Each Run shells out to the "
        "CLI; output streams below."
    ),
}

# Sentinel shown in the Shape "Peaks:" dropdown to chain peak→shape in one run.
_CHAIN_OPTION = "⟵ run peak algorithm above first"


class ProgramsTab(QWidget):
    def __init__(self, project_root, scan=None, bin_size=3):
        super().__init__()
        self.project_root = str(Path(project_root).resolve())
        self.scan = scan
        self.bin_size = bin_size

        lay = QVBoxLayout(self)

        # ---- existing bins (drives every step below) --------------------
        # The top dropdown lists only the bins that already exist for this scan;
        # it selects which one Peak/Shape/Combined operate on. New bins are made
        # in the Data Prep box below and then appear here automatically.
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Existing bins:</b>"))
        self.bin_combo = QComboBox()
        self.bin_combo.setToolTip(
            "Bins already built for this scan. Selecting one drives Peak Finding, "
            "Shape Finding and the per-bin context below.")
        self.bin_combo.currentTextChanged.connect(self._on_bin_changed)
        top.addWidget(self.bin_combo)
        self.bins_status = QLabel("")
        self.bins_status.setStyleSheet("color:#888; font-size:0.9em; padding-left:8px;")
        top.addWidget(self.bins_status)
        top.addStretch()
        lay.addLayout(top)

        # ---- Data prep: build bins --------------------------------------
        # The "Create bins" button depends only on the spin box here (not the
        # Existing-bins dropdown above): type any size — e.g. 2, 4, 7 — build it,
        # and it shows up in the dropdown once detected.
        bins_box = QGroupBox("Data Prep  (build binned HDF5 from raw frames)")
        bl = QHBoxLayout(bins_box)
        bl.addWidget(QLabel("New bin size (N×N):"))
        self.make_bin_spin = QSpinBox()
        self.make_bin_spin.setRange(1, 99)
        self.make_bin_spin.setValue(bin_size if bin_size else 3)
        self.make_bin_spin.setToolTip(
            "Bin size to build (NxN). Type any value; after building it appears "
            "in “Existing bins” above.")
        bl.addWidget(self.make_bin_spin)
        make_bins_btn = QPushButton("Create bins")
        make_bins_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        make_bins_btn.setMinimumHeight(40)
        make_bins_btn.clicked.connect(self._make_bins)
        bl.addWidget(make_bins_btn)
        bl.addStretch()
        lay.addWidget(bins_box)

        # ---- Peak Finding ------------------------------------------------
        peak_box = QGroupBox("Peak Finding  (Phase 1: per-bin detection)")
        pl = QHBoxLayout(peak_box)
        pl.addWidget(QLabel("Algorithm:"))
        self.peak_algo = QComboBox()
        pl.addWidget(self.peak_algo, 1)
        run_peaks = QPushButton("Run")
        run_peaks.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        run_peaks.setMinimumHeight(40)
        run_peaks.clicked.connect(self._run_peaks)
        pl.addWidget(run_peaks)
        lay.addWidget(peak_box)

        # ---- Shape Finding ----------------------------------------------
        shape_box = QGroupBox("Shape Finding  (Phase 2: link + gaussian filter)")
        sl = QHBoxLayout(shape_box)
        sl.addWidget(QLabel("Peaks:"))
        self.shape_src = QComboBox()
        sl.addWidget(self.shape_src, 1)
        run_shapes = QPushButton("Run")
        run_shapes.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        run_shapes.setMinimumHeight(40)
        run_shapes.clicked.connect(self._run_shapes)
        sl.addWidget(run_shapes)
        lay.addWidget(shape_box)

        # ---- Combined (peak + shape in one pass) ------------------------
        comb_box = QGroupBox("Combined  (peak + shape in one per-frame pass · 1×1)")
        cb = QHBoxLayout(comb_box)
        cb.addWidget(QLabel("Algorithm:"))
        self.combined_algo = QComboBox()
        cb.addWidget(self.combined_algo, 1)
        run_combined = QPushButton("Run")
        run_combined.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        run_combined.setMinimumHeight(40)
        run_combined.clicked.connect(self._run_combined)
        cb.addWidget(run_combined)
        lay.addWidget(comb_box)

        # ---- CVEvolve + lineage -----------------------------------------
        cve_row = QHBoxLayout()
        cve_btn = QPushButton("Use CVEvolve…")
        cve_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        cve_btn.setMinimumHeight(40)
        cve_btn.clicked.connect(self._open_cvevolve)
        cve_row.addWidget(cve_btn)
        lin_btn = QPushButton("Show lineage")
        lin_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        lin_btn.setMinimumHeight(40)
        lin_btn.setToolTip("Print the provenance (bin → algorithm chain) of every "
                           "result JSON for the active scan.")
        lin_btn.clicked.connect(self._show_lineage)
        cve_row.addWidget(lin_btn)
        cve_row.addStretch()
        lay.addLayout(cve_row)

        # ---- Console -----------------------------------------------------
        self.console = JobConsole()
        lay.addWidget(self.console, 1)

        self.update_context(scan, bin_size)

    # ----- context ----------------------------------------------------
    def update_context(self, scan, bin_size):
        self.scan = scan
        self.bin_size = bin_size
        self._populate_bins(select=bin_size)
        self._refresh_algos()
        self._refresh_peak_sources()
        self._refresh_bins_status()

    def _populate_bins(self, select=None):
        """Fill the Existing-bins dropdown from the bins built for this scan.

        Selects ``select`` if present, else the last context bin, else the first.
        """
        from ._embed import existing_bins
        dm = DataManager(self.project_root, scan=self.scan)
        bins = existing_bins(dm, self.scan)
        self.bin_combo.blockSignals(True)
        self.bin_combo.clear()
        if bins:
            self.bin_combo.addItems([f"{b}x{b}" for b in bins])
            target = next((b for b in (select, self.bin_size) if b in bins),
                          bins[0])
            self.bin_combo.setCurrentText(f"{target}x{target}")
            self.bin_size = target
        else:
            self.bin_combo.addItem("(no bins — create one)")
        self.bin_combo.blockSignals(False)

    def _cur_bin(self):
        try:
            return int(self.bin_combo.currentText().split("x")[0])
        except (ValueError, AttributeError):
            return None

    def _on_bin_changed(self, *_):
        bs = self._cur_bin()
        if bs is not None:
            self.bin_size = bs
        self._refresh_algos()
        self._refresh_peak_sources()
        self._refresh_bins_status()

    def _refresh_bins_status(self, *_):
        from ._embed import bins_status_text
        bs = self._cur_bin()
        if bs is None:
            self.bins_status.setText("no bins built — create one in Data Prep")
            return
        dm = DataManager(self.project_root, scan=self.scan)
        self.bins_status.setText(bins_status_text(dm, bs, self.scan))

    def _refresh_algos(self):
        dm = DataManager(self.project_root, scan=self.scan)
        bs = self._cur_bin()
        dets = dm.list_detectors(bs) or dm.list_detectors()
        self.peak_algo.clear()
        if not dets:
            self.peak_algo.addItem("(none found)")
            return
        for d in dets:
            # Show "name (F1 0.79)"; keep the bare name + perframe flag as data.
            self.peak_algo.addItem(format_detector_label(d),
                                   (d["name"], d.get("pipeline") == "perframe"))
        # Combined (peak+shape) library — names carry scores, data = bare name.
        self.combined_algo.clear()
        combined = dm.list_combined()
        if combined:
            for d in combined:
                self.combined_algo.addItem(format_detector_label(d), d["name"])
        else:
            self.combined_algo.addItem("(none found)")

    def _selected_peak_algo(self):
        """(name, is_unbinned) for the selected peak detector, or (None, False)."""
        data = self.peak_algo.currentData()
        if isinstance(data, tuple):
            return data
        text = self.peak_algo.currentText()
        return (None if text.startswith("(") else text, False)

    def _refresh_peak_sources(self):
        """List saved *_peaks.json in Labels/<scan>/ as shape-finding inputs.

        The first option chains peak→shape: it runs the peak algorithm selected
        above, then shape-finds its output, all in one process.
        """
        self.shape_src.clear()
        dm = DataManager(self.project_root, scan=self.scan)
        ldir = dm.labels_dir(self.scan)
        bs = self._cur_bin()
        found = [_CHAIN_OPTION]
        if ldir.is_dir():
            for p in sorted(ldir.glob(f"*_peaks_{bs}x{bs}.json")):
                found.append(p.stem.replace(f"_peaks_{bs}x{bs}", ""))
        self.shape_src.addItems(found)

    # ----- actions ----------------------------------------------------
    def _run_peaks(self):
        if not self.scan:
            self.console._append("[no active scan — load data in Setup first]\n")
            return
        if self._cur_bin() is None:
            self.console._append("[no bins — create one in Data Prep first]\n")
            return
        algo, unbinned = self._selected_peak_algo()
        if unbinned:
            self.console._append(
                f"[{algo} is a per-frame (unbinned) detector — it can't run in "
                "the binned peak pipeline yet. Pick a binned detector.]\n")
            return
        args = ["peaks", "--root", self.project_root, "--scan", str(self.scan),
                "--bin-size", str(self._cur_bin())]
        if algo:
            args += ["--algorithm", algo]
        self.console.run(args)

    def _make_bins(self):
        if not self.scan:
            self.console._append("[no active scan — load data in Setup first]\n")
            return
        bs = self.make_bin_spin.value()  # Create bins depends only on this spin box.
        self.console.run(["make-bins", "--root", self.project_root,
                          "--scan", str(self.scan),
                          "--bin-size", str(bs)],
                         on_finished=lambda code: self._on_bins_built(bs))

    def _on_bins_built(self, bs):
        """After a build, surface the new size in the Existing-bins dropdown."""
        self._populate_bins(select=bs)
        self._refresh_algos()
        self._refresh_peak_sources()
        self._refresh_bins_status()

    def _run_combined(self):
        if not self.scan:
            self.console._append("[no active scan — load data in Setup first]\n")
            return
        algo = self.combined_algo.currentData() or self.combined_algo.currentText()
        if not algo or algo.startswith("("):
            self.console._append("[no combined algorithm available]\n")
            return
        self.console.run(["run-combined", "--root", self.project_root,
                          "--scan", str(self.scan), "--algorithm", algo])

    def _show_lineage(self):
        if not self.scan:
            self.console._append("[no active scan — load data in Setup first]\n")
            return
        self.console.run(["lineage", "--root", self.project_root,
                          "--scan", str(self.scan)])

    def _open_cvevolve(self):
        bs = self._cur_bin()
        if bs is None:
            self.console._append("[no bins — create one in Data Prep first]\n")
            return
        from .cvevolve_dialog import CVEvolveDialog
        CVEvolveDialog(self.project_root, scan=self.scan,
                       bin_size=bs, parent=self).exec_()

    def _run_shapes(self):
        if not self.scan:
            self.console._append("[no active scan — load data in Setup first]\n")
            return
        if self._cur_bin() is None:
            self.console._append("[no bins — create one in Data Prep first]\n")
            return
        src = self.shape_src.currentText()
        # Chain option: run the peak algorithm above, then shapes, in one process.
        if src == _CHAIN_OPTION:
            peak_algo, unbinned = self._selected_peak_algo()
            if unbinned:
                self.console._append(
                    f"[{peak_algo} is a per-frame (unbinned) detector — it can't "
                    "run in the binned pipeline yet. Pick a binned detector.]\n")
                return
            args = ["run-pipeline", "--root", self.project_root,
                    "--scan", str(self.scan),
                    "--bin-size", str(self._cur_bin())]
            if peak_algo:
                args += ["--algorithm", peak_algo]
            self.console.run(args)
            return
        if not src or src.startswith("("):
            self.console._append("[no peak set — run Peak Finding first]\n")
            return
        args = ["shapes", "--root", self.project_root, "--scan", str(self.scan),
                "--bin-size", str(self._cur_bin()), "--peak-algo", src]
        self.console.run(args)


def make_tab(project_root=".", scan=None, bin_size=3):
    return ProgramsTab(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
