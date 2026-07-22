"""Programs tab — run the pipeline. Every Run button shells out to the CLI."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout, QWidget, QSizePolicy,
)

from ..config import DataManager, format_detector_label
from ._console import ConsolePanel

TAB_META = {
    "title": "Programs",
    "order": 20,
    "takes_bin_size": True,
    "scan_dependent": False,
    "general": (
        "Run the two-stage pipeline over one or more scans at once. Pick the "
        "target scans in the Scans box (Ctrl/Shift-click for several) and one or "
        "more algorithms in each box (also Ctrl/Shift-click). Data Prep builds "
        "the binned HDF5 from raw frames (grid mapping → bins). Peak Finding runs "
        "a detector over every bin (Phase 1). Shape Finding links peaks across "
        "bins and keeps gaussian-like features (Phase 2); pick “run peak "
        "algorithm above first” to chain peaks→shapes in one process. Combined "
        "runs a per-frame (1×1) algorithm that does peak+shape in one pass. Each "
        "Run fans out over the selected scans × algorithms as a queue of CLI "
        "jobs; output streams to a console tab below. A Run reuses the active "
        "console if it's idle, or opens a new console tab if it's busy — so you "
        "can start a second run without waiting (no need for a second GUI)."
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

        # ---- target scans (multi-select; drives every Run below) --------
        # Ctrl/Shift-click to run over several scans at once. Every Run fans
        # out over (selected scans × selected algorithms) as a queue of CLI
        # jobs. Defaults to the active scan chosen in the header.
        scans_box = QGroupBox("Scans  (Ctrl/Shift-click to run over several)")
        scl = QVBoxLayout(scans_box)
        self.scans_list = QListWidget()
        self.scans_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.scans_list.setMaximumHeight(110)
        self.scans_list.setToolTip(
            "Scans this run applies to. Ctrl/Shift-click for several; with none "
            "selected the active header scan is used. The list honors the "
            "Setup → “Choose scans to show…” filter.")
        scl.addWidget(self.scans_list)
        srow = QHBoxLayout()
        self.scans_summary = QLabel("")
        self.scans_summary.setStyleSheet("color:#888; font-size:0.9em;")
        srow.addWidget(self.scans_summary, 1)
        b_sall = QPushButton("Select all")
        b_snone = QPushButton("Select none")
        b_sall.clicked.connect(self.scans_list.selectAll)
        b_snone.clicked.connect(self.scans_list.clearSelection)
        srow.addWidget(b_sall)
        srow.addWidget(b_snone)
        scl.addLayout(srow)
        self.scans_list.itemSelectionChanged.connect(self._refresh_scans_summary)
        lay.addWidget(scans_box)

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
        self.peak_algo = self._make_algo_list(
            "Peak detectors. Ctrl/Shift-click to run several over each scan.")
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
        self.shape_algo = self._make_algo_list(
            "Shape algorithms. Ctrl/Shift-click to run several over each scan.")
        sl.addWidget(self.shape_algo, 1)
        speaks = QVBoxLayout()
        speaks.addWidget(QLabel("Peaks:"))
        self.shape_src = QComboBox()
        self.shape_src.setToolTip(
            "Input peak set for shape finding. “run peak algorithm above first” "
            "chains peaks→shapes per scan (required when several scans/algorithms "
            "are selected, since saved peak sets are per-scan).")
        speaks.addWidget(self.shape_src)
        speaks.addStretch()
        sl.addLayout(speaks, 1)
        run_shapes = QPushButton("Run")
        run_shapes.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        run_shapes.setMinimumHeight(40)
        run_shapes.clicked.connect(self._run_shapes)
        sl.addWidget(run_shapes)
        lay.addWidget(shape_box)

        # ---- Combined (peak + shape in one pass) ------------------------
        comb_box = QGroupBox("Combined  (peak + shape in one per-frame pass · 1×1)")
        cb = QHBoxLayout(comb_box)
        self.combined_algo = self._make_algo_list(
            "Combined per-frame algorithms. Ctrl/Shift-click to run several.")
        cb.addWidget(self.combined_algo, 1)
        run_combined = QPushButton("Run")
        run_combined.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        run_combined.setMinimumHeight(40)
        run_combined.clicked.connect(self._run_combined)
        cb.addWidget(run_combined)
        lay.addWidget(comb_box)

        # ---- Territorial reference (skew-free source of truth) -----------
        # One button for the whole territory-grid → bin → peaks → shapes
        # (--variant territory, 1×1) chain, so the Territory Map / Device-View
        # territorial reference is buildable from the GUI (not CLI-only).
        terr_box = QGroupBox(
            "Territorial reference  (skew-free source of truth · 1×1)")
        tb = QHBoxLayout(terr_box)
        tb.addWidget(QLabel("Target frames/territory:"))
        self.terr_target = QSpinBox()
        self.terr_target.setRange(1, 999)
        self.terr_target.setValue(9)
        self.terr_target.setToolTip(
            "Frames grouped per territory before it stops growing (small ≈ 1×1 "
            "resolution, large = higher per-cell SNR). 9 is the default reference.")
        tb.addWidget(self.terr_target)
        build_terr = QPushButton("Build territorial reference")
        build_terr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        build_terr.setMinimumHeight(40)
        build_terr.setToolTip(
            "Run territory-grid → bin → peaks → shapes (--variant territory) for "
            "the selected scans. Builds the skew-free cell-model map the Territory "
            "Map tab and the Device-View “Territorial reference available →” button "
            "read. Heavy: reads raw frames (needs the raw scan dir mounted).")
        build_terr.clicked.connect(self._build_territory)
        tb.addWidget(build_terr)
        lay.addWidget(terr_box)

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
        # A tabbed panel of consoles: each Run reuses the active console if it's
        # idle, else opens a new console tab, so several jobs run at once in one
        # window (no more opening a second GUI to run a second thing).
        self.console = ConsolePanel()
        lay.addWidget(self.console, 1)

        self.update_context(scan, bin_size)

    # ----- widgets ----------------------------------------------------
    @staticmethod
    def _make_algo_list(tooltip):
        """A compact multi-select algorithm list (Ctrl/Shift-click)."""
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lst.setMaximumHeight(96)
        lst.setToolTip(tooltip)
        return lst

    # ----- context ----------------------------------------------------
    def update_context(self, scan, bin_size):
        self.scan = scan
        self.bin_size = bin_size
        self._populate_scans_list()
        self._populate_bins(select=bin_size)
        self._refresh_algos()
        self._refresh_peak_sources()
        self._refresh_bins_status()

    def _populate_scans_list(self):
        """Fill the target-scans list (honors Setup's visible-scan filter).

        Preserves an existing multi-selection across refreshes; if nothing is
        selected yet, defaults to the active header scan.
        """
        dm = DataManager(self.project_root, scan=self.scan)
        scans = dm.discover_scans(selected_only=True)
        prev = set(self._selected_scans(fallback=False))
        self.scans_list.blockSignals(True)
        self.scans_list.clear()
        for name in scans:
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, name)
            self.scans_list.addItem(it)
            if (name in prev) or (not prev and name == self.scan):
                it.setSelected(True)
        self.scans_list.blockSignals(False)
        self._refresh_scans_summary()

    def _refresh_scans_summary(self):
        names = self._selected_scans(fallback=False)
        if not names:
            self.scans_summary.setText(
                f"→ active scan only ({self.scan or '—'})")
        else:
            shown = ", ".join(names[:6]) + (" …" if len(names) > 6 else "")
            self.scans_summary.setText(f"{len(names)} selected: {shown}")

    def _selected_scans(self, fallback=True):
        """Selected target scans; falls back to the active scan when none."""
        names = [self.scans_list.item(i).data(Qt.UserRole)
                 for i in range(self.scans_list.count())
                 if self.scans_list.item(i).isSelected()]
        if names or not fallback:
            return names
        return [self.scan] if self.scan else []

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
        # Peak (Phase 1): data = (bare name, is_perframe).
        dets = dm.list_detectors(bs) or dm.list_detectors()
        self._fill_algo_list(
            self.peak_algo,
            [(format_detector_label(d), (d["name"], d.get("pipeline") == "perframe"))
             for d in dets])
        # Shape (Phase 2) library — names carry scores, data = bare name.
        self._fill_algo_list(
            self.shape_algo,
            [(format_detector_label(d), d["name"]) for d in dm.list_shapes()])
        # Combined (peak+shape) library — names carry scores, data = bare name.
        self._fill_algo_list(
            self.combined_algo,
            [(format_detector_label(d), d["name"]) for d in dm.list_combined()])

    @staticmethod
    def _fill_algo_list(lst, entries):
        """Populate a multi-select algorithm list, preserving prior selection.

        ``entries`` is a list of ``(label, data)``. When nothing was selected
        before, the first (highest-scoring) entry is pre-selected so a plain Run
        works without a manual pick.
        """
        prev = {lst.item(i).data(Qt.UserRole)
                for i in range(lst.count()) if lst.item(i).isSelected()}
        lst.blockSignals(True)
        lst.clear()
        if not entries:
            it = QListWidgetItem("(none found)")
            it.setFlags(Qt.NoItemFlags)
            lst.addItem(it)
            lst.blockSignals(False)
            return
        for label, data in entries:
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, data)
            lst.addItem(it)
        restored = False
        for i in range(lst.count()):
            if lst.item(i).data(Qt.UserRole) in prev:
                lst.item(i).setSelected(True)
                restored = True
        if not restored and lst.count():
            lst.item(0).setSelected(True)
        lst.blockSignals(False)

    def _selected_peak_algos(self):
        """List of (name, is_unbinned) for the selected peak detectors."""
        out = []
        for i in range(self.peak_algo.count()):
            it = self.peak_algo.item(i)
            data = it.data(Qt.UserRole)
            if it.isSelected() and isinstance(data, tuple):
                out.append(data)
        return out

    @staticmethod
    def _selected_names(lst):
        """Selected bare algorithm names from a shape/combined list."""
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count())
                if lst.item(i).isSelected() and lst.item(i).data(Qt.UserRole)]

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
        scans = self._selected_scans()
        if not scans:
            self.console._append("[no scan selected — load data in Setup first]\n")
            return
        if self._cur_bin() is None:
            self.console._append("[no bins — create one in Data Prep first]\n")
            return
        algos = self._selected_peak_algos()
        if not algos:
            self.console._append("[no peak algorithm selected]\n")
            return
        bs = self._cur_bin()
        jobs, skipped = [], []
        for name in scans:
            for algo, unbinned in algos:
                if unbinned:
                    if algo not in skipped:
                        skipped.append(algo)
                    continue
                jobs.append(["peaks", "--root", self.project_root,
                             "--scan", str(name), "--bin-size", str(bs),
                             "--algorithm", algo])
        notes = []
        if skipped:
            notes.append(
                f"[skipping per-frame (unbinned) detector(s): {', '.join(skipped)} "
                "— they can't run in the binned peak pipeline]")
        if not jobs:
            if notes:
                self.console._append(notes[0] + "\n")
            return
        self._run_jobs(jobs, scans, len(algos), notes=notes)

    def _make_bins(self):
        scans = self._selected_scans()
        if not scans:
            self.console._append("[no scan selected — load data in Setup first]\n")
            return
        bs = self.make_bin_spin.value()  # Create bins depends only on this spin box.
        jobs = [["make-bins", "--root", self.project_root, "--scan", str(n),
                 "--bin-size", str(bs)] for n in scans]
        self.console.run_many(jobs, on_all_finished=lambda _n: self._on_bins_built(bs))

    def _on_bins_built(self, bs):
        """After a build, surface the new size in the Existing-bins dropdown."""
        self._populate_bins(select=bs)
        self._refresh_algos()
        self._refresh_peak_sources()
        self._refresh_bins_status()

    def _run_combined(self):
        scans = self._selected_scans()
        if not scans:
            self.console._append("[no scan selected — load data in Setup first]\n")
            return
        algos = self._selected_names(self.combined_algo)
        if not algos:
            self.console._append("[no combined algorithm selected]\n")
            return
        jobs = [["run-combined", "--root", self.project_root, "--scan", str(n),
                 "--algorithm", a] for n in scans for a in algos]
        self._run_jobs(jobs, scans, len(algos))

    def _build_territory(self):
        scans = self._selected_scans()
        if not scans:
            self.console._append("[no scan selected — load data in Setup first]\n")
            return
        target = self.terr_target.value()
        jobs = [["territory-build", "--root", self.project_root, "--scan", str(n),
                 "--target-size", str(target)] for n in scans]
        self.console.run_many(
            jobs, on_all_finished=lambda _n: self._refresh_algos())

    def _show_lineage(self):
        scans = self._selected_scans()
        if not scans:
            self.console._append("[no scan selected — load data in Setup first]\n")
            return
        jobs = [["lineage", "--root", self.project_root, "--scan", str(n)]
                for n in scans]
        self.console.run_many(jobs)

    def _open_cvevolve(self):
        bs = self._cur_bin()
        if bs is None:
            self.console._append("[no bins — create one in Data Prep first]\n")
            return
        from .cvevolve_dialog import CVEvolveDialog
        CVEvolveDialog(self.project_root, scan=self.scan,
                       bin_size=bs, parent=self).exec_()

    def _run_shapes(self):
        scans = self._selected_scans()
        if not scans:
            self.console._append("[no scan selected — load data in Setup first]\n")
            return
        if self._cur_bin() is None:
            self.console._append("[no bins — create one in Data Prep first]\n")
            return
        bs = self._cur_bin()
        # None in the list = let the CLI fall back to its default shape algo.
        shape_algos = self._selected_names(self.shape_algo) or [None]
        src = self.shape_src.currentText()
        jobs, notes = [], []
        # Chain: run the selected peak algorithm(s), then shapes, per scan.
        if src == _CHAIN_OPTION:
            peak_algos = self._selected_peak_algos() or [(None, False)]
            skipped = []
            for name in scans:
                for peak_algo, unbinned in peak_algos:
                    if unbinned:
                        if peak_algo not in skipped:
                            skipped.append(peak_algo)
                        continue
                    for sa in shape_algos:
                        args = ["run-pipeline", "--root", self.project_root,
                                "--scan", str(name), "--bin-size", str(bs)]
                        if peak_algo:
                            args += ["--algorithm", peak_algo]
                        if sa:
                            args += ["--shape-algo", sa]
                        jobs.append(args)
            if skipped:
                notes.append(
                    f"[skipping per-frame (unbinned) detector(s): "
                    f"{', '.join(skipped)}]")
        else:
            if not src or src.startswith("("):
                self.console._append("[no peak set — run Peak Finding first]\n")
                return
            # An explicit (per-scan-named) peak set only makes sense for one scan.
            if len(scans) > 1:
                self.console._append(
                    "[multiple scans selected — switch Peaks to “run peak "
                    "algorithm above first” to chain per scan, since saved peak "
                    "sets are per-scan]\n")
                return
            for name in scans:
                for sa in shape_algos:
                    args = ["shapes", "--root", self.project_root,
                            "--scan", str(name), "--bin-size", str(bs),
                            "--peak-algo", src]
                    if sa:
                        args += ["--algorithm", sa]
                    jobs.append(args)
        if not jobs:
            return
        self._run_jobs(jobs, scans, max(1, len(jobs) // max(1, len(scans))),
                       notes=notes)

    def _run_jobs(self, jobs, scans, n_algos, notes=None):
        """Run a fan-out queue, announcing the scan × algorithm spread.

        ``notes`` are extra lines printed above the run (e.g. skipped detectors);
        they are passed as the console header so they survive the log clear.
        """
        header = "\n".join(notes) if notes else None
        if len(jobs) > 1:
            spread = (f"[queuing {len(jobs)} jobs: {len(scans)} scan(s) × "
                      f"{n_algos} algorithm(s)]")
            header = f"{header}\n{spread}" if header else spread
            self.console.run_many(jobs, header=header)
        else:
            self.console.run(jobs[0], header=header)


def make_tab(project_root=".", scan=None, bin_size=3):
    return ProgramsTab(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
