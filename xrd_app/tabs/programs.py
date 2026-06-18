"""Programs tab — run the pipeline. Every Run button shells out to the CLI."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ._console import JobConsole

TAB_META = {
    "title": "Programs",
    "order": 20,
    "takes_bin_size": True,
    "scan_dependent": False,
    "general": (
        "Run the two-stage pipeline. Peak Finding runs a detector over every bin "
        "(Phase 1). Shape Finding links peaks across bins and keeps gaussian-like "
        "features (Phase 2). Each Run shells out to the CLI; output streams below."
    ),
}

_BIN_SIZES = [1, 3, 4, 5]


class ProgramsTab(QWidget):
    def __init__(self, project_root, scan=None, bin_size=3):
        super().__init__()
        self.project_root = str(Path(project_root).resolve())
        self.scan = scan
        self.bin_size = bin_size

        lay = QVBoxLayout(self)

        # ---- Peak Finding ------------------------------------------------
        peak_box = QGroupBox("Peak Finding  (Phase 1: per-bin detection)")
        pl = QHBoxLayout(peak_box)
        pl.addWidget(QLabel("Algorithm:"))
        self.peak_algo = QComboBox()
        pl.addWidget(self.peak_algo, 1)
        pl.addWidget(QLabel("Bin:"))
        self.peak_bin = QComboBox()
        self.peak_bin.addItems([f"{b}x{b}" for b in _BIN_SIZES])
        pl.addWidget(self.peak_bin)
        run_peaks = QPushButton("Run")
        run_peaks.clicked.connect(self._run_peaks)
        pl.addWidget(run_peaks)
        lay.addWidget(peak_box)

        # ---- Shape Finding ----------------------------------------------
        shape_box = QGroupBox("Shape Finding  (Phase 2: link + gaussian filter)")
        sl = QHBoxLayout(shape_box)
        sl.addWidget(QLabel("Peaks:"))
        self.shape_src = QComboBox()
        sl.addWidget(self.shape_src, 1)
        sl.addWidget(QLabel("Bin:"))
        self.shape_bin = QComboBox()
        self.shape_bin.addItems([f"{b}x{b}" for b in _BIN_SIZES])
        sl.addWidget(self.shape_bin)
        run_shapes = QPushButton("Run")
        run_shapes.clicked.connect(self._run_shapes)
        sl.addWidget(run_shapes)
        lay.addWidget(shape_box)

        # ---- CVEvolve ----------------------------------------------------
        cve_row = QHBoxLayout()
        cve_btn = QPushButton("Use CVEvolve…")
        cve_btn.clicked.connect(self._open_cvevolve)
        cve_row.addWidget(cve_btn)
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
        if bin_size in _BIN_SIZES:
            self.peak_bin.setCurrentText(f"{bin_size}x{bin_size}")
            self.shape_bin.setCurrentText(f"{bin_size}x{bin_size}")
        self._refresh_algos()
        self._refresh_peak_sources()

    def _bin(self, combo):
        return int(combo.currentText().split("x")[0])

    def _refresh_algos(self):
        dm = DataManager(self.project_root, scan=self.scan)
        bs = self._bin(self.peak_bin)
        names = [d["name"] for d in dm.list_detectors(bs)] or \
                [d["name"] for d in dm.list_detectors()]
        self.peak_algo.clear()
        self.peak_algo.addItems(names or ["(none found)"])

    def _refresh_peak_sources(self):
        """List saved *_peaks.json in Labels/<scan>/ as shape-finding inputs."""
        self.shape_src.clear()
        dm = DataManager(self.project_root, scan=self.scan)
        ldir = dm.labels_dir(self.scan)
        bs = self._bin(self.shape_bin)
        found = []
        if ldir.is_dir():
            for p in sorted(ldir.glob(f"*_peaks_{bs}x{bs}.json")):
                found.append(p.stem.replace(f"_peaks_{bs}x{bs}", ""))
        self.shape_src.addItems(found or ["(run peak finding first)"])

    # ----- actions ----------------------------------------------------
    def _run_peaks(self):
        if not self.scan:
            self.console._append("[no active scan — load data in Setup first]\n")
            return
        algo = self.peak_algo.currentText()
        args = ["peaks", "--root", self.project_root, "--scan", str(self.scan),
                "--bin-size", str(self._bin(self.peak_bin))]
        if algo and not algo.startswith("("):
            args += ["--algorithm", algo]
        self.console.run(args)

    def _open_cvevolve(self):
        from .cvevolve_dialog import CVEvolveDialog
        CVEvolveDialog(self.project_root, scan=self.scan,
                       bin_size=self._bin(self.shape_bin), parent=self).exec_()

    def _run_shapes(self):
        if not self.scan:
            self.console._append("[no active scan — load data in Setup first]\n")
            return
        src = self.shape_src.currentText()
        if not src or src.startswith("("):
            self.console._append("[no peak set — run Peak Finding first]\n")
            return
        args = ["shapes", "--root", self.project_root, "--scan", str(self.scan),
                "--bin-size", str(self._bin(self.shape_bin)), "--peak-algo", src]
        self.console.run(args)


def make_tab(project_root=".", scan=None, bin_size=3):
    return ProgramsTab(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
