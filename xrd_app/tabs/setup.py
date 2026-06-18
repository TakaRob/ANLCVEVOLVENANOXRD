"""Setup tab — project summary, data loading, calibration (MVP)."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ._console import JobConsole

TAB_META = {
    "title": "Setup",
    "order": 10,
    "takes_bin_size": False,
    "scan_dependent": False,
    "general": (
        "Create/open a project, load scan data (a single .hdf5 → its scan dir, or "
        "a Scans/ parent → all scans), and set calibration (tth.tiff now; a "
        ".poni→tth conversion button is reserved). The active scan drives every "
        "other tab."
    ),
}


class SetupTab(QWidget):
    def __init__(self, project_root, scan=None, bin_size=3):
        super().__init__()
        self.project_root = str(Path(project_root).resolve())
        self.scan = scan

        lay = QVBoxLayout(self)

        # ---- project summary --------------------------------------------
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-family: monospace;")
        box = QGroupBox("Project")
        bl = QVBoxLayout(box)
        bl.addWidget(self.summary)
        lay.addWidget(box)

        # ---- data loading -----------------------------------------------
        load_box = QGroupBox("Load Data")
        ll = QHBoxLayout(load_box)
        b_file = QPushButton("Select scan .hdf5…")
        b_file.clicked.connect(self._pick_file)
        b_dir = QPushButton("Select Scans directory…")
        b_dir.clicked.connect(self._pick_dir)
        ll.addWidget(b_file)
        ll.addWidget(b_dir)
        ll.addStretch()
        lay.addWidget(load_box)

        # ---- calibration ------------------------------------------------
        cal_box = QGroupBox("Calibration & Reflections")
        cl = QHBoxLayout(cal_box)
        b_tth = QPushButton("Load tth.tiff…")
        b_tth.clicked.connect(self._load_tth)
        b_poni = QPushButton("Convert .poni → tth…")
        b_poni.clicked.connect(self._convert_poni)
        b_refl = QPushButton("Manual reflections…")
        b_refl.clicked.connect(self._open_reflections)
        cl.addWidget(b_tth)
        cl.addWidget(b_poni)
        cl.addWidget(b_refl)
        cl.addStretch()
        lay.addWidget(cal_box)

        self.console = JobConsole()
        lay.addWidget(self.console, 1)

        self.update_context(scan, bin_size)

    # ----- context ----------------------------------------------------
    def update_context(self, scan, bin_size=3):
        self.scan = scan
        self._refresh_summary()

    def _refresh_summary(self):
        dm = DataManager(self.project_root, scan=self.scan)
        cfg = dm.config
        shape = cfg.get("detector", "shape")
        scans = dm.discover_scans()
        lines = [
            f"name:   {cfg.get('name')}",
            f"root:   {dm.root}",
            f"active: {dm.scan_name or '—'}",
            f"frame:  {tuple(shape) if shape else '— (load data)'}",
            f"scans:  {len(scans)}  {', '.join(scans) if scans else ''}",
            f"tth:    {dm.tth_map()}",
        ]
        from ..core import io
        warn = io.slow_mount_warning(dm.binned_dir_root)
        if warn:
            lines.append(f"\n⚠ {warn}")
        self.summary.setText("\n".join(lines))

    # ----- actions ----------------------------------------------------
    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a scan .hdf5 file", self.project_root,
            "HDF5 (*.h5 *.hdf5)")
        if path:
            self.console.run(["scan-detect", "--root", self.project_root,
                              "--scan-file", path])

    def _pick_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select the Scans directory", self.project_root)
        if path:
            self.console.run(["scan-detect", "--root", self.project_root,
                              "--scans-dir", path])

    def _load_tth(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select tth.tiff", self.project_root, "TIFF (*.tif *.tiff)")
        if path:
            self.console.run(["link", "--root", self.project_root, "--tth", path])

    def _convert_poni(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a pyFAI .poni file", self.project_root, "PONI (*.poni)")
        if path:
            args = ["convert-poni", "--root", self.project_root, "--poni", path]
            if self.scan:
                args += ["--scan", str(self.scan)]
            self.console.run(args)

    def _open_reflections(self):
        from .reflection_popup import open_reflection_dialog
        dlg = open_reflection_dialog(self.project_root, scan=self.scan, parent=self)
        dlg.exec_()
        self._refresh_summary()


def make_tab(project_root=".", scan=None, bin_size=3):
    return SetupTab(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
