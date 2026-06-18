"""[Use CVEvolve] popup: build a dev/holdout split, then run CVEvolve.

Pick a development set by bins (verified labels, or an algorithm's peak/shape
set — which sets the evolved kind), choose a holdout percentage with a live
count, build the seeded split, then optionally launch CVEvolve. Every action
shells out to the CLI through an embedded console.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
)

from ..config import DataManager
from ._console import JobConsole


class CVEvolveDialog(QDialog):
    def __init__(self, project_root, scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.project_root = str(Path(project_root).resolve())
        self.scan = scan
        self.bin_size = bin_size
        self.setWindowTitle("Use CVEvolve")
        self.resize(820, 620)

        lay = QVBoxLayout(self)

        # ---- Development set --------------------------------------------
        dev_box = QGroupBox("Pick Development Set (by bins)")
        form = QFormLayout(dev_box)
        self.kind = QComboBox()
        self.kind.addItems(["verified", "peaks", "shapes"])
        self.kind.currentTextChanged.connect(self._refresh_algos)
        form.addRow("Source:", self.kind)

        self.algo = QComboBox()
        form.addRow("Algorithm:", self.algo)

        self.holdout = QDoubleSpinBox()
        self.holdout.setRange(0, 90)
        self.holdout.setValue(20)
        self.holdout.setSuffix(" %")
        form.addRow("Holdout:", self.holdout)

        self.seed = QSpinBox()
        self.seed.setRange(0, 10_000)
        self.seed.setValue(42)
        form.addRow("Seed:", self.seed)

        self.count_lbl = QLabel("(build to see counts)")
        form.addRow("Split:", self.count_lbl)

        build_btn = QPushButton("Build Split")
        build_btn.clicked.connect(self._build)
        form.addRow(build_btn)
        lay.addWidget(dev_box)

        # ---- Run CVEvolve ----------------------------------------------
        run_box = QGroupBox("Run CVEvolve")
        rl = QFormLayout(run_box)
        cfg_row = QHBoxLayout()
        self.config = QLineEdit()
        self.config.setPlaceholderText("CVEvolve config.yaml")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_config)
        cfg_row.addWidget(self.config)
        cfg_row.addWidget(browse)
        rl.addRow("Config:", cfg_row)

        self.engine = QComboBox()
        self.engine.addItems(["podman", "docker", "local"])
        rl.addRow("Engine:", self.engine)

        run_btn = QPushButton("Run CVEvolve")
        run_btn.clicked.connect(self._run)
        rl.addRow(run_btn)
        lay.addWidget(run_box)

        self.console = JobConsole()
        lay.addWidget(self.console, 1)

        self._refresh_algos()

    # ----- helpers ----------------------------------------------------
    def _refresh_algos(self, *_):
        kind = self.kind.currentText()
        self.algo.clear()
        if kind == "verified":
            self.algo.setEnabled(False)
            self.algo.addItem("(reviewed bins)")
            return
        self.algo.setEnabled(True)
        dm = DataManager(self.project_root, scan=self.scan)
        ldir = dm.labels_dir(self.scan)
        bs = f"{self.bin_size}x{self.bin_size}"
        pat = f"*_{kind}_{bs}.json"
        found = []
        if ldir.is_dir():
            for p in sorted(ldir.glob(pat)):
                found.append(p.stem.replace(f"_{kind}_{bs}", ""))
        self.algo.addItems(found or [f"(run {kind} first)"])

    def _pick_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CVEvolve config.yaml", self.project_root, "YAML (*.yaml *.yml)")
        if path:
            self.config.setText(path)

    def _build(self):
        args = ["build-holdout", "--root", self.project_root,
                "--source", self.kind.currentText(),
                "--bin-size", str(self.bin_size),
                "--holdout-pct", str(self.holdout.value()),
                "--seed", str(self.seed.value())]
        if self.scan:
            args += ["--scan", str(self.scan)]
        if self.kind.currentText() != "verified":
            algo = self.algo.currentText()
            if algo.startswith("("):
                self.count_lbl.setText("no algorithm output found — run it first")
                return
            args += ["--algorithm", algo]
        self.count_lbl.setText("building…")
        self.console.run(args)

    def _run(self):
        cfg = self.config.text().strip()
        if not cfg:
            self.console._append("[pick a CVEvolve config.yaml first]\n")
            return
        self.console.run(["run-cvevolve", "--root", self.project_root,
                          "--config", cfg, "--engine", self.engine.currentText()])
