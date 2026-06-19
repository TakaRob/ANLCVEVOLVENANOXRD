"""Dialog for the Save Algorithm action (used by View/Label and Shape/Verify).

Collects a base detector + sensitivity (+ optional noise reduction) and writes a
runnable wrapper into the algorithm library via ``core.save_algorithm``.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from ..config import DataManager, format_detector_label
from ..core import save_algorithm

_NOISE_OPTIONS = ["(none)", "gaussian"]


class SaveAlgorithmDialog(QDialog):
    def __init__(self, project_root, bin_size=3, kind="peak", parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self.bin_size = bin_size
        self.kind = kind
        self.setWindowTitle(f"Save Algorithm ({kind})")

        lay = QVBoxLayout(self)
        form = QFormLayout()

        dm = DataManager(project_root)
        dets = dm.list_detectors(bin_size) or dm.list_detectors()
        # A wrapper delegates to the binned detector contract, so per-frame
        # (unbinned) detectors can't be a base.
        dets = [d for d in dets if d.get("pipeline") != "perframe"]
        self.base = QComboBox()
        for d in dets:
            self.base.addItem(format_detector_label(d), d["name"])
        if not dets:
            self.base.addItem("5x5_tophat_band_adaptive_snr",
                              "5x5_tophat_band_adaptive_snr")
        form.addRow("Base detector:", self.base)

        self.sens = QDoubleSpinBox()
        self.sens.setRange(0.1, 50.0)
        self.sens.setDecimals(2)
        self.sens.setSingleStep(0.5)
        self.sens.setValue(4.0)
        form.addRow("Sensitivity (SNR):", self.sens)

        self.noise = QComboBox()
        self.noise.addItems(_NOISE_OPTIONS)
        form.addRow("Noise reduction:", self.noise)

        self.bin = QComboBox()
        self.bin.addItems([f"{b}x{b}" for b in (1, 3, 4, 5)])
        self.bin.setCurrentText(f"{bin_size}x{bin_size}")
        form.addRow("Bin size:", self.bin)

        self.name = QLineEdit()
        self.name.setPlaceholderText("(auto: <base>__sens<NN>__nr-<...>)")
        form.addRow("Name (optional):", self.name)
        lay.addLayout(form)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        btns = QHBoxLayout()
        btns.addStretch()
        save = QPushButton("Save")
        save.clicked.connect(self._save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(save)
        btns.addWidget(cancel)
        lay.addLayout(btns)

    def _save(self):
        nr = self.noise.currentText()
        nr = None if nr == "(none)" else nr
        bs = int(self.bin.currentText().split("x")[0])
        try:
            out = save_algorithm.save_algorithm(
                self.base.currentData() or self.base.currentText(),
                sensitivity=self.sens.value(),
                bin_size=bs, noise_reduction=nr,
                name=self.name.text().strip() or None, kind=self.kind,
                source="manual")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        QMessageBox.information(
            self, "Saved",
            f"Saved {out.stem} → {out}\n\n"
            f"Run with: xrd-app peaks --bin-size {bs} --algorithm {out.stem}")
        self.accept()
