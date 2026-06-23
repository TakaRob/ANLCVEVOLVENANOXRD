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

# Radial background models from NoiseReduction/noise_reduction_algorithms.py.
_NOISE_OPTIONS = ["(none)", "gaussian", "split_gaussian", "skewed_gaussian",
                  "fourier"]


class SaveAlgorithmDialog(QDialog):
    def __init__(self, project_root, bin_size=3, kind="peak", parent=None,
                 sensitivity=None, noise_reduction=None, noise_strength=1.0,
                 noise_shift=0.0, log_scale=False, clip_lo_pct=None,
                 clip_hi_pct=None, compact=False):
        super().__init__(parent)
        self.project_root = project_root
        self.bin_size = bin_size
        self.kind = kind
        self.compact = compact
        # Prefill values used in compact mode (taken from the live GUI view).
        self._sensitivity = sensitivity if sensitivity is not None else 4.0
        self._noise_reduction = noise_reduction
        self._noise_strength = noise_strength
        self._noise_shift = noise_shift
        self._log_scale = log_scale
        self._clip_lo_pct = clip_lo_pct
        self._clip_hi_pct = clip_hi_pct
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

        # Compact mode (View/Label "Run on displayed image" flow): sensitivity,
        # noise reduction and bin size are taken from the live view, so we only
        # ask for the base detector + name and show the baked values as a hint.
        if not compact:
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
        if compact:
            if self._noise_reduction:
                nr_txt = (f"{self._noise_reduction} "
                          f"(strength {self._noise_strength:g}, "
                          f"shift {self._noise_shift:g})")
            else:
                nr_txt = "none"
            disp = []
            if self._log_scale:
                disp.append("log")
            if self._clip_lo_pct is not None and self._clip_hi_pct is not None:
                disp.append(
                    f"clip {self._clip_lo_pct:g}–{self._clip_hi_pct:g}%")
            disp_txt = (" · " + ", ".join(disp)) if disp else ""
            self.status.setStyleSheet("color: #888; font-style: italic;")
            self.status.setText(
                f"Using current view: sensitivity {self._sensitivity:g} · "
                f"{bin_size}x{bin_size} · noise: {nr_txt}{disp_txt}")
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
        if self.compact:
            nr = self._noise_reduction
            bs = int(self.bin_size)
            sens = self._sensitivity
            strength = self._noise_strength
            shift = self._noise_shift
            log_scale = self._log_scale
            clip_lo, clip_hi = self._clip_lo_pct, self._clip_hi_pct
        else:
            nr = self.noise.currentText()
            nr = None if nr == "(none)" else nr
            bs = int(self.bin.currentText().split("x")[0])
            sens = self.sens.value()
            strength, shift = 1.0, 0.0
            log_scale, clip_lo, clip_hi = False, None, None
        try:
            out = save_algorithm.save_algorithm(
                self.base.currentData() or self.base.currentText(),
                sensitivity=sens,
                bin_size=bs, noise_reduction=nr,
                noise_strength=strength, noise_shift=shift,
                log_scale=log_scale, clip_lo_pct=clip_lo, clip_hi_pct=clip_hi,
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
