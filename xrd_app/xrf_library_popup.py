"""Read-only emission-line library browser for XRF spectrum labeling."""

from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QVBoxLayout,
)

from .core import xrf as xrf_core


class XRFLineLibraryDialog(QDialog):
    """Browse known emission lines within an editable keV interval."""

    def __init__(self, energy_range=(0.0, 15.0), parent=None):
        super().__init__(parent)
        self.setWindowTitle("XRF emission-line library")
        self.resize(620, 720)

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Energy range:"))
        self.minimum = QDoubleSpinBox()
        self.minimum.setRange(0.0, 100.0)
        self.minimum.setDecimals(3)
        self.minimum.setSuffix(" keV")
        self.minimum.setValue(float(energy_range[0]))
        controls.addWidget(self.minimum)
        controls.addWidget(QLabel("to"))
        self.maximum = QDoubleSpinBox()
        self.maximum.setRange(0.0, 100.0)
        self.maximum.setDecimals(3)
        self.maximum.setSuffix(" keV")
        self.maximum.setValue(float(energy_range[1]))
        controls.addWidget(self.maximum)
        layout.addLayout(controls)

        layout.addWidget(QLabel(
            "Select a line to inspect it. This reference does not modify the manual ROI table."
        ))
        self.lines = QListWidget()
        self.lines.currentItemChanged.connect(self._show_details)
        layout.addWidget(self.lines, 1)
        self.details = QLabel("Select an emission line.")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.minimum.valueChanged.connect(self._populate)
        self.maximum.valueChanged.connect(self._populate)
        self._populate()

    def _populate(self):
        low, high = sorted((self.minimum.value(), self.maximum.value()))
        self.lines.clear()
        for line in sorted(xrf_core.EMISSION_LINES, key=lambda item: item["energy_ev"]):
            energy_kev = float(line["energy_ev"]) / 1000.0
            if low <= energy_kev <= high:
                item = QListWidgetItem(
                    f"{line['element']:>2}  {line['line']:<2}    {energy_kev:8.3f} keV"
                )
                item.setData(256, dict(line))
                self.lines.addItem(item)
        self.details.setText(
            f"{self.lines.count()} library lines between {low:.3f} and {high:.3f} keV."
        )

    def _show_details(self, item):
        if item is None:
            return
        line = item.data(256)
        self.details.setText(
            f"Element: {line['element']}    Line: {line['line']}    "
            f"Energy: {line['energy_ev'] / 1000.0:.6f} keV"
        )
