"""Library-backed material ROI editor for the standalone XRF application."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .core import xrf as xrf_core
from .core import xrf_selection


class XRFMaterialPopup(QDialog):
    """Edit selected material ROIs beside the calibrated summed spectrum."""

    def __init__(self, selection, parent=None):
        super().__init__(parent)
        self.selection = selection
        self.calibration = selection["attrs"].get("energy_calibration") or {}
        self.setWindowTitle("XRF material library and integration ranges")
        self.resize(1350, 720)

        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        body.addWidget(self.canvas, 3)

        side = QVBoxLayout()
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Range units:"))
        self.units = QComboBox()
        self.units.addItems(["keV", "pixel"])
        self.units.currentTextChanged.connect(self._convert_table)
        mode_row.addWidget(self.units)
        side.addLayout(mode_row)

        side.addWidget(QLabel("<b>Predicted library lines near observed peaks</b>"))
        self.predictions = QListWidget()
        self.predictions.itemDoubleClicked.connect(lambda _: self._add_prediction())
        side.addWidget(self.predictions, 1)
        add_button = QPushButton("Add selected prediction")
        add_button.clicked.connect(self._add_prediction)
        side.addWidget(add_button)

        side.addWidget(QLabel("<b>Selected material integration ranges</b>"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Material", "Low", "High"])
        side.addWidget(self.table, 1)
        manual_button = QPushButton("Add manual material")
        manual_button.clicked.connect(self._add_manual)
        side.addWidget(manual_button)
        remove_button = QPushButton("Remove selected material")
        remove_button.clicked.connect(self._remove_selected)
        side.addWidget(remove_button)
        body.addLayout(side, 2)
        layout.addLayout(body)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._populate_predictions()
        self._populate_selected()
        self._draw()

    def _populate_predictions(self):
        spectrum = self.selection.get("spectrum")
        if spectrum is None:
            return
        counts = spectrum["summed_counts"]
        energy = spectrum["energy_kev"]
        smoothed = np.log10(np.maximum(counts, 1.0))
        from scipy.signal import find_peaks
        found, properties = find_peaks(smoothed, prominence=0.05, distance=15)
        order = np.argsort(properties["prominences"])[::-1][:20]
        seen = set()
        for index in order:
            pixel = int(found[index])
            observed_kev = float(energy[pixel])
            line = xrf_core.nearest_emission_line(observed_kev * 1000.0, tol_ev=200.0)
            if line is None:
                continue
            name = line["element"]
            if name in seen:
                continue
            seen.add(name)
            item = QListWidgetItem(
                f"{name}: {line['line']} {line['energy_ev'] / 1000:.3f} keV "
                f"(observed {observed_kev:.3f} keV)"
            )
            item.setData(Qt.UserRole, {
                "name": name,
                "center_kev": observed_kev,
                "line": xrf_core.emission_line_name(line),
            })
            self.predictions.addItem(item)

    def _populate_selected(self):
        for name, material in self.selection["materials"].items():
            attrs = material["attrs"]
            bounds = attrs.get("energy_range_kev")
            if bounds is None:
                bounds = xrf_selection.pixel_to_kev(attrs["pixel_range"], self.calibration)
            self._append_row(name, bounds[0], bounds[1])

    def _append_row(self, name, low, high):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(name)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{float(low):.6g}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{float(high):.6g}"))

    def _add_prediction(self):
        item = self.predictions.currentItem()
        if item is None:
            return
        data = item.data(Qt.UserRole)
        center = data["center_kev"]
        if self.units.currentText() == "keV":
            low, high = center - 0.15, center + 0.15
        else:
            low, high = xrf_selection.kev_to_pixel(
                [center - 0.15, center + 0.15], self.calibration
            )
        self._append_row(data["name"], low, high)
        self._draw()

    def _add_manual(self):
        if self.units.currentText() == "keV":
            low, high = 1.0, 1.3
        else:
            low, high = xrf_selection.kev_to_pixel([1.0, 1.3], self.calibration)
        self._append_row("New material", low, high)
        self._draw()

    def _remove_selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self._draw()

    def _convert_table(self, units):
        for row in range(self.table.rowCount()):
            values = [float(self.table.item(row, column).text()) for column in (1, 2)]
            converted = (
                xrf_selection.pixel_to_kev(values, self.calibration)
                if units == "keV" else xrf_selection.kev_to_pixel(values, self.calibration)
            )
            for column, value in zip((1, 2), converted):
                self.table.item(row, column).setText(f"{float(value):.6g}")
        self._draw()

    def definitions(self):
        definitions = {}
        units = self.units.currentText()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip()
            low, high = sorted(float(self.table.item(row, column).text()) for column in (1, 2))
            old = self.selection["materials"].get(name, {}).get("attrs", {})
            definitions[name] = {
                "display_name": name,
                "minimum_counts": old.get("minimum_counts"),
                "energy_range_kev" if units == "keV" else "pixel_range": [low, high],
            }
        return definitions

    def _draw(self):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        spectrum = self.selection.get("spectrum")
        if spectrum is not None:
            axis.plot(spectrum["energy_kev"], spectrum["summed_counts"], color="black", lw=0.8)
            for name, definition in self.definitions().items():
                bounds = definition.get("energy_range_kev")
                if bounds is None:
                    bounds = xrf_selection.pixel_to_kev(definition["pixel_range"], self.calibration)
                axis.axvspan(*bounds, alpha=0.2, label=name)
            axis.set_yscale("log")
            if self.table.rowCount():
                axis.legend(fontsize=8)
        axis.set(xlabel="Calibrated energy (keV)", ylabel="Summed counts")
        axis.grid(alpha=0.2)
        self.canvas.draw_idle()
