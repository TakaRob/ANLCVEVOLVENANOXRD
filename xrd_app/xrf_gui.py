"""Standalone read-only GUI for finalized XRF project selections."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from . import workspace
from .core import xrf as xrf_core
from .core import xrf_selection
from .xrf_project import XRFProject


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)


class XRFAnalysisWindow(QMainWindow):
    """Choose an xrd-app project, then inspect its XRF add-on selections."""

    def __init__(self, project_root=None):
        super().__init__()
        self.project = None
        self.selection = None
        self.setWindowTitle("XRF Analysis")
        self.resize(1350, 850)

        self.main_tabs = QTabWidget()
        self.setup_page = self._build_setup_page()
        self.analysis_page = self._build_analysis_page()
        self.main_tabs.addTab(self.setup_page, "Setup")
        self.main_tabs.addTab(self.analysis_page, "Analysis")
        self.main_tabs.setTabEnabled(1, False)
        self.setCentralWidget(self.main_tabs)

        self._refresh_projects()
        if project_root is not None:
            self.open_project(project_root)

    def _button(self, text, callback):
        button = QPushButton(text)
        button.setMinimumHeight(40)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        button.clicked.connect(callback)
        return button

    def _build_setup_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        project_box = QGroupBox("Project")
        project_layout = QVBoxLayout(project_box)

        workspace_row = QHBoxLayout()
        workspace_row.addWidget(QLabel("Workspace:"))
        self.workspace_label = QLabel()
        self.workspace_label.setStyleSheet("font-family: monospace;")
        workspace_row.addWidget(self.workspace_label, 1)
        workspace_row.addWidget(self._button("Use launch directory", self._use_launch_directory))
        workspace_row.addWidget(self._button("Change...", self._choose_workspace))
        project_layout.addLayout(workspace_row)

        project_row = QHBoxLayout()
        project_row.addWidget(QLabel("Project:"))
        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(280)
        project_row.addWidget(self.project_combo, 1)
        project_row.addWidget(self._button("Open", self._open_selected))
        project_row.addWidget(self._button("New project...", self._new_project))
        project_row.addWidget(self._button("Open other...", self._browse_project))
        project_layout.addLayout(project_row)

        self.project_summary = QLabel(
            "No project open. Choose a workspace, then create or open a project."
        )
        self.project_summary.setWordWrap(True)
        self.project_summary.setStyleSheet("font-family: monospace;")
        project_layout.addWidget(self.project_summary)
        layout.addWidget(project_box)

        self.load_box = QGroupBox("Load Data")
        load_layout = QVBoxLayout(self.load_box)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button("Select scan folder...", self._load_scan_folder))
        buttons.addWidget(self._button("Select scan set...", self._load_scan_set))
        buttons.addWidget(self._button("Load processed selection...", self._load_processed_file))
        load_layout.addLayout(buttons)
        load_layout.addWidget(self._button("Process registered raw ME7...", self._process_raw_me7))
        self.data_status = QLabel("Open or create a project before loading data.")
        self.data_status.setWordWrap(True)
        self.data_status.setStyleSheet("font-family: monospace;")
        load_layout.addWidget(self.data_status)
        self.load_box.setEnabled(False)
        layout.addWidget(self.load_box)

        self.calibration_box = QGroupBox("Energy Calibration")
        calibration_layout = QFormLayout(self.calibration_box)
        self.calibration_quadratic = QDoubleSpinBox()
        self.calibration_quadratic.setDecimals(12)
        self.calibration_quadratic.setRange(-1.0, 1.0)
        self.calibration_linear = QDoubleSpinBox()
        self.calibration_linear.setDecimals(9)
        self.calibration_linear.setRange(-100.0, 100.0)
        self.calibration_offset = QDoubleSpinBox()
        self.calibration_offset.setDecimals(9)
        self.calibration_offset.setRange(-100.0, 100.0)
        calibration_layout.addRow("Quadratic (keV/pixel^2):", self.calibration_quadratic)
        calibration_layout.addRow("Linear (keV/pixel):", self.calibration_linear)
        calibration_layout.addRow("Offset (keV):", self.calibration_offset)
        calibration_layout.addRow(self._button("Save calibration", self._save_calibration))
        self.calibration_box.setEnabled(False)
        layout.addWidget(self.calibration_box)
        layout.addStretch()
        return page

    def _build_analysis_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QHBoxLayout()
        header.addWidget(QLabel("Scan:"))
        self.scan_combo = QComboBox()
        header.addWidget(self.scan_combo)
        self.status = QLabel()
        self.status.setStyleSheet("color: #555; padding-left: 12px;")
        header.addWidget(self.status, 1)
        layout.addLayout(header)

        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.addTab(self._build_label_spectrum_page(), "Label Spectrum")
        self.analysis_tabs.addTab(self._build_intensity_cut_page(), "Intensity Cut")
        layout.addWidget(self.analysis_tabs, 1)
        self.scan_combo.currentTextChanged.connect(self._load_scan)
        return page

    def _build_label_spectrum_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.spectrum_canvas = PlotCanvas()
        left_layout.addWidget(self.spectrum_canvas, 3)
        table_header = QHBoxLayout()
        table_header.addWidget(QLabel("<b>Manual material integration ranges</b>"))
        self.roi_units = QComboBox()
        self.roi_units.addItems(["keV", "pixel"])
        self.roi_units.currentTextChanged.connect(self._convert_roi_table)
        table_header.addWidget(self.roi_units)
        table_header.addStretch()
        left_layout.addLayout(table_header)
        self.roi_table = QTableWidget(0, 3)
        self.roi_table.setHorizontalHeaderLabels(["Material", "Low", "High"])
        left_layout.addWidget(self.roi_table, 1)
        table_buttons = QHBoxLayout()
        table_buttons.addWidget(self._button("Add material row", self._add_roi_row))
        table_buttons.addWidget(self._button("Remove selected row", self._remove_roi_rows))
        table_buttons.addWidget(self._button("Apply material ranges", self._apply_roi_table))
        table_buttons.addWidget(self._button("Save selection", self._save_selection))
        left_layout.addLayout(table_buttons)
        splitter.addWidget(left)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.addWidget(QLabel("<b>Predicted materials</b>"))
        sidebar_layout.addWidget(QLabel(
            "Predictions annotate the spectrum only. Material rows remain manual."
        ))
        self.prediction_list = QListWidget()
        sidebar_layout.addWidget(self.prediction_list, 1)
        sidebar_layout.addWidget(self._button("Predict materials", self._predict_materials))
        sidebar_layout.addWidget(self._button("Show library...", self._show_library))
        sidebar_layout.addStretch()
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        return page

    def _build_intensity_cut_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Material:"))
        self.cut_material_combo = QComboBox()
        self.cut_material_combo.currentTextChanged.connect(self._load_cut_material)
        controls.addWidget(self.cut_material_combo)
        controls.addWidget(QLabel("Minimum counts:"))
        self.cut_minimum = QDoubleSpinBox()
        self.cut_minimum.setRange(0.0, 1e15)
        self.cut_minimum.setDecimals(3)
        self.cut_minimum.setKeyboardTracking(False)
        self.cut_minimum.valueChanged.connect(self._preview_cut)
        controls.addWidget(self.cut_minimum)
        self.cut_summary = QLabel()
        controls.addWidget(self.cut_summary, 1)
        controls.addWidget(self._button("Save selection", self._save_selection))
        layout.addLayout(controls)
        self.cut_canvas = PlotCanvas()
        layout.addWidget(self.cut_canvas, 1)
        return page

    def _refresh_projects(self):
        current_workspace = workspace.get_workspace()
        self.workspace_label.setText(
            str(current_workspace) if current_workspace else "(not set - click Change...)"
        )
        roots = workspace.discover_projects()
        self.project_combo.clear()
        if not roots:
            self.project_combo.addItem("(no projects found)", None)
            return
        duplicate_names = {
            root.name for root in roots if sum(other.name == root.name for other in roots) > 1
        }
        for root in roots:
            label = root.name
            if root.name in duplicate_names or root.parent != current_workspace:
                label = f"{root.name} - {root.parent}"
            if not (root / "XRF" / "xrf_config.yaml").exists():
                label += " (XRF add-on not created)"
            self.project_combo.addItem(label, str(root))
        if self.project is not None:
            current = str(self.project.root)
            for index in range(self.project_combo.count()):
                if self.project_combo.itemData(index) == current:
                    self.project_combo.setCurrentIndex(index)
                    break

    def _use_launch_directory(self):
        workspace.set_workspace(workspace.get_launch_directory())
        self._refresh_projects()

    def _choose_workspace(self):
        start = str(workspace.get_workspace() or workspace.get_launch_directory())
        path = QFileDialog.getExistingDirectory(self, "Choose project workspace", start)
        if path:
            workspace.set_workspace(path)
            self._refresh_projects()

    def _new_project(self):
        current_workspace = workspace.get_workspace()
        if current_workspace is None:
            QMessageBox.information(self, "No workspace", "Choose a workspace first.")
            return
        name, accepted = QInputDialog.getText(self, "New project", "Project name:")
        name = name.strip()
        if not accepted or not name:
            return
        try:
            root = workspace.create_project(name, current_workspace)
            XRFProject.load(root).create_addon()
        except (FileExistsError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not create project", str(exc))
            return
        self.open_project(root, prompt_addon=False)

    def _open_selected(self):
        root = self.project_combo.currentData()
        if root:
            self.open_project(root)

    def _browse_project(self):
        start = str(workspace.get_workspace() or Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Open an xrd-app project (contains config.yaml)", start
        )
        if not path:
            return
        if not workspace.is_project(path):
            QMessageBox.warning(self, "Not a project", "That folder has no config.yaml.")
            return
        self.open_project(path)

    def _run_load_data(self, source, scan=None):
        if self.project is None:
            return
        from click.testing import CliRunner
        from .xrf_cli import main

        command = ["load-data", "--root", str(self.project.root), "--source", str(source)]
        if scan is not None:
            command.extend(["--scan", str(scan)])
        result = CliRunner().invoke(main, command, catch_exceptions=False)
        if result.exit_code:
            QMessageBox.critical(self, "Could not load XRF data", result.output)
            return
        self.data_status.setText(result.output.strip())
        self.project = XRFProject.load(self.project.root)
        self.open_project(self.project.root, prompt_addon=False)

    def _load_scan_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select a scan folder (containing ME7/)", str(self.project.root)
        )
        if path:
            self._run_load_data(path)

    def _load_scan_set(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select a folder containing Scan_*/ directories", str(self.project.root)
        )
        if path:
            scans = XRFProject.discover_scan_folders(path)
            if not scans:
                QMessageBox.information(
                    self, "No XRF scans found", "No Scan_*/ME7/scan_*.h5 data were found."
                )
                return
            summary = "\n".join(
                f"{item['name']}: {item['n_files']} files, {item['n_points']:,} points"
                for item in scans
            )
            answer = QMessageBox.question(
                self, "Register XRF scan set?",
                f"Register these {len(scans)} scans?\n\n{summary}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self._run_load_data(path)

    def _load_processed_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select canonical XRF selection", str(self.project.addon_root),
            "XRF selection (*.h5)",
        )
        if path:
            self._run_load_data(path)

    def _process_raw_me7(self):
        if self.project is None:
            return
        raw_scans = [
            name for name, record in (self.project.data.get("scans") or {}).items()
            if record.get("me7_dir")
        ]
        if not raw_scans:
            QMessageBox.information(self, "No raw ME7", "Load a raw ME7 folder first.")
            return
        scan, accepted = QInputDialog.getItem(
            self, "Process raw ME7", "Scan:", raw_scans, 0, False
        )
        if not accepted:
            return
        answer = QMessageBox.question(
            self, "Process raw ME7?",
            "This reads all registered ME7 spectra and may take several minutes. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        from click.testing import CliRunner
        from .xrf_cli import main
        result = CliRunner().invoke(
            main,
            ["process-raw", "--root", str(self.project.root), "--scan", scan],
            catch_exceptions=False,
        )
        if result.exit_code:
            QMessageBox.critical(self, "Raw XRF processing failed", result.output)
            return
        self.data_status.setText(result.output.strip())
        self.project = XRFProject.load(self.project.root)
        self.open_project(self.project.root, prompt_addon=False)

    def open_project(self, project_root, prompt_addon=True):
        project = XRFProject.load(project_root)
        if not project.xrd_exists():
            QMessageBox.warning(self, "Not a project", "That folder has no xrd-app config.yaml.")
            return False
        if not project.exists():
            if prompt_addon:
                answer = QMessageBox.question(
                    self,
                    "Create XRF add-on?",
                    f"{project.root} has no XRF add-on.\n\nCreate XRF/ now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer != QMessageBox.Yes:
                    return False
            project.create_addon()
        project.discover_processed()
        self.project = project
        workspace.set_last_project(project.root)
        self.setWindowTitle(
            f"XRF Analysis - {project.data.get('name', project.root.name)}"
        )
        self.main_tabs.setTabEnabled(1, True)
        self.load_box.setEnabled(True)
        self.calibration_box.setEnabled(True)
        calibration = project.data.get("calibration") or {}
        self.calibration_quadratic.setValue(float(calibration.get("quadratic_kev", 0.0)))
        self.calibration_linear.setValue(float(calibration.get("linear_kev", 0.01)))
        self.calibration_offset.setValue(float(calibration.get("offset_kev", 0.0)))
        self.project_summary.setText(
            f"name: {project.data.get('name')}\n"
            f"root: {project.root}\n"
            f"XRF:  {project.addon_root}"
        )
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
        scans = sorted((project.data.get("scans") or {}).keys())
        self.scan_combo.addItems(scans)
        active = project.data.get("active_scan")
        if active in scans:
            self.scan_combo.setCurrentText(active)
        self.scan_combo.blockSignals(False)
        self.cut_material_combo.clear()
        self.roi_table.setRowCount(0)
        self.prediction_list.clear()
        self.selection = None
        raw_scans = [
            name for name, record in (project.data.get("scans") or {}).items()
            if record.get("me7_dir")
        ]
        finalized = [
            name for name, record in (project.data.get("scans") or {}).items()
            if record.get("selection")
        ]
        self.data_status.setText(
            f"Raw ME7 scans: {', '.join(raw_scans) or 'none'}\n"
            f"Finalized selections: {', '.join(finalized) or 'none'}"
        )
        if finalized:
            if self.scan_combo.currentText() not in finalized:
                self.scan_combo.setCurrentText(finalized[0])
            self._load_scan(self.scan_combo.currentText())
            self.main_tabs.setCurrentIndex(1)
        else:
            self.status.setText(
                "Raw source registered but not processed." if raw_scans
                else "No XRF data loaded. Use Setup -> Load Data."
            )
        self._refresh_projects()
        return True

    def _load_scan(self, scan):
        if not scan:
            return
        record = (self.project.data.get("scans") or {}).get(scan, {})
        path = Path((record.get("selection") or {}).get("path", self.project.selection_path(scan)))
        try:
            self.selection = xrf_selection.load(path)
        except (KeyError, OSError, ValueError) as exc:
            self.selection = None
            self.status.setText(f"Could not load {path}: {exc}")
            return
        self.cut_material_combo.blockSignals(True)
        self.cut_material_combo.clear()
        self.cut_material_combo.addItems(sorted(self.selection["materials"]))
        self.cut_material_combo.blockSignals(False)
        info = xrf_selection.summary(self.selection)
        self.status.setText(
            f"{info['n_registered_frames']:,} registered frames | "
            f"{len(info['materials'])} materials | {path}"
        )
        self._refresh_roi_table()
        self._draw_spectrum()
        self._load_cut_material(self.cut_material_combo.currentText())

    def _save_calibration(self):
        if self.project is None:
            return
        calibration = {
            "quadratic_kev": self.calibration_quadratic.value(),
            "linear_kev": self.calibration_linear.value(),
            "offset_kev": self.calibration_offset.value(),
        }
        self.project.set_calibration(calibration)
        if self.selection is not None and self.selection.get("spectrum") is not None:
            self.selection["attrs"]["energy_calibration"] = calibration
            self.selection["spectrum"]["energy_kev"] = xrf_selection.pixel_to_kev(
                np.arange(self.selection["spectrum"]["summed_counts"].size), calibration
            )
            self._draw_spectrum()
        self.data_status.setText("Saved project XRF energy calibration.")

    def _refresh_roi_table(self):
        self.roi_table.setRowCount(0)
        if self.selection is None:
            return
        calibration = self.selection["attrs"].get("energy_calibration") or {}
        for name, material in sorted(self.selection["materials"].items()):
            bounds = material["attrs"].get("energy_range_kev")
            if bounds is None:
                bounds = xrf_selection.pixel_to_kev(
                    material["attrs"]["pixel_range"], calibration
                )
            row = self.roi_table.rowCount()
            self.roi_table.insertRow(row)
            for column, value in enumerate((name, *bounds)):
                self.roi_table.setItem(row, column, QTableWidgetItem(
                    str(value) if column == 0 else f"{float(value):.6g}"
                ))

    def _add_roi_row(self):
        row = self.roi_table.rowCount()
        self.roi_table.insertRow(row)
        defaults = ("New material", 1.0, 1.3) if self.roi_units.currentText() == "keV" \
            else ("New material", 0.0, 30.0)
        for column, value in enumerate(defaults):
            self.roi_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _remove_roi_rows(self):
        rows = sorted({index.row() for index in self.roi_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.roi_table.removeRow(row)
        self._draw_spectrum()

    def _convert_roi_table(self, units):
        if self.selection is None:
            return
        calibration = self.selection["attrs"].get("energy_calibration") or {}
        for row in range(self.roi_table.rowCount()):
            values = [float(self.roi_table.item(row, column).text()) for column in (1, 2)]
            converted = xrf_selection.pixel_to_kev(values, calibration) if units == "keV" \
                else xrf_selection.kev_to_pixel(values, calibration)
            for column, value in zip((1, 2), converted):
                self.roi_table.item(row, column).setText(f"{float(value):.6g}")
        self._draw_spectrum()

    def _roi_definitions(self):
        definitions = {}
        units = self.roi_units.currentText()
        for row in range(self.roi_table.rowCount()):
            name = self.roi_table.item(row, 0).text().strip()
            if not name:
                continue
            low, high = sorted(
                float(self.roi_table.item(row, column).text()) for column in (1, 2)
            )
            old = self.selection["materials"].get(name, {}).get("attrs", {})
            definitions[name] = {
                "display_name": name,
                "minimum_counts": old.get("minimum_counts"),
                "energy_range_kev" if units == "keV" else "pixel_range": [low, high],
            }
        return definitions

    def _apply_roi_table(self):
        if self.selection is None:
            return
        record = (self.project.data.get("scans") or {}).get(self.scan_combo.currentText(), {})
        me7_dir = record.get("me7_dir")
        if not me7_dir:
            QMessageBox.warning(
                self, "Raw ME7 required",
                "Applying manual material ranges requires the raw ME7 source for this scan.",
            )
            return
        try:
            self.selection = xrf_selection.integrate_material_rois(
                self.selection, me7_dir, self._roi_definitions()
            )
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not integrate material ranges", str(exc))
            return
        self.cut_material_combo.blockSignals(True)
        self.cut_material_combo.clear()
        self.cut_material_combo.addItems(sorted(self.selection["materials"]))
        self.cut_material_combo.blockSignals(False)
        self._refresh_roi_table()
        self._draw_spectrum()
        self._load_cut_material(self.cut_material_combo.currentText())

    def _predict_materials(self):
        self.prediction_list.clear()
        if self.selection is None or self.selection.get("spectrum") is None:
            return
        from scipy.signal import find_peaks
        spectrum = self.selection["spectrum"]
        counts = np.log10(np.maximum(spectrum["summed_counts"], 1.0))
        peaks, properties = find_peaks(counts, prominence=0.05, distance=15)
        order = np.argsort(properties["prominences"])[::-1][:20]
        self._predicted_lines = []
        seen = set()
        for index in order:
            observed = float(spectrum["energy_kev"][peaks[index]])
            line = xrf_core.nearest_emission_line(observed * 1000.0, tol_ev=200.0)
            if line is None:
                continue
            key = (line["element"], line["line"])
            if key in seen:
                continue
            seen.add(key)
            self._predicted_lines.append((observed, line))
            self.prediction_list.addItem(QListWidgetItem(
                f"{line['element']} {line['line']}: {line['energy_ev'] / 1000:.3f} keV "
                f"(observed {observed:.3f})"
            ))
        self._draw_spectrum()

    def _show_library(self):
        from .xrf_library_popup import XRFLineLibraryDialog
        energy_range = (0.0, 15.0)
        if self.selection is not None and self.selection.get("spectrum") is not None:
            energy = self.selection["spectrum"]["energy_kev"]
            energy_range = (max(0.0, float(np.nanmin(energy))), float(np.nanmax(energy)))
        XRFLineLibraryDialog(energy_range, self).exec_()

    def _draw_spectrum(self):
        self.spectrum_canvas.figure.clear()
        axis = self.spectrum_canvas.figure.add_subplot(111)
        if self.selection is None or self.selection.get("spectrum") is None:
            axis.text(0.5, 0.5, "No processed spectrum", ha="center", va="center")
            self.spectrum_canvas.draw_idle()
            return
        spectrum = self.selection["spectrum"]
        axis.plot(spectrum["energy_kev"], spectrum["summed_counts"], color="black", lw=0.8)
        calibration = self.selection["attrs"].get("energy_calibration") or {}
        for name, definition in self._roi_definitions().items():
            bounds = definition.get("energy_range_kev")
            if bounds is None:
                bounds = xrf_selection.pixel_to_kev(definition["pixel_range"], calibration)
            axis.axvspan(*bounds, alpha=0.2, label=name)
        for observed, line in getattr(self, "_predicted_lines", []):
            axis.axvline(observed, color="0.45", linestyle=":", linewidth=0.8)
            index = int(np.argmin(np.abs(spectrum["energy_kev"] - observed)))
            axis.annotate(
                f"{line['element']} {line['line']}",
                (observed, spectrum["summed_counts"][index]),
                xytext=(3, 5), textcoords="offset points", rotation=55, fontsize=8,
            )
        axis.set(
            title="Fully summed calibrated XRF spectrum",
            xlabel="Energy (keV)", ylabel="Summed counts", yscale="log",
        )
        if self.roi_table.rowCount():
            axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        self.spectrum_canvas.draw_idle()

    def _load_cut_material(self, name):
        if self.selection is None or name not in self.selection["materials"]:
            return
        minimum = self.selection["materials"][name]["attrs"].get("minimum_counts")
        self.cut_minimum.blockSignals(True)
        self.cut_minimum.setValue(0.0 if minimum is None else float(minimum))
        self.cut_minimum.blockSignals(False)
        self._preview_cut()

    def _preview_cut(self):
        if self.selection is None:
            return
        name = self.cut_material_combo.currentText()
        if name not in self.selection["materials"]:
            return
        material = self.selection["materials"][name]
        values = material["intensity"]
        finite = np.isfinite(values)
        minimum = self.cut_minimum.value()
        keep = finite & (values >= minimum)
        material["keep"] = keep
        material["attrs"]["minimum_counts"] = minimum
        self.cut_summary.setText(
            f"retained {keep.sum():,}/{finite.sum():,} | cut {100 * (~keep & finite).sum() / max(1, finite.sum()):.2f}%"
        )
        figure = self.cut_canvas.figure
        figure.clear()
        axes = figure.subplots(1, 2)
        finite_values = values[finite]
        bins = 150
        if finite_values.size and np.allclose(finite_values, np.round(finite_values)) \
                and finite_values.max() - finite_values.min() <= 500:
            bins = np.arange(np.floor(finite_values.min()), np.ceil(finite_values.max()) + 2) - 0.5
        axes[0].hist(finite_values, bins=bins, log=True, color="#2b6f9f", alpha=0.85)
        axes[0].axvline(minimum, color="red", linestyle=":", linewidth=2)
        axes[0].set(
            title=f"{name}: positions vs integrated counts",
            xlabel="Integrated XRF counts", ylabel="Positions",
        )
        x, y = self.selection["frames"]["x"], self.selection["frames"]["y"]
        spatial = np.isfinite(x) & np.isfinite(y)
        axes[1].scatter(x[spatial], y[spatial], c="white", s=3, edgecolors="none")
        selected = spatial & keep
        artist = axes[1].scatter(
            x[selected], y[selected], c=values[selected], s=3,
            cmap="viridis", edgecolors="none",
        )
        axes[1].set(
            title=f"{name}: retained XRF intensity map",
            xlabel="X position", ylabel="Corrected Y position",
        )
        axes[1].set_aspect("equal")
        if selected.any():
            figure.colorbar(artist, ax=axes[1], label="Integrated XRF counts")
        self.cut_canvas.draw_idle()

    def _save_selection(self):
        if self.selection is None or self.project is None:
            return
        self._preview_cut()
        scan = self.scan_combo.currentText()
        path = self.project.selection_path(scan)
        try:
            xrf_selection.save(path, self.selection)
            info = xrf_selection.summary(self.selection)
            self.project.register_selection(
                scan, path, info["materials"], info["selection_hash"]
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save selection", str(exc))
            return
        self.status.setText(f"Saved {path}")


def launch(project_root=None):
    """Launch XRF setup and analysis; project selection can happen in the GUI."""
    application = QApplication.instance() or QApplication(sys.argv)
    window = XRFAnalysisWindow(project_root)
    window.show()
    return application.exec_()
