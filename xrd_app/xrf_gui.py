"""Standalone read-only GUI for finalized XRF project selections."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from . import workspace
from .core import xrf as xrf_core
from .core import xrf_selection
from .gui.lifecycle import start_process, stop_process
from .xrf_project import XRFProject


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, scroll_zoom=False):
        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        if scroll_zoom:
            self.mpl_connect("scroll_event", self._zoom_at_cursor)

    def _zoom_at_cursor(self, event):
        axis = event.inaxes
        if axis is None or event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25

        def scaled_limits(limits, center, scale):
            if scale == "log":
                limits = np.log10(limits)
                center = np.log10(center)
                return 10 ** (center + (limits - center) * factor)
            return center + (np.asarray(limits) - center) * factor

        axis.set_xlim(scaled_limits(axis.get_xlim(), event.xdata, axis.get_xscale()))
        axis.set_ylim(scaled_limits(axis.get_ylim(), event.ydata, axis.get_yscale()))
        self.draw_idle()


class XRFAnalysisWindow(QMainWindow):
    """Choose an xrd-app project, then inspect its XRF add-on selections."""

    def __init__(self, project_root=None):
        super().__init__()
        self.project = None
        self.selection = None
        self._spectrum_process = None
        self._spectrum_output = []
        self._spectrum_pending = ""
        self._spectrum_scan = None
        self._spectrum_total = 0
        self._shared_roi_window = None
        self._link_process = None
        self._link_output = []
        self._link_pending = ""
        self._link_then_load = None
        self._selection_saved = False
        self._restoring_state = False
        self.setWindowTitle("XRF Analysis")
        self.resize(1350, 850)

        self.main_tabs = QTabWidget()
        self.setup_page = self._build_setup_page()
        self.analysis_page = self._build_analysis_page()
        self.main_tabs.addTab(self.setup_page, "Setup")
        self.main_tabs.addTab(self.analysis_page, "Analysis")
        self.main_tabs.setTabEnabled(1, False)
        self.main_tabs.currentChanged.connect(self._save_state)
        self.setCentralWidget(self.main_tabs)

        self._refresh_projects()
        project_root = project_root or workspace.get_last_xrf_project()
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
        offset_row = QHBoxLayout()
        offset_row.addWidget(self._button("Load position offset JSON...", self._load_position_offset))
        self.position_offset_status = QLabel("Position offset: not loaded")
        self.position_offset_status.setWordWrap(True)
        offset_row.addWidget(self.position_offset_status, 1)
        load_layout.addLayout(offset_row)
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
        self.analysis_tabs.addTab(self._build_label_spectrum_page(), "Complete ME7 Spectrum")
        self.analysis_tabs.addTab(self._build_intensity_cut_page(), "Intensity Cut")
        self.analysis_tabs.addTab(self._build_roi_shape_page(), "XRD ROI > Shape Check")
        self.analysis_tabs.currentChanged.connect(self._save_state)
        layout.addWidget(self.analysis_tabs, 1)
        self.scan_combo.currentTextChanged.connect(self._load_scan)
        self.scan_combo.currentTextChanged.connect(self._save_state)
        return page

    def _build_label_spectrum_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        process_row = QHBoxLayout()
        self.complete_spectrum_button = self._button(
            "Compute and save complete ME7 spectrum...", self._process_raw_me7
        )
        process_row.addWidget(self.complete_spectrum_button)
        self.spectrum_process_status = QLabel("Register raw ME7 in Setup to compute a spectrum.")
        self.spectrum_process_status.setWordWrap(True)
        process_row.addWidget(self.spectrum_process_status, 1)
        left_layout.addLayout(process_row)
        self.spectrum_progress = QProgressBar()
        self.spectrum_progress.setFormat("%v / %m frames (%p%)")
        self.spectrum_progress.setVisible(False)
        left_layout.addWidget(self.spectrum_progress)
        self.spectrum_canvas = PlotCanvas(scroll_zoom=True)
        self.spectrum_canvas.setToolTip(
            "Scroll over the graph to zoom at the cursor; use the toolbar to pan or reset."
        )
        self.spectrum_toolbar = NavigationToolbar2QT(self.spectrum_canvas, page)
        left_layout.addWidget(self.spectrum_toolbar)
        left_layout.addWidget(self.spectrum_canvas, 3)
        table_header = QHBoxLayout()
        table_header.addWidget(QLabel("<b>Manual material integration ranges</b>"))
        table_header.addStretch()
        left_layout.addLayout(table_header)
        self.roi_table = QTableWidget(0, 5)
        self.roi_table.setHorizontalHeaderLabels([
            "Material", "Pixel low", "Pixel high", "keV low", "keV high"
        ])
        self.roi_table.itemChanged.connect(self._sync_roi_table_range)
        left_layout.addWidget(self.roi_table, 1)
        table_buttons = QHBoxLayout()
        table_buttons.addWidget(self._button("Add material row", self._add_roi_row))
        table_buttons.addWidget(self._button("Remove selected row", self._remove_roi_rows))
        table_buttons.addWidget(self._button("Apply material ranges", self._apply_roi_table))
        self.spectrum_save_button = self._button("Save selection", self._save_selection)
        table_buttons.addWidget(self.spectrum_save_button)
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
        sidebar_layout.addWidget(self._button("Predict from library", self._predict_materials))
        sidebar_layout.addWidget(self._button("View library...", self._show_library))
        sidebar_layout.addStretch()
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        return page

    def _build_roi_shape_page(self):
        self.roi_shape_page = QWidget()
        layout = QVBoxLayout(self.roi_shape_page)
        controls = QHBoxLayout()
        self.roi_shape_load = self._button("Open ROI > Shape Check", self._open_roi_shape_check)
        controls.addWidget(self.roi_shape_load)
        self.roi_shape_status = QLabel(
            "Create the linked .h5 on Intensity Cut before opening this check."
        )
        self.roi_shape_status.setWordWrap(True)
        controls.addWidget(self.roi_shape_status, 1)
        layout.addLayout(controls)
        self.roi_shape_host = QWidget()
        self.roi_shape_host_layout = QVBoxLayout(self.roi_shape_host)
        self.roi_shape_host_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_shape_host_layout.addStretch()
        layout.addWidget(self.roi_shape_host, 1)
        return self.roi_shape_page

    def _build_intensity_cut_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Material:"))
        previous_material = QPushButton("<")
        previous_material.setToolTip("Previous material")
        previous_material.setMaximumWidth(40)
        previous_material.clicked.connect(lambda: self._step_cut_material(-1))
        controls.addWidget(previous_material)
        self.cut_material_combo = QComboBox()
        self.cut_material_combo.currentTextChanged.connect(self._load_cut_material)
        self.cut_material_combo.currentTextChanged.connect(self._save_state)
        controls.addWidget(self.cut_material_combo)
        next_material = QPushButton(">")
        next_material.setToolTip("Next material")
        next_material.setMaximumWidth(40)
        next_material.clicked.connect(lambda: self._step_cut_material(1))
        controls.addWidget(next_material)
        controls.addWidget(QLabel("Minimum counts:"))
        self.cut_minimum = QDoubleSpinBox()
        self.cut_minimum.setRange(0.0, 1e15)
        self.cut_minimum.setDecimals(3)
        self.cut_minimum.setKeyboardTracking(False)
        self.cut_minimum.valueChanged.connect(self._cut_changed)
        self.cut_minimum.valueChanged.connect(self._save_state)
        controls.addWidget(self.cut_minimum)
        self.cut_summary = QLabel()
        controls.addWidget(self.cut_summary, 1)
        self.cut_save_button = self._button("Save selection", self._save_selection)
        controls.addWidget(self.cut_save_button)
        self.create_linked_xrd_button = self._button(
            "Create Linked .h5 File", self._create_linked_xrd
        )
        controls.addWidget(self.create_linked_xrd_button)
        layout.addLayout(controls)
        self.cut_canvas = PlotCanvas()
        self.cut_canvas.mpl_connect("button_press_event", self._set_cut_from_histogram)
        self.cut_canvas.setToolTip("Click the histogram to set the minimum-count threshold.")
        layout.addWidget(self.cut_canvas, 1)
        self.link_dataset_status = QLabel(
            "Per-position XRF counts load automatically; Minimum counts updates both views."
        )
        self.link_dataset_status.setWordWrap(True)
        layout.addWidget(self.link_dataset_status)
        self.link_dataset_progress = QProgressBar()
        self.link_dataset_progress.setFormat("%v / %m frames (%p%)")
        self.link_dataset_progress.setVisible(False)
        layout.addWidget(self.link_dataset_progress)
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

    def _load_position_offset(self):
        if self.project is None:
            return
        start = (self.project.data.get("data_sources") or {}).get(
            "position_offset", str(self.project.root)
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select position offset JSON", str(start), "JSON (*.json)"
        )
        if not path:
            return
        try:
            xrf_core.load_position_offsets(path)
            path = self.project.set_position_offset(path)
        except (KeyError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid position offset", str(exc))
            return
        self.position_offset_status.setText(f"Position offset: {Path(path).resolve()}")

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
        self.complete_spectrum_button.setText("Loading complete ME7 spectrum...")
        self.complete_spectrum_button.setEnabled(False)
        self.spectrum_progress.setRange(0, 0)
        self.spectrum_progress.setFormat("Counting ME7 spectra...")
        self.spectrum_progress.setVisible(True)
        self.spectrum_process_status.setText(
            f"{scan}: counting ME7 spectra before summing. XRD is not read in this step."
        )
        self._spectrum_scan = scan
        self._spectrum_output = []
        self._spectrum_pending = ""
        self._spectrum_process = QProcess(self)
        self._spectrum_process.setProcessChannelMode(QProcess.MergedChannels)
        self._spectrum_process.readyReadStandardOutput.connect(self._on_spectrum_output)
        self._spectrum_process.finished.connect(self._on_spectrum_finished)
        self._spectrum_process.errorOccurred.connect(self._on_spectrum_error)
        start_process(
            self._spectrum_process, sys.executable,
            ["-m", "xrd_app.xrf_cli", "process-raw", "--root",
             str(self.project.root), "--scan", scan],
        )

    def _on_spectrum_output(self):
        data = bytes(self._spectrum_process.readAllStandardOutput()).decode(
            "utf-8", "replace"
        )
        self._spectrum_output.append(data)
        text = self._spectrum_pending + data
        lines = text.split("\n")
        self._spectrum_pending = lines.pop()
        for line in lines:
            match = re.search(r"PROGRESS\s+(\d+)/(\d+)\s+files", line)
            if match:
                processed, total = map(int, match.groups())
                self._spectrum_total = total
                self.spectrum_progress.setRange(0, total)
                self.spectrum_progress.setValue(processed)
                self.spectrum_progress.setFormat("%v / %m files (%p%)")
                self.spectrum_process_status.setText(
                    f"{self._spectrum_scan}: reading and summing ME7 spectra, "
                    f"{processed:,} / {total:,} files"
                )

    def _on_spectrum_error(self, error):
        if error == QProcess.FailedToStart:
            self._finish_spectrum_controls(hide_progress=True)
            status = f"Processing failed to start: {self._spectrum_process.errorString()}"
            self.spectrum_process_status.setText(status)
            QMessageBox.critical(self, "Raw XRF processing failed", status)

    def _on_spectrum_finished(self, code, _status):
        scan = self._spectrum_scan
        output = "".join(self._spectrum_output).strip()
        self._finish_spectrum_controls(hide_progress=code != 0)
        if code != 0:
            status = output or f"Processing failed with exit code {code}."
            self.spectrum_process_status.setText(status)
            QMessageBox.critical(self, "Raw XRF processing failed", status)
            return
        self.project = XRFProject.load(self.project.root)
        self.open_project(self.project.root, prompt_addon=False)
        self.scan_combo.setCurrentText(scan)
        self._load_scan(scan)
        self.main_tabs.setCurrentIndex(1)
        self.analysis_tabs.setCurrentIndex(0)
        total_files = self._spectrum_total
        self.spectrum_progress.setRange(0, total_files)
        self.spectrum_progress.setValue(total_files)
        self.spectrum_progress.setFormat("Complete: %v / %m files (%p%)")
        self.spectrum_process_status.setText(
            f"Complete ME7 spectrum saved for {scan}: {total_files:,} / "
            f"{total_files:,} files"
        )

    def _finish_spectrum_controls(self, hide_progress=False):
        self.complete_spectrum_button.setText("Compute and save complete ME7 spectrum...")
        self.complete_spectrum_button.setEnabled(True)
        if hide_progress:
            self.spectrum_progress.setVisible(False)

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
        state = self._load_state()
        self._restoring_state = True
        workspace.set_last_project(project.root)
        workspace.set_last_xrf_project(project.root)
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
        active = state.get("active_scan") or project.data.get("active_scan")
        if active in scans:
            self.scan_combo.setCurrentText(active)
        self.scan_combo.blockSignals(False)
        self.cut_material_combo.clear()
        self.roi_table.setRowCount(0)
        self.prediction_list.clear()
        self.selection = None
        self._set_selection_saved(False)
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
        position_offset = project.position_offset_path()
        self.position_offset_status.setText(
            f"Position offset: {position_offset if position_offset.exists() else 'not loaded'}"
        )
        self.spectrum_process_status.setText(
            "Complete spectrum saved; recompute if raw inputs or calibration changed."
            if finalized else "Ready to compute complete ME7 spectrum."
            if raw_scans else "Register raw ME7 in Setup to compute a spectrum."
        )
        if finalized:
            if self.scan_combo.currentText() not in finalized:
                self.scan_combo.setCurrentText(finalized[0])
            self._load_scan(self.scan_combo.currentText())
            material = state.get("material")
            if material and self.cut_material_combo.findText(material) >= 0:
                self.cut_material_combo.setCurrentText(material)
            minimum = state.get("minimum_counts")
            if minimum is not None:
                self.cut_minimum.setValue(float(minimum))
            main_tab = int(state.get("main_tab", 1))
            self.main_tabs.setCurrentIndex(main_tab if 0 <= main_tab < self.main_tabs.count() else 1)
            analysis_tab = int(state.get("analysis_tab", 0))
            if 0 <= analysis_tab < self.analysis_tabs.count():
                self.analysis_tabs.setCurrentIndex(analysis_tab)
        else:
            self.status.setText(
                "Raw source registered but not processed." if raw_scans
                else "No XRF data loaded. Use Setup -> Load Data."
            )
        geometry = state.get("geometry") or {}
        width, height = geometry.get("width"), geometry.get("height")
        if width and height:
            self.resize(max(640, int(width)), max(480, int(height)))
        self._restoring_state = False
        self._refresh_projects()
        return True

    def _state_path(self):
        return self.project.path("metadata_dir") / "gui_state.json"

    def _load_state(self):
        if self.project is None:
            return {}
        path = self._state_path()
        try:
            with path.open() as stream:
                return json.load(stream) or {}
        except (OSError, ValueError):
            return {}

    def _save_state(self, *_args):
        if self.project is None or self._restoring_state:
            return
        state = {
            "active_scan": self.scan_combo.currentText(),
            "main_tab": self.main_tabs.currentIndex(),
            "analysis_tab": self.analysis_tabs.currentIndex(),
            "material": self.cut_material_combo.currentText(),
            "minimum_counts": self.cut_minimum.value(),
            "geometry": {"width": self.width(), "height": self.height()},
        }
        try:
            path = self._state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            with temporary.open("w") as stream:
                json.dump(state, stream, indent=2)
            temporary.replace(path)
        except OSError:
            pass

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
        linked = bool(self.selection["attrs"].get("linked_dataset"))
        self.link_dataset_status.setText(
            "Per-position XRF counts ready; press Create Linked .h5 File to finalize this cut."
            if linked else "Create Linked .h5 File will build the XRF/XRD frame registration."
        )
        self.roi_shape_load.setEnabled(False)
        self.roi_shape_status.setText(
            "Create the linked .h5 on Intensity Cut before opening this check."
        )
        active_calibration = self.selection["attrs"].get("energy_calibration")
        if active_calibration and "quadratic_kev" in active_calibration:
            self.calibration_quadratic.setValue(float(active_calibration["quadratic_kev"]))
            self.calibration_linear.setValue(float(active_calibration["linear_kev"]))
            self.calibration_offset.setValue(float(active_calibration["offset_kev"]))
        info = xrf_selection.summary(self.selection)
        self.status.setText(
            f"{info['n_registered_frames']:,} registered frames | "
            f"{len(info['materials'])} materials | {path}"
        )
        self._refresh_roi_table()
        self._set_selection_saved(True)
        self._draw_spectrum()
        self._load_cut_material(self.cut_material_combo.currentText())

    def _set_selection_saved(self, saved):
        self._selection_saved = saved
        style = "background-color: #2e7d32; color: white; font-weight: bold;" if saved else ""
        text = "Selection saved" if saved else "Save selection"
        for button in (self.spectrum_save_button, self.cut_save_button):
            button.setText(text)
            button.setStyleSheet(style)

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
            self._refresh_roi_table()
            self._set_selection_saved(False)
            self._draw_spectrum()
        self.data_status.setText("Saved project XRF energy calibration.")

    def _refresh_roi_table(self):
        self.roi_table.blockSignals(True)
        self.roi_table.setRowCount(0)
        if self.selection is not None:
            calibration = self.selection["attrs"].get("energy_calibration") or {}
            for name, material in sorted(self.selection["materials"].items()):
                pixels = material["attrs"].get("pixel_range")
                energies = material["attrs"].get("energy_range_kev")
                if pixels is None and energies is not None:
                    pixels = xrf_selection.kev_to_pixel(energies, calibration)
                if pixels is None:
                    pixels = [0, 1]
                if energies is None:
                    energies = xrf_selection.pixel_to_kev(pixels, calibration)
                row = self.roi_table.rowCount()
                self.roi_table.insertRow(row)
                values = (name, *pixels, *energies)
                for column, value in enumerate(values):
                    self.roi_table.setItem(row, column, QTableWidgetItem(
                        str(value) if column == 0 else f"{float(value):.6g}"
                    ))
        self.roi_table.blockSignals(False)

    def _add_roi_row(self):
        calibration = self.selection["attrs"].get("energy_calibration") or {}
        pixels = [0.0, 30.0]
        energies = xrf_selection.pixel_to_kev(pixels, calibration)
        row = self.roi_table.rowCount()
        self.roi_table.blockSignals(True)
        self.roi_table.insertRow(row)
        for column, value in enumerate(("New material", *pixels, *energies)):
            self.roi_table.setItem(row, column, QTableWidgetItem(
                str(value) if column == 0 else f"{float(value):.6g}"
            ))
        self.roi_table.blockSignals(False)
        self._set_selection_saved(False)
        self._draw_spectrum()

    def _remove_roi_rows(self):
        rows = sorted({index.row() for index in self.roi_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.roi_table.removeRow(row)
        if rows:
            self._set_selection_saved(False)
        self._draw_spectrum()

    def _sync_roi_table_range(self, item):
        if self.selection is None:
            return
        self._set_selection_saved(False)
        if item.column() == 0:
            self._draw_spectrum()
            return
        source_columns = (1, 2) if item.column() in (1, 2) else (3, 4)
        target_columns = (3, 4) if source_columns == (1, 2) else (1, 2)
        try:
            values = [
                float(self.roi_table.item(item.row(), column).text())
                for column in source_columns
            ]
        except (AttributeError, ValueError):
            return
        calibration = self.selection["attrs"].get("energy_calibration") or {}
        converted = (
            xrf_selection.pixel_to_kev(values, calibration)
            if source_columns == (1, 2)
            else xrf_selection.kev_to_pixel(values, calibration)
        )
        self.roi_table.blockSignals(True)
        for column, value in zip(target_columns, converted):
            target = self.roi_table.item(item.row(), column)
            if target is None:
                target = QTableWidgetItem()
                self.roi_table.setItem(item.row(), column, target)
            target.setText(f"{float(value):.6g}")
        self.roi_table.blockSignals(False)
        self._draw_spectrum()

    def _roi_definitions(self):
        definitions = {}
        for row in range(self.roi_table.rowCount()):
            name = self.roi_table.item(row, 0).text().strip()
            if not name:
                continue
            pixel_low, pixel_high = sorted(
                float(self.roi_table.item(row, column).text()) for column in (1, 2)
            )
            kev_low, kev_high = sorted(
                float(self.roi_table.item(row, column).text()) for column in (3, 4)
            )
            old = self.selection["materials"].get(name, {}).get("attrs", {})
            definitions[name] = {
                "display_name": name,
                "minimum_counts": old.get("minimum_counts"),
                "pixel_range": [pixel_low, pixel_high],
                "energy_range_kev": [kev_low, kev_high],
            }
        return definitions

    def _apply_roi_table(self):
        if self.selection is None:
            return
        definitions = self._roi_definitions()
        size = self.selection["frames"]["global_frame_index"].size
        self.selection["materials"] = {
            name: {
                "intensity": np.full(size, np.nan),
                "keep": np.ones(size, dtype=bool),
                "attrs": attrs,
            }
            for name, attrs in definitions.items()
        }
        self.selection["attrs"]["linked_dataset"] = False
        self.selection = xrf_selection.validate(self.selection)
        self.cut_material_combo.blockSignals(True)
        self.cut_material_combo.clear()
        self.cut_material_combo.addItems(sorted(self.selection["materials"]))
        self.cut_material_combo.blockSignals(False)
        self.link_dataset_status.setText(
            "Material ranges changed; Create Linked .h5 File will rebuild the link."
        )
        self.roi_shape_load.setEnabled(False)
        self.roi_shape_status.setText(
            "Create the linked .h5 on Intensity Cut before opening this check."
        )
        self._set_selection_saved(False)
        self._draw_spectrum()

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
            color = "#2e7d32" if self._selection_saved else "#1976d2"
            axis.axvspan(*bounds, color=color, alpha=0.28, label=name)
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

    def _create_linked_dataset(self):
        if self.selection is None or self._link_process is not None:
            return
        material = self.cut_material_combo.currentText()
        if material in self.selection["materials"]:
            self.selection["materials"][material]["attrs"]["minimum_counts"] = (
                self.cut_minimum.value()
            )
        scan = self.scan_combo.currentText()
        definitions_path = self.project.cache_scan_dir(scan) / "material_ranges.json"
        definitions_path.parent.mkdir(parents=True, exist_ok=True)
        with definitions_path.open("w") as stream:
            json.dump(self._roi_definitions(), stream, indent=2)
        self.link_dataset_progress.setRange(0, 0)
        self.link_dataset_progress.setFormat("Matching ME7/XRD frames...")
        self.link_dataset_progress.setVisible(True)
        self.link_dataset_status.setText(
            f"{scan}: matching ME7 and XRD acquisition frames."
        )
        self._link_output = []
        self._link_pending = ""
        self._link_process = QProcess(self)
        self._link_process.setProcessChannelMode(QProcess.MergedChannels)
        self._link_process.readyReadStandardOutput.connect(self._on_link_output)
        self._link_process.finished.connect(self._on_link_finished)
        start_process(
            self._link_process, sys.executable,
            ["-m", "xrd_app.xrf_cli", "link-dataset", "--root",
             str(self.project.root), "--scan", scan, "--definitions",
             str(definitions_path)],
        )

    def _on_link_output(self):
        data = bytes(self._link_process.readAllStandardOutput()).decode("utf-8", "replace")
        self._link_output.append(data)
        lines = (self._link_pending + data).split("\n")
        self._link_pending = lines.pop()
        for line in lines:
            match = re.search(r"PROGRESS\s+(\d+)/(\d+)\s+frames", line)
            if match:
                done, total = map(int, match.groups())
                self.link_dataset_progress.setRange(0, total)
                self.link_dataset_progress.setValue(done)
                self.link_dataset_progress.setFormat("%v / %m frames (%p%)")
                self.link_dataset_status.setText(
                    f"Integrating material ranges: {done:,} / {total:,} frames"
                )

    def _on_link_finished(self, code, _status):
        output = "".join(self._link_output).strip()
        self._link_process = None
        if code != 0:
            self._link_then_load = None
            self.create_linked_xrd_button.setEnabled(True)
            self.create_linked_xrd_button.setText("Create Linked .h5 File")
            self.link_dataset_progress.setVisible(False)
            self.link_dataset_status.setText(output or f"XRF/XRD linking failed (exit {code}).")
            return
        scan = self.scan_combo.currentText()
        self._load_scan(scan)
        total = self.selection["frames"]["global_frame_index"].size
        self.link_dataset_progress.setRange(0, total)
        self.link_dataset_progress.setValue(total)
        self.link_dataset_progress.setFormat("Complete: %v / %m frames (%p%)")
        self.link_dataset_status.setText(
            f"Per-position XRF counts ready for {total:,} matched frames."
        )
        material = self._link_then_load
        self._link_then_load = None
        if material:
            self.cut_material_combo.setCurrentText(material)
        self._finalize_linked_h5()
        self._preview_cut()

    def _create_linked_xrd(self):
        if self.selection is None or self._link_process is not None:
            return
        material = self.cut_material_combo.currentText()
        if material not in self.selection["materials"]:
            return
        self._save_selection()
        if not self.selection["attrs"].get("linked_dataset"):
            self._link_then_load = material
            self.create_linked_xrd_button.setEnabled(False)
            self.create_linked_xrd_button.setText("Creating Linked .h5 File...")
            self._create_linked_dataset()
            return
        self._finalize_linked_h5()

    def _finalize_linked_h5(self):
        material = self.cut_material_combo.currentText()
        try:
            result = xrf_selection.activate_xrd_roi_project(
                self.project.root, self.scan_combo.currentText(), material
            )
        except (KeyError, OSError, ValueError) as exc:
            self.create_linked_xrd_button.setEnabled(True)
            self.create_linked_xrd_button.setText("Create Linked .h5 File")
            self.link_dataset_status.setText(f"Could not create linked .h5: {exc}")
            return
        self.create_linked_xrd_button.setEnabled(False)
        self.create_linked_xrd_button.setText("Created Linked .h5")
        self.roi_shape_load.setEnabled(True)
        self.link_dataset_status.setText(
            f"XRF complete: linked .h5 created for {result['selected_frames']:,} retained "
            "frames; no XRD images were read."
        )
        self.roi_shape_status.setText(
            "Linked .h5 ready. Open ROI > Shape only to confirm the handoff."
        )

    def _open_roi_shape_check(self):
        if self.selection is None or not self.selection["attrs"].get("linked_dataset"):
            self.roi_shape_status.setText(
                "Create the linked .h5 on Intensity Cut before opening this check."
            )
            return
        if self.create_linked_xrd_button.text() != "Created Linked .h5":
            self.roi_shape_status.setText(
                "Press Create Linked .h5 File on Intensity Cut to finalize this cut first."
            )
            return
        if self._shared_roi_window is not None:
            self._shared_roi_window.close()
            self._shared_roi_window.deleteLater()
        while self.roi_shape_host_layout.count():
            item = self.roi_shape_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        from .gui.roi_shape import ROIShapeWindow
        self._shared_roi_window = ROIShapeWindow(
            self.project.root, scan=self.scan_combo.currentText(), bin_size=3, embedded=False
        )
        self.roi_shape_host_layout.addWidget(self._shared_roi_window)
        self.roi_shape_status.setText(
            "Standard xrd-app ROI > Shape opened from the linked .h5 cut."
        )

    def _step_cut_material(self, amount):
        count = self.cut_material_combo.count()
        if count:
            self.cut_material_combo.setCurrentIndex(
                (self.cut_material_combo.currentIndex() + amount) % count
            )

    def _set_cut_from_histogram(self, event):
        if (event.button != 1 or event.xdata is None
                or not self.cut_canvas.figure.axes
                or event.inaxes is not self.cut_canvas.figure.axes[0]):
            return
        self.cut_minimum.setValue(max(0.0, float(event.xdata)))

    def _load_cut_material(self, name):
        if self.selection is None or name not in self.selection["materials"]:
            return
        self._mark_link_pending()
        minimum = self.selection["materials"][name]["attrs"].get("minimum_counts")
        self.cut_minimum.blockSignals(True)
        self.cut_minimum.setValue(0.0 if minimum is None else float(minimum))
        self.cut_minimum.blockSignals(False)
        self._preview_cut()

    def _mark_link_pending(self):
        if self.create_linked_xrd_button.text() == "Created Linked .h5":
            self.create_linked_xrd_button.setEnabled(True)
            self.create_linked_xrd_button.setText("Create Linked .h5 File")
            self.roi_shape_load.setEnabled(False)
            self.roi_shape_status.setText(
                "The active cut changed; create the linked .h5 again before checking it."
            )

    def _cut_changed(self):
        self._set_selection_saved(False)
        self._mark_link_pending()
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
        material["attrs"]["minimum_counts"] = minimum
        if not finite.any():
            self.cut_summary.setText("Loading per-position XRF counts...")
            self.cut_canvas.figure.clear()
            axis = self.cut_canvas.figure.add_subplot(111)
            axis.text(0.5, 0.5, "Loading XRF histogram...", ha="center", va="center")
            self.cut_canvas.draw_idle()
            return
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
        axes[0].hist(finite_values, bins=bins, log=True, color="tab:blue", alpha=0.85)
        axes[0].axvline(
            minimum, color="tab:red", linestyle=":", linewidth=2,
            label=f"minimum = {minimum:g}",
        )
        axes[0].legend()
        axes[0].set(
            title=(f"{name}: scan positions vs integrated counts\n"
                   f"cut = {100 * (~keep & finite).sum() / max(1, finite.sum()):.2f}%"),
            xlabel="Integrated XRF counts per scan position", ylabel="Scan positions",
        )
        x, y = self.selection["frames"]["x"], self.selection["frames"]["y"]
        spatial = np.isfinite(x) & np.isfinite(y)
        axes[1].scatter(x[spatial], y[spatial], c="white", s=3, edgecolors="none")
        selected = spatial & keep
        artist = axes[1].scatter(
            x[selected], y[selected], c=values[selected], s=3,
            cmap="viridis", edgecolors="none",
        )
        axes[1].set_facecolor("white")
        axes[1].set(
            title=f"{name}: integrated XRF counts in real space",
            xlabel="X position (um)", ylabel="Corrected Y position (um)",
        )
        axes[1].set_aspect("equal")
        if selected.any():
            figure.colorbar(artist, ax=axes[1], label="Integrated XRF counts")
        self.cut_canvas.draw_idle()

    def closeEvent(self, event):  # noqa: N802 (Qt signature)
        self._save_state()
        if self._shared_roi_window is not None:
            self._shared_roi_window.close()
            self._shared_roi_window = None
        stop_process(self._spectrum_process)
        stop_process(self._link_process)
        self._spectrum_process = None
        self._link_process = None
        super().closeEvent(event)

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
        self._set_selection_saved(True)
        self._draw_spectrum()
        self.status.setText(f"Selection saved: {path}")


def launch(project_root=None):
    """Launch XRF setup and analysis; project selection can happen in the GUI."""
    application = QApplication.instance() or QApplication(sys.argv)
    window = XRFAnalysisWindow(project_root)
    window.show()
    return application.exec_()
