"""Report tab: configure and run the headless PDF report engine."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout,
    QWidget,
)

from ..config import DataManager
from ..core import catalogs
from ._console import JobConsole
from ._embed import existing_bins, placeholder

TAB_META = {
    "title": "Report",
    "order": 100,
    "takes_bin_size": False,
    "scan_dependent": False,
    "general": (
        "Create a landscape PDF slide deck from selected scan/bin/catalog targets. "
        "Preview processes only the first selected scan. Missing artifacts produce "
        "an unavailable page and generation continues; ROI maps can be calculated "
        "on demand from the selected catalog's largest features."
    ),
}


class _TargetRow(QWidget):
    def __init__(self, dm, scan, selected=False, active_bin=3):
        super().__init__()
        self.dm = dm
        self.scan = scan
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox(scan)
        self.enabled.setChecked(selected)
        self.enabled.setMinimumWidth(130)
        layout.addWidget(self.enabled)
        self.bin = QComboBox()
        bins = set(existing_bins(dm, scan))
        bins.update(catalogs.available_bins(dm.labels_dir(scan)))
        for size in sorted(bins):
            self.bin.addItem(f"{size}x{size}", size)
        index = self.bin.findData(active_bin)
        self.bin.setCurrentIndex(index if index >= 0 else 0)
        self.bin.currentIndexChanged.connect(self._populate_catalogs)
        layout.addWidget(self.bin)
        self.catalog = QComboBox()
        self.catalog.setMinimumWidth(420)
        layout.addWidget(self.catalog, 1)
        self._populate_catalogs()

    def _populate_catalogs(self):
        previous = self.catalog.currentData()
        self.catalog.clear()
        size = self.bin.currentData()
        for path in catalogs.feature_sources(self.dm.labels_dir(self.scan), size):
            self.catalog.addItem(path.name, str(path))
        index = self.catalog.findData(previous)
        if index >= 0:
            self.catalog.setCurrentIndex(index)
        if not self.catalog.count():
            self.catalog.addItem("(default / unavailable)", None)

    def target_argument(self):
        value = f"{self.scan}:{self.bin.currentData()}"
        catalog = self.catalog.currentData()
        return f"{value}:{catalog}" if catalog else value


class ReportTab(QWidget):
    def __init__(self, project_root, scan=None, bin_size=3):
        super().__init__()
        self.project_root = str(Path(project_root).resolve())
        self.dm = DataManager(project_root)
        self._rows = []
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        intro = QLabel(
            "<b>PDF report deck</b><br>Select each scan's bin size and feature catalog, "
            "then choose the Sarah Data-style pages to include.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        targets = QGroupBox("Scans and feature maps")
        targets_layout = QVBoxLayout(targets)
        self.target_list = QListWidget()
        self.target_list.setMinimumHeight(150)
        targets_layout.addWidget(self.target_list)
        for name in self.dm.discover_scans(selected_only=True):
            row = _TargetRow(self.dm, name, selected=(name == scan), active_bin=bin_size)
            item = QListWidgetItem()
            item.setSizeHint(row.sizeHint())
            self.target_list.addItem(item)
            self.target_list.setItemWidget(item, row)
            self._rows.append(row)
        root.addWidget(targets)

        pages = QGroupBox("Pages")
        grid = QGridLayout(pages)
        self.summed = QCheckBox("Show scan summed detector images")
        self.all_reflections = QCheckBox(
            "Show all reflections: summed detector image + device feature map")
        self.by_reflection = QCheckBox(
            "Show features by reflection: detector image + device feature map")
        self.top_features = QCheckBox("Show top features by size (intensity map)")
        for checkbox in (self.summed, self.all_reflections, self.by_reflection,
                         self.top_features):
            checkbox.setChecked(True)
        self.source_images = QCheckBox("Include source-bin detector image for top features")
        self.source_images.setChecked(True)
        self.rois = QCheckBox("Create or show ROI images for selected peaks")
        self.calculate_rois = QCheckBox("Calculate missing ROI maps on demand (exact)")
        self.territory = QCheckBox("Print territorial map for each reflection")
        grid.addWidget(self.summed, 0, 0, 1, 2)
        grid.addWidget(self.all_reflections, 1, 0, 1, 2)
        grid.addWidget(self.by_reflection, 2, 0, 1, 2)
        grid.addWidget(self.top_features, 3, 0)
        self.top_count = QSpinBox()
        self.top_count.setRange(1, 50)
        self.top_count.setValue(5)
        self.top_count.setPrefix("Top ")
        self.top_count.setSuffix(" / reflection")
        grid.addWidget(self.top_count, 3, 1)
        self.override = QCheckBox("Override five-feature maximum")
        grid.addWidget(self.override, 4, 1)
        grid.addWidget(self.source_images, 4, 0)
        grid.addWidget(self.rois, 5, 0)
        grid.addWidget(self.calculate_rois, 5, 1)
        grid.addWidget(self.territory, 6, 0, 1, 2)
        self.calculate_rois.setEnabled(False)
        self.rois.toggled.connect(self.calculate_rois.setEnabled)
        root.addWidget(pages)

        actions = QHBoxLayout()
        self.preview_button = QPushButton("Preview first scan")
        self.generate_button = QPushButton("Generate PDF")
        self.preview_button.clicked.connect(lambda: self._run(preview=True))
        self.generate_button.clicked.connect(lambda: self._run(preview=False))
        actions.addWidget(self.preview_button)
        actions.addWidget(self.generate_button)
        actions.addStretch()
        root.addLayout(actions)

        self.console = JobConsole()
        root.addWidget(self.console, 1)

    def update_context(self, scan=None, bin_size=None):
        for row in self._rows:
            if row.scan == scan and not any(candidate.enabled.isChecked() for candidate in self._rows):
                row.enabled.setChecked(True)

    def _arguments(self, output, preview):
        arguments = ["report", "--root", self.project_root, "--output", output]
        selected = [row for row in self._rows if row.enabled.isChecked()]
        if not selected:
            raise ValueError("Select at least one scan")
        for row in selected:
            arguments.extend(["--target", row.target_argument()])
        if preview:
            arguments.append("--preview")
        flags = [
            (self.summed, "--summed-images", "--no-summed-images"),
            (self.all_reflections, "--all-reflections", "--no-all-reflections"),
            (self.by_reflection, "--features-by-reflection", "--no-features-by-reflection"),
            (self.top_features, "--top-features", "--no-top-features"),
            (self.source_images, "--source-images", "--no-source-images"),
            (self.rois, "--roi-images", "--no-roi-images"),
            (self.territory, "--territory-maps", "--no-territory-maps"),
        ]
        for checkbox, yes, no in flags:
            arguments.append(yes if checkbox.isChecked() else no)
        arguments.extend(["--top-count", str(self.top_count.value())])
        if self.override.isChecked():
            arguments.append("--allow-more-than-five")
        if self.calculate_rois.isChecked():
            arguments.append("--calculate-rois")
        return arguments

    def _run(self, preview=False):
        if self.top_count.value() > 5 and not self.override.isChecked():
            self.console.log.setPlainText(
                "Top features is capped at five per reflection. Enable the override to continue.")
            return
        default = Path(self.project_root) / "Figures" / (
            "report_preview.pdf" if preview else "xrd_report.pdf")
        output, _ = QFileDialog.getSaveFileName(
            self, "Save report PDF", str(default), "PDF files (*.pdf)")
        if not output:
            return
        try:
            arguments = self._arguments(output, preview)
        except ValueError as error:
            self.console.log.setPlainText(str(error))
            return
        self.console.run(arguments, cwd=self.project_root,
                         header="Preview uses only the first selected scan.\n" if preview else None)


def make_tab(project_root=".", scan=None, bin_size=3):
    try:
        return ReportTab(project_root, scan=scan, bin_size=bin_size)
    except Exception as error:
        return placeholder("Could not load Report.", f"{type(error).__name__}: {error}")
