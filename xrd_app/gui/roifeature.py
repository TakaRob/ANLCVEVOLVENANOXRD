"""Standalone first-use launcher for manual ROI feature discovery."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)

from .. import workspace
from ..config import DataManager
from ..core import io, roi_map
from .roi_shape import ROIShapeWindow
from .viewer import DetectorView


class _RawSumWorker(QThread):
    progress = pyqtSignal(int, int)
    complete = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, scan_folder):
        super().__init__()
        self.scan_folder = Path(scan_folder)

    def run(self):
        try:
            info = io.discover_scans(self.scan_folder, deep=False)
            if len(info) != 1:
                raise ValueError(f"Expected one scan; found {len(info)}")
            scan = info[0]
            files = io.scan_h5_files(scan["frames_dir"],
                                     DataManager.scan_number_of(scan["name"]) or 0)
            if not files:
                files = sorted(Path(scan["frames_dir"]).glob("*.h5")) + \
                        sorted(Path(scan["frames_dir"]).glob("*.hdf5"))
            image = None
            total = len(files)
            for i, path in enumerate(files):
                with h5py.File(path, "r") as handle:
                    dataset = handle[io.H5_DATASET]
                    for frame in dataset:
                        values = np.asarray(frame, dtype=np.float64)
                        image = values if image is None else image + values
                self.progress.emit(i + 1, total)
            if image is None:
                raise FileNotFoundError("No readable XRD frames found")
            image = np.clip(image, 0, 1e9)
            self.complete.emit(image)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CreateProjectDialog(QDialog):
    """Save-time project details for a standalone raw scan."""

    def __init__(self, scan_name, scan_folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create project to save ROI features")
        self.resize(720, 260)
        form = QFormLayout(self)
        intro = QLabel(
            "Shape catalogs need a project, detector 2θ calibration, and a spatial "
            "grid. The raw scan remains where it is; the project stores metadata, "
            "bins, and saved features.")
        intro.setWordWrap(True)
        form.addRow(intro)
        self.name = QLineEdit(f"{scan_name}-Project")
        form.addRow("Project name", self.name)
        self.parent_dir = QLineEdit(str(workspace.get_workspace() or Path(scan_folder).parent))
        form.addRow("Project parent", self._browse_row(self.parent_dir, True))
        self.tth = QLineEdit()
        form.addRow("tth.tiff (required)", self._browse_row(self.tth, False, "TIFF (*.tif *.tiff)"))
        self.positions = QLineEdit()
        form.addRow("Positions (optional*)", self._browse_row(
            self.positions, False, "Positions (*.csv *.h5 *.hdf5)"))
        positions_note = QLabel(
            "* Optional for a clean one-HDF5-file-per-scan-row raster; required "
            "for irregular scans so spatial features are linked correctly.")
        positions_note.setWordWrap(True)
        form.addRow("", positions_note)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse_row(self, edit, directory, name_filter=""):
        host = QWidget(); row = QHBoxLayout(host); row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse...")
        def browse():
            if directory:
                path = QFileDialog.getExistingDirectory(self, "Choose project parent", edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Choose file", edit.text(), name_filter)
            if path:
                edit.setText(path)
        button.clicked.connect(browse)
        row.addWidget(edit, 1); row.addWidget(button)
        return host

    def _validate(self):
        if not self.name.text().strip():
            QMessageBox.information(self, "Project name", "Enter a project name.")
            return
        if not Path(self.parent_dir.text()).is_dir():
            QMessageBox.information(self, "Project parent", "Choose an existing parent folder.")
            return
        if not Path(self.tth.text()).is_file():
            QMessageBox.information(self, "2-theta map", "Choose the detector tth.tiff calibration.")
            return
        if self.positions.text().strip() and not Path(self.positions.text()).is_file():
            QMessageBox.information(self, "Positions", "The selected positions file does not exist.")
            return
        self.accept()

    def values(self):
        return {"name": self.name.text().strip(), "project_parent": self.parent_dir.text(),
                "tth_path": self.tth.text(),
                "positions_path": self.positions.text().strip() or None}


class RawScanWindow(QMainWindow):
    """Pre-project state: choose a raw scan and compute its detector grand sum."""

    def __init__(self, source=None, bin_size=3):
        super().__init__()
        self.bin_size = int(bin_size)
        self.scan_folder = None
        self.scan_name = None
        self.summed_image = None
        self.roi = None
        self.roi_item = None
        self.worker = None
        self.setWindowTitle("xrd-app ROI Feature")
        self.resize(1100, 850)
        host = QWidget(); lay = QVBoxLayout(host)
        lay.addWidget(QLabel(
            "Start directly from a raw Scan_NNNN folder. Compute the fully summed "
            "detector image now; a normal project is requested only when features "
            "are saved."))
        row = QHBoxLayout()
        self.path = QLineEdit()
        choose = QPushButton("Select scan or project folder...")
        choose.clicked.connect(self._choose)
        row.addWidget(self.path, 1); row.addWidget(choose)
        lay.addLayout(row)
        self.compute = QPushButton("Compute NPZ sum")
        self.compute.clicked.connect(self._compute)
        lay.addWidget(self.compute)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.detector = DetectorView()
        self.detector.drag_enabled = True
        self.detector.set_drag_callback(self._roi_selected)
        self.detector.setVisible(False)
        lay.addWidget(self.detector, 1)
        self.save_feature = QPushButton("Save selected feature...")
        self.save_feature.setEnabled(False)
        self.save_feature.clicked.connect(self._save_selected)
        lay.addWidget(self.save_feature)
        self.status = QLabel("Select a folder to begin."); self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch()
        self.setCentralWidget(host)
        if source:
            self._open_source(source)

    def _choose(self):
        path = QFileDialog.getExistingDirectory(self, "Select project or Scan_NNNN folder",
                                                self.path.text() or str(Path.home()))
        if path:
            self._open_source(path)

    def _open_source(self, path):
        path = Path(path).resolve()
        self.path.setText(str(path))
        if workspace.is_project(path):
            self._open_project(path)
            return
        try:
            found = io.discover_scans(path, deep=False)
            if len(found) != 1:
                raise ValueError(f"Choose one scan folder; found {len(found)} scans")
            self.scan_folder = Path(found[0]["dir"])
            self.scan_name = found[0]["name"]
            self.status.setText(
                f"Raw scan ready: {self.scan_name}, {found[0]['n_files']} files, "
                f"~{found[0]['n_frames']} frames. Press Compute NPZ sum.")
            self.compute.setEnabled(True)
        except Exception as exc:
            self.status.setText(f"Could not open folder: {exc}")

    def _open_project(self, root, scan=None):
        window = ROIShapeWindow(root, scan=scan, bin_size=self.bin_size)
        window.setAttribute(window.WA_DeleteOnClose, False)
        self._next_window = window
        window.show()
        self.close()

    def _compute(self):
        if self.scan_folder is None:
            return
        self.compute.setEnabled(False); self.progress.setVisible(True); self.progress.setValue(0)
        self.status.setText("Summing raw detector frames...")
        self.worker = _RawSumWorker(self.scan_folder)
        self.worker.progress.connect(lambda i, n: self.progress.setValue(int(100 * i / n) if n else 0))
        self.worker.failed.connect(self._sum_failed)
        self.worker.complete.connect(self._sum_complete)
        self.worker.start()

    def _sum_failed(self, message):
        self.compute.setEnabled(True)
        self.status.setText(f"Could not compute sum: {message}")

    def _sum_complete(self, image):
        self.summed_image = image
        try:
            np.savez_compressed(self.scan_folder / "reflection_sum.npz",
                                image=image.astype(np.float32), is_raw=True, max_bins=0)
        except OSError:
            pass
        self.progress.setValue(100)
        finite = image[np.isfinite(image)]
        vmin, vmax = np.percentile(finite, [2, 99.7]) if finite.size else (0, 1)
        self.detector.show_image(image, float(vmin), float(vmax), "inferno")
        self.detector.set_title(f"{self.scan_name}: fully summed detector image")
        self.detector.setVisible(True)
        self.compute.setEnabled(True)
        self.status.setText(
            "Full detector sum is ready. Drag around one feature; saving it will "
            "ask where to create the project.")

    def _roi_selected(self, x0, y0, x1, y1, rect):
        if self.roi_item is not None:
            self.roi_item.remove()
        self.roi_item = rect
        self.roi = roi_map.normalize_roi((x0, y0, x1, y1))
        self.save_feature.setEnabled(True)
        self.status.setText(
            f"Selected detector ROI {self.roi}. Press Save selected feature to "
            "create a project and run peak/shape finding.")

    def _save_selected(self):
        if self.roi is None:
            return
        dialog = CreateProjectDialog(self.scan_name, self.scan_folder, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            from ..core.roi_project import create_from_scan
            root, scan = create_from_scan(
                self.scan_folder, bin_size=self.bin_size, summed_image=self.summed_image,
                **dialog.values())
        except Exception as exc:
            QMessageBox.critical(self, "Could not create project", f"{type(exc).__name__}: {exc}")
            return
        window = ROIShapeWindow(root, scan=scan, bin_size=self.bin_size)
        window.roi = self.roi
        window.roi_label.setText(", ".join(str(v) for v in self.roi))
        window._draw_roi()
        self._next_window = window
        window.show()
        self.close()


def launch(source=None, bin_size=3):
    from ..app import _harden_env_for_remote_x
    _harden_env_for_remote_x()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = RawScanWindow(source=source, bin_size=bin_size)
    window.show()
    return app.exec_()
