"""Headless tests for GUI resource lifecycle helpers."""

import os
import json
import shutil
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QProcess, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QLabel, QListWidget, QMainWindow, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QWidget,
)

from xrd_app import workspace
from xrd_app.app import MainWindow
from xrd_app.config import DataManager
from xrd_app.gui.roi_shape import ROIShapeWindow

from xrd_app.gui.device_map import QRangeSlider
from xrd_app.gui.labeling import LabelingTool
from xrd_app.gui import lifecycle
from xrd_app.gui.lifecycle import dispose_widget, start_process, stop_process, stop_thread
from xrd_app.tabs import _console
from xrd_app.tabs._console import JobConsole


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_startup_restores_last_project_and_gui_state(tmp_path, monkeypatch):
    _app()
    project = tmp_path / "project"
    metadata = project / "Metadata"
    metadata.mkdir(parents=True)
    (project / "config.yaml").write_text("name: restored project\n")
    (metadata / "gui_state.json").write_text(json.dumps({
        "active_scan": "Scan_0002",
        "bin_size": 5,
        "current_tab": 0,
    }))
    settings_dir = tmp_path / ".settings"
    monkeypatch.setattr("xrd_app.workspace.SETTINGS_DIR", settings_dir)
    monkeypatch.setattr("xrd_app.workspace.SETTINGS_PATH", settings_dir / "settings.json")
    monkeypatch.setattr("xrd_app.app._discover_tabs", lambda only=None: [])
    monkeypatch.setattr("xrd_app.app.DataManager.discover_scans",
                        lambda self, selected_only=False: ["Scan_0001", "Scan_0002"])
    workspace.set_last_project(project)

    window = MainWindow()

    assert window.project_root == str(project.resolve())
    assert window.scan == "Scan_0002"
    assert window.bin_size == 5
    window.close()


def test_tab_bin_context_persists_and_drives_scan_rebuild(tmp_path, monkeypatch):
    _app()
    (tmp_path / "Metadata").mkdir()
    (tmp_path / "config.yaml").write_text("name: GUI test\n")
    built = []

    class Tab(QWidget):
        bin_size_changed = pyqtSignal(int)

        def __init__(self, bin_size):
            super().__init__()
            self._bin_size = bin_size

        def current_bin_size(self):
            return self._bin_size

        def select_bin(self, bin_size):
            self._bin_size = bin_size
            self.bin_size_changed.emit(bin_size)

    def make_tab(project_root=".", scan=None, bin_size=3):
        built.append((scan, bin_size))
        container = QWidget()
        container._embedded_window = Tab(bin_size)
        return container

    module = SimpleNamespace(__name__="test_bin_tab", make_tab=make_tab)
    meta = {"title": "Bin tab", "order": 1, "scan_dependent": True}
    monkeypatch.setattr("xrd_app.app._discover_tabs", lambda only=None: [(module, meta)])
    monkeypatch.setattr("xrd_app.app.DataManager.discover_scans",
                        lambda self, selected_only=False: ["Scan_0001", "Scan_0002"])

    window = MainWindow(tmp_path, scan="Scan_0001", bin_size=3, fresh=True)
    first = window._content[0]._embedded_window
    first.select_bin(5)

    assert window.bin_size == 5
    assert json.loads((tmp_path / "Metadata" / "gui_state.json").read_text())["bin_size"] == 5

    window.scan_combo.setCurrentText("Scan_0002")

    assert built == [("Scan_0001", 3), ("Scan_0002", 5)]
    assert window._content[0]._embedded_window.current_bin_size() == 5
    window.close()


def test_feature_size_slider_uses_logarithmic_mapping():
    _app()
    slider = QRangeSlider(1, 100, log_scale=True)
    slider.resize(216, slider.height())  # 200 px track plus 8 px margins

    assert slider._val_to_x(10) == 108
    assert slider._x_to_val(108) == 10


def test_range_slider_remains_linear_by_default():
    _app()
    slider = QRangeSlider(1, 100)
    slider.resize(216, slider.height())

    assert slider._x_to_val(108) in (50, 51)


def test_dispose_widget_closes_before_deferred_delete():
    _app()
    closed = []

    class Window(QWidget):
        def closeEvent(self, event):  # noqa: N802
            closed.append(True)
            super().closeEvent(event)

    dispose_widget(Window())
    assert closed == [True]


def test_stop_thread_requests_interruption_and_waits():
    class Worker(QThread):
        def run(self):
            while not self.isInterruptionRequested():
                self.msleep(1)

    worker = Worker()
    worker.start()
    assert stop_thread(worker)
    assert not worker.isRunning()


def test_start_process_uses_setsid_when_available(monkeypatch):
    process = QProcess()
    calls = []
    monkeypatch.setattr(
        lifecycle.shutil, "which",
        lambda name: "/usr/bin/setsid" if name == "setsid" else name)
    monkeypatch.setattr(process, "start", lambda program, args: calls.append((program, args)))

    start_process(process, "/some/python", ["-m", "xrd_app.cli"])

    assert calls == [("/usr/bin/setsid", ["/some/python", "-m", "xrd_app.cli"])]
    assert process.property(lifecycle._PROCESS_GROUP_PROPERTY) is True


def test_start_process_falls_back_without_setsid(monkeypatch):
    process = QProcess()
    calls = []
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: None)
    monkeypatch.setattr(process, "start", lambda program, args: calls.append((program, args)))

    start_process(process, "/some/python", ["--version"])

    assert calls == [("/some/python", ["--version"])]
    assert process.property(lifecycle._PROCESS_GROUP_PROPERTY) is False


@pytest.mark.skipif(os.name != "posix" or shutil.which("setsid") is None,
                    reason="requires POSIX process groups and setsid")
def test_stop_process_cleans_term_resistant_descendant():
    process = QProcess()
    child_code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        "print(p.pid,flush=True);"
        "time.sleep(60)"
    )
    start_process(process, sys.executable, ["-c", parent_code])
    assert process.waitForReadyRead(3000)
    child_pid = int(bytes(process.readLine()).decode().strip())

    assert stop_process(process, timeout_ms=200)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"descendant process {child_pid} survived group cancellation")


def _roi_window():
    window = ROIShapeWindow.__new__(ROIShapeWindow)
    QMainWindow.__init__(window)
    window.sum_process = None
    window._detect_process = None
    window._batch_process = None
    window._save_process = None
    window._output_buffers = {}
    window.pending = []
    window.pending_list = QListWidget()
    window.run_shapes_btn = QPushButton()
    window.run_btn = QPushButton()
    window.detect_btn = QPushButton()
    window.remove_btn = QPushButton()
    window.cancel_btn = QPushButton()
    window.bin_combo = QComboBox()
    window.compute_sum_btn = QPushButton()
    window.progress = QProgressBar()
    window.status = QLabel()
    window.log = QPlainTextEdit()
    window._grand_sum_cached = lambda: False
    return window


def test_labeling_finds_nearest_readable_built_bins_without_raw_frames(tmp_path):
    binned = tmp_path / "Binned" / "Scan_0007"
    binned.mkdir(parents=True)
    with h5py.File(binned / "xrd_3x3_bins.h5", "w") as handle:
        handle.create_dataset("0_0", data=np.ones((2, 2)))
    with h5py.File(binned / "xrd_5x5_bins.h5", "w") as handle:
        handle.create_dataset("metadata", data=np.ones(2))
    (binned / "xrd_2x2_bins.h5").write_text("unreadable")

    window = LabelingTool.__new__(LabelingTool)
    window.dm = DataManager(tmp_path, scan=7)
    window.bin_size = 1

    assert window._available_binned_size(exclude=1) == 3

    loaded = []

    def load(size):
        if size == 1:
            raise RuntimeError("raw share unavailable")
        loaded.append(size)
        window.bin_size = size

    window._load_bin_data_for_size = load
    window._load_initial_bin_data()
    assert loaded == [3]
    assert window.bin_size == 3
    assert "opened built 3x3" in window._startup_source_note


def test_roi_job_controls_and_overlap_guard(monkeypatch):
    _app()
    window = _roi_window()
    process = QProcess(window)
    window._batch_process = process
    window._update_job_controls()

    assert window.cancel_btn.isEnabled()
    assert not window.run_shapes_btn.isEnabled()
    assert not window.run_btn.isEnabled()
    assert not window.compute_sum_btn.isEnabled()

    started = []
    monkeypatch.setattr("xrd_app.gui.roi_shape.start_process", lambda *args: started.append(args))
    window._run_detected_shapes()
    assert started == []
    assert "already active" in window.status.text()

    window._batch_process = None
    window._update_job_controls()
    assert not window.cancel_btn.isEnabled()
    assert window.run_shapes_btn.isEnabled()


def test_roi_batch_malformed_output_cleans_running_entries(tmp_path):
    _app()
    window = _roi_window()
    process = QProcess(window)
    entry = {"roi": (1, 2, 3, 4), "status": "running", "process": process}
    window.pending = [entry]
    window._batch_process = process
    window._output_buffers[process] = ""
    preview = tmp_path / "preview.json"
    preview.write_text("not json")

    window._on_batch_finished(process, [entry], preview, 0, QProcess.NormalExit)

    assert entry["status"] == "failed"
    assert entry["process"] is None
    assert not window.cancel_btn.isEnabled()
    assert "Could not load" in window.status.text()


def test_roi_batch_buffers_split_progress_lines(monkeypatch):
    _app()
    window = _roi_window()
    process = QProcess(window)
    chunks = iter((b"PROGRESS 1/", b"4\nmessage\n"))
    monkeypatch.setattr(process, "readAllStandardOutput", lambda: next(chunks))
    window._output_buffers[process] = ""

    window._on_batch_output(process)
    assert window.progress.value() == -1
    window._on_batch_output(process)

    assert window.progress.value() == 25
    assert window.log.toPlainText() == "message"


def test_roi_crop_grid_shape_uses_metadata_then_loaded_keys(monkeypatch):
    _app()
    window = _roi_window()
    window.bin_size = 3
    window.scan = "Scan_0203"
    window.preview_feature = None
    window.spatial_keys = []
    window.heatmap = SimpleNamespace(_grid_data=None)
    requested = []

    class DM:
        def grid_mapping(self, **kwargs):
            requested.append(kwargs)
            return "grid.json"

    window.dm = DM()
    monkeypatch.setattr(
        "xrd_app.gui.roi_shape.io.validate_grid_mapping_bin_size",
        lambda path, size: {"n_bin_rows": 20, "n_bin_cols": 30})
    assert window._active_grid_shape() == (20, 30)
    assert requested == [{"bin_size": 3, "scan": "Scan_0203"}]

    monkeypatch.setattr(
        "xrd_app.gui.roi_shape.io.validate_grid_mapping_bin_size",
        lambda path, size: (_ for _ in ()).throw(FileNotFoundError()))
    window.spatial_keys = ["0_0", "4_7", "bad"]
    assert window._active_grid_shape() == (5, 8)


def test_roi_new_feature_draws_full_grid_outline(monkeypatch):
    _app()
    window = _roi_window()
    window.bin_size = 3
    window.preview_feature = None
    window.sample_grid_rect = None
    drawn = []
    window.heatmap = SimpleNamespace(
        img=SimpleNamespace(clear=lambda: None), _grid_data=None,
        fit_to_rect=lambda *args: drawn.append(("fit", args)),
        plot=SimpleNamespace(setTitle=lambda *args, **kwargs: None),
        set_markers=lambda *args: None,
    )
    monkeypatch.setattr(window, "_active_grid_shape", lambda: (20, 30))
    monkeypatch.setattr(window, "_draw_sample_grid_outline",
                        lambda rows, cols: drawn.append(("outline", (rows, cols))))
    monkeypatch.setattr(window, "_draw_sample_crop", lambda: None)

    window._render_feature()

    assert window.heatmap._grid_data.shape == (20, 30)
    assert ("outline", (20, 30)) in drawn
    assert ("fit", (-0.5, -0.5, 30, 20)) in drawn


def test_roi_manual_crop_interprets_raw_and_binned_coordinates(monkeypatch):
    _app()
    window = _roi_window()
    window.bin_size = 3
    entry = {"roi": (1, 2, 3, 4), "sample_crop_raw": None, "status": "detected"}
    window.pending = [entry]
    window.pending_list.addItem("ROI")
    window.pending_list.setCurrentRow(0)
    window.crop_coords = QComboBox()
    window.crop_coords.addItems(["Raw coordinates", "Binned coordinates"])
    window.crop_inputs = [QSpinBox() for _ in range(4)]
    window.crop_label = QLabel()
    window.apply_crop_btn = QPushButton()
    window.clear_crop_btn = QPushButton()
    window.crop_sample_cb = SimpleNamespace(
        setChecked=lambda checked: None, isChecked=lambda: True)
    window.heatmap = SimpleNamespace(vb=object(), drag_enabled=False)
    window.sample_crop_rect = None
    window.preview_feature = None
    monkeypatch.setattr(window, "_active_grid_shape", lambda: (20, 30))
    monkeypatch.setattr(window, "_draw_sample_crop", lambda: None)
    monkeypatch.setattr(window, "_render_feature", lambda: None)

    for field, value in zip(window.crop_inputs, (3, 6, 12, 15)):
        field.setValue(value)
    window._apply_manual_crop()
    assert entry["sample_crop_raw"] == (3, 6, 12, 15)
    assert window._active_sample_crop(entry) == (1, 2, 4, 5)

    window.crop_coords.setCurrentIndex(1)
    for field, value in zip(window.crop_inputs, (2, 3, 7, 9)):
        field.setValue(value)
    window._apply_manual_crop()
    assert entry["sample_crop_raw"] == (6, 9, 21, 27)
    assert window._active_sample_crop(entry) == (2, 3, 7, 9)


def test_roi_ready_feature_recrops_without_clearing_scan(monkeypatch):
    _app()
    window = _roi_window()
    entry = {
        "roi": (1, 2, 3, 4), "sample_crop_raw": (3, 6, 12, 15),
        "status": "ready", "feature": {"intensity_profile": {"0_0": {}}},
    }
    window.pending = [entry]
    window.pending_list.addItem("ROI")
    window.pending_list.setCurrentRow(0)
    window.crop_coords = QComboBox()
    window.crop_coords.addItems(["Raw coordinates", "Binned coordinates"])
    window.crop_inputs = [QSpinBox() for _ in range(4)]
    window.crop_label = QLabel()
    window.apply_crop_btn = QPushButton()
    window.clear_crop_btn = QPushButton()
    window.crop_sample_cb = SimpleNamespace(isChecked=lambda: True)
    window.heatmap = SimpleNamespace(drag_enabled=True)
    window.status = QLabel()
    monkeypatch.setattr(
        window, "_active_sample_crop",
        lambda selected=None: (1, 2, 4, 5)
        if (selected or entry).get("sample_crop_raw") is not None else None)
    monkeypatch.setattr(window, "_draw_sample_crop", lambda: None)
    monkeypatch.setattr(window, "_render_feature", lambda: None)

    window._update_crop_controls()
    assert window.heatmap.drag_enabled
    assert window.apply_crop_btn.isEnabled()
    assert window.clear_crop_btn.isEnabled()
    assert "locked" not in window.crop_label.text()

    feature = entry["feature"]
    window._clear_sample_crop()
    assert entry["sample_crop_raw"] is None
    assert entry["feature"] is feature
    assert entry["status"] == "ready"


def test_roi_heatmap_click_selects_requested_bin(monkeypatch):
    _app()
    window = _roi_window()
    window.spatial_keys = ["0_0", "2_3", "4_1"]
    window.spatial_index = 0
    window.spatial_label = QLabel()
    window.image_mode = QComboBox()
    window.image_mode.addItems(["sum", "bin"])
    monkeypatch.setattr(window, "_ensure_source", lambda: object())
    monkeypatch.setattr(window, "_load_detector_image", lambda *_: None)

    window._heatmap_clicked(2, 3)

    assert window.spatial_index == 1
    assert window.image_mode.currentIndex() == 1
    assert "2_3" in window.spatial_label.text()


def test_job_console_adds_transient_environment(monkeypatch):
    _app()
    console = JobConsole()
    captured = {}

    def fake_start(process, program, arguments=()):
        captured["key"] = process.processEnvironment().value("ARGO_API_KEY")

    monkeypatch.setattr(_console, "start_process", fake_start)
    console.run(["--help"], env={"ARGO_API_KEY": "temporary-secret"})

    assert captured["key"] == "temporary-secret"
    assert "temporary-secret" not in console.log.toPlainText()
    console._proc = None


def test_job_console_handles_process_start_failure(monkeypatch):
    app = _app()
    console = JobConsole()
    monkeypatch.setattr(sys, "executable", "/definitely/missing/python")

    console.run(["--help"])
    deadline = time.monotonic() + 2
    while console._proc is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert console._proc is None
    assert "failed to start" in console.log.toPlainText()
    assert console.status.text().startswith("failed")
