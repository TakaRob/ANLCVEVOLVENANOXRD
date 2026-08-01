"""Headless tests for GUI resource lifecycle helpers."""

import os
import json
import shutil
import sys
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QProcess, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QLabel, QListWidget, QMainWindow, QPlainTextEdit,
    QProgressBar, QPushButton, QWidget,
)

from xrd_app.app import MainWindow
from xrd_app.gui.roi_shape import ROIShapeWindow
from xrd_app.gui import lifecycle
from xrd_app.gui.lifecycle import dispose_widget, start_process, stop_process, stop_thread
from xrd_app.tabs._console import JobConsole


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


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
